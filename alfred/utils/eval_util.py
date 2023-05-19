import os
import json
import queue
import torch
import shutil
import filelock
import numpy as np
import sys

from PIL import Image
from termcolor import colored

from alfred.gen import constants
from alfred.env.thor_env import ThorEnv
from alfred.nn.enc_visual import FeatureExtractor
from alfred.utils import data_util, model_util
import Levenshtein

THR_SCORE = 0.5


def setup_scene(env, traj_data, reward_type='dense', test_split=False):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name, silent=True)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))
    # setup task for reward
    if not test_split:
        env.set_task(traj_data, reward_type=reward_type)


def load_agent(model_path, dataset_info, device):
    '''
    load a pretrained agent and its feature extractor
    '''
    learned_model, _ = model_util.load_model(model_path, device)
    model = learned_model.model
    model.eval()
    model.args.device = device
    extractor = FeatureExtractor(
        archi=dataset_info['visual_archi'],
        device=device,
        checkpoint=dataset_info['visual_checkpoint'],
        compress_type=dataset_info['compress_type'])
    return model, extractor


def load_object_predictor(args):
    if args.object_predictor is None:
        return None
    return FeatureExtractor(
        archi='maskrcnn', device=args.device,
        checkpoint=args.object_predictor, load_heads=True)


def worker_loop(
        evaluate_function,
        dataset,
        trial_queue,
        log_queue,
        args,
        cuda_device=None):
    '''
    evaluation loop
    '''
    if cuda_device:
        torch.cuda.set_device(cuda_device)
        args.device = 'cuda:{}'.format(cuda_device)
    # start THOR
    env = ThorEnv(x_display=args.x_display)
    # master may ask to evaluate different models
    model_path_loaded = None
    object_predictor = load_object_predictor(args)

    if args.num_workers == 0:
        num_success, num_trials_done, num_trials = 0, 0, trial_queue.qsize()
    try:
        while True:
            trial_uid, dataset_idx, model_path = trial_queue.get(timeout=3)
            if model_path != model_path_loaded:
                if model_path_loaded is not None:
                    del model, extractor
                    torch.cuda.empty_cache()
                model, extractor = load_agent(
                    model_path,
                    dataset.dataset_info,
                    args.device)
                dataset.vocab_translate = model.vocab_out
                model_path_loaded = model_path
            log_entry = evaluate_function(
                env, model, dataset, extractor, trial_uid, dataset_idx, args,
                object_predictor)
            if (args.debug or args.num_workers == 0) and 'success' in log_entry:
                if 'subgoal_action' in log_entry:
                    trial_type = log_entry['subgoal_action']
                else:
                    trial_type = 'full task'
                print(colored('Trial {}: {} ({})'.format(
                    trial_uid, 'success' if log_entry['success'] else 'fail',
                    trial_type), 'green' if log_entry['success'] else 'red'))
            if args.num_workers == 0 and 'success' in log_entry:
                num_trials_done += 1
                num_success += int(log_entry['success'])
                print('{:4d}/{} trials are done (current SR = {:.1f})'.format(
                    num_trials_done, num_trials, 100 * num_success / num_trials_done))
            log_queue.put((log_entry, trial_uid, model_path))
    except queue.Empty:
        pass
    # stop THOR
    env.stop()


def get_model_paths(args):
    '''
    check which models need to be evaluated
    '''
    model_paths = []
    if args.eval_range is None:
        # evaluate only the latest checkpoint
        model_paths.append(
            os.path.join(constants.ET_LOGS, args.exp, args.checkpoint))
    else:
        # evaluate a range of epochs
        for model_epoch in range(*args.eval_range):
            model_path = os.path.join(
                constants.ET_LOGS, args.exp,
                'model_{:02d}.pth'.format(model_epoch))
            if os.path.exists(model_path):
                model_paths.append(model_path)
    for idx, model_path in enumerate(model_paths):
        if os.path.islink(model_path):
            model_paths[idx] = os.readlink(model_path)
    if len(model_paths) == 0:
        raise ValueError('No models are found for evaluation')
    return model_paths


def get_sr(eval_json, eval_type):
    if eval_type == 'task':
        return eval_json['results']['all']['success']['success_rate']
    sr_sum, sr_count = 0, 0
    for _, sg_dict in sorted(eval_json['results'].items()):
        sr_sum += sg_dict['successes']
        sr_count += sg_dict['evals']
    return sr_sum / sr_count


def overwrite_eval_json(results, model_path, eval_type, args, dataset_id):
    '''
    append results to an existing eval.json or create a new one
    '''
    # eval_json: eval_epoch / eval_split / {'subgoal','task'} / {'normal','fast_epoch'}
    # see if eval.json file alredy existed
    eval_json = {}
    eval_json_path = os.path.join(os.path.dirname(model_path), 'eval.json')
    lock = filelock.FileLock(eval_json_path + '.lock')
    with lock:
        if os.path.exists(eval_json_path):
            with open(eval_json_path, 'r') as eval_json_file:
                eval_json = json.load(eval_json_file)
    eval_epoch = os.path.basename(model_path)
    if eval_epoch not in eval_json:
        eval_json[eval_epoch] = {}
    if dataset_id not in eval_json[eval_epoch]:
        eval_json[eval_epoch][dataset_id] = {}
    if eval_type not in eval_json[eval_epoch][dataset_id]:
        eval_json[eval_epoch][dataset_id][eval_type] = {}
    eval_mode = 'normal' if not args.fast_epoch else 'fast_epoch'
    if eval_mode in eval_json[eval_epoch][dataset_id][eval_type]:
        print('WARNING: the evaluation was already done')
        sr_new = len(results['successes']) / (len(results['successes']) + len(results['failures']))
        prev_res = eval_json[eval_epoch][dataset_id][eval_type][eval_mode]
        sr_old = len(prev_res['successes']) / (len(prev_res['successes']) + len(prev_res['failures']))
        print('Previous success rate = {:.1f}, new success rate = {:.1f}'.format(
            100 * sr_old, 100 * sr_new))
    eval_json[eval_epoch][dataset_id][eval_type][eval_mode] = results
    # make a backup copy of eval.json file before writing
    save_with_backup(eval_json, eval_json_path, lock)
    print('Evaluation is saved to {}'.format(eval_json_path))


def save_with_backup(obj, file_path, lock):
    with lock:
        # make a backup copy of results file before writing and swipe it after
        file_path_back = file_path + '.back'
        file_path_back_back = file_path + '.back.back'
        # always remove the second backup
        if os.path.exists(file_path_back_back):
            os.remove(file_path_back_back)
        # rename the first backup to the second one
        if os.path.exists(file_path_back):
            os.rename(file_path_back, file_path_back_back)
        # put the original file as the first backup
        if os.path.exists(file_path):
            os.rename(file_path, file_path_back)
        # write the updated content to the file path
        with open(file_path, 'w') as file_opened:
            json.dump(obj, file_opened, indent=2, sort_keys=True)


def disable_lock_logs():
    lock_logger = filelock.logger()
    lock_logger.setLevel(30)

def get_closest_object(obj, obj_list):
    max_sim = 0
    max_idx = 0
    for ref in obj_list:
        if Levenshtein.ratio(obj, ref) > max_sim:
            max_sim =  Levenshtein.ratio(obj, ref) 
            # max_idx = obj_list.index(ref)
    return ref #, max_idx

def extract_rcnn_pred(_class, obj_predictor, env, verbose=False, is_idx=True):
    '''
    extract a pixel mask using a pre-trained MaskRCNN
    '''

    env.reset_angle()
    for rotate_action in ["RotateRight_90"] * 4:  # 四方を仔細にチェック
        _ = env.va_interact(rotate_action, interact_mask=None,smooth_nav=False) 
        for look_action in ["LookUp_15", "LookDown_15","LookDown_15","LookUp_15"]:
            _ = env.va_interact(look_action, interact_mask=None,smooth_nav=False)
            rcnn_pred = obj_predictor.predict_objects(Image.fromarray(env.last_event.frame))
            obj_list = obj_predictor.vocab_obj.to_dict()["index2word"]
            if not is_idx and _class not in obj_list:
                class_name = get_closest_object(_class, obj_list)
                class_idx = obj_predictor.vocab_obj.word2index(class_name)
                print("<Could not find class>", _class, class_name)
            else:
            # import sys; sys.exit()
                class_name = obj_predictor.vocab_obj.index2word(_class) if is_idx else _class
                class_idx = obj_predictor.vocab_obj.word2index(_class) if not is_idx else _class

            candidates = list(filter(lambda p: p.label == class_name and p.score > THR_SCORE, rcnn_pred))
            # print(f"detected bbox ({class_name};{[str(int(c.score * 100)) + '%' for c in candidates]}):",[c.box for c in candidates])
            target = None
            if verbose:
                visible_objs = [
                    obj for obj in env.last_event.metadata['objects']
                    if obj['visible'] and obj['objectId'].startswith(class_name + '|')]
                print('Agent prediction = {}, detected {} objects (visible {})'.format(
                    class_name, len(candidates), len(visible_objs)))
            if len(candidates) > 0:
                if env.last_interaction[0] == class_idx and env.last_interaction[1] is not None:
                    last_center = np.array(env.last_interaction[1].nonzero()).mean(axis=1)
                    cur_centers = np.array(
                        [np.array(c.mask[0].nonzero()).mean(axis=1) for c in candidates])
                    distances = ((cur_centers - last_center)**2).sum(axis=1)
                    index = np.argmin(distances)
                    mask = candidates[index].mask[0]
                    target = candidates[index]
                else:
                    index = np.argmax([p.score for p in candidates])
                    mask = candidates[index].mask[0]
                    target = candidates[index]
            else:
                mask = None
            if mask is not None:
                return mask, target
    return mask, target

def move_behind(env,args):
    actions = ["RotateRight_90","RotateRight_90","MoveAhead_25","RotateRight_90","RotateRight_90"]
    for action in actions:
        step_success, _, target_instance_id, err, api_action = env.va_interact(
                action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
            

def rule_based_planner(action,llm_data,obj_predictor,m_pred,env,m_out,model,args):
    target = None
    is_rule_based_target = True
    will_execute = True

    llm_action = llm_data[-1][0]
    llm_target = llm_data[-1][1]
    llm_destination = llm_data[-1][2]

    if llm_action == "PickupObject":
        obj_list = obj_predictor.vocab_obj.to_dict()["index2word"]  
        if llm_target not in obj_list:
            class_name = get_closest_object(llm_target, obj_list)
            obj = obj_predictor.vocab_obj.word2index(class_name) if model_util.has_interaction(llm_action) else None
            print("<Could not find class>", llm_target, class_name)
        else:
            obj = obj_predictor.vocab_obj.word2index(llm_target) if model_util.has_interaction(llm_action) else None
        mask, target = extract_rcnn_pred(llm_target, obj_predictor, env, verbose=False, is_idx=False)
        if mask is not None:
            m_pred['mask_rcnn'] = mask
            action = "PickupObject"
            action = obstruction_detection(action, env, m_out, model.vocab_out, args.debug)
            m_pred['action'] = action
            print("== rule-based action selection ==     ", end="")
            print(f"target: {llm_target}, action: {llm_action}")
        else:
            will_execute = False

    elif llm_action == "PutObject":
        # obj = vocab['word'].word2index(llm_destination.lower()) if model_util.has_interaction(action) else None
        obj_list = obj_predictor.vocab_obj.to_dict()["index2word"]  
        if llm_destination not in obj_list:
            class_name = get_closest_object(llm_destination, obj_list)
            obj = obj_predictor.vocab_obj.word2index(class_name) if model_util.has_interaction(llm_action) else None
            print("<Could not find class>", llm_destination, class_name)
        else:
            obj = obj_predictor.vocab_obj.word2index(llm_destination) if model_util.has_interaction(llm_action) else None

        mask, target = extract_rcnn_pred(llm_destination, obj_predictor, env, verbose=False, is_idx=False)
        if mask is not None:
            m_pred['mask_rcnn'] = mask
            action = "PutObject"
            action = obstruction_detection(action, env, m_out, model.vocab_out, args.debug)
            m_pred['action'] = action
            print("== rule-based action selection ==     ", end="")
            print(f"destination: {llm_destination}, action: {llm_action}")
        else:
            will_execute = False

    elif llm_action == "OpenObject" or llm_action == "CloseObject" :
        if llm_destination != "None":
            moveto_obj = llm_destination
        elif llm_target != "None":       
            moveto_obj = llm_target
        obj_list = obj_predictor.vocab_obj.to_dict()["index2word"]  
        if llm_target not in obj_list:
            class_name = get_closest_object(moveto_obj, obj_list)
            obj = obj_predictor.vocab_obj.word2index(class_name) if model_util.has_interaction(llm_action) else None
            print("<Could not find class>", moveto_obj, class_name)
        else:
            obj = obj_predictor.vocab_obj.word2index(moveto_obj) if model_util.has_interaction(llm_action) else None
        
        mask, target = extract_rcnn_pred(moveto_obj, obj_predictor, env, verbose=False, is_idx=False)
        if mask is not None:
            m_pred['mask_rcnn'] = mask
            action = llm_action
            m_pred['action'] = action
            print("== rule-based action selection ==     ", end="")
            print(f"destination: {moveto_obj}, action: {llm_action}")
        else:
            will_execute = False
    
    elif llm_action == "ToggleObjectOn" or llm_action == "ToggleObjectOff":
        if llm_target != "None":       
            moveto_obj = llm_target
        elif llm_destination != "None":
            moveto_obj = llm_destination
        obj_list = obj_predictor.vocab_obj.to_dict()["index2word"]  
        if llm_target not in obj_list:
            class_name = get_closest_object(moveto_obj, obj_list)
            obj = obj_predictor.vocab_obj.word2index(class_name) if model_util.has_interaction(llm_action) else None
            print("<Could not find class>", moveto_obj, class_name)
        else:
            obj = obj_predictor.vocab_obj.word2index(moveto_obj) if model_util.has_interaction(llm_action) else None
        
        mask, target = extract_rcnn_pred(moveto_obj, obj_predictor, env, verbose=False, is_idx=False)
        if mask is not None:
            m_pred['mask_rcnn'] = mask
            action = llm_action
            m_pred['action'] = action
            print("== rule-based action selection ==     ", end="")
            print(f"destination: {moveto_obj}, action: {llm_action}")
        else:
            will_execute = False
    else:
        is_rule_based_target = False
        will_execute = False

    return will_execute, is_rule_based_target, m_pred, target, obj, action, mask

# step and compute model confusion
def agent_step_mc(
        model, input_dict, vocab, prev_action, env, args, num_fails, obj_predictor, rcnn_pred=None, subgoal_instr=None, llm_data=None, subgoal_idx=None,action_idx=None):
    '''
    environment step based on model prediction
    '''
    # forward model
    with torch.no_grad():
        m_out = model.step(input_dict, vocab, prev_action=prev_action)
    
    mc_array = torch.nn.functional.softmax(m_out['action'], 2).topk(2)[0][0][0].cpu().detach().numpy()

    m_pred = model_util.extract_action_preds(
        m_out, model.pad, vocab['action_low'], clean_special_tokens=False)[0]
    action = m_pred['action']
    action = obstruction_detection(action, env, m_out, model.vocab_out, args.debug)
    
    if args.debug:
        print("Predicted action: {}".format(action))

    mask = None
    obj = m_pred['object'][0][0] if model_util.has_interaction(action) else None
    original_obj = obj_predictor.vocab_obj.index2word(obj) if obj else None

    # rule-based action selection
    for data in llm_data:
        if data[0] != "MoveTo":
            llm_target = data[1]
            llm_action = data[0]
            llm_destination = data[2]

    episode_end = False
    if action == constants.STOP_TOKEN:
        if llm_action == "MoveTo":
            episode_end = True
        else:
            action = "MoveAhead_25"
            action = obstruction_detection(action, env, m_out, model.vocab_out, args.debug)

    will_execute = True
    is_rule_based_target = False
    if prev_action != llm_action:
        will_execute, is_rule_based_target, m_pred, target, obj, action, mask \
            = rule_based_planner(action,llm_data,obj_predictor,m_pred,env,m_out,model,args)

    xx = obj_predictor.vocab_obj.index2word(obj) if obj else None
    print(f"{subgoal_idx+1:02}_{action_idx:03} -> subgoal : {subgoal_instr},  action: {action}, llm_output: {llm_data}, object: {xx}")

    # use the predicted action
    # episode_end = (action == constants.STOP_TOKEN)
    api_action = None
    # constants.TERMINAL_TOKENS was originally used for subgoal evaluation
    target_instance_id = ''

    # 前回のactionが所望のmanipulationかつ成功していたら強制stop
    if prev_action == llm_action and env.last_event.metadata['lastActionSuccess']:
        episode_end = True
        action = constants.STOP_TOKEN
        return episode_end, str(action), num_fails, target_instance_id, api_action, mc_array

    step_success = True

 
    if not episode_end:
        failed_rule = not will_execute
        if not is_rule_based_target or not (failed_rule and (action == "PickupObject" or action == "PutObject")): # TODO: ここの条件分岐チェック
            step_success, _, target_instance_id, err, api_action = env.va_interact(
                action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            env.last_interaction = (obj, mask)
        else:
            action = prev_action

        if not step_success:
            print(f"+++ ERROR: {err}")
            if target:
                box = target.box
                c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
                frame = env.last_event.frame
                H, W, _ = frame.shape
                centroid_x = (c1[0] + c2[0]) / 2
                print("frame",frame.shape,"target",c1,c2,centroid_x)
                action = []
                pos = ""
                if centroid_x < H / 4: # left 
                    actions = ["MoveAhead_25","RotateLeft_90"]
                    pos = "left"
                elif centroid_x > H * 3 / 4: # right
                    actions = ["MoveAhead_25","RotateRight_90"]
                    pos = "right"
                else: # center
                    actions = ["MoveAhead_25"]
                    pos = "center"
                
                failed_step = 0
                max_failed_step = 3
                while failed_step < max_failed_step:
                    for action in actions: # 目標領域の位置に従って，回転 or 前進
                        print(f"=== ERROR RECOVERY[({failed_step}/{max_failed_step})] (action: {action}, pos: {pos})===")
                        step_success, _, target_instance_id, err, api_action = env.va_interact(
                            action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                        env.last_interaction = (obj, mask)
                        if not step_success:
                            print("failed:",err)

                        if not step_success and action == "MoveAhead_25":
                            move_behind(env,args)
                            step_success, _, target_instance_id, err, api_action = env.va_interact(
                                action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                            env.last_interaction = (obj, mask)
                    
                    if pos == "center":
                        break

                    # 上記行動によって改善したか？
                    if prev_action != llm_action and pos != "center":
                        will_execute, is_rule_based_target, m_pred, target, obj, action, mask \
                            = rule_based_planner(action,llm_data,obj_predictor,m_pred,env,m_out,model,args)
                        step_success, _, target_instance_id, err, api_action = env.va_interact(
                            action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                        env.last_interaction = (obj, mask)
                        if is_rule_based_target and not step_success:
                            if pos == "left":
                                actions = ["RotateRight_90","MoveAhead_25","RotateLeft_90"]
                            elif pos == "right":
                                actions = ["RotateLeft_90","MoveAhead_25","RotateRight_90"]
                            failed_step += 1
                            if failed_step > max_failed_step:
                                break
                        else:
                            break

            num_fails += 1
            if num_fails >= args.max_fails:
                if args.debug:
                    print("Interact API failed {} times; latest error '{}'".format(
                        num_fails, err))
                episode_end = True
    return episode_end, str(action), num_fails, target_instance_id, api_action, mc_array

def agent_step(
        model, input_dict, vocab, prev_action, env, args, num_fails, obj_predictor):
    '''
    environment step based on model prediction
    '''
    # forward model
    with torch.no_grad():
        m_out = model.step(input_dict, vocab, prev_action=prev_action)
    
    m_pred = model_util.extract_action_preds(
        m_out, model.pad, vocab['action_low'], clean_special_tokens=False)[0]
    action = m_pred['action']
    if args.debug:
        print("Predicted action: {}".format(action))

    mask = None
    obj = m_pred['object'][0][0] if model_util.has_interaction(action) else None
    if obj is not None:
        # get mask from a pre-trained RCNN
        assert obj_predictor is not None
        mask, target = extract_rcnn_pred(
            obj, obj_predictor, env, args.debug)
        m_pred['mask_rcnn'] = mask
    # remove blocking actions
    action = obstruction_detection(
        action, env, m_out, model.vocab_out, args.debug)
    m_pred['action'] = action

    # use the predicted action
    episode_end = (action == constants.STOP_TOKEN)
    api_action = None
    # constants.TERMINAL_TOKENS was originally used for subgoal evaluation
    target_instance_id = ''
    if not episode_end:
        step_success, _, target_instance_id, err, api_action = env.va_interact(
            action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
        env.last_interaction = (obj, mask)
        if not step_success:
            num_fails += 1
            if num_fails >= args.max_fails:
                if args.debug:
                    print("Interact API failed {} times; latest error '{}'".format(
                        num_fails, err))
                episode_end = True
    return episode_end, str(action), num_fails, target_instance_id, api_action


def expert_step(action, masks, model, input_dict, vocab, prev_action, env, args):
    '''
    environment step based on expert action
    '''
    mask = masks.pop(0).float().numpy()[0] if model_util.has_interaction(
        action) else None
    # forward model
    if not args.no_model_unroll:
        with torch.no_grad():
            model.step(input_dict, vocab, prev_action=prev_action)
        prev_action = (action if not args.no_teacher_force else None)
    # execute expert action
    step_success, _, _, err, _ = env.va_interact(
        action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
    if not step_success:
        print("expert initialization failed")
        return True, prev_action
    # update transition reward
    _, _ = env.get_transition_reward()
    return False, prev_action


def get_observation(event, extractor, id=None, subgoal_idx=None, action_idx=None):
    '''
    get environment observation
    '''
    frames = extractor.featurize([Image.fromarray(event.frame)], batch=1)

    return frames

#追加
def get_observation_clip(event, extractor):
    '''
    get environment observation
    '''
    frames = extractor.featurize_clip([Image.fromarray(event.frame)])
    #変更
    # feat = data_util.extract_features(images, extractor)
    # feat_clip = data_util.extract_features_clip(images, extractor)
    return frames


def load_language(
        dataset, task, dataset_key, model_args, extractor, subgoal_idx=None,
        test_split=False):
    '''
    load language features from the dataset and unit-test the feature extractor
    '''
    # load language features
    feat_numpy = dataset.load_features(task, subgoal_idx=subgoal_idx)
    # test extractor with the frames
    if not test_split:
        frames_expert = dataset.load_frames(dataset_key)
        model_util.test_extractor(task['root'], extractor, frames_expert)
    if not test_split and 'frames' in dataset.ann_type:
        # frames will be used as annotations
        feat_numpy['frames'] = frames_expert
    _, input_dict, _ = data_util.tensorize_and_pad(
        [(task, feat_numpy)], model_args.device, dataset.pad)
    return input_dict

def load_language_qa(
        dataset, task, dataset_key, model_args, extractor, num_qa, subgoal_idx=None,
        test_split=False):
    '''
    load language features from the dataset and add the qa
    '''
    # load language features
    feat_numpy = dataset.load_features(task, subgoal_idx=subgoal_idx)
    feat_numpy['lang'] += num_qa
    # test extractor with the frames
    if not test_split:
        frames_expert = dataset.load_frames(dataset_key)
        model_util.test_extractor(task['root'], extractor, frames_expert)
    if not test_split and 'frames' in dataset.ann_type:
        # frames will be used as annotations
        feat_numpy['frames'] = frames_expert
    _, input_dict, _ = data_util.tensorize_and_pad(
        [(task, feat_numpy)], model_args.device, dataset.pad)
    return input_dict

def load_expert_actions(dataset, task, dataset_key, subgoal_idx):
    '''
    load actions and masks for expert initialization
    '''
    expert_dict = dict()
    expert_dict['actions'] = [
        a['discrete_action'] for a in task['plan']['low_actions']
        if a['high_idx'] < subgoal_idx]
    expert_dict['masks'] = dataset.load_masks(dataset_key)
    return expert_dict


def read_task_data(task, subgoal_idx=None):
    '''
    read data from the traj_json
    '''
    # read general task info
    repeat_idx = task['repeat_idx']
    task_dict = {'repeat_idx': repeat_idx,
                 'type': task['task_type'],
                 'task': '/'.join(task['root'].split('/')[-3:-1])}
    # read subgoal info
    if subgoal_idx is not None:
        task_dict['subgoal_idx'] = subgoal_idx
        task_dict['subgoal_action'] = task['plan']['high_pddl'][
            subgoal_idx]['discrete_action']['action']
    return task_dict


def obstruction_detection(action, env, m_out, vocab_out, verbose):
    '''
    change 'MoveAhead' action to a turn in case if it has failed previously
    '''
    if action != 'MoveAhead_25':
        return action
    if env.last_event.metadata['lastActionSuccess']:
        return action
    dist_action = m_out['action'][0][0].detach().cpu()
    idx_rotateR = vocab_out.word2index('RotateRight_90')
    idx_rotateL = vocab_out.word2index('RotateLeft_90')
    action = 'RotateLeft_90' if dist_action[idx_rotateL] > dist_action[idx_rotateR] else 'RotateRight_90'
    if verbose:
        print("Blocking action is changed to: {}".format(action))
    return action