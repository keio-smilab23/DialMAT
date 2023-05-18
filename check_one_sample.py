import argparse
import json
import logging
import os
from io import open

import colored_traceback.always
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from pathlib import Path

from alfred.data.preprocessor import *
from alfred.env.thor_env import ThorEnv
# from alfred.data.zoo.alfred import Dataset
from alfred.eval.eval_subgoals import *
from alfred.gen import constants
from alfred.nn.enc_visual import FeatureExtractor
from alfred.utils import data_util, model_util
from alfred.utils.eval_util import *
from seq2seq_questioner_multimodel import *
from utils import *

# c.f. https://github.com/xfgao/DialFRED/issues/2
def def_value(): return None

class BDataset(TorchDataset):
    def __init__(self, json_paths,  name, partition, args, ann_type):
        path = os.path.join(constants.ET_DATA, name)
        self.partition = partition
        self.name = name
        self.args = args
        if ann_type not in ('lang', 'frames', 'lang_frames'):
            raise ValueError('Unknown annotation type: {}'.format(ann_type))
        self.ann_type = ann_type
        self.test_mode = True
        self.pad = 0
        self.vocab = data_util.load_vocab("lmdb_augmented_human_subgoal", "tests_seen")


        # read information about the dataset
        self.dataset_info = data_util.read_dataset_info(name)
        if self.dataset_info['visual_checkpoint']:
            print('The dataset was recorded using model at {}'.format(
                self.dataset_info['visual_checkpoint']))

        # load data
        self.json_paths = json_paths
        self._length = self.load_data(path)
        if self.args.fast_epoch:
            self._length = 16
        print('{} dataset size = {}'.format(partition, self._length))

        # load vocabularies for input language and output actions
        vocab = data_util.load_vocab(name, ann_type)
        self.vocab_in = vocab['word']
        out_type = 'action_low' if args.model == 'transformer' else 'action_high'
        self.vocab_out = vocab[out_type]
        # if several datasets are used, we will translate outputs to this vocab later
        self.vocab_translate = None

    def load_data(self, path, feats=True, masks=True, jsons=True):
        '''
        load data
        '''
        # do not open the lmdb database open in the main process, do it in each thread
        if feats:
            self.feats_lmdb_path = os.path.join(path, self.partition, 'feats')
        if masks:
            self.masks_lmdb_path = os.path.join(path, self.partition, 'masks')

        # load jsons with pickle and parse them
        if jsons:
            jsons = []
            for path in self.json_paths:
                with open(path, "r") as f:
                    import json
                    jsons.append((path, json.load(f)))
            self.jsons_and_keys = []
            print("Loading jsons ... ")
            for idx in tqdm(range(len(jsons))):
                path, js = jsons[idx]
                key = path.split("/")[-1][:-len(".json")]
                # print(key)
                # compatibility with the evaluation
                if 'task' in js and isinstance(js['task'], str):
                    pass
                else:
                    # js['task'] = '/'.join(js['root'].split('/')[-3:-1])
                    js['task'] = '/'.join(path.split('/')[-3:-1])
                # add dataset idx and partition into the json
                js['dataset_name'] = self.name
                
                if 'num' not in js:
                    pp = Preprocessor(self.vocab, is_test_split=True) # TODO: vocabここ合ってるかわからん これだと無限に時間かかる (vocab=Noneだと全体で1hくらいで終わる)
                    pp.process_language(js, js, 0)
                    path = os.environ['DF_ROOT'] + f"/testset/dialfred_testset_final/{key}.json"
                    assert 'num' in js
                    with open(path, "w") as f:
                        f.write(json.dumps(js, indent=4))

                self.jsons_and_keys.append((js, key))
                # if the dataset has script annotations, do not add identical data
                # if len(set([str(j['ann']['instr']) for j in task_jsons])) == 1:
                #     break

        # return the true length of the loaded data
        return len(self.jsons_and_keys) if jsons else None

    def load_frames(self, key):
        '''
        load image features from the disk
        '''
        if not hasattr(self, 'feats_lmdb'):
            self.feats_lmdb, self.feats = self.load_lmdb(
                self.feats_lmdb_path)
        feats_bytes = self.feats.get(key)
        feats_numpy = np.frombuffer(
            feats_bytes, dtype=np.float32).reshape(self.dataset_info['feat_shape'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            frames = torch.tensor(feats_numpy)
        return frames

    def load_lmdb(self, lmdb_path):
        '''
        load lmdb (should be executed in each worker on demand)
        '''
        database = lmdb.open(
            lmdb_path, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=252)
        cursor = database.begin(write=False)
        return database, cursor

    def __len__(self):
        '''
        return dataset length
        '''
        return self._length

    def __getitem__(self, idx):
        '''
        get item at index idx
        '''
        raise NotImplementedError

    @property
    def id(self):
        return self.partition + ':' + self.name + ';' + self.ann_type

    def __del__(self):
        '''
        close the dataset
        '''
        if hasattr(self, 'feats_lmdb'):
            self.feats_lmdb.close()
        if hasattr(self, 'masks_lmdb'):
            self.masks_lmdb.close()

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.id)


class Dataset(BDataset):
    def __init__(self, json_paths, name, partition, args, ann_type):
        super().__init__(json_paths, name, partition, args, ann_type)
        # preset values
        self.max_subgoals = constants.MAX_SUBGOALS
        self._load_features = True
        self._load_frames = True
        # load the vocabulary for object classes
        self.vocab_obj = torch.load(os.path.join(
            constants.ET_ROOT, constants.OBJ_CLS_VOCAB))

    def load_data(self, path):
        return super().load_data(
            path, feats=True, masks=True, jsons=True)

    def __getitem__(self, idx):
        task_json, key = self.jsons_and_keys[idx]
        feat_dict = {}
        if self._load_features:
            feat_dict = self.load_features(task_json)
        if self._load_frames:
            feat_dict['frames'] = self.load_frames(key)
        return task_json, feat_dict

    # def load_masks(self, key):
    #     '''
    #     load interaction masks from the disk
    #     '''
    #     if not hasattr(self, 'masks_lmdb'):
    #         self.masks_lmdb, self.masks = self.load_lmdb(
    #             self.masks_lmdb_path)
    #     masks_bytes = self.masks.get(key)
    #     masks_list = pickle.load(BytesIO(masks_bytes))
    #     masks = [data_util.decompress_mask_bytes(m) for m in masks_list]
    #     return masks

    def load_features(self, task_json, subgoal_idx=None, append_no_op=False):
        '''
        load features from task_json
        '''
        feat = dict()
        # language inputs
        feat['lang'] = self.load_lang(task_json, subgoal_idx)

        # action outputs
        if not self.test_mode:
            # low-level action
            if append_no_op:
                assert len(task_json['num']['action_low']
                           ) == 1, 'Subgoal level data.'
                if task_json['num']['action_low'][0][-1]['action'] != 11:
                    no_op_high_idx = task_json['num']['action_low'][0][0]['high_idx']
                    task_json['num']['action_low'][0].append({
                        'high_idx': no_op_high_idx,
                        'action': 11, 'action_high_args': {},
                        'centroid': [-1, -1],
                        'mask': None,
                        'valid_interact': 0
                    })
            # feat['action'] = Dataset.load_action(
            #     task_json, self.vocab_out, self.vocab_translate)
            # low-level valid interact
            feat['action_valid_interact'] = [
                a['valid_interact'] for a in sum(
                    task_json['num']['action_low'], [])]
            feat['object'] = Dataset.load_object_classes(
                task_json, self.vocab_obj)
            assert len(feat['object']) == sum(feat['action_valid_interact']), \
                f"{feat['object']}, {feat['action_valid_interact']}"

        # auxillary outputs
        if not self.test_mode:
            # subgoal completion supervision
            if self.args.subgoal_aux_loss_wt > 0:
                feat['subgoals_completed'] = np.array(
                    task_json['num']['low_to_high_idx']) / self.max_subgoals
            # progress monitor supervision
            if self.args.progress_aux_loss_wt > 0:
                num_actions = len(task_json['num']['low_to_high_idx'])
                goal_progress = [(i + 1) / float(num_actions)
                                 for i in range(num_actions)]
                feat['goal_progress'] = goal_progress
        return feat

    @staticmethod
    def numericalize(vocab, words, train=True):
        '''
        converts words to unique integers
        '''
        if not train:
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else '<<pad>>' for w in words]
        return vocab.word2index(words, train=train)

    def load_lang(self, task_json, subgoal_idx=None):
        '''
        load numericalized language from task_json
        '''

        if subgoal_idx is None:
            lang_num_goal = task_json['num']['lang_goal']
            lang_num = lang_num_goal + sum(task_json['num']['lang_instr'], [])
        else:
            lang_num = task_json['num']['lang_instr'][subgoal_idx]

        return lang_num

    @staticmethod
    def load_action(task_json, vocab_orig, vocab_translate, action_type='action_low'):
        '''
        load action as a list of tokens from task_json
        '''
        if action_type == 'action_low':
            # load low actions
            lang_action = [
                [a['action'] for a in a_list]
                for a_list in task_json['num']['action_low']]
        elif action_type == 'action_high':
            # load high actions
            lang_action = [
                [a['action']] + a['action_high_args']
                for a in task_json['num']['action_high']]
        else:
            raise NotImplementedError(
                'Unknown action_type {}'.format(action_type))
        lang_action = sum(lang_action, [])
        # translate actions to the provided vocab if needed
        if vocab_translate and not vocab_orig.contains_same_content(
                vocab_translate):
            lang_action = model_util.translate_to_vocab(
                lang_action, vocab_orig, vocab_translate)
        return lang_action

    @staticmethod
    def load_object_classes(task_json, vocab=None):
        '''
        load object classes for interactive actions
        '''
        object_classes = []
        for action in task_json['plan']['low_actions']:
            if model_util.has_interaction(action['api_action']['action']):
                obj_key = ('receptacleObjectId'
                           if 'receptacleObjectId' in action['api_action']
                           else 'objectId')
                object_class = action['api_action'][obj_key].split('|')[0]
                object_classes.append(
                    object_class if vocab is None
                    else vocab.word2index(object_class))
        return object_classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ITER = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
WEIGHT_DECAY = 0.0005
DROPOUT_RATIO = 0.5
BIDIRECTIONAL = False
WORD_EMBEDDING_SIZE = 256
ACTION_EMBEDDING_SIZE = 32
TARGET_EMBEDDING_SIZE = 32
HIDDEN_SIZE = 512
MAX_INPUT_LENGTH = 160
MAX_LENGTH = 2
REWARD_INVALID = -0.1
REWARD_QUESTION = -0.05
REWARD_TIME = -0.01
REWARD_SUC = 1.0

SOS_token = 0
EOS_token = 1


recent_id = None
action_idx = None
VERBOSE = False


def extractFeatureOnline(env, extractor):
    event = env.last_event
    feat = get_observation(event, extractor)
    avg_pool = nn.AvgPool2d(7)
    feat = torch.unsqueeze(torch.squeeze(avg_pool(feat)), 0)
    return feat


def iters(json_paths, args, lang, dataset, encoder, decoder, critic, performer, extractor, all_ans, split_id, max_steps, print_every=1, save_every=100, use_qa_everytime=False):
    env = ThorEnv(x_display=1)
    obj_predictor = FeatureExtractor(
        archi='maskrcnn', device=device, checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

    loc_ans, app_ans, dir_ans = all_ans
    all_rewards = []
    succ = []
    all_query = []
    all_instr = []
    object_found = []
    sg_pairs = []
    num_q = []
    all_pws = []
    # assume we only have one instruction
    instr_id = 0
    it = 0

    jsons_data = []
    for path in json_paths:
        with open(path, "r") as f:
            jsons_data.append((path, json.load(f)))

    # first sample a subgoal and get the instruction and image feature
    for dataset_idx, (path, task_json) in enumerate(tqdm(jsons_data)):
        print("   ", dataset_idx)
        setup_scene(env, task_json)
        # performer.reset_for_both()
        if args.clip_image:
            performer.reset_for_clip()
        elif args.clip_resnet:
            performer.reset_for_both()
        else:
            performer.reset()
        num_subgoal = len(task_json["turk_annotations"]
                          ["anns"][0]["high_descs"])
        meta_data = []

        # initialize environment
        # traj_data, traj_key = dataset.jsons_and_keys[dataset_idx]
        # setup_scene(env, traj_data)
        # performer.reset()

        vocab = {'word': dataset.vocab_in, 'action_low': performer.vocab_out}
        init_states = (None, vocab, None, False)
        _, _, _, init_failed = init_states

        # execute actions and save the metadata for each subgoal
        for subgoal_idx in range(num_subgoal):
            current_query = []
            current_object_found = []
            all_log = []
            current_num_q = 0
            t_agent = 0
            num_fails = 0
            episode_end = False
            # set up the performer for expect actions first
            trial_uid = "pad:" + str(0) + ":" + str(subgoal_idx)
            dataset_idx_qa = 0 + dataset_idx
            # init_states = reset(env, performer, dataset, extractor,
            #                     trial_uid, dataset_idx_qa, args, obj_predictor)

            # vocab = {'word': dataset.vocab_in, 'action_low': performer.vocab_out}
            # init_states = (None, vocab, None, False)
            # _, _, _, init_failed = init_states
            

            task, trial = task_json["turk_annotations"]["anns"][0]["task_desc"], task_json["turk_annotations"]["anns"][0]["task_desc"]
            sg_pairs.append([task, trial, subgoal_idx])
            # orig_instr = normalizeString(turk_annos[0]["high_descs"][subgoal_idx]).lower().replace(",", "").replace(".", "")
            qa = ""
            reward = 0
            if subgoal_idx == 0:
                interm_states = None
            pws = 0.0
            t_agent_old = 0
            orig_instr = normalizeString(task_json["turk_annotations"]["anns"][0]["high_descs"][subgoal_idx]).lower(
            ).replace(",", "").replace(".", "")
            all_instr.append(orig_instr)
            while True:
                #questionerを用いない場合.全てのQAを使う.
                if use_qa_everytime:
                    if task in app_ans and trial in app_ans[task] and subgoal_idx in app_ans[task][trial] and 0 in app_ans[task][trial][subgoal_idx]:
                        ans_sg = app_ans[task][trial][subgoal_idx][0]
                        if len(ans_sg) > 0:
                            for key, value in ans_sg.items():
                                if "ans" in value.keys():
                                    query = "<<app>> " + key
                                    ans = value["ans"]
                                    if isinstance(query, str) and isinstance(ans, str):
                                        qa1 = query + " " + ans
                                    else:
                                        qa1 = ""
                                else:
                                    qa1 = ""
                        else:
                            qa1 = ""
                    else:
                        logging.info("invalid answer for %s, %s, %s" % (task, trial, subgoal_idx))
                        qa1 = ""

                    if task in loc_ans and trial in loc_ans[task] and subgoal_idx in loc_ans[task][trial] and 0 in loc_ans[task][trial][subgoal_idx]:
                        ans_sg = loc_ans[task][trial][subgoal_idx][0]
                        if len(ans_sg) > 0:
                            for key, value in ans_sg.items():
                                if "ans" in value.keys() and "obj_id" in value.keys():
                                    query = "<<loc>> " + key
                                    obj_id = value["obj_id"]
                                    event = env.last_event
                                    metadata = event.metadata
                                    odata = get_obj_data(metadata, obj_id)
                                    if odata is None:
                                        qa2 = ""
                                    else:
                                        oname = key
                                        recs = odata["parentReceptacles"]
                                        rel_ang = get_obj_direction(metadata, odata)
                                        ans = objLocAns(oname, rel_ang, recs)
                                        if isinstance(query, str) and isinstance(ans, str):
                                            qa2 = query + " " + ans
                                        else:
                                            qa2 = ""
                                else:
                                    qa2 = ""
                        else:
                            qa2 = ""
                    else:
                        qa2 = ""

                    if task in dir_ans and trial in dir_ans[task] and subgoal_idx in dir_ans[task][trial]:
                        query = "<<dir>> "
                        target_pos = dir_ans[task][trial][subgoal_idx]["target_pos"]
                        event = env.last_event
                        cur_metadata = event.metadata
                        targ_metadata = {'agent':{'position':target_pos}}
                        rel_ang, rel_pos = get_agent_direction(cur_metadata, targ_metadata)
                        ans = dirAns(rel_ang, rel_pos).lower()
                        if isinstance(query, str) and isinstance(ans, str):
                            qa3 = query + " " + ans
                        else:
                            qa3 = ""
                    else:
                        qa3 = ""

                    reward += REWARD_QUESTION
                    current_object_found.append(True)
                    current_num_q += 1

                    qa = qa1 + " " + qa2 + " " + qa3
                    current_query.append(qa)

                #questionerを用いた場合
                else:                
                    # use online feature extractor instead
                    f_t = extractFeatureOnline(env, extractor)
                    dialog = orig_instr
                    input_tensor = torch.unsqueeze(torch.squeeze(
                        tensorFromSentence(lang, dialog)), 0).to(device)
                    input_length = input_tensor.size(1)

                    # infer question based on the instruction
                    encoder.init_state(input_tensor)
                    seq_lengths = torch.from_numpy(np.array([input_length]))
                    ctx, h_t, c_t = encoder(input_tensor, seq_lengths)
                    decoder_input = torch.tensor([[SOS_token]], device=device)
                    decoded_words = []

                    # decode the question
                    for di in range(MAX_LENGTH):
                        h_t, c_t, alpha, logit = decoder(
                            decoder_input, f_t, h_t, c_t, ctx)
                        # record the value V(s)
                        value = critic(h_t)
                        dist = Categorical(F.softmax(logit, dim=-1))
                        selected_word = dist.sample()
                        decoded_words.append(lang.index2word[selected_word.item()])
                        decoder_input = selected_word.detach().to(device)

                    # set up query and answer
                    repeat_idx = -1
                    ans = ""
                    task_id = path.split("/")[-1][:-len(".json")]
                    # for appearance answer, we can directly use the saved ones
                    if decoded_words[0] == "appearance":
                        query = "<<app>> " + decoded_words[1]
                        if task_id in app_ans  and subgoal_idx in app_ans[task_id] and 0 in app_ans[task_id][subgoal_idx]:
                            ans_sg = app_ans[task_id][subgoal_idx][0]
                            if decoded_words[1] in ans_sg and ans_sg[decoded_words[1]]["ans"] is not None:
                                ans += ans_sg[decoded_words[1]]["ans"]
                            else:
                                ans += "invalid"
                        else:
                            logging.info("invalid answer for %s, %s, %s" %
                                        (task_id, trial, subgoal_idx))
                            ans += "invalid"

                    # for location answer, we need to construct a new one using current metadata
                    elif decoded_words[0] == "location":
                        query = "<<loc>> " + decoded_words[1]
                        if task_id in loc_ans  and subgoal_idx in loc_ans[task_id] and 0 in loc_ans[task_id][subgoal_idx]:
                            ans_sg = loc_ans[task_id][subgoal_idx][0]
                            if decoded_words[1] in ans_sg:
                                ans += ans_sg[decoded_words[1]]["ans"]
                            else:
                                ans += "invalid"
                        else:
                            ans += "invalid"

                    elif decoded_words[0] == "direction":
                        query = "<<dir>> "
                        if task_id in dir_ans and subgoal_idx in dir_ans[task_id]:
                            ans += dir_ans[task_id][subgoal_idx]["ans"]
                        else:
                            ans += "invalid"
                    elif decoded_words[0] == "none" or decoded_words[0] == "EOS":
                        query = "none"
                    # else:
                    #     query = "<<invalid>>"

                    # for invalid query, give a negative reward and use the instruction only
                    if "invalid" in query or "invalid" in ans:
                        reward += REWARD_INVALID
                        current_object_found.append(False)
                        qa = ""
                    # for asking each question, we add a small negative reward
                    elif not query == "none":
                        reward += REWARD_QUESTION
                        current_object_found.append(True)
                        current_num_q += 1
                        qa = query + " " + ans
                    # for not asking question, there is no penalty
                    else:
                        current_object_found.append(True)
                        qa = ""

                    current_query.append(query + " " + ans)

                qa = qa.lower().replace(",", "").replace(".", "")
                if VERBOSE:
                    print(query,ans)
                    print("QA:",qa)

                # performer rollout for some steps
                with torch.no_grad():
                    interm_states = step(env, performer, dataset, extractor,
                                         trial_uid, dataset_idx_qa, args, obj_predictor, init_states, interm_states, qa, int(path[-9:-5]), num_rollout=5)

                # if log_entry['success']:
                #     reward += REWARD_SUC
                #     done = 1.0
                #     pws = log_entry['success_spl']
                # else:
                #     done = 0.0

                t_agent, num_fails, _, mc_lists, episode_end, _ = interm_states
                # a penalty for each time step
                reward += REWARD_TIME * (t_agent - t_agent_old)
                t_agent_old = t_agent
                # print(t_agent, args.max_steps, num_fails, args.max_fails, episode_end)
                if t_agent > args.max_steps or num_fails > args.max_fails or episode_end or init_failed or len(current_query) > 100:
                    # print(t_agent, args.max_steps, num_fails, args.max_fails, episode_end)
                    break

            all_rewards.append(reward)
            all_pws.append(pws)
            # record the next value V(s')
            object_found.append(current_object_found)
            all_query.append(current_query)
            num_q.append(current_num_q)

            if it % print_every == 0:
                logging.info("task, trial, subgoals: %s" % sg_pairs[-1])
                logging.info("instruction: %s" % all_instr[-1])
                logging.info("questions: %s" % all_query[-1])
                logging.info("number of questions: %s" % num_q[-1])

            m_data = env.last_event.metadata
            m_data["pose_discrete"] = env.last_event.pose_discrete
            meta_data.append(m_data)

            it += 1
            save_path_full = os.path.join(
                os.environ['DF_ROOT'] + "/submission/" + path[-9:])
            with open(save_path_full, "w") as f:
                json.dump(meta_data, f, sort_keys=True, indent=4)

    env.stop()


def setup_scene(env, traj_data):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name,silent=True)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))
    # env.set_task(traj_data, reward_type="dense")


def reset(
        env, model, dataset, extractor, trial_uid, dataset_idx, args, obj_predictor):
    # set up the evaluation
    # load trajectory data from the dataset
    traj_data, traj_key = dataset.jsons_and_keys[dataset_idx]
    r_idx, subgoal_idx = int(trial_uid.split(
        ':')[1]), int(trial_uid.split(':')[2])
    # reset model and setup scene
    if args.clip_image:
        model.reset_for_clip()
    elif args.clip_resnet:
        model.reset_for_both()
    else:
        model.reset()
        
    # model.reset()
    # print(traj_data)

    if subgoal_idx == 0:
        setup_scene(env, traj_data)
    
        
    vocab = {'word': dataset.vocab_in, 'action_low': model.vocab_out}
    # load language features, expert initialization and task info
    # task_info = eval_util.read_task_data(traj_data, subgoal_idx)
    # if args.debug:
    # print(task_info)

    prev_action, subgoal_success, init_failed = None, False, False

    # # expert teacher-forcing upto subgoal, get expert action
    # for a_expert in expert_dict['actions']:
    #     input_dict['frames'] = eval_util.get_observation(env.last_event, extractor)
    #     init_failed, prev_action = eval_util.expert_step(
    #         a_expert['action'], expert_dict['masks'], model,
    #         input_dict, vocab, prev_action, env, args)
    #     if init_failed:
    #         break

    init_states = (None, vocab, prev_action, init_failed)
    
    return init_states


def draw_bbox(roi_list, image):
    import cv2
    image = copy.deepcopy(image)
    for roi in roi_list:
        box, label, score = roi.box, roi.label, roi.score
        c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
        display_txt = "%s: %.1f%%" % (label, 100 * score)
        tl = 2
        color = (0, 0, 255)
        cv2.rectangle(image, c1, c2, color, thickness=tl)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1)  # filled
        cv2.putText(image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return image


def step(env, model, dataset, extractor, trial_uid, dataset_idx, args, obj_predictor, init_states, interm_states, qa, id, num_rollout=5):
    # modification of evaluate_subgoals: add qa and skip init

    # add initial states from expert initialization
    _, vocab, prev_action, init_failed = init_states

    # load trajectory data from the dataset
    traj_data, traj_key = dataset.jsons_and_keys[dataset_idx]
    r_idx, subgoal_idx = int(trial_uid.split(
        ':')[1]), int(trial_uid.split(':')[2])
    # if not traj_data['repeat_idx'] == r_idx:
    #     print(traj_data)

    # load language features and append numericalized qa
    num_qa = []
    for w in qa.split():
        if w in vocab['word']._word2index:
            num_qa.append(vocab['word'].word2index(w, train=False))
        else:
            print("word not in vocab: ", w)
    input_dict = eval_util.load_language_qa(
        dataset, traj_data, traj_key, model.args, extractor, num_qa, subgoal_idx, test_split=True)
    # print(input_dict)

    if interm_states == None:
        t_agent, num_fails, reward = 0, 0, 0
        mc_lists = []
        episode_end = False
    else:
        t_agent, num_fails, reward, mc_lists, episode_end, prev_action = interm_states

    subgoal_success = False
    if not init_failed:
        # this should be set during the teacher-forcing but sometimes it fails
        # print(env.task)
        # env.task.goal_idx = subgoal_idx
        # env.task.finished = subgoal_idx - 1
        t_current = 0
        while t_agent < args.max_steps and t_current < num_rollout:
            # print(t_agent)
            # get an observation and do an agent step
            global recent_id, action_idx
            if recent_id != id:
                recent_id = id
                action_idx = 1
            else:
                action_idx += 1

            input_dict['frames'] = [eval_util.get_observation(env.last_event, extractor, id=id, subgoal_idx=subgoal_idx, action_idx=action_idx), 
                                    eval_util.get_observation_clip(env.last_event, extractor)]

            # save image
            if id != None and subgoal_idx != None and action_idx != None:
                dir = f"./qualitative/test/{id:04}"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                frame = Image.fromarray(env.last_event.frame)
                rcnn_pred = obj_predictor.predict_objects(Image.fromarray(env.last_event.frame))
                frame_np = draw_bbox(rcnn_pred,np.array(frame))
                frame = Image.fromarray(frame_np)
                frame.save(f"{dir}/{subgoal_idx+1:02}_{action_idx:03}.png")

            # 現在のlow-level instructionを取得
            subgoal_instr = traj_data['turk_annotations']['anns'][0]['high_descs'][subgoal_idx]

            # print(prev_action, )
            episode_end, prev_action, num_fails, _, _, mc_array = eval_util.agent_step_mc(
                model, input_dict, vocab, prev_action, env, args,
                num_fails, obj_predictor, rcnn_pred, subgoal_instr)
            mc_lists.append(mc_array[0] - mc_array[1])
            # get rewards and subgoal success
            # reward += env.get_transition_reward()[0]
            # subgoal_success = (env.get_subgoal_idx() == subgoal_idx)
            t_agent += 1
            t_current += 1
            # break if stop is predicted, args.max_fails is reached or success
            if episode_end or subgoal_success:
                break

    interm_states = [t_agent, num_fails, reward,
                     mc_lists, episode_end, prev_action]
    return interm_states


class Critic(nn.Module):
    def __init__(self, rnn_dim=512, dropout=0.5):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


def test(args, json_paths):
    np.random.seed(0)
    data_split = "unseen"
    train_id = 1
    logging.basicConfig(filename='./logs/rl_anytime_eval_' +
                        data_split + str(train_id) + '.log', level=logging.INFO)

    use_qa_everytime = args.use_qa_everytime

    if use_qa_everytime:
        encoder = None
        decoder = None
        critic = None
        lang = None
    else:
        # load pretrained questioner
        test_csv_fn = "./data/hdc_input_augmented.csv"
        lang = prepareDataTest(test_csv_fn)
        enc_hidden_size = HIDDEN_SIZE//2 if BIDIRECTIONAL else HIDDEN_SIZE
        encoder = EncoderLSTM(lang.n_words, WORD_EMBEDDING_SIZE, enc_hidden_size,
                            DROPOUT_RATIO, bidirectional=BIDIRECTIONAL).to(device)
        decoder = AttnDecoderLSTM(lang.n_words, lang.n_words,
                                ACTION_EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATIO).to(device)
        critic = Critic().to(device)
        finetune_questioner_fn = args.questioner_path
        checkpt = torch.load(finetune_questioner_fn)
        encoder.load_state_dict(checkpt["encoder"])
        decoder.load_state_dict(checkpt["decoder"])
        critic.load_state_dict(checkpt["critic"])

    # load dataset and pretrained performer
    data_name = "lmdb_augmented_human_subgoal"

    model_path = args.performer_path
    model_args = model_util.load_model_args(model_path)
    model_args.debug = False
    model_args.no_model_unroll = False
    model_args.no_teacher_force = False
    model_args.smooth_nav = False
    model_args.max_steps = 1000
    model_args.max_fails = 10
    # TODO:
    # dataset = Dataset(data_name, "valid_"+data_split, model_args, "lang")
    dataset = Dataset(json_paths, data_name, "tests_" +
                      data_split, model_args, "lang")
    performer, extractor = load_agent(model_path, dataset.dataset_info, device)
    dataset.vocab_translate = performer.vocab_out
    dataset.vocab_in.name = "lmdb_augmented_human_subgoal"


    loc_ans_fn = "testset/loc_testset_final.pkl"
    app_ans_fn = "testset/appear_testset_final.pkl"
    dir_ans_fn = "testset/direction_testset_final.pkl"
    with open(loc_ans_fn, "rb") as f:
        loc_ans = pickle.load(f)
    with open(app_ans_fn, "rb") as f:
        app_ans = pickle.load(f)
    with open(dir_ans_fn, "rb") as f:
        dir_ans = pickle.load(f)
    
    # print(json.dumps(dir_ans,indent=4))
    all_ans = [loc_ans, app_ans, dir_ans]

    iters(json_paths, model_args, lang, dataset, encoder, decoder, critic, performer, extractor,
          all_ans, split_id=data_split + str(train_id), max_steps=1000, print_every=1, save_every=10, use_qa_everytime=use_qa_everytime)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest="mode", type=str, default="eval")
    parser.add_argument("--questioner-path", dest="questioner_path", type=str,
                        default="./logs/pretrained/questioner_anytime_finetuned.pt")
    parser.add_argument("--performer-path", dest="performer_path",
                        type=str, default="./logs/pretrained/performer/latest.pth")
    parser.add_argument("--use_qa_everytime", action='store_true')
    parser.add_argument("--test_id", type=str, default="0001")
    args = parser.parse_args()
    # path to testset json file
    # input_jsons = [str(path) for path in Path(os.environ['DF_ROOT'] + "/testset/dialfred_testset_final/").glob("*.json")]
    input_jsons = [f"testset/dialfred_testset_final/{int(args.test_id):04}.json"]
    test(args, input_jsons)


if __name__ == "__main__":
    main()
