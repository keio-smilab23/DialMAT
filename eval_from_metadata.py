import os
from itertools import count
import time
import math
import json
import matplotlib.pyplot as plt
import unicodedata
import string
import re
import random
import csv
import pickle
from io import open
import logging
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

from alfred.gen import constants
from alfred.env.thor_env import ThorEnv
from alfred.nn.enc_visual import FeatureExtractor
from alfred.utils import data_util, model_util
from alfred.utils.eval_util import *
from alfred.data.zoo.alfred import AlfredDataset
from alfred.eval.eval_subgoals import *
from seq2seq_questioner_multimodel import *
from utils import *
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ITER = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
WEIGHT_DECAY = 0.0005
DROPOUT_RATIO= 0.5
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

def def_value():
	return None

class Critic(nn.Module):
    def __init__(self, rnn_dim = 512, dropout=0.5):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

# extract resnet feature from current observation
def extractFeatureOnline(env, extractor):
    event = env.last_event
    feat = get_observation(event, extractor)
    avg_pool = nn.AvgPool2d(7)
    feat = torch.unsqueeze(torch.squeeze(avg_pool(feat)), 0)
    return feat

#追加
def extractFeatureOnlineClip(env, extractor):
    event = env.last_event
    feat = get_observation_clip(event, extractor)
    return feat

@dataclass
class PsuedoState:
    pose_discrete: any
    metadata: any

def extract_state(metadata)-> PsuedoState:
    pose_discrete = metadata["pose_discrete"]
    state = PsuedoState(pose_discrete,metadata)
    return state

def evaluate_from_metadata(original_subgoal_idx, env, model, dataset, extractor, trial_uid, dataset_idx, args, obj_predictor, init_states, interm_states, qa, num_rollout=5):
    task_info, vocab, prev_action, init_failed, expert_dict = init_states
    
    traj_data, traj_key = dataset.jsons_and_keys[dataset_idx]
    r_idx, subgoal_idx = int(trial_uid.split(':')[1]), int(trial_uid.split(':')[2])
    if not traj_data['repeat_idx'] == r_idx:
        print(traj_data)
    
    # load language features and append numericalized qa
    num_qa = []
    for w in qa.split():
        if w in vocab['word']._word2index:
            num_qa.append(vocab['word'].word2index(w, train=False))
        else:
            print("word not in vocab: ", w)
    input_dict = eval_util.load_language_qa(dataset, traj_data, traj_key, model.args, extractor, num_qa, subgoal_idx)
    # print(input_dict)
    
    if interm_states == None:
        t_agent, t_expert, num_fails, reward = 0, len(expert_dict['actions']), 0, 0
        mc_lists = []
        episode_end = False
    else:
        t_agent, t_expert, num_fails, reward, mc_lists, episode_end, prev_action = interm_states

    subgoal_success = False
    traj_key_str = traj_key.decode("utf-8") 
    print("traj_key",traj_key_str,dataset_idx)
    if not init_failed:
        with open(f"submission_file/{traj_key_str}.json","r") as f: # TODO: ここにjsonを渡してあげてください (for koreさん)
            metadata_list = json.load(f)
            metadata = metadata_list[original_subgoal_idx]
            state = extract_state(metadata)
            subgoal_success = env.task.check_subgoal_is_done(state)

    interm_states = [t_agent, t_expert, num_fails, reward, mc_lists, episode_end, prev_action]
    return subgoal_success, interm_states


def evalIters(args, lang, dataset, encoder, decoder, critic, performer, extractor, all_ans, split_id, max_steps, print_every=1, save_every=100):
    start = time.time()
    env = ThorEnv(x_display=1)
    obj_predictor = FeatureExtractor(archi='maskrcnn', device=device,
        checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

    loc_ans, app_ans, dir_ans = all_ans
    num_subgoals = len(dataset.jsons_and_keys)
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

    # record the dataset index for the original instruction without qas
    data_instruct_list = []
    for i in range(len(dataset.jsons_and_keys)):
        traj_data, traj_key = dataset.jsons_and_keys[i]
        if traj_data["repeat_idx"] == 0:
            data_instruct_list.append(i)
    print("dataset length", len(data_instruct_list))
    # rough estimation of total number of subgoals
    n_iters = len(data_instruct_list) * 4

    # first sample a subgoal and get the instruction and image feature
    for dataset_idx in data_instruct_list:
        task_json = dataset.jsons_and_keys[dataset_idx]
        turk_annos = task_json[0]["turk_annotations"]["anns"]
        subgoal_idxs = [sg['high_idx'] for sg in task_json[0]['plan']['high_pddl']]
        # ignore the last subgoal which is often the padding one
        subgoal_idxs = subgoal_idxs[:-1]
        print("subgoal_idxs:",subgoal_idxs)
        for subgoal_idx in subgoal_idxs:
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
            init_states = evaluate_subgoals_start_qa(
                env, performer, dataset, extractor, trial_uid, dataset_idx_qa, args, obj_predictor)
            _, _, _, init_failed, _ = init_states

            task, trial = task_json[0]['task'].split("/")
            pair = (None, None, task, trial, subgoal_idx)
            sg_pairs.append([task, trial, subgoal_idx])
            orig_instr = normalizeString(turk_annos[0]["high_descs"][subgoal_idx]).lower().replace(",", "").replace(".", "")
            qa = ""
            reward = 0
            all_instr.append(orig_instr)
            interm_states = None
            pws = 0.0
            t_agent_old = 0

            # use online feature extractor instead
            f_t = extractFeatureOnline(env, extractor)
            # f_t = extractFeatureOnlineClip(env, extractor)
            dialog = orig_instr
            input_tensor = torch.unsqueeze(torch.squeeze(tensorFromSentence(lang, dialog)), 0).to(device)
            input_length = input_tensor.size(1)

            # infer question based on the instruction
            encoder.init_state(input_tensor)
            seq_lengths = torch.from_numpy(np.array([input_length]))
            ctx, h_t, c_t = encoder(input_tensor, seq_lengths)
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoded_words = []

            # decode the question
            for di in range(MAX_LENGTH):
                h_t, c_t, alpha, logit = decoder(decoder_input, f_t, h_t, c_t, ctx)
                # record the value V(s)
                value = critic(h_t)
                dist = Categorical(F.softmax(logit, dim=-1))
                selected_word = dist.sample()
                decoded_words.append(lang.index2word[selected_word.item()])
                decoder_input = selected_word.detach().to(device)

            # set up query and answer
            repeat_idx = -1
            ans = ""
            
            # for appearance answer, we can directly use the saved ones
            if decoded_words[0] == "appearance":
                query = "<<app>> " + decoded_words[1]
                if task in app_ans and trial in app_ans[task] and subgoal_idx in app_ans[task][trial] and 0 in app_ans[task][trial][subgoal_idx]:
                    ans_sg = app_ans[task][trial][subgoal_idx][0]
                    if decoded_words[1] in ans_sg and ans_sg[decoded_words[1]]["ans"] is not None:
                        ans += ans_sg[decoded_words[1]]["ans"]
                    else:
                        ans += "invalid"
                else:
                    logging.info("invalid answer for %s, %s, %s" % (task, trial, subgoal_idx))
                    ans += "invalid"
            # # for location answer, we need to construct a new one using current metadata
            elif decoded_words[0] == "location":
                query = "<<loc>> " + decoded_words[1]
                if task in loc_ans and trial in loc_ans[task] and subgoal_idx in loc_ans[task][trial] and 0 in loc_ans[task][trial][subgoal_idx]:
                    ans_sg = loc_ans[task][trial][subgoal_idx][0]
                    if decoded_words[1] in ans_sg:
                        obj_id = ans_sg[decoded_words[1]]["obj_id"]
                        event = env.last_event
                        metadata = event.metadata
                        odata = get_obj_data(metadata, obj_id)
                        if odata is None:
                            ans += "invalid"
                        else:
                            oname = decoded_words[1]
                            recs = odata["parentReceptacles"]
                            rel_ang = get_obj_direction(metadata, odata)
                            ans += objLocAns(oname, rel_ang, recs)
                    else:
                        ans += "invalid"
                else:
                    ans += "invalid"
            # # for direction answer, we can directly use the saved ones
            elif decoded_words[0] == "direction":
                query = "<<dir>> "
                if task in dir_ans and trial in dir_ans[task] and subgoal_idx in dir_ans[task][trial]:
                    target_pos = dir_ans[task][trial][subgoal_idx]["target_pos"]
                    event = env.last_event
                    cur_metadata = event.metadata
                    targ_metadata = {'agent':{'position':target_pos}}
                    rel_ang, rel_pos = get_agent_direction(cur_metadata, targ_metadata)
                    ans += dirAns(rel_ang, rel_pos).lower()
                else:
                    ans += "invalid"
            elif decoded_words[0] == "none" or decoded_words[0] == "EOS":
                query = "none"
            else:
                query = "<<invalid>>"
            
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
                #変更
                # qa = qa1 + " " + qa2 + " " + qa3
            # for not asking question, there is no penalty
            else:
                current_object_found.append(True)
                qa = ""

            current_query.append(query + " " + ans)
            qa = qa.lower().replace(",", "").replace(".", "")

            # metadataをjsonから読み出して評価
            sg_success, interm_states = evaluate_from_metadata(subgoal_idx, env, performer, dataset, extractor, \
                trial_uid, dataset_idx_qa, args, obj_predictor, init_states, interm_states, qa, num_rollout=5)

            if sg_success:
                reward += REWARD_SUC
                done = 1.0
                # pws = log_entry['success_spl']
            else:
                done = 0.0

            t_agent, _, num_fails, _, mc_lists, episode_end, _ = interm_states
            # a penalty for each time step
            reward += REWARD_TIME * (t_agent - t_agent_old)
            t_agent_old = t_agent
             

            succ.append(done)
            all_rewards.append(reward)
            all_pws.append(pws)
            # all_log.append(log_entry)
            # record the next value V(s')
            object_found.append(current_object_found)
            all_query.append(current_query)
            num_q.append(current_num_q)

            if it % print_every == 0:
                logging.info("task, trial, subgoals: %s" % sg_pairs[-1])
                logging.info("instruction: %s" % all_instr[-1])
                logging.info("questions: %s" % all_query[-1])
                logging.info("number of questions: %s" % num_q[-1])
                logging.info('%s (%d %d%%) reward %.4f, SR %.4f, pws %.4f' % \
                    (timeSince(start, (it+1) / n_iters), (it+1), (it+1) / n_iters * 100, \
                    np.mean(all_rewards), np.mean(succ), np.mean(all_pws)))

            
            if it % save_every == 0:
                with open("./logs/questioner_rl/eval_questioner_anytime_"+split_id+".pkl", "wb") as pkl_f:
                    pickle.dump([all_rewards, succ, all_query, all_instr, sg_pairs, num_q, all_pws], pkl_f)
            
            it += 1
        
    env.stop()


def evalModel(args):
    np.random.seed(0)
    # data_split = "unseen"
    data_split = "pseudo_test"
    # data_split = "valid_seen"
    train_id = 1
    logging.basicConfig(filename='./logs/rl_anytime_eval_'+ data_split + str(train_id) + '.log', level=logging.INFO)

    # load pretrained questioner
    test_csv_fn = "./data/hdc_input_augmented.csv"
    lang = prepareDataTest(test_csv_fn)
    enc_hidden_size = HIDDEN_SIZE//2 if BIDIRECTIONAL else HIDDEN_SIZE
    encoder = EncoderLSTM(lang.n_words, WORD_EMBEDDING_SIZE, enc_hidden_size, \
        DROPOUT_RATIO, bidirectional=BIDIRECTIONAL).to(device)
    decoder = AttnDecoderLSTM(lang.n_words, lang.n_words, \
        ACTION_EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATIO).to(device)
    critic = Critic().to(device)
    finetune_questioner_fn = args.questioner_path
    checkpt = torch.load(finetune_questioner_fn)
    encoder.load_state_dict(checkpt["encoder"])
    decoder.load_state_dict(checkpt["decoder"])
    critic.load_state_dict(checkpt["critic"])

    # load dataset and pretrained performer
    data_name = "lmdb_augmented_human_subgoal_fixed"
    # data_name = "lmdb_augmented_human_subgoal_temp"
    # data_name = "lmdb_augmented_human_subgoal_splited"
    model_path = args.performer_path
    model_args = model_util.load_model_args(model_path)
    model_args.debug = False
    model_args.no_model_unroll = False
    model_args.no_teacher_force = False
    model_args.smooth_nav = False
    model_args.max_steps = 1000
    model_args.max_fails = 10
    #追加 新しい要素
    model_args.clip_image = args.clip_image
    model_args.clip_resnet = args.clip_resnet
    #変更
    # dataset = AlfredDataset(data_name, "valid_"+data_split, model_args, "lang")
    dataset = AlfredDataset(data_name, data_split, model_args, "lang")
    performer, extractor = load_agent(model_path, dataset.dataset_info, device)
    dataset.vocab_translate = performer.vocab_out
    dataset.vocab_in.name = "lmdb_augmented_human_subgoal_fixed"
    # dataset.vocab_in.name = "lmdb_augmented_human_subgoal_temp"
    # dataset.vocab_in.name = "lmdb_augmented_human_subgoal_splited"

    # load answers
    loc_ans_fn = "./data/answers/loc_augmented.pkl"
    app_ans_fn = "./data/answers/appear_augmented.pkl"
    dir_ans_fn = "./data/answers/direction_augmented.pkl"
    with open(loc_ans_fn, "rb") as f:
        loc_ans = pickle.load(f)
    with open(app_ans_fn, "rb") as f:
        app_ans = pickle.load(f)
    with open(dir_ans_fn, "rb") as f:
        dir_ans = pickle.load(f)
    all_ans = [loc_ans, app_ans, dir_ans]
    evalIters(model_args, lang, dataset, encoder, decoder, critic, performer, extractor, all_ans, split_id=data_split + str(train_id), max_steps=1000, print_every=1, save_every=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", dest="mode", type=str, default="eval")
    parser.add_argument("--questioner-path", dest="questioner_path", type=str, default="./logs/pretrained/questioner_anytime_finetuned.pt")
    parser.add_argument("--performer-path", dest="performer_path", type=str, default="./logs/pretrained/performer/latest.pth")
    parser.add_argument("--clip_image", dest="clip_image", type=bool, default=False)
    parser.add_argument("--clip_resnet", dest="clip_resnet", type=bool, default=False)
    parser.add_argument("--wandb_run", type=str, default="tmp_run")

    args = parser.parse_args()
    evalModel(args)

if __name__ == '__main__':
    main()