import os
import torch
import pickle
import numpy as np
import glob
from io import BytesIO

from alfred.gen import constants
from alfred.data.zoo.base import BaseDataset
from alfred.utils import data_util, model_util
from tqdm import tqdm

def tqdm_wrapper(self, x): return tqdm([y for y in x])


class AlfredDataset(BaseDataset):
    def __init__(self, name, partition, args, ann_type):
        super().__init__(name, partition, args, ann_type)
        # preset values
        self.max_subgoals = constants.MAX_SUBGOALS
        self._load_features = True
        self._load_frames = True
        # load the vocabulary for object classes
        self.vocab_obj = torch.load(os.path.join(
            constants.ET_ROOT, constants.OBJ_CLS_VOCAB))

    def load_data(self, path):
        print("Loading data ...")
        return super().load_data(
            path, feats=True, masks=True, jsons=True)

    def __getitem__(self, idx):
        task_json, key = self.jsons_and_keys[idx]
        # print("key",key) key b'003108'
        # print("task_json",task_json)
        # print("task_json.keys()",task_json.keys()) #task_json.keys() dict_keys(['images', 'pddl_params', 'plan', 'scene', 'task', 'task_id', 'task_type', 'turk_annotations', 'root', 'split', 'repeat_idx', 'num', 'ann', 'dataset_name'])

        # feats_dict_from_task_json = self.load_features_from_task_json(task_json)

        # for k, v in feats_dict_from_task_json.items():
        #     feats_dict_from_task_json[k] = torch.from_numpy(v)

        task_path = os.path.join(constants.ET_DATA, 'generated_2.1.0', task_json['split'], task_json['task'])

        feat_dict = {}
        if self._load_features:
            feat_dict = self.load_features(task_json)
        if self._load_frames:
            feat_dict['frames'] = self.load_frames(key)
        return task_path, task_json, feat_dict

    def load_features_from_task_json(self, task_json):
        task = task_json['task_type'] #'task': 'look_at_obj_in_light-Pillow-None-DeskLamp-205/trial_T20190909_014933_134232'
        split = task_json['split']

        #./data/generated_2.1.0/{split}/{task}/feats/deberta/*.pthからデータを取得
        data_paths = glob.glob(os.path.join(constants.ET_DATA,  "generated_2.1.0", split, task, "feats", "deberta", "*.pth"))
        feats_deberta = []
        for data_path in tqdm_wrapper(data_paths):
            deberta = torch.load(data_path) #(1, num_words, 768)
            feats_deberta.append(deberta)
        feats_deberta = torch.cat(feats_deberta, dim=0) #(num_sentences, num_words, 768)

        #./data/generated_2.1.0/{split}/{task}/feats/text_clip/*.pthからデータを取得
        data_paths = glob.glob(os.path.join(constants.ET_DATA,  "generated_2.1.0", split, task, "feats", "text_clip", "*.pth"))
        feats_text_clip = []
        for data_path in tqdm_wrapper(data_paths):
            text_clip = torch.load(data_path)
            feats_text_clip.append(text_clip)
        feats_text_clip = torch.cat(feats_text_clip, dim=0) #(num_sentences, num_words, 768)

        #./data/generated_2.1.0/{split}/{task}/feats/mask_rcnn/bbox/*.pthからデータを取得
        data_paths = glob.glob(os.path.join(constants.ET_DATA,  "generated_2.1.0", split, task, "feats", "mask_rcnn", "bbox", "*.pth"))
        feats_mask_rcnn_bbox = []
        for data_path in tqdm_wrapper(data_paths):
            mask_rcnn_bbox = torch.load(data_path)
            feats_mask_rcnn_bbox.append(mask_rcnn_bbox)
        feats_mask_rcnn_bbox = torch.cat(feats_mask_rcnn_bbox, dim=0) #(num_sentences, num_words, 4)

        #./data/generated_2.1.0/{split}/{task}/feats/mask_rcnn/label/*.pthからデータを取得
        data_paths = glob.glob(os.path.join(constants.ET_DATA,  "generated_2.1.0", split, task, "feats", "mask_rcnn", "label", "*.pth"))
        feats_mask_rcnn_label = []
        for data_path in tqdm_wrapper(data_paths):
            mask_rcnn_label = torch.load(data_path)
            feats_mask_rcnn_label.append(mask_rcnn_label)
        feats_mask_rcnn_label = torch.cat(feats_mask_rcnn_label, dim=0) #(num_sentences, num_words, 1)

        return {"deberta": feats_deberta, "text_clip": feats_text_clip, "mask_rcnn_bbox": feats_mask_rcnn_bbox, "mask_rcnn_label": feats_mask_rcnn_label}

    def load_masks(self, key):
        '''
        load interaction masks from the disk
        '''
        if not hasattr(self, 'masks_lmdb'):
            self.masks_lmdb, self.masks = self.load_lmdb(
                self.masks_lmdb_path)
        masks_bytes = self.masks.get(key)
        masks_list = pickle.load(BytesIO(masks_bytes))
        masks = [data_util.decompress_mask_bytes(m) for m in masks_list]
        return masks

    def load_features(self, task_json, subgoal_idx=None, append_no_op=False):
        '''
        load features from task_json
        '''
        print("Loading features ...")
        feat = dict()
        # language inputs
        feat['lang'] = AlfredDataset.load_lang(task_json, subgoal_idx)

        # action outputs
        if not self.test_mode:
            # low-level action
            if append_no_op:
                assert len(task_json['num']['action_low']) == 1, 'Subgoal level data.'
                if task_json['num']['action_low'][0][-1]['action'] != 11:
                    no_op_high_idx = task_json['num']['action_low'][0][0]['high_idx']
                    task_json['num']['action_low'][0].append({
                        'high_idx': no_op_high_idx, 
                        'action': 11, 'action_high_args': {}, 
                        'centroid': [-1, -1], 
                        'mask': None, 
                        'valid_interact': 0
                    })
            feat['action'] = AlfredDataset.load_action(
                task_json, self.vocab_out, self.vocab_translate)
            # low-level valid interact
            feat['action_valid_interact'] = [
                a['valid_interact'] for a in sum(
                    task_json['num']['action_low'], [])]
            feat['object'] = AlfredDataset.load_object_classes(task_json, self.vocab_obj)
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
    def load_lang(task_json, subgoal_idx=None):
        '''
        load numericalized language from task_json
        '''
        #追加
        #tips/jsons_pklにあるように、task_jsonsは複数タスクのjsonがキーとともに格納されている。
        #キーはb'000001'のような形式。言語はそのままエンコードされてしまっている.
        # print("task_json",task_json)
        
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
        if 'action_low' not in task_json['num']:
            print("task_json['num]",task_json['num'])
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
            raise NotImplementedError('Unknown action_type {}'.format(action_type))
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