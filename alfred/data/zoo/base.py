import os
import pickle
import lmdb
import torch
import warnings
import numpy as np

from torch.utils.data import Dataset as TorchDataset
from copy import deepcopy
from tqdm import tqdm

from alfred.gen import constants
from alfred.utils import data_util


class BaseDataset(TorchDataset):
    def __init__(self, name, partition, args, ann_type):
        path = os.path.join(constants.ET_DATA, name)
        self.partition = partition
        self.name = name
        self.args = args
        if ann_type not in ('lang', 'frames', 'lang_frames'):
            raise ValueError('Unknown annotation type: {}'.format(ann_type))
        self.ann_type = ann_type
        self.test_mode = False
        self.pad = 0

        # read information about the dataset
        self.dataset_info = data_util.read_dataset_info(name)
        if self.dataset_info['visual_checkpoint']:
            print('The dataset was recorded using model at {}'.format(
                self.dataset_info['visual_checkpoint']))

        # load data
        self._length = self.load_data(path)
        if self.args.fast_epoch:
            self._length = 16
        print('{} dataset size = {}'.format(partition, self._length))

        # load vocabularies for input language and output actions
        vocab = data_util.load_vocab(name, ann_type)
        self.vocab_in = vocab['word']
        #追加
        #vocab_in : Vocab(899)
        out_type = 'action_low' if args.model == 'transformer' else 'action_high'
        #追加
        #vocab_out : Vocab(17)
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
            print("Loading jsons.pkl ... (it might be huge) ")
            with open(os.path.join(path, self.partition, 'jsons.pkl'), 'rb') as jsons_file:
                jsons = pickle.load(jsons_file)
            self.jsons_and_keys = []
            for idx in tqdm(range(len(jsons))):
                key = '{:06}'.format(idx).encode('ascii')
                task_jsons = jsons[key]
                for json in task_jsons:
                    # compatibility with the evaluation
                    if 'task' in json and isinstance(json['task'], str):
                        pass
                    else:
                        json['task'] = '/'.join(json['root'].split('/')[-3:-1])
                    # add dataset idx and partition into the json
                    json['dataset_name'] = self.name
                    self.jsons_and_keys.append((json, key))
                    # if the dataset has script annotations, do not add identical data
                    if len(set([str(j['ann']['instr']) for j in task_jsons])) == 1:
                        break

        # return the true length of the loaded data
        return len(self.jsons_and_keys) if jsons else None

    def load_frames(self, key):
        '''
        load image features from the disk
        '''
        if not hasattr(self, 'feats_lmdb'):
            self.feats_lmdb, self.feats = self.load_lmdb(
                self.feats_lmdb_path)
        # feats_bytes = self.feats.get(key)
        # feats_numpy = np.frombuffer(
        #     feats_bytes, dtype=np.float32).reshape(self.dataset_info['feat_shape'])
        #変更(only clip)
        feats_list = pickle.loads(self.feats.get(key))
        #追加
        #feats_numpy: ex. [99, 512, 7, 7]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #変更
            # frames = torch.tensor(feats_numpy)
        return feats_list
        # return frames

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