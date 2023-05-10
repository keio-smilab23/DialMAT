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

def main():
    env = ThorEnv(x_display=1)
    traj_data = json.load(open("testset/dialfred_testset_final/0001.json", "r"))
    
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']
    
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    env.step(dict(traj_data['scene']['init_action']))

if __name__ == "__main__":
    main()