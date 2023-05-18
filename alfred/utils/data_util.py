import os
import re
import copy
import json
import glob
import torch
import lmdb
import shutil
import pickle
import warnings
import numpy as np

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn import CosineSimilarity
from copy import deepcopy
from vocab import Vocab
from pathlib import Path
import clip

from alfred.gen import constants
from alfred.gen.utils import image_util
from alfred.utils import helper_util, model_util

#追加
from .model_util import tokens_to_lang

def read_images(image_path_list):
    images = []
    for image_path in image_path_list:
        image_orig = Image.open(image_path)
        images.append(image_orig.copy())
        image_orig.close()
    return images


def read_traj_images(json_path, image_folder):
    root_path = json_path.parents[0]
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
    image_names = [None] * len(json_dict['plan']['low_actions'])
    for im_idx, im_dict in enumerate(json_dict['images']):
        if image_names[im_dict['low_idx']] is None:
            image_names[im_dict['low_idx']] = im_dict['image_name']
    before_last_image = json_dict['images'][-1]['image_name']
    last_image = '{:09d}.png'.format(int(before_last_image.split('.')[0]) + 1)
    image_names.append(last_image)
    fimages = [root_path / image_folder / im for im in image_names]
    if not any([os.path.exists(path) for path in fimages]):
        # maybe images were compressed to .jpg instead of .png
        fimages = [Path(str(path).replace('.png', '.jpg')) for path in fimages]
    if not all([os.path.exists(path) for path in fimages]):
        return None
    assert len(fimages) > 0
    # this reads on images (works with render_trajs.py)
    # fimages = sorted(glob.glob(os.path.join(root_path, image_folder, '*.png')))
    try:
        images = read_images(fimages)
    except:
        return None
    return images


def extract_features(images, extractor):
    if images is None:
        return None
    feat = extractor.featurize(images, batch=8)
    return feat.cpu()


#追加
def extract_clip_features(images, extractor):
    if images is None:
        return None
    feat = extractor.featurize_clip(images)
    return feat.cpu()

def extract_save_region_features(images, obj_predictor, clip_preprocess,  clip_model, subgoal_words, num_of_use=5, batch_size=8):
    '''
    get environment observation

    returns list of tensors of shape (subgoal_words * 5, 768)
    '''
    cossim = CosineSimilarity(dim=1, eps=1e-6)

    original_images = images

    all_bboxes = []
    all_labels = []

    #まずsubgoal_wordsをCLIP特徴量に変換
    tokenized_subgoal_labels = clip.tokenize(subgoal_words).to("cuda")
    subgoal_words_clip = clip_model.encode_text(tokenized_subgoal_labels) #(len(subgoal_words), 768)

    subgoal_words_dict = {subgoal_word: [] for  subgoal_word in enumerate(subgoal_words)}
    #imagesをbatch sizeに分割
    images = [original_images[i:i+batch_size] for i in range(0, len(original_images), batch_size)]
    for images_batch in images:
        images_batch_tensor = torch.stack([F.to_tensor(img) for img in images_batch]).to(torch.device('cuda'))
        outputs = obj_predictor.model.model(images_batch_tensor)
        bboxes = []
        labels = []

        #以下は各画像に対しての処理
        for idx, output in enumerate(outputs):
            _bbox = []
            _label = []
            _label_clip = []
            #各bboxにおいて、各subgoal_wordsに対しての類似度をclipで計算し、最も類似度が高いbboxをそれぞれのsubgoal_wordsに対して5つ選択する

            #以下は各bboxに対しての処理
            for pred_idx in range(len(output['scores'])):
                label = obj_predictor.model.vocab_pred[output['labels'][pred_idx].cpu().item()]

                #labelをclip特徴量に変換
                tokenized_labels = clip.tokenize(label).to("cuda")
                label_clip = clip_model.encode_text(tokenized_labels) #(len(label), 768)
                _label_clip.append(label_clip)

                #label_clipとsubgoal_words_clipの類似度を全て計算
                for idx, subgoal_word in enumerate(subgoal_words):
                    subgoal_words_dict[subgoal_word].append(cossim(label_clip, subgoal_words_clip[idx]).cpu().item())

                box = output['boxes'][pred_idx].cpu().numpy()
                
                c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
                region = np.array(images_batch[idx])
                region = region[c1[0]:c1[1], c2[0]:c2[1],:]
                if region.shape[0] * region.shape[1] > 0:
                    _bbox.append(Image.fromarray(region))
                    _label.append(label)

            top_k_indexs = {}
            #各labelにおいて、各subgoal_wordとの類似度が最も高いものを5つ選択する
            for subgoal_word in subgoal_words:
                #各subgoal_wordに対して、類似度が高い順にソート
                sort_idx = np.argsort(subgoal_words_dict[subgoal_word])[::-1]
                #類似度が高いindexを5つ取得
                sort_idx = sort_idx[:num_of_use]
                #この5つのindexに対応するboxとlabelを取得する
                top_k_indexs[subgoal_word] = sort_idx
                for box, label in zip(np.array(_bbox)[sort_idx], np.array(_label)[sort_idx]):
                    bboxes.append(box)
                    labels.append(label)


def get_maskrcnn_features(image, obj_predictor, clip_preprocess,  clip_model, subgoal_words, subgoal_words_clip, num_of_use=5, batch_size=8):
    '''
    get environment observation

    returns list of tensors of shape (subgoal_words * 5, 768)
    '''
    cossim = CosineSimilarity(dim=0, eps=1e-6)

    #まずsubgoal_wordsをCLIP特徴量に変換
    subgoal_words_idx_dict = {subgoal_word: np.array([]) for  subgoal_word in subgoal_words}

    #top5つの類似度を保存する
    bboxes = []
    labels = []

    image = F.to_tensor(image).to(torch.device("cuda"))
    output = obj_predictor.model.model(image[None])[0]

    #以下は各画像に対しての処理
    _bbox = []
    _label = []
    _label_clip = []

    #各bboxにおいて、各subgoal_wordsに対しての類似度をclipで計算し、最も類似度が高いbboxをそれぞれのsubgoal_wordsに対して5つ選択する

    #以下は各bboxに対しての処理
    for pred_idx in range(len(output['scores'])):
        label = obj_predictor.model.vocab_pred[output['labels'][pred_idx].cpu().item()]

        box = output['boxes'][pred_idx].detach().cpu().numpy() 
        
        c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
        region = image.detach().cpu().numpy()
        # regio = image.cpu().numpy()
        region = region[c1[0]:c1[1], c2[0]:c2[1],:]
        if region.shape[0] * region.shape[1] > 0:
            #labelをclip特徴量に変換
            tokenized_labels = clip.tokenize(label).to("cuda")
            label_clip = clip_model.encode_text(tokenized_labels) #(len(label), 768)
            _label_clip.append(label_clip)

            # print("label_clip[0]", label_clip[0].shape)
            #label_clipとsubgoal_words_clipの類似度を全て計算
            for idx, subgoal_word in enumerate(subgoal_words):
                # print("subgoal_words_clip[idx]", subgoal_words_clip[idx].shape)
                np.append(subgoal_words_idx_dict[subgoal_word], cossim(label_clip[0], subgoal_words_clip[idx]).detach().cpu().item())

            _bbox.append(Image.fromarray(region, mode="RGB"))
            _label.append(label)

    top_k_indexs = {}
    #各labelにおいて、各subgoal_wordとの類似度が最も高いものを5つ選択する
    for subgoal_word in subgoal_words:
        #各subgoal_wordに対して、類似度が高い順にソート
        sort_idx = np.argsort(subgoal_words_idx_dict[subgoal_word])[::-1]
        #類似度が高いindexを5つ取得
        sort_idx = sort_idx[:num_of_use]
        #この5つのindexに対応するboxとlabelを取得する
        top_k_indexs[subgoal_word] = sort_idx
        # for box, label in zip(np.array(_bbox)[sort_idx], np.array(_label_clip)[sort_idx]):
        for i in range(len(_bbox)):
            bboxes.append(_bbox[i])
            labels.append(_label[i])
    
    #各boxに対して、CLIP特徴量を計算する
    if len(bboxes) > 0:
        feats = encode_clip_image(clip_preprocess, clip_model, bboxes, device="cuda:0")
    else:
        feats = torch.zeros(1, 768).to("cuda")
    
    return feats
        
        

def encode_clip_image(clip_preprocess, clip_model, images, device="cuda:0"):
    images = [clip_preprocess(image).unsqueeze(0).to("cuda") for image in images]
    with torch.no_grad():
        feats = [clip_model.encode_image(image) for image in images]

        #featsがからであれば、0を入れる
        if len(feats) == 0:
            feats = torch.zeros(1, 768).to(device)
        feats = torch.cat(feats, dim=0) # [len(images),768]

    return feats

    # torch.save(all_bboxes, os.path.join(input_path, "maskrcnn_bbox.pth"))
    # torch.save(all_labels, os.path.join(input_path, "maskrcnn_label.pth"))



    # # frame = Image.fromarray(images)
    # feats = []
    # labels = []
    # for image in images:
    #     rcnn_pred = obj_predictor.predict_objects(image)
    #     regions = []
    #     scores = []
    #     label = []
    #     for pred in rcnn_pred:
    #         box = pred.box
    #         c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
    #         regio = np.array(image)
    #         regio = region[c1[0]:c1[1], c2[0]:c2[1],:]
    #         if region.shape[0] * region.shape[1] > 0:
    #             regions.append(Image.fromarray(region))
    #             label.append(pred.label)
    #             scores.append(pred.score)
        
    #     #scoreの上位3つを抽出
    #     if len(scores) > 2:
    #         top3_idx = np.argsort(scores)[::-1][:3]
    #         regions = [regions[i] for i in top3_idx]
    #         label = [label[i] for i in top3_idx]
    #     else:
    #         #zeroを追加
    #         regions = regions + [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(3 - len(regions))]
    #         label = label + ["" for _ in range(3 - len(label))]
            

    #     feat = extractor.featurize_clip(regions).unsqueeze(0)
    #     feats.append(feat)
    #     label = extractor.tokenize_featurize_clip(label)
    #     labels.append(label)
    
    # if len(feats) > 0: 
    #     feats = torch.cat(feats, dim=0)
    #     labels = torch.cat(labels, dim=0)
    # else:
    #     feats = torch.zeros(len(images), 3, 768)
    #     labels = torch.zeros(len(images), 3, 768)
    # # print("feats.shape:", feats.shape)
    # # print("len(labels):", len(labels))
    # return feats, labels

def decompress_mask_alfred(mask_compressed_alfred):
    '''
    decompress mask array from ALFRED compression (initially contained in jsons)
    '''
    mask = np.zeros((constants.DETECTION_SCREEN_WIDTH,
                     constants.DETECTION_SCREEN_HEIGHT))
    for start_idx, run_len in mask_compressed_alfred:
        for idx in range(start_idx, start_idx + run_len):
            mask[idx // constants.DETECTION_SCREEN_WIDTH,
                 idx % constants.DETECTION_SCREEN_HEIGHT] = 1
    return mask


def decompress_mask_bytes(mask_bytes):
    '''
    decompress mask given as a binary string and cast them to tensors (for optimization)
    '''
    mask_pil = image_util.decompress_image(mask_bytes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = transforms.ToTensor()(mask_pil)
    return mask


def process_masks(traj):
    masks = []
    for action_low in traj['plan']['low_actions']:
        if 'mask' in action_low['discrete_action']['args']:
            mask = decompress_mask_alfred(
                action_low['discrete_action']['args'].pop('mask'))
            masks.append(mask)
        else:
            masks.append(None)

    masks_compressed = []
    for mask in masks:
        if mask is not None:
            mask = image_util.compress_image(mask.astype('int32'))
        masks_compressed.append(mask)
    return masks_compressed


def process_traj(traj_orig, traj_path, r_idx, preprocessor):
    # copy trajectory
    traj = traj_orig.copy()
    # root & split
    traj['root'] = str(traj_path)
    partition = traj_path.parents[2 if 'tests_' not in str(traj_path) else 1].name
    traj['split'] = partition
    traj['repeat_idx'] = r_idx
    # numericalize actions for train/valid splits
    if ('test' not in partition) or (partition == 'pseudo_test'): # expert actions are not available for the test set
        preprocessor.process_actions(traj_orig, traj)
    # numericalize language
    preprocessor.process_language(traj_orig, traj, r_idx)
    return traj


def gather_feats(files, output_path):
    print('Writing features to LMDB')
    if output_path.is_dir():
        shutil.rmtree(output_path)
    lmdb_feats = lmdb.open(str(output_path), 700*1024**3, writemap=True)
    with lmdb_feats.begin(write=True) as txn_feats:
        for idx, path in tqdm(enumerate(files)):
            #変更
            # traj_feats = torch.load(path).numpy()
            # txn_feats.put('{:06}'.format(idx).encode('ascii'), traj_feats.tobytes())
            traj_feats = torch.load(path)
            value = pickle.dumps(traj_feats)
            txn_feats.put('{:06}'.format(idx).encode('ascii'), value)
    lmdb_feats.close()


def gather_masks(files, output_path):
    print('Writing masks to LMDB')
    if output_path.is_dir():
        shutil.rmtree(output_path)
    lmdb_masks = lmdb.open(str(output_path), 50*1024**3, writemap=True)
    with lmdb_masks.begin(write=True) as txn_masks:
        for idx, path in tqdm(enumerate(files)):
            with open(path, 'rb') as f:
                masks_list = pickle.load(f)
                masks_list = [el for el in masks_list if el is not None]
                masks_buffer = BytesIO()
                pickle.dump(masks_list, masks_buffer)
                txn_masks.put('{:06}'.format(idx).encode('ascii'), masks_buffer.getvalue())
    lmdb_masks.close()


def gather_jsons(files, output_path):
    print('Writing JSONs to PKL')
    if output_path.exists():
        os.remove(output_path)
    jsons = {}
    for idx, path in tqdm(enumerate(files)):
        with open(path, 'rb') as f:
            jsons_idx = pickle.load(f)
            jsons['{:06}'.format(idx).encode('ascii')] = jsons_idx
    with output_path.open('wb') as f:
        pickle.dump(jsons, f)


def get_preprocessor(PreprocessorClass, subgoal_ann, lock, vocab_path=None):
    if vocab_path is None:
        init_words = ['<<pad>>', '<<seg>>', '<<goal>>', '<<mask>>']
    else:
        init_words = []
    vocabs_with_lock = {
        'word': helper_util.VocabWithLock(deepcopy(init_words), lock),
        'action_low': helper_util.VocabWithLock(deepcopy(init_words), lock),
        'action_high': helper_util.VocabWithLock(deepcopy(init_words), lock)}
    if vocab_path is not None:
        vocabs_loaded = torch.load(vocab_path)
        for vocab_name, vocab in vocabs_with_lock.items():
            loaded_dict = vocabs_loaded[vocab_name].to_dict()
            for i, w in enumerate(loaded_dict['index2word']):
                vocab.word2index(w, train=True)
                vocab.counts[w] = loaded_dict['counts'][w]

    preprocessor = PreprocessorClass(vocabs_with_lock, subgoal_ann)
    return preprocessor


def tensorize_and_pad(batch, device, pad):
    '''
    cast values to torch tensors, put them to the correct device and pad sequences
    '''
    device = torch.device(device)
    input_dict, gt_dict, feat_dict = dict(), dict(), dict()
    task_path, traj_data, feat_list = list(zip(*batch))
    for key in feat_list[0].keys():
        feat_dict[key] = [el[key] for el in feat_list]
    # check that all samples come from the same dataset
    assert len(set([t['dataset_name'] for t in traj_data])) == 1
    # feat_dict keys that start with these substrings will be assigned to input_dict
    input_keys = {'lang', 'frames'}
    # the rest of the keys will be assigned to gt_dict

    for k, v in feat_dict.items():
        dict_assign = input_dict if any([k.startswith(s) for s in input_keys]) else gt_dict
        if k.startswith('lang'):
            # no preprocessing should be done here
            seqs = [torch.tensor(vv if vv is not None else [pad, pad], device=device).long() for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
            dict_assign['lengths_' + k] = torch.tensor(list(map(len, seqs)))
            length_max_key = 'length_' + k + '_max'
            if ':' in k:
                # for translated length keys (e.g. lang:lmdb/1x_det) we should use different names
                length_max_key = 'length_' + k.split(':')[0] + '_max:' + ':'.join(k.split(':')[1:])
            dict_assign[length_max_key] = max(map(len, seqs))
        elif k in {'object'}:
            # convert lists with object indices to tensors
            seqs = [torch.tensor(vv, device=device, dtype=torch.long)
                    for vv in v if len(vv) > 0]
            dict_assign[k] = seqs
        elif k in {'goal_progress', 'subgoals_completed'}:
            # auxillary padding
            seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
        elif k in {'frames'}:
            # 追加
            # ex. seqs[0]:[44, 512, 7, 7]
            # frames features were loaded from the disk as tensors
            # print("v[0][0].shape", v[0][0].shape)# torch.Size([27, 512, 7, 7])
            # print("v[0][1].shape", v[0][1].shape)#torch.Size([27, 768])
            # print("v[1][0].shape", v[1][0].shape)#torch.Size([72, 512, 7, 7])
            # print("v[1][1].shape", v[1][1].shape)#torch.Size([72, 768])
            seqs = [vv[0].clone().detach().to(device).type(torch.float) for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            # print("pad_seq.shape",pad_seq.shape)#torch.Size([2, 72, 512, 7, 7])
            # dict_assign[k] = pad_seq
            #pad_seqとv[0][1]とv[1][1]を結合
            if len(v) == 1:
                pad_seq = [pad_seq, v[0][1]]
            else:
                pad_seq = [pad_seq, v[0][1], v[1][1]]
            # print("pad_seq[0].shape",pad_seq[0].shape)
            dict_assign[k] = pad_seq
            dict_assign['lengths_' + k] = torch.tensor(list(map(len, seqs)))
            dict_assign['length_' + k + '_max'] = max(map(len, seqs))
        else:
            # default: tensorize and pad sequence
            seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq

    return task_path, traj_data, input_dict, gt_dict


def sample_batches(iterators, device, pad, args):
    '''
    sample a batch from each iterator, return Nones if the iterator is empty
    '''
    batches_dict = {}
    for dataset_id, iterator in iterators.items():
        try:
            batches = next(iterator)
        except StopIteration as e:
            return None
        dataset_name = dataset_id.split(':')[1]
        task_path, traj_data, input_dict, gt_dict = tensorize_and_pad(
            batches, device, pad)
        batches_dict[dataset_name] = (task_path, traj_data, input_dict, gt_dict)
    return batches_dict


def load_vocab(name, ann_type='lang'):
    '''
    load a vocabulary from the dataset
    '''
    path = os.path.join(constants.ET_DATA, name, constants.VOCAB_FILENAME)
    vocab_dict = torch.load(path)
    #追加
    #vocab: {'word': Vocab(899), 'action_low': Vocab(17), 'action_high': Vocab(104)}
    # result = tokens_to_lang([45, 8, 5, 202, 194, 31, 41, 16, 5, 193, 195, 196], vocab_dict['word'])
    # print("result: {}".format(result))
    # print("vocab: {}".format(vocab_dict['word'].index2word([45, 8, 5, 202, 194, 31, 41, 16, 5, 193, 195, 196])))
    # set name and annotation types
    for vocab in vocab_dict.values():
        vocab.name = name
        vocab.ann_type = ann_type
    return vocab_dict


def get_feat_shape(visual_archi, compress_type=None):
    '''
    Get feat shape depending on the training archi and compress type
    '''
    if visual_archi == 'fasterrcnn':
        # the RCNN model should be trained with min_size=224
        feat_shape = (-1, 2048, 7, 7)
    elif visual_archi == 'maskrcnn':
        # the RCNN model should be trained with min_size=800
        feat_shape = (-1, 2048, 10, 10)
    elif visual_archi == 'resnet18':
        feat_shape = (-1, 512, 7, 7)
    else:
        raise NotImplementedError('Unknown archi {}'.format(visual_archi))

    if compress_type is not None:
        if not re.match(r'\d+x', compress_type):
            raise NotImplementedError('Unknown compress type {}'.format(compress_type))
        compress_times = int(compress_type[:-1])
        feat_shape = (
            feat_shape[0], feat_shape[1] // compress_times,
            feat_shape[2], feat_shape[3])
    return feat_shape


def feat_compress(feat, compress_type):
    '''
    Compress features by channel average pooling
    '''
    assert re.match(r'\d+x', compress_type) and len(feat.shape) == 4
    times = int(compress_type[:-1])
    assert feat.shape[1] % times == 0
    feat = feat.reshape((
        feat.shape[0], times,
        feat.shape[1] // times,
        feat.shape[2],
        feat.shape[3]))
    feat = feat.mean(dim=1)
    return feat


def read_dataset_info(data_name):
    '''
    Read dataset a feature shape and a feature extractor checkpoint path
    '''
    path = os.path.join(constants.ET_DATA, data_name, 'params.json')
    with open(path, 'r') as f_params:
        params = json.load(f_params)
    return params
