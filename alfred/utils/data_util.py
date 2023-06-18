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
import torchvision
import torch.nn.functional as FF
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from alfred.gen import constants
from alfred.gen.utils import image_util
from alfred.utils import helper_util, model_util

import torch.multiprocessing as multiprocessing

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

def save_features_to_path(output_path, feature):
    """
    Save features to a file.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "wb") as f:
        pickle.dump(feature, f)

def load_features_from_path(path):
    """
    Load features from a file.
    """

    #まずpathにファイルがあるかを確認する
    if os.path.exists(path):
        #あれば読み込む
        with open(path, "rb") as f:
            features = pickle.load(f)
    else:
        return None
        
    return features
        
def reshape_with_pad(x,shape):
    if x.shape[0] < shape[0]:
        zeros = torch.zeros(shape[0] - x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(x.device)
        x = torch.cat([x, zeros])
    return x

def encode_clip_image(clip_preprocess, clip_model, images, device="cuda:0"):
    images = [clip_preprocess(image).unsqueeze(0).to("cuda") for image in images]
    with torch.no_grad():
        feats = [clip_model.encode_image(image) for image in images]

        #featsがからであれば、0を入れる
        if len(feats) == 0:
            feats = torch.zeros(1, 768).to(device)
        feats = torch.cat(feats, dim=0) # [len(images),768]

    return feats


def get_maskrcnn_resnet_features(images_path, obj_predictor,  clip_model, subgoal_words, subgoal_limit=8, num_of_use=2, batch_size=8):
    '''
    get environment observation
    
    get_maskrcnn_featuresの処理をbatch化したもの

    returns list of tensors of shape (subgoal_words * 5, 768)
    '''
    images = [Image.open(img_path) for img_path in images_path]

    # len(subgoal_words) <= subgoal_limit
    if len(subgoal_words) > subgoal_limit:
        subgoal_words = subgoal_words[:subgoal_limit]

    subgoal_words_clip = clip.tokenize(subgoal_words).to("cuda")
    subgoal_words_clip = clip_model.encode_text(subgoal_words_clip) # (N,D)

    original_images = images

    all_bboxes = []

    #divide into batch
    images = [original_images[i:i+batch_size] for i in range(0, len(original_images), batch_size)]

    for images_batch in images:
        tensor_images = torch.stack([F.to_tensor(img).cuda() for img in images_batch]).to(torch.device('cuda'))
        outputs = obj_predictor.model.model(tensor_images)

        regions, labels = [], []

        #for single image
        for idx, output in enumerate(outputs):
            _regions, _labels = [], []

            #for single bbox
            for pred_idx in range(len(output['scores'])):
                label = obj_predictor.model.vocab_pred[output['labels'][pred_idx].cpu().item()]
                box = output['boxes'][pred_idx].detach().cpu().numpy() 
                mask = output['masks'][pred_idx].detach().cpu().numpy()
                image = tensor_images[idx]
                c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
                region = tensor_images[idx,:,c1[0]:c1[1], c2[0]:c2[1]] # TODO: H,W逆かも
                if region.shape[1] * region.shape[2] > 0:
                        region_resized = torchvision.transforms.functional.resize(region,size=(300,300))
                        _labels.append(label)
                        _regions.append(region_resized.unsqueeze(0).cuda())
                        print("")

            if len(_regions) == 0:
                feats, _labels = torch.zeros(1, len(subgoal_words), num_of_use,  2048, 25, 25).cuda(), torch.zeros(1, len(subgoal_words), num_of_use, 768).cuda()
                regions.append(feats)
                labels.append(_labels)
                continue
            
            _regions = torch.cat(_regions,dim=0) # (N,3,224,224)

            tokenized_labels = clip.tokenize(_labels).to("cuda")
            label_clip = clip_model.encode_text(tokenized_labels) # (N,D)
            cossim = FF.cosine_similarity(subgoal_words_clip.unsqueeze(1), label_clip, dim=-1) # (M,D) @ (D,N)
            sort_idxs = torch.argsort(cossim, dim=1, descending=True)
            #sort_idxsの各行において、subgoal_words_clipの各単語との類似度が高い順にソートされている
            #上位num_of_use個のインデックスを取得
            sort_idxs = sort_idxs[:,:num_of_use]

            # 各行で一番大きい値が格納されたインデックスを取得
            # sort_idxs = torch.argmax(sort_idxs, dim=1)

            # print("subgoal_words_clip", subgoal_words_clip.shape)
            #top1bboxes
            top_bboxes = torch.cat([_regions[sort_idxs[i]].unsqueeze(0) for i in range(sort_idxs.shape[0])],dim=0)
            # top_label_clip = torch.cat([label_clip[sort_idxs[i]].unsqueeze(0) for i in range(sort_idxs.shape[0])],dim=0)

            if len(top_bboxes) == 0:
                feats, _labels = torch.zeros(1, len(subgoal_words), num_of_use, 2048, 25, 25).cuda(), torch.zeros(1, len(subgoal_words), num_of_use, 768).cuda()
                regions.append(feats)
                # labels.append(_labels)
                continue
            
            
            with torch.no_grad():
                batch_images = top_bboxes.view(-1, 3, 300, 300) # (subgoal_words * num_of_use, 3, 224, 224)
                # feats = clip_model.encode_image(batch_images) # (subgoal_words * num_of_use,768)
                #batch_imagesを<PIL.Image.Image image mode=RGB size=300x300>のような形式のリストに変換
                batch_images = [transforms.ToPILImage()(img) for img in batch_images]

                feats = extract_features(batch_images, obj_predictor)
                feats = feats.view(len(subgoal_words), sort_idxs.shape[1], 2048, 25, 25) # (subgoal_words, num_of_use, 768)
                if sort_idxs.shape[1] < num_of_use:
                    # top_label_clip = torch.cat([top_label_clip, torch.zeros(top_label_clip.shape[0], num_of_use-sort_idxs.shape[1], 768).cuda()], dim=1)
                    feats = torch.cat([feats, torch.zeros(feats.shape[0], num_of_use-sort_idxs.shape[1], 2048, 25, 25).cuda()], dim=1)
            regions.append(feats.unsqueeze(0).cuda()) #(1, subgoal_words, num_of_use, 768)
            # labels.append(top_label_clip.unsqueeze(0))#(1, subgoal_words, num_of_use, 768)
        all_bboxes.append(torch.cat(regions,dim=0)) #(batch_size, subgoal_words, num_of_use, 768)
        # all_labels.append(torch.cat(labels,dim=0)) #(batch_size,subgoal_words, num_of_use, 768)
    
    all_bboxes = torch.cat(all_bboxes,dim=0) #(len(images), subgoal_words, num_of_use, 768)
    # all_labels = torch.cat(all_labels,dim=0) #(len(images), subgoal_words, num_of_use, 768)
    print("all_bboxes", all_bboxes.shape)
    return all_bboxes.cuda()


def get_maskrcnn_features(image, obj_predictor, clip_model, subgoal_words, num_of_use=1, subgoal_limit=4):
    '''
    get environment observation
    
    subgoal_words_clip: (subgoal_nums, D)
    各bboxにおいて、各subgoal_wordsに対しての類似度をclipで計算し、最も類似度が高いbboxをそれぞれのsubgoal_wordsに対して5つ選択する

    returns list of tensors of shape (subgoal_words * 5, 768)
    '''
    subgoal_words_clip = clip.tokenize(subgoal_words).to("cuda")
    subgoal_words_clip = clip_model.encode_text(subgoal_words_clip) # (N,D)

    image = F.to_tensor(image).cuda()
    output = obj_predictor.model.model(image[None])[0]


    #for single image
    _regions, _labels = [], []

    #for single bbox
    for pred_idx in range(len(output['scores'])):
        label = obj_predictor.model.vocab_pred[output['labels'][pred_idx].cpu().item()]
        box = output['boxes'][pred_idx].detach().cpu().numpy() 
        c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
        region = image[:,c1[0]:c1[1], c2[0]:c2[1]] # TODO: H,W逆かも
        if region.shape[1] * region.shape[2] > 0:
                region_resized = torchvision.transforms.functional.resize(region,size=(224,224))
                _labels.append(label)
                _regions.append(region_resized.unsqueeze(0))

    if len(_regions) == 0:
        feats, _labels = torch.zeros(1, subgoal_limit, num_of_use, 768).cuda(), torch.zeros(1, subgoal_limit, num_of_use, 768).cuda()
        return feats, _labels, torch.tensor([subgoal_limit]).cuda()

    _regions = torch.cat(_regions,dim=0) # (N,3,224,224)

    tokenized_labels = clip.tokenize(_labels).to("cuda")
    label_clip = clip_model.encode_text(tokenized_labels) # (N,D)
    cossim = FF.cosine_similarity(subgoal_words_clip.unsqueeze(1), label_clip, dim=-1) # (M,D) @ (D,N)
    sort_idxs = torch.argsort(cossim, dim=1, descending=True)
    #sort_idxsの各行において、subgoal_words_clipの各単語との類似度が高い順にソートされている
    #上位num_of_use個のインデックスを取得
    sort_idxs = sort_idxs[:,:num_of_use]

    # 各行で一番大きい値が格納されたインデックスを取得
    # sort_idxs = torch.argmax(sort_idxs, dim=1)

    # print("subgoal_words_clip", subgoal_words_clip.shape)
    #top1bboxes
    top_bboxes = torch.cat([_regions[sort_idxs[i]].unsqueeze(0) for i in range(sort_idxs.shape[0])],dim=0)
    top_label_clip = torch.cat([label_clip[sort_idxs[i]].unsqueeze(0) for i in range(sort_idxs.shape[0])],dim=0)

    if len(top_bboxes) == 0:
        feats, _labels = torch.zeros(1, subgoal_limit, num_of_use, 768).cuda(), torch.zeros(1, subgoal_limit, num_of_use, 768).cuda()
        return feats, _labels, torch.tensor([subgoal_limit]).cuda()
    
    # c.f. https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
    my_transform = Compose([
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    with torch.no_grad():
        batch_images = my_transform(top_bboxes) # (subgoal_words, num_of_use, 3, 224, 224)
        batch_images = batch_images.view(-1, 3, 224, 224) # (subgoal_words * num_of_use, 3, 224, 224)
        feats = clip_model.encode_image(batch_images) # (subgoal_words * num_of_use,768)
        feats = feats.view(len(subgoal_words), sort_idxs.shape[1], 768) # (subgoal_words, num_of_use, 768)
        if sort_idxs.shape[1] < num_of_use:
            top_label_clip = torch.cat([top_label_clip, torch.zeros(top_label_clip.shape[0], num_of_use-sort_idxs.shape[1], 768).cuda()], dim=1)
            feats = torch.cat([feats, torch.zeros(feats.shape[0], num_of_use-sort_idxs.shape[1], 768).cuda()], dim=1)
        # subgoal_words.shape[0]がsubgoal_limitよりも小さい場合には、0でpaddingする
        if subgoal_words_clip.shape[0] < subgoal_limit:
            top_label_clip = torch.cat([top_label_clip, torch.zeros(subgoal_limit-subgoal_words_clip.shape[0], num_of_use, 768).cuda()], dim=0)
            feats = torch.cat([feats, torch.zeros(subgoal_limit-subgoal_words_clip.shape[0], num_of_use, 768).cuda()], dim=0)

        top_label_clip = top_label_clip.unsqueeze(0)
        feats = feats.unsqueeze(0)

    if len(subgoal_words) < subgoal_limit:
        length_subgoal_words = len(subgoal_words)
    else:
        length_subgoal_words = subgoal_limit
        
    return feats, top_label_clip, torch.tensor([length_subgoal_words]).cuda()

def get_maskrcnn_features_with_mask(image,  extractor_bbox, pretrained_resnet, clip_model, subgoal_words, subgoal_limit=4):
    '''
    get environment observation
    
    get_maskrcnn_featuresの処理をbatch化したもの

    returns list of tensors of shape (subgoal_words * 5, 768)
    '''
    # len(subgoal_words) <= subgoal_limit
    if len(subgoal_words) > subgoal_limit:
        subgoal_words = subgoal_words[:subgoal_limit]

    subgoal_words_clip = clip.tokenize(subgoal_words).to("cuda")
    subgoal_words_clip = clip_model.encode_text(subgoal_words_clip) # (N,D)

    # ResNetの前処理を適用
    resnet_preprocess = transforms.Compose([
        transforms.ToPILImage(),  # テンソルをPIL画像に変換
        transforms.Resize((224, 224)),  # 画像サイズをリサイズ
        transforms.ToTensor(),  # PIL画像をテンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    #divide into batch

    tensor_image = F.to_tensor(image).cuda() 
    output = extractor_bbox.model.model(tensor_image[None])[0]

    _regions, _labels, _mask = [], [], []

    #for single bbox
    for pred_idx in range(len(output['scores'])):
        label = extractor_bbox.model.vocab_pred[output['labels'][pred_idx].cpu().item()]
        box = output['boxes'][pred_idx].detach().cpu().numpy() 
        c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
        region = tensor_image[:, c1[0]:c1[1], c2[0]:c2[1]] # TODO: H,W逆かも
        if region.shape[1] * region.shape[2] > 0:
                region_resized = torchvision.transforms.functional.resize(region,size=(224,224))
                _labels.append(label)
                _regions.append(region_resized.unsqueeze(0))
                mask = output['masks'][pred_idx].detach().cpu().numpy()
                #to tensor
                mask = torch.from_numpy(mask)
                mask_rgb = torch.cat([mask, mask, mask], dim=0) 
                masked_image = tensor_image.cuda() * mask_rgb.cuda()
                # masked_image = image.cuda() * mask_rgb.cuda()
                _mask.append(masked_image.unsqueeze(0))

    if len(_regions) == 0:
        return torch.zeros(1, subgoal_limit, 768).cuda(), torch.zeros(1, subgoal_limit, 768).cuda(),\
             torch.zeros(1, subgoal_limit, 1000).cuda(), 0
    
    _regions = torch.cat(_regions,dim=0) # (N,3,224,224)
    _mask = torch.cat(_mask,dim=0) # (N, 1, 300,300)
    # print("_mask", _mask)
    tokenized_labels = clip.tokenize(_labels).to("cuda")
    label_clip = clip_model.encode_text(tokenized_labels) # (N,D)
    cossim = FF.cosine_similarity(subgoal_words_clip.unsqueeze(1).float(), label_clip.float(), dim=-1) # (M,D) @ (D,N)
    sort_idxs = torch.argsort(cossim, dim=1, descending=True)

    use_index = []
    for i in range(sort_idxs.shape[0]):
        use_index.append(int(sort_idxs[i][0]))

    #重複があったら1,2,...のようにインデックスをずらして取得、重複がなければそのまま
    for i in range(len(use_index)):
        # temp_list = use_index.copy()
        # temp_list.pop(i)
        if use_index[i] in use_index[:i]:
            for j in range(int(sort_idxs.shape[1])):
                if int(sort_idxs[i][j]) not in use_index[:i]:
                    use_index[i] = int(sort_idxs[i][j])
                    break
    # top_bboxes = torch.cat([_regions[sort_idxs[i]].unsqueeze(0) for i in range(sort_idxs.shape[0])],dim=0)
    top_bboxes = torch.cat([_regions[use_index[i]].unsqueeze(0) for i in range(len(use_index))],dim=0)
    top_label_clip = torch.cat([label_clip[use_index[i]].unsqueeze(0) for i in range(len(use_index))],dim=0)
    if len(top_bboxes) == 0:
        return torch.zeros(1, subgoal_limit, 768).cuda(), torch.zeros(1, subgoal_limit, 768).cuda(),\
             torch.zeros(1, subgoal_limit, 1000).cuda(), 0

    # 特徴量の結合
    # print("output_resnet.shape", output_resnet.shape)
    # c.f. https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
    my_transform = Compose([
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    with torch.no_grad():
        # batch_images = top_bboxes.view(-1, 3, 224, 224) # (subgoal_words * num_of_use, 3, 224, 224)
        # batch_images_maskrcnn = [transforms.ToPILImage()(img) for img in batch_images]

        # feats_mask = extract_features(batch_images_maskrcnn, extractor_bbox)

        batch_images = my_transform(top_bboxes) # (subgoal_words, num_of_use, 3, 224, 224)
        batch_images = batch_images.view(-1, 3, 224, 224) # (subgoal_words * num_of_use, 3, 224, 224)
        feats_clip = clip_model.encode_image(batch_images) # (subgoal_words * num_of_use,768)
        feats_clip = feats_clip.view(len(use_index), 768) # (subgoal_words, num_of_use, 768)

        mask = torch.stack([resnet_preprocess(_mask[use_index[i]]) for i in range(len(use_index))])
        output_resnet = pretrained_resnet(mask.cuda()) # [subgoal_word, 1000]
        
        if subgoal_limit > len(use_index):
            top_label_clip = torch.cat([top_label_clip, torch.zeros(subgoal_limit - len(use_index), 768).cuda()], dim=0)
            # feats_mask = torch.cat([feats_mask.cuda(), torch.zeros(subgoal_limit - len(use_index), 4, 25, 25).cuda()], dim=0)
            feats_clip = torch.cat([feats_clip, torch.zeros(subgoal_limit - len(use_index), 768).cuda()], dim=0)
            output_resnet = torch.cat([output_resnet, torch.zeros(subgoal_limit - len(use_index), 1000).cuda()], dim=0)
    
    return feats_clip.unsqueeze(0).cpu(), top_label_clip.unsqueeze(0).cpu(), output_resnet.unsqueeze(0).cpu(), len(use_index)

def get_maskrcnn_features_similarity_lmdb(images, obj_predictor, pretrained_resnet, clip_model, subgoal_words, subgoal_limit=5, num_of_use=1, batch_size=7):
    '''
    get environment observation
    
    get_maskrcnn_featuresの処理をbatch化したもの

    returns list of tensors of shape (subgoal_words * 5, 768)
    '''
    # len(subgoal_words) <= subgoal_limit
    if len(subgoal_words) > subgoal_limit:
        subgoal_words = subgoal_words[:subgoal_limit]

    subgoal_words_clip = clip.tokenize(subgoal_words).to("cuda")
    subgoal_words_clip = clip_model.encode_text(subgoal_words_clip) # (N,D)

    original_images = images

    all_bboxes_clip, all_labels, all_masks, all_lengths = [], [], [], []

    # ResNetの前処理を適用
    resnet_preprocess = transforms.Compose([
        transforms.ToPILImage(),  # テンソルをPIL画像に変換
        transforms.Resize((224, 224)),  # 画像サイズをリサイズ
        transforms.ToTensor(),  # PIL画像をテンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    #divide into batch
    images = [original_images[i:i+batch_size] for i in range(0, len(original_images), batch_size)]

    for images_batch in images:
        tensor_images = torch.stack([F.to_tensor(img).cuda() for img in images_batch]).to(torch.device('cuda'))
        outputs = obj_predictor.model.model(tensor_images)

        regions_clip, labels, masks = [], [], []

        #for single image
        for idx, output in enumerate(outputs):
            _regions, _labels, _mask= [], [], []

            #for single bbox
            for pred_idx in range(len(output['scores'])):
                label = obj_predictor.model.vocab_pred[output['labels'][pred_idx].cpu().item()]
                box = output['boxes'][pred_idx].detach().cpu().numpy() 
                c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
                region = tensor_images[idx,:,c1[0]:c1[1], c2[0]:c2[1]] # TODO: H,W逆かも
                image = tensor_images[idx]
                if region.shape[1] * region.shape[2] > 0:
                        region_resized = torchvision.transforms.functional.resize(region,size=(224,224))
                        _labels.append(label)
                        _regions.append(region_resized.unsqueeze(0))
                        mask = output['masks'][pred_idx].detach().cpu().numpy()
                        #to tensor
                        mask = torch.from_numpy(mask)
                        mask_rgb = torch.cat([mask, mask, mask], dim=0) 
                        masked_image = image.cuda() * mask_rgb.cuda()
                        _mask.append(masked_image.unsqueeze(0))

            if len(_regions) == 0:
                region_clip, _labels = torch.zeros(1, subgoal_limit, 768).cuda(), torch.zeros(1, subgoal_limit, 768).cuda()
                # region_maskrcnn = torch.zeros(1, subgoal_limit, 4, 25, 25).cuda()
                # regions_maskrcnn.append(region_maskrcnn)
                regions_clip.append(region_clip)
                labels.append(_labels)
                masks.append(torch.zeros(1, subgoal_limit, 1000).cuda())
                all_lengths.append(0)
                continue
            
            _regions = torch.cat(_regions,dim=0) # (N,3,224,224)
            _mask = torch.cat(_mask,dim=0) # (N, 1, 300,300)
            # print("_mask", _mask)
            tokenized_labels = clip.tokenize(_labels).to("cuda")
            label_clip = clip_model.encode_text(tokenized_labels) # (N,D)
            cossim = FF.cosine_similarity(subgoal_words_clip.unsqueeze(1).float(), label_clip.float(), dim=-1) # (M,D) @ (D,N)
            sort_idxs = torch.argsort(cossim, dim=1, descending=True)

            #sort_idxsの各行において、subgoal_words_clipの各単語との類似度が高い順にソートされている
            #上位num_of_use個のインデックスを取得
            # sort_idxs = sort_idxs[:,:]
            # 各行で一番大きい値が格納されたインデックスを取得
            # sort_idxs = torch.argmax(sort_idxs, dim=1)

            # print("subgoal_words_clip", subgoal_words_clip.shape)
            #top1bboxes
            use_index = []
            for i in range(sort_idxs.shape[0]):
                use_index.append(int(sort_idxs[i][0]))

            #重複があったら1,2,...のようにインデックスをずらして取得、重複がなければそのまま
            for i in range(len(use_index)):
                # temp_list = use_index.copy()
                # temp_list.pop(i)
                if use_index[i] in use_index[:i]:
                    for j in range(int(sort_idxs.shape[1])):
                        if int(sort_idxs[i][j]) not in use_index[:i]:
                            use_index[i] = int(sort_idxs[i][j])
                            break
            # top_bboxes = torch.cat([_regions[sort_idxs[i]].unsqueeze(0) for i in range(sort_idxs.shape[0])],dim=0)
            top_bboxes = torch.cat([_regions[use_index[i]].unsqueeze(0) for i in range(len(use_index))],dim=0)
            top_label_clip = torch.cat([label_clip[use_index[i]].unsqueeze(0) for i in range(len(use_index))],dim=0)
            if len(top_bboxes) == 0:
                region_clip, _labels = torch.zeros(1, subgoal_limit, 768).cuda(), torch.zeros(1, subgoal_limit, 768).cuda()
                # region_maskrcnn = torch.zeros(1, subgoal_limit, 4, 25, 25).cuda()
                # regions_maskrcnn.append(region_maskrcnn)
                regions_clip.append(region_clip)
                labels.append(_labels)
                masks.append(torch.zeros(1, subgoal_limit, 1000).cuda())
                all_lengths.append(0)
                continue

                # 特徴量の結合
                # print("output_resnet.shape", output_resnet.shape)
            # c.f. https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
            my_transform = Compose([
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            
            with torch.no_grad():
                # batch_images = top_bboxes.view(-1, 3, 224, 224) # (subgoal_words * num_of_use, 3, 224, 224)
                #batch_imagesを<PIL.Image.Image image mode=RGB size=300x300>のような形式のリストに変換
                # batch_images_maskrcnn = [transforms.ToPILImage()(img) for img in batch_images]

                # feats_mask = extract_features(batch_images_maskrcnn, obj_predictor)
                # feats_mask = feats_mask.view(len(subgoal_words), sort_idxs.shape[1], 512, 25, 25) # (subgoal_words, num_of_use, 768)
                batch_images = my_transform(top_bboxes) # (subgoal_words, num_of_use, 3, 224, 224)
                batch_images = batch_images.view(-1, 3, 224, 224) # (subgoal_words * num_of_use, 3, 224, 224)
                feats_clip = clip_model.encode_image(batch_images) # (subgoal_words * num_of_use,768)
                feats_clip = feats_clip.view(len(use_index), 768) # (subgoal_words, num_of_use, 768)

                mask = torch.stack([resnet_preprocess(_mask[use_index[i]]) for i in range(len(use_index))])
                output_resnet = pretrained_resnet(mask.cuda()) # [subgoal_word, 1000]
                
                if subgoal_limit > len(use_index):
                    top_label_clip = torch.cat([top_label_clip, torch.zeros(subgoal_limit - len(use_index), 768).cuda()], dim=0)
                    # feats_mask = torch.cat([feats_mask.cuda(), torch.zeros(subgoal_limit - len(use_index), 4, 25, 25).cuda()], dim=0)
                    feats_clip = torch.cat([feats_clip, torch.zeros(subgoal_limit - len(use_index), 768).cuda()], dim=0)
                    output_resnet = torch.cat([output_resnet, torch.zeros(subgoal_limit - len(use_index), 1000).cuda()], dim=0)
            # regions.append(feats.unsqueeze(0))
            labels.append(top_label_clip.unsqueeze(0))
            # regions_maskrcnn.append(feats_mask.unsqueeze(0).cuda()) 
            regions_clip.append(feats_clip.unsqueeze(0))
            all_lengths.append(len(use_index))
            masks.append(output_resnet.unsqueeze(0))

        # all_bboxes.append(torch.cat(regions,dim=0)) #(batch_size, subgoal_words, num_of_use, 768)
        all_labels.append(torch.cat(labels,dim=0)) #(batch_size,subgoal_words, num_of_use, 768)
        # all_bboxes_maskrcnn.append(torch.cat(regions_maskrcnn,dim=0))
        all_bboxes_clip.append(torch.cat(regions_clip,dim=0))
        all_masks.append(torch.cat(masks,dim=0))

    # all_bboxes_maskrcnn = torch.cat(all_bboxes_maskrcnn,dim=0) 
    all_bboxes_clip = torch.cat(all_bboxes_clip,dim=0) 
    all_labels = torch.cat(all_labels,dim=0)
    all_masks = torch.cat(all_masks,dim=0) 
    all_lengths = torch.tensor(all_lengths)
    return all_bboxes_clip.cpu(), all_labels.cpu(), all_masks.cpu(), all_lengths.cpu()



def encode_clip_image(clip_preprocess, clip_model, images, device="cuda:0"):
    images = [clip_preprocess(image).unsqueeze(0).to("cuda") for image in images]
    with torch.no_grad():
        feats = [clip_model.encode_image(image) for image in images]

        #featsがからであれば、0を入れる
        if len(feats) == 0:
            feats = torch.zeros(1, 768).to(device)
        feats = torch.cat(feats, dim=0) # [len(images),768]

    return feats

    return feats if feats.shape[0] > 0 else torch.zeros(1, 768).cuda()
        
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

def zero_pad_feats(feats, device):
    '''
    pad feats with zeros to max_len
    '''
    # 入力テンソルの定義（可変長のリスト）

    # 各次元の最大サイズを計算
    max_dim0 = max(tensor.size(0) for tensor in feats)
    max_dim1 = max(tensor.size(1) for tensor in feats)

    # ゼロパディングを適用して、結果のテンソルを初期化
    padded_tensor = torch.zeros(len(feats), max_dim0, max_dim1, 2, 768).to(device)

    # 各要素のテンソルをゼロパディングして結果のテンソルに代入
    for i, tensor in enumerate(feats):
        padded_tensor[i, :tensor.size(0), :tensor.size(1)] = tensor

    return padded_tensor

def zero_pad_feats_maskrcnn(feats, device):
    '''
    pad feats with zeros to max_len
    '''
    # 入力テンソルの定義（可変長のリスト）

    # 各次元の最大サイズを計算
    max_dim0 = max(tensor.size(0) for tensor in feats)
    max_dim1 = max(tensor.size(1) for tensor in feats)

    # ゼロパディングを適用して、結果のテンソルを初期化
    padded_tensor = torch.zeros(len(feats), max_dim0, max_dim1, 1, 2048, 25, 25).to(device)

    # 各要素のテンソルをゼロパディングして結果のテンソルに代入
    for i, tensor in enumerate(feats):
        padded_tensor[i, :tensor.size(0), :tensor.size(1)] = tensor

    return padded_tensor

def tensorize_and_pad(batch, device, pad):
    '''
    cast values to torch tensors, put them to the correct device and pad sequences
    '''
    device = torch.device(device)
    input_dict, gt_dict, feat_dict = dict(), dict(), dict()
    if len(list(zip(*batch))) == 3:
        task_path, traj_data, feat_list = list(zip(*batch))
    else:
        traj_data, feat_list = list(zip(*batch))
        task_path = ""

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
            # resnet = pad_sequence([vv[0].clone().detach().to(device).type(torch.float) for vv in v], batch_first=True, padding_value=pad)
            # print("pad_seq.shape",pad_seq.shape)#torch.Size([2, 72, 512, 7, 7])
            clip = pad_sequence([vv[0].clone().detach().to(device).type(torch.float) for vv in v], batch_first=True,padding_value=pad)

            bbox = pad_sequence([vv[1].clone().detach().to(device).type(torch.float) for vv in v], batch_first=True,padding_value=pad)
            label = pad_sequence([vv[2].clone().detach().to(device).type(torch.float) for vv in v], batch_first=True,padding_value=pad)
            mask = pad_sequence([vv[3].clone().detach().to(device).type(torch.float) for vv in v], batch_first=True,padding_value=pad)
            #bboxの次元は[batch, len_img, len_subword, 5, 1024]であるが、pad_sequenceを使うと[batch, max_img, max_subword, 5, 1024]になる
            dict_assign['lengths_subword'] = [copy.deepcopy(vv[4].tolist()) for vv in v]
            # [49, 9, 5, 1024]次元のtensorと[49, 4, 5, 1024]次元のtensorをゼロパディングして[49, 9, 5, 1024]次元のtensorにする

            # 入力テンソルをリストとしてまとめる
            # bbox = zero_pad_feats([vv[2].clone().detach().to(device).type(torch.float) for vv in v], device)
            # label = zero_pad_feats([vv[3].clone().detach().to(device).type(torch.float) for vv in v], device)
            # bbox = pad_sequence([vv[2].clone().detach().to(device).type(torch.float) for vv in v], batch_first=True,padding_value=pad)
            # label = pad_sequence([vv[3].clone().detach().to(device).type(torch.float) for vv in v], batch_first=True,padding_value=pad)

            dict_assign[k] = [clip, bbox, label, mask]
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
        # feat_shape = (-1, 2048, 10, 10)
        feat_shape = (-1, 2048, 25, 25)
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
