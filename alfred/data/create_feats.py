"""
$DATA/generated_2.1.0/{split}/{task}/raw_images/*.jpgから画像を読み込み、
それを特徴量に変換し、.pthファイルとして保存する.deberta, text_clip, mask_rcnn_bbox, mask_rcnn_label
の4つの特徴量を作成する
"""
import os
import glob
import types
import copy
import pickle

from PIL import Image
import numpy as np
import torch
import clip
from tqdm import tqdm

from torchvision.transforms import functional as F
from transformers import AutoTokenizer, AutoModel
from alfred.gen import constants
from alfred.nn.enc_visual import FeatureExtractor
from alfred.data.zoo.alfred import AlfredDataset
from alfred.utils.data_util import tokens_to_lang

# def create_feats():


def get_region_feats(image_path, obj_predictor, num_of_use=10):
    '''
    get environment observation
    '''
    #image_pathから画像を読み込む
    frame = Image.open(image_path).convert('RGB')
    frame_array = np.array(frame)
    rcnn_pred = obj_predictor.predict_objects(frame)
    regions = []
    labels = []
    scores = []
    for pred in rcnn_pred:
        box = pred.box
        c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
        region = copy.deepcopy(frame_array)
        region = region[c1[0]:c1[1], c2[0]:c2[1],:]
        if region.shape[0] * region.shape[1] > 0:
            regions.append(Image.fromarray(region))
            labels.append(pred.label)
            scores.append(pred.score)

    #scoreでソートして、上位{num_of_use}個のregionを取得する
    if len(regions) > num_of_use:
        indices = np.argsort(scores)[::-1][:num_of_use]
        regions = [regions[i] for i in indices]
        labels = [labels[i] for i in indices]
    else:
        #zero埋めする
        regions = regions + [Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)) for _ in range(num_of_use - len(regions))]
        labels = labels + ["" for _ in range(num_of_use - len(labels))]

    return regions, labels

def encode_clip_image(clip_preprocess, clip_model, images, device="cuda:0"):
    images = [clip_preprocess(image).unsqueeze(0).to("cuda") for image in images]
    with torch.no_grad():
        feats = [clip_model.encode_image(image) for image in images]
        feats = torch.cat(feats, dim=0) # [len(images),768]

    return feats
    
def main():
    clip_model, clip_preprocess = clip.load("ViT-L/14", device="cuda")
    for params in clip_model.parameters():
        params.requires_grad = False

    # # if args.deberta:
    deberta_model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base").cuda()
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    for param in deberta_model.parameters():
        param.requires_grad = False

    obj_predictor = FeatureExtractor(archi='maskrcnn', device="cuda",
    checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)
    obj_predictor.model.eval()

    is_process_batch = True

    jsons = {}

    # for split in ['train', 'valid_seen', 'pseudo_valid', 'pseudo_test']:
        
        
    with torch.no_grad():
        # for split in ["train", "valid_seen", "pseudo_valid", "pseudo_test"]:
        if is_process_batch:
            batch_size = 40
            num_of_use = 5
            for split in ["train"]:
                #./data/generated_2.1.0/{split}/*/*/feats/deberta/*.pthからデータを取得
                input_paths = glob.glob(os.path.join(constants.ET_DATA,  "generated_2.1.0", split, "*/*"))
                for input_path in tqdm(input_paths):
                    #image_pathsからbatch_size個のパスを取得し、image_pathsから削除する
                    original_images = [Image.open(image_path) for image_path in glob.glob(os.path.join(input_path, "raw_images/*.png"))]

                    all_bboxes = []
                    all_labels = []

                    #imagesをbatch sizeに分割
                    # print("len(images): ", len(original_images))
                    images = [original_images[i:i+batch_size] for i in range(0, len(original_images), batch_size)]
                    for images_batch in images:
                        images_batch_tensor = torch.stack([F.to_tensor(img) for img in images_batch]).to(torch.device('cuda'))
                        outputs = obj_predictor.model.model(images_batch_tensor)
                        bboxes = []
                        labels = []

                        for idx, output in enumerate(outputs):
                            _bbox = []
                            _label = []
                            for pred_idx in range(len(output['scores'])):
                                score = output['scores'][pred_idx].cpu().item()
                                if score < 0.4:
                                    continue
                                box = output['boxes'][pred_idx].cpu().numpy()
                                label = obj_predictor.model.vocab_pred[output['labels'][pred_idx].cpu().item()]
                                
                                c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
                                region = np.array(images_batch[idx])
                                region = region[c1[0]:c1[1], c2[0]:c2[1],:]
                                if region.shape[0] * region.shape[1] > 0:
                                    _bbox.append(Image.fromarray(region))
                                    _label.append(label)

                            if len(_bbox) < num_of_use:
                                #featを0埋め
                                _bbox = _bbox + [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(num_of_use - len(_bbox))]
                                #labelを0埋め
                                _label = _label + ["" for _ in range(num_of_use - len(_label))]

                            else:
                                _bbox = _bbox[:num_of_use]
                                _label = _label[:num_of_use]
                            #ここでbbox: [num_of_use, 224, 224, 3], label: [num_of_use]が得られる

                            bboxes += _bbox
                            labels += _label
                        
                        #ここでbboxes: [batch_size * num_of_use, 224, 224, 3], labels: [batch_size * num_of_use]が得られる
                        #ここでbboxesをclip_imageによって特徴量に変換する
                        # bboxes = encode_clip_image(clip_preprocess, clip_model, bboxes) # (batch_size * num_of_uses, 768)
                        # tokenized_labels = clip.tokenize(feats_mask_rcnn_label).to("cuda")
                        # labels = clip_model.encode_text(tokenized_labels) #(30, 768)
                        # #ここでbboxes: [batch_size , num_of_use, 768], labels: [batch_size , num_of_use, 768]に戻す
                        # bboxes = bboxes.reshape(len(images_batch), num_of_use, 768)
                        # labels = labels.reshape(len(images_batch), num_of_use, 768)

                        all_bboxes += bboxes
                        all_labels += labels

                    #ここでall_bboxes: [len(images) * num_of_use, 224, 224, 3], all_labels: [len(images) * num_of_use]が得られる
                    #ここでall_bboxesをclip_imageによって特徴量に変換する
                    all_bboxes = encode_clip_image(clip_preprocess, clip_model, all_bboxes) # (batch_size * num_of_uses, 768)
                    tokenized_labels = clip.tokenize(all_labels).to("cuda")
                    all_labels = clip_model.encode_text(tokenized_labels) #(30, 768)

                    #ここでall_bboxes: [len(images), num_of_use, 768], all_labels: [len(images), num_of_use, 768]に戻す
                    all_bboxes = all_bboxes.reshape(len(original_images), num_of_use, 768)
                    all_labels = all_labels.reshape(len(original_images), num_of_use, 768)

                    # print("all_bboxes.shape: ", all_bboxes.shape)
                    # print("all_labels.shape: ", all_labels.shape)

                    torch.save(all_bboxes, os.path.join(input_path, "maskrcnn_bbox.pth"))
                    torch.save(all_labels, os.path.join(input_path, "maskrcnn_label.pth"))



                    # # get bbox and label for each image
                    # for i in range(len(preds)):

                    #     # preds[i]は画像iに対する予測結果
                    #     for j in range(len(preds[i])):
                    #         box = preds[i][j].box
                    #         c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
                    #         regio = np.array(images[i])
                    #         regin = region[c1[0]:c1[1], c2[0]:c2[1],:]
                    #         if region.shape[0] * region.shape[1] > 0:
                    #             bboxes.append(Image.fromarray(region))
                    #             labels.append(preds[i][j].label)

                    #     #もしbboxesが空であれば
                    #     if len(bboxes) < 30:
                    #         #featを0埋め
                    #         bboxes = bboxes + [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(30 - len(feat))]
                    #         #labelを0埋め
                    #         labels = labels + ["" for _ in range(30 - len(label))]

                    #     else:
                    #         bboxes = bboxes[:30]
                    #         labels = labels[:30]

                    #     bboxes = encode_clip_image(clip_preprocess, clip_model, bboxes) #ex. (64, 768)

                    #     tokenized = clip.tokenize(label).to("cuda")
                    #     label = clip_model.encode_text(tokenized)

                    #     feats_mask_rcnn_bbox.append(bboxes)
                    #     feats_mask_rcnn_label.append(label)
                    
                    # #feats_mask_rcnn_bbox,feats_mask_rcnn_labelをtensorに変換
                    # feats_mask_rcnn_bbox = torch.stack(feats_mask_rcnn_bbox, dim=0)
                    # feats_mask_rcnn_label = torch.stack(feats_mask_rcnn_label, dim=0)

                    # print("feats_mask_rcnn_bbox.shape", feats_mask_rcnn_bbox.shape)
                    # print("feats_mask_rcnn_label.shape", feats_mask_rcnn_label.shape)

                    # # 特徴量を保存する
                    # torch.save(feats_mask_rcnn_bbox, os.path.join(input_path, "maskrcnn_bbox.pth"))
                    # torch.save(feats_mask_rcnn_label, os.path.join(input_path, "maskrcnn_label.pth"))
        else:   
            for split in ["train", "valid_seen", "pseudo_valid", "pseudo_test"]:
            # for split in ["train"]:
                #./data/generated_2.1.0/{split}/*/*/feats/deberta/*.pthからデータを取得
                input_paths = glob.glob(os.path.join(constants.ET_DATA,  "generated_2.1.0", split, "*/*"))
                for input_path in tqdm(input_paths):

                    feats_mask_rcnn_bbox = []
                    feats_mask_rcnn_label = []

                    image_paths = glob.glob(os.path.join(input_path, "raw_images/*.png"))

                    len_images = len(image_paths)
                    for image_path in image_paths:
                    
                        bboxes, labels = get_region_feats(image_path, obj_predictor, 5)

                        feats_mask_rcnn_bbox += bboxes
                        feats_mask_rcnn_label += labels

                        #bbox : len(bboxes) 

                    tokenized_labels = clip.tokenize(feats_mask_rcnn_label).to("cuda")
                    feats_mask_rcnn_label = clip_model.encode_text(tokenized_labels) #(30, 768)
                    feats_mask_rcnn_bbox = encode_clip_image(clip_preprocess, clip_model, feats_mask_rcnn_bbox) #(30, 768)

                    #feats_mask_rcnn_bbox,feats_mask_rcnn_labelは(len_images * 30, 768)になっているので、(len_images, 30, 768)に変換する
                    feats_mask_rcnn_bbox = feats_mask_rcnn_bbox.reshape(len_images, 5, 768)
                    feats_mask_rcnn_label = feats_mask_rcnn_label.reshape(len_images, 5, 768)

                    #feats_mask_rcnn_bbox,feats_mask_rcnn_labelをtensorに変換

                    print("feats_mask_rcnn_bbox.shape", feats_mask_rcnn_bbox.shape)
                    print("feats_mask_rcnn_label.shape", feats_mask_rcnn_label.shape)

                    # 特徴量を保存する
                    torch.save(feats_mask_rcnn_bbox, os.path.join(input_path, "maskrcnn_bbox.pth"))
                    torch.save(feats_mask_rcnn_label, os.path.join(input_path, "maskrcnn_label.pth"))

                
        #次にdeberta, text_clipの特徴量を作成する
        # for split in ["pseudo_valid"]:
        #     pickle_path = os.path.join(constants.ET_DATA, "lmdb_augmented_human_subgoal", split, "jsons.pkl")
            
        #     with open(pickle_path, 'rb') as f:
        #         # pickleファイルからオブジェクトを復元
        #         jsons = pickle.load(f)

        #     # print("jsons.keys()", jsons[0].keys())
        #     print("len", len(jsons))
        #     for k, v in jsons.items():
        #         print("key", v[0].keys()) #key dict_keys(['images', 'pddl_params', 'plan', 'scene', 'task', 'task_id', 'task_type', 'turk_annotations', 'root', 'split', 'repeat_idx', 'num', 'ann'])
        #         print("root", v[0]["root"]) 
        #         # print("task", v[0]["task"])
        #         # print("task_type", v[0]["task_type"])
        #         lang_num = AlfredDataset.load_lang(v[0])
        #         print("lang_num", lang_num)
        #         print("tokens_to_lang", tokens_to_lang(lang_num))
        #         #v[0]["root"]の最後の/より前の部分を取得する
        #         root_path = v[0]["root"]
        #         root_path = root_path.split("/traj_data.json")[0]


            
            







if __name__ == "__main__":
    main()




























