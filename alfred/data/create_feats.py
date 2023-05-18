"""
$DATA/generated_2.1.0/{split}/{task}/raw_images/*.jpgから画像を読み込み、
それを特徴量に変換し、.pthファイルとして保存する.deberta, text_clip, mask_rcnn_bbox, mask_rcnn_label
の4つの特徴量を作成する
"""
import os
import glob
import copy
import pickle

from PIL import Image
import numpy as np
import torch
import clip
from tqdm import tqdm

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

    is_process_batch = False
        
    with torch.no_grad():
        # for split in ["train", "valid_seen", "pseudo_valid", "pseudo_test"]:
        if is_process_batch:
            for split in ["pseudo_test"]:
                #./data/generated_2.1.0/{split}/*/*/feats/deberta/*.pthからデータを取得
                input_paths = glob.glob(os.path.join(constants.ET_DATA,  "generated_2.1.0", split, "*/*"))
                for input_path in input_paths:

                    feats_mask_rcnn_bbox = []
                    feats_mask_rcnn_label = []

                    image_paths = glob.glob(os.path.join(input_path, "raw_images/*.png"))
                    images = [Image.open(image_path) for image_path in image_paths]
                    preds = obj_predictor.predict_objects_batch(images)

                    # get bbox and label for each image
                    for i in range(len(preds)):
                        bboxes = []
                        labels = []

                        # preds[i]は画像iに対する予測結果
                        for j in range(len(preds[i])):
                            box = preds[i][j].box
                            c1, c2 = (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item()))
                            region = np.array(images[i])
                            region = region[c1[0]:c1[1], c2[0]:c2[1],:]
                            if region.shape[0] * region.shape[1] > 0:
                                bboxes.append(Image.fromarray(region))
                                labels.append(preds[i][j].label)

                        #もしbboxesが空であれば
                        if len(bboxes) < 30:
                            #featを0埋め
                            bboxes = bboxes + [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(30 - len(feat))]
                            #labelを0埋め
                            labels = labels + ["" for _ in range(30 - len(label))]

                        else:
                            bboxes = bboxes[:30]
                            labels = labels[:30]

                        bboxes = encode_clip_image(clip_preprocess, clip_model, bboxes) #ex. (64, 768)

                        tokenized = clip.tokenize(label).to("cuda")
                        label = clip_model.encode_text(tokenized)

                        feats_mask_rcnn_bbox.append(bboxes)
                        feats_mask_rcnn_label.append(label)
                    
                    #feats_mask_rcnn_bbox,feats_mask_rcnn_labelをtensorに変換
                    feats_mask_rcnn_bbox = torch.stack(feats_mask_rcnn_bbox, dim=0)
                    feats_mask_rcnn_label = torch.stack(feats_mask_rcnn_label, dim=0)

                    print("feats_mask_rcnn_bbox.shape", feats_mask_rcnn_bbox.shape)
                    print("feats_mask_rcnn_label.shape", feats_mask_rcnn_label.shape)

                    # 特徴量を保存する
                    torch.save(feats_mask_rcnn_bbox, os.path.join(input_path, "maskrcnn_bbox.pth"))
                    torch.save(feats_mask_rcnn_label, os.path.join(input_path, "maskrcnn_label.pth"))
        else:   
            # for split in ["train", "valid_seen", "pseudo_valid", "pseudo_test"]:
            for split in ["train"]:
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




























