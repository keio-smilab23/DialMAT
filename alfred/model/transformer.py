#追加
import re
import os
import copy
import glob


import pickle
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
#追加
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

import clip
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

from alfred.model import base
from alfred.nn.enc_lang import EncoderLang
from alfred.nn.enc_visual import FeatureFlat
from alfred.nn.enc_vl import EncoderVL
from alfred.nn.encodings import DatasetLearnedEncoding
from alfred.nn.dec_object import ObjectClassifier
from alfred.utils import model_util
from alfred.utils.data_util import tokens_to_lang, get_maskrcnn_features_batch

#追加
from alfred.model import mat
from alfred.nn.enc_visual import FeatureExtractor

class Model(base.Model):
    def __init__(self, args, embs_ann, vocab_out, pad, seg, dim_size: int=768, rho1=0.9, rho2=0.999, lr_reduce=2e-3 / 8e-5, step_size=4):
        '''
        transformer agent
        '''
        super().__init__(args, embs_ann, vocab_out, pad, seg)

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

        #追加
        # if args.clip_image or args.clip_text or args.clip_resnet:
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device="cuda")
        for params in self.clip_model.parameters():
            params.requires_grad = False

        # if args.deberta:
        self.deberta_model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base").cuda()
        self.deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        for param in self.deberta_model.parameters():
            param.requires_grad = False

        # if args.mat_action:
        self.mat = mat.AdversarialPerturbationAdder(dim_size)

        self.obj_predictor = FeatureExtractor(archi='maskrcnn', device="cuda",
        checkpoint="./logs/pretrained/maskrcnn_model.pth", load_heads=True)

        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)
        # pre-encoder for language tokens
        self.encoder_lang = EncoderLang(
            args.encoder_lang['layers'], args, embs_ann)
        # feature embeddings
        self.vis_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape,
            output_size=args.demb)
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None
        if args.enc['dataset']:
            self.dataset_enc = DatasetLearnedEncoding(args.demb, args.data['train'])
        # embeddings for actions
        self.emb_action = nn.Embedding(len(vocab_out), args.demb)
        # dropouts
        self.dropout_action = nn.Dropout2d(args.dropout['transformer']['action'])

        # decoder parts
        encoder_output_size = args.demb
        # if self.args.clip_resnet:
        #     self.dec_action = nn.Linear(
        #         encoder_output_size, args.demb)
        # else:
        self.dec_action = nn.Linear(
            encoder_output_size, args.demb)
        
        self.dec_object = ObjectClassifier(encoder_output_size)

        # skip connection for object predictions
        self.object_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape,
            output_size=args.demb)

        # progress monitoring heads
        if self.args.progress_aux_loss_wt > 0:
            self.dec_progress = nn.Linear(encoder_output_size, 1)
        if self.args.subgoal_aux_loss_wt > 0:
            self.dec_subgoal = nn.Linear(encoder_output_size, 1)

        # final touch
        self.init_weights()

        # self.reset()
        if args.clip_image:
            # self.reset_for_clip()
            self.reset_for_both()
        elif args.clip_resnet:
            self.reset_for_both()
        else:
            self.reset()

    def extract_nouns(self, text):
        nouns = []
        # 単語をトークン化
        words = word_tokenize(text)
        # 単語の品詞タグ付け
        tagged_words = nltk.pos_tag(words)
        
        for word, tag in tagged_words:
            # 名詞のみを抽出
            if tag.startswith('NN'):
                nouns.append(word)
        
        #nounsをアルファベット順にする
        nouns.sort()
        return nouns

    #追加
    def get_subgoal_text(self, tokens, vocab):
        sentences_list = []
        for token in tokens:
            # tokens to text
            text = tokens_to_lang(token.tolist(), vocab)
            # 先頭の"<<"の前の文字列を取得
            text = text.split("<<")[0]
            nouns = self.extract_nouns(text)
            sentences_list.append(nouns)
            #batchsize分のリストができる

        return sentences_list
    
    def token_to_sentence_list(self, tokens, vocab):
        sentences_list = []
        for token in tokens:
            # tokens to text
            text = tokens_to_lang(token.tolist(), vocab)
            # remove tokens enclosed in << >>
            text = re.sub(r'<<.*?>>', '.', text)

            # split text into sentences using period as delimiter
            sentences = text.split('.')

            # remove leading and trailing white space from each sentence
            sentences = [s.strip() for s in sentences if s.strip() and (s != "" or s != " " or s != ", ")]
            sentences_list.append(sentences)
        return sentences_list
    
    def encode_deberta(self, task_path, sentences, device="cuda:0"):
        """
        Encode two sentences with the deberta model.
        """

        text_features_list = []
        lengths_list = []
        
        if not self.args.update_feat:
            text_features, lengths_list = self.load_features_from_path(task_path, "deberta.pth")
        
            if text_features is not None and lengths_list is not None:
                return text_features, lengths_list

        for i in range(len(sentences)):
            sentence = sentences[i]
            # 全ての文を.で結合
            sentence = ".".join(sentence)

            tokenized = self.deberta_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
            text_features = self.deberta_model(tokenized).last_hidden_state.squeeze(0).to(device) #ex. (64, 768)
            # print("text_features.shape: ", text_features.shape)

            text_features_list.append(text_features)
            lengths_list.append(text_features.shape[0])

        text_features = pad_sequence(text_features_list, batch_first=True)

        self.save_features_to_paths(task_path, "deberta.pth", [text_features, torch.tensor(lengths_list).to(device)])

        print("text_features.shape: ", text_features.shape)
        print("lengths_list: ", lengths_list)

        return text_features, torch.tensor(lengths_list).to(device)
    
    def embed_maskrcnn(self, task_path, subgoal_words_list):
        '''
        get environment observation
        '''
        # frame = Image.fromarray(images)

        if not self.args.update_feat:
            text_features_bbox, _ = self.load_features_from_path(task_path, "maskrcnn_bbox.pth")
            text_features_label, _ = self.load_features_from_path(task_path, "maskrcnn_label.pth")

            # if features are already saved, return them
            if text_features_bbox != None and text_features_label != None:
                return text_features_bbox, text_features_label

        # get images from task_path

        all_feats = []
        all_labels = []

        #for each batch
        for idx, single_path in enumerate(task_path):
            image_paths = glob.glob(os.path.join(single_path, "raw_images", "*.png"))
            image_paths.sort()

            feats = []

            tokenized_subgoal_labels = clip.tokenize(subgoal_words_list[idx]).to("cuda")
            subgoal_words_clip = self.clip_model.encode_text(tokenized_subgoal_labels) #(len(subgoal_words), 768)

            feats, labels = get_maskrcnn_features_batch(image_paths, self.obj_predictor, self.clip_preprocess, self.clip_model, subgoal_words_list[idx], subgoal_words_clip)
            # print("len subgoal_words", len(subgoal_words_list[idx]))

            # for image_path in image_paths:
            #     image = Image.open(image_path).convert('RGB')
            #     # for each image
            #     # TODO: ここに無限時間掛かる
            #     feat = get_maskrcnn_features(image, self.obj_predictor, self.clip_preprocess, self.clip_model, subgoal_words_list[idx], subgoal_words_clip)
            #     # feat = []
            #     feats.append(feat)
                # print("feat.shape: ", feat.shape)

            # print("len subgoal_words", len(subgoal_words_list[idx]))

            # for image_path in image_paths:
            #     image = Image.open(image_path).convert('RGB')
            #     # for each image
            #     feat = get_maskrcnn_features_batch(image, self.obj_predictor, self.clip_preprocess, self.clip_model, subgoal_words_list[idx], subgoal_words_clip)
            #     feats.append(feat)
            #     # print("feat.shape: ", feat.shape)

            #     file_name = os.path.basename(image_path)
            #     #image_pathのfile_name(拡張子以外)にmarkrcnnという文字列を追加
            #     output_path = os.path.join(single_path, "raw_images", file_name.replace(".png", "_maskrcnn.pth"))
            #     self.save_features_to_path(output_path, feat)
            
            all_feats.append(feats)
            all_labels.append(labels)
        
        return all_feats, all_labels

    def encode_clip_image(self, images, device="cuda:0"):
        images = [self.clip_preprocess(image).unsqueeze(0).to("cuda") for image in images]
        with torch.no_grad():
            feats = [self.clip_model.encode_image(image) for image in images]
            feats = torch.cat(feats, dim=0) # [len(images),768]

        return feats
    #追加
    def encode_clip_text(self, task_path, sentences, device="cuda:0"):
        """
        Encode sentences with the CLIP model.
        """

        if not self.args.update_feat:
            text_features, lengths_list = self.load_features_from_path(task_path, "clip_text.pth")

            # if features are already saved, return them
            if text_features != None:
                return text_features, lengths_list

        text_features_list = []
        lengths_list = []

        for i in range(len(sentences)):
            tokenized = clip.tokenize(sentences[i]).to(device)
            text_features = self.clip_model.encode_text(tokenized)
            text_features_list.append(text_features)
            lengths_list.append(tokenized.shape[0])

        text_features = pad_sequence(text_features_list, batch_first=True)

        self.save_features_to_paths(task_path, "clip_text.pth", [text_features, torch.tensor(lengths_list).to(device)])

        print("encode_clip_text", text_features.shape)
        print("encode_clip_text length", torch.tensor(lengths_list).to(device).shape)
        return text_features, torch.tensor(lengths_list).to(device)

    #追加
    def concat_embeddings_lang(self, emb_lang, lengths_lang, emb_other, lengths_other, device="cuda:0"):
        if len(lengths_lang) == 1:
            # temp = torch.cat([emb_lang[0, :lengths_lang[0], :], emb_other.unsqueeze(1)[0, :lengths_other[0], :]], dim=0)
            temp = torch.cat([emb_lang[0, :lengths_lang[0], :], emb_other[0, :lengths_other[0], :]], dim=0)
            return temp.unsqueeze(0), torch.tensor([temp.shape[0]]).to(device)

        #上記はemb_langのshapeが[2, max, 768]であることを前提としているが、それを一般化する
        temp_list = []
        for i in range(len(lengths_lang)):
            lang = emb_lang[i, :lengths_lang[i], :]
            other = emb_other[i, :lengths_other[i], :]
            temp = torch.cat([lang, other], dim=0)
            temp_list.append(temp)
        
        emb_lang = pad_sequence(temp_list, batch_first=True, padding_value=0)

        #make length tensor
        lengths = torch.tensor([temp.shape[0] for temp in temp_list]).to(device)

        return emb_lang, lengths

    def concat_embeddings_frames(self, emb_frames, emb_other, lengths_frames, lengths_other, device):
        #emb_frames:[batch, max, 768], emb_other:[batch, max, 768], lengths_frames:[size * batch], lengths_other:[size * batch]
        temp_list = []
        for i in range(len(lengths_frames)):
            frames = emb_frames[i, :lengths_frames[i], :]
            other = emb_other[i, :lengths_other[i], :]
            temp = torch.cat([frames, other], dim=0)
            temp_list.append(temp)
        
        emb_frame = pad_sequence(temp_list, batch_first=True, padding_value=0)

        lengths = torch.tensor([temp.shape[0] for temp in temp_list]).to(device)

        return emb_frame, lengths
    
    def load_features_from_path(self, paths, file_name, return_length=False):
        """
        Load features from a file.
        """

        features = []
        if return_length:
            lengths = []
        for path in paths:
            path = os.path.join(path, file_name)

            #まずpathにファイルがあるかを確認する
            if os.path.exists(path):
                #あれば読み込む
                with open(path, "rb") as f:
                    feature = pickle.load(f)
                    features.append(feature)
                    if return_length:
                        lengths.append(feature.shape[0])
            else:
                return None, None
            
        features = pad_sequence(features, batch_first=True)
        if return_length:
            lengths = torch.tensor(lengths)
            return features, lengths
        else:
            return features
        
    def save_features_to_paths(self, paths, file_name, features):
        """
        Save features to files.
        """
        for idx, path in enumerate(paths):
            path = os.path.join(path, file_name)
        
            if len(features.shape) == 3:
                save_features = features[idx,:,:]
            elif len(features.shape) == 4:
                 save_features = features[idx,:,:,:]

            # save_features = features[idx,:,:].unsqueeze(0) #mask_rcnnでは( max_len(image), 30(max), 768), clipでは(max_len(text), 768), debertaでは(max_len(text), 768)
            if isinstance(save_features, torch.Tensor):
                print("save_{}: {}".format(file_name, save_features.shape))
            elif isinstance(save_features, list):
                print("save_{}: {}".format(file_name, save_features[0].shape))

            # Write features to file
            with open(path, "wb") as f:
                pickle.dump(save_features, f)

    def save_features_to_path(self, output_path, feature):
        """
        Save features to a file.
        """
        save_feature = feature #mask_rcnnでは(1, max_len(image), 30(max), 768), clipでは(1, max_len(text), 768), debertaでは(1, max_len(text), 768)

        with open(output_path, "wb") as f:
            pickle.dump(save_feature, f)

    def forward(self, task_path, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        # embed language
        output = {}
        
        subgoal_words_list = self.get_subgoal_text(inputs['lang'], vocab)

        if self.args.update_feat:
            with torch.no_grad():

                emb_maskrcnn_bbox, emb_maskrcnn_label = self.embed_maskrcnn(task_path, subgoal_words_list)
                # emb_clip, lengths_clip = self.embed_clip(task_path)
                # emb_deberta, lengths_deberta = self.embed_deberta(task_path)

                return output
        
        else:
            emb_maskrcnn_bbox, emb_maskrcnn_label = self.embed_maskrcnn(task_path)

        #変更(CLIPのtext情報も用いる)
        if self.args.clip_text:
            emb_lang, lengths_lang = self.embed_lang(inputs['lang'], vocab)
            #emb_lang:[batch, max, 768](2つのデータの大きい方をmaxに入れる), lengths_lang:[batch](emb_langのbatch個のデータの長さを持つ.)
            
            # token to sentence
            sentences = self.token_to_sentence_list(inputs['lang'], vocab)
            
            # encode clip
            emb_clip, lengths_clip = self.encode_clip_text(sentences, device=inputs['lang'].device)
            
            # concat clip and lang
            emb_lang, lengths_lang = self.concat_embeddings_lang(emb_lang, lengths_lang, emb_clip, lengths_clip, device=inputs['lang'].device)

            emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang

        elif self.args.deberta:
            emb_lang, lengths_lang = self.embed_lang(inputs['lang'], vocab)

            sentences = self.token_to_sentence_list(inputs['lang'], vocab)

            emb_deberta, lengths_deberta = self.encode_deberta(task_path, sentences, device=inputs['lang'].device)

            emb_lang, lengths_lang = self.concat_embeddings_lang(emb_lang, lengths_lang, emb_deberta, lengths_deberta, device=inputs['lang'].device)

            emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang
    
        elif self.args.clip_deberta:
            emb_lang, lengths_lang = self.embed_lang(inputs['lang'], vocab)

            sentences = self.token_to_sentence_list(inputs['lang'], vocab)

            # encode clip
            emb_clip, lengths_clip = self.encode_clip_text(sentences, device=inputs['lang'].device)

            # encode deberta
            emb_deberta, lengths_deberta = self.encode_deberta(task_path, sentences, device=inputs['lang'].device)

            # concat clip and lang
            emb_lang, lengths_lang = self.concat_embeddings_lang(emb_lang, lengths_lang, emb_clip, lengths_clip, device=inputs['lang'].device)

            # concat deberta and lang
            emb_lang, lengths_lang = self.concat_embeddings_lang(emb_lang, lengths_lang, emb_deberta, lengths_deberta, device=inputs['lang'].device)

            emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang


        else:
            emb_lang, lengths_lang = self.embed_lang(inputs['lang'], vocab)

            emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang



        #変更(CLIPのimage情報のみを用いる)
        if self.args.clip_image or self.args.clip_resnet:
            inputs_frames = [inputs['frames'][i] for i in range(len(inputs['frames'])) if i != 0]

            emb_clip = pad_sequence(inputs_frames, batch_first=True, padding_value=0).to("cuda")

            if len(emb_clip.shape) == 4:
                emb_clip = emb_clip.squeeze(0)

            emb_resnet, emb_object = self.embed_frames(inputs['frames'][0])

            #clipとresnetのどちらも使う場合
            if self.args.clip_resnet:
                emb_frames = torch.cat([emb_clip, emb_resnet], dim=1)
                # emb_frames, lengths_frames = self.concat_embeddings_frames(emb_clip, emb_resnet, inputs['lengths_frames'], inputs['lengths_frames'], device=inputs['frames'][0].device)
                lengths_frames = torch.tensor([inputs['length_frames_max'], inputs['length_frames_max']])
                length_frames_max = inputs['length_frames_max']
                lengths_actions = inputs['lengths_frames'].clone()
                if self.args.clip_object:
                    emb_object = emb_clip
            #clipのみの場合
            else:
                emb_frames = emb_clip
                lengths_frames = inputs['lengths_frames']
                length_frames_max = inputs['length_frames_max']
                lengths_actions = lengths_frames.clone()


            # emb_frames, lengths_frames = self.concat_embeddings_frames(emb_frames, inputs['region_feats'], inputs['lengths_frames'],device=inputs['frames'][0].device)
        else:
            #元々
            # embed frames and actions
            if len(inputs["frames"]) == 1: #推論時
                if len(inputs["frames"].shape) != 5:
                    inputs["frames"] = inputs["frames"].unsqueeze(0)
                emb_frames, emb_object = self.embed_frames(inputs['frames'])
            
            else:
                emb_frames, emb_object = self.embed_frames(inputs['frames'][0])
            lengths_frames = inputs['lengths_frames']
            length_frames_max = inputs['length_frames_max']
            lengths_actions = lengths_frames.clone()

        #emb_frames:[2, max_, 768], lengths_frames:[2],emb_actions:[2,max_,768] (langのmaxとは違う), ex. inputs['frames']: [2, 72, 512, 7, 7]
        emb_actions = self.embed_actions(inputs['action'])

        if self.args.mat_text:
            emb_lang = self.mat(emb_lang)
        if self.args.mat_image:
            emb_frames = self.mat(emb_frames)
        if self.args.mat_action:
            emb_actions = self.mat(emb_actions)
        # if self.args.mat_object:
        #     emb_object = self.mat(emb_object)

        #変更
        if not (self.args.clip_image or self.args.clip_resnet):
            assert emb_frames.shape == emb_actions.shape

        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang, emb_frames, emb_actions, lengths_lang,
            lengths_frames, lengths_actions, length_frames_max, is_clip_resnet=self.args.clip_resnet)
        # use outputs corresponding to visual frames for prediction only
        if self.args.clip_resnet:
            encoder_out_visual = encoder_out[
                :, lengths_lang.max().item():
                lengths_lang.max().item() + length_frames_max * 2]
        else:
            encoder_out_visual = encoder_out[
                :, lengths_lang.max().item():
                lengths_lang.max().item() + length_frames_max]
            
        if self.args.clip_resnet:
            #encoder_out_visualを半分にして足し合わせる
            middle_shape = encoder_out_visual.shape[1] // 2
            encoder_out_visual = encoder_out_visual[:, :middle_shape, :] + encoder_out_visual[:, middle_shape:,:]

        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_emb_flat = self.dec_action(decoder_input)
        action_flat = action_emb_flat.mm(self.emb_action.weight.t())
        action = action_flat.view(
            *encoder_out_visual.shape[:2], *action_flat.shape[1:])
        
        # get the output objects
        emb_object_flat = emb_object.view(-1, self.args.demb)

        decoder_input = decoder_input + emb_object_flat
        
        object_flat = self.dec_object(decoder_input)

        objects = object_flat.view(
            *encoder_out_visual.shape[:2], *object_flat.shape[1:])
        output.update({'action': action, 'object': objects})

        # (optionally) get progress monitor predictions
        if self.args.progress_aux_loss_wt > 0:
            progress = torch.sigmoid(self.dec_progress(encoder_out_visual))
            output['progress'] = progress
        if self.args.subgoal_aux_loss_wt > 0:
            subgoal = torch.sigmoid(self.dec_subgoal(encoder_out_visual))
            output['subgoal'] = subgoal
        return output

    def embed_lang(self, lang_pad, vocab):
        '''
        take a list of annotation tokens and extract embeddings with EncoderLang
        '''
        assert lang_pad.max().item() < len(vocab)
        embedder_lang = self.embs_ann[vocab.name]
        emb_lang, lengths_lang = self.encoder_lang(
            lang_pad, embedder_lang, vocab, self.pad)
        if self.args.detach_lang_emb:
            emb_lang = emb_lang.clone().detach()
        return emb_lang, lengths_lang

    def embed_frames(self, frames_pad):
        '''
        take a list of frames tensors, pad it, apply dropout and extract embeddings
        '''
        # print("frames_pad.shape", frames_pad.shape) # torch.Size([2, 72, 512, 7, 7])
        self.dropout_vis(frames_pad)
        frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).view(
            *frames_pad.shape[:2], -1)
        frames_pad_emb_skip = self.object_feat(
            frames_4d).view(*frames_pad.shape[:2], -1)
        # print("frames_pad_emb.shape", frames_pad_emb.shape) # torch.Size([2, 72, 768])
        # print("frames_pad_emb_skip.shape", frames_pad_emb_skip.shape) # torch.Size([2, 72, 768])
        return frames_pad_emb, frames_pad_emb_skip

    def embed_actions(self, actions):
        '''
        embed previous actions
        '''
        emb_actions = self.emb_action(actions)
        emb_actions = self.dropout_action(emb_actions)
        return emb_actions

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = torch.zeros(1, 0, *self.visual_tensor_shape)
        self.action_traj = torch.zeros(1, 0).long()

    def reset_for_clip(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = torch.zeros(1, 0, 768)
        self.action_traj = torch.zeros(1, 0).long()

    def reset_for_both(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = [torch.zeros(1, 0, *self.visual_tensor_shape), torch.zeros(1, 0, 768)]
        self.action_traj = torch.zeros(1, 0).long()
    
    def step(self, input_dict, vocab, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''
        if self.args.clip_image or self.args.clip_resnet:
            frames = input_dict['frames']
            device = frames[0].device
        else:
            frames = input_dict['frames'][0]
            device = frames.device

        region_feats = input_dict['regions']

        #もともと
        # device = frames.device
        #変更(clipとresnet両方を用いる場合)
        if prev_action is not None:
            prev_action_int = vocab['action_low'].word2index(prev_action)
            prev_action_tensor = torch.tensor(prev_action_int)[None, None].to(device)
            self.action_traj = torch.cat(
                (self.action_traj.to(device), prev_action_tensor), dim=1)

        #変更
        if self.args.clip_image or self.args.clip_resnet:
            self.frames_traj = [torch.cat(
                (self.frames_traj[0].to(device), frames[0][None]), dim=1), torch.cat(
                (self.frames_traj[1].to(device), frames[1][None]), dim=1)]
            frames = copy.deepcopy(self.frames_traj)
            if self.args.clip_resnet:
                lengths_frames = torch.tensor([self.frames_traj[0].size(1)])
                length_frames_max=self.frames_traj[0].size(1)
            else:
                lengths_frames = torch.tensor([self.frames_traj[1].size(1)])
                length_frames_max=self.frames_traj[1].size(1)
        else:
            self.frames_traj = torch.cat(
                (self.frames_traj.to(device), frames[None]), dim=1)
            frames = self.frames_traj.clone()
            lengths_frames = torch.tensor([self.frames_traj.size(1)])
            length_frames_max=self.frames_traj.size(1)
        
        # at timestep t we have t-1 prev actions so we should pad them
        action_traj_pad = torch.cat(
            (self.action_traj.to(device),
             torch.zeros((1, 1)).to(device).long()), dim=1)
        
        model_out = self.forward(
            vocab=vocab['word'],
            lang=input_dict['lang'],
            region_feats=region_feats,
            lengths_lang=input_dict['lengths_lang'],
            length_lang_max=input_dict['length_lang_max'],
            frames=frames,
            lengths_frames=lengths_frames,
            length_frames_max=length_frames_max,
            action=action_traj_pad)
        step_out = {}
        for key, value in model_out.items():
            # return only the last actions, ignore the rest
            step_out[key] = value[:, -1:]
        return step_out

    def compute_batch_loss(self, model_out, gt_dict):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # action loss
        action_pred = model_out['action'].view(-1, model_out['action'].shape[-1])
        action_gt = gt_dict['action'].view(-1)
        pad_mask = (action_gt != self.pad)
        action_loss = F.cross_entropy(action_pred, action_gt, reduction='none')
        action_loss *= pad_mask.float()
        action_loss = action_loss.mean()
        losses['action'] = action_loss * self.args.action_loss_wt

        # object classes loss
        object_pred = model_out['object']
        object_gt = torch.cat(gt_dict['object'], dim=0)
        interact_idxs = gt_dict['action_valid_interact'].view(-1).nonzero(
            as_tuple=False).view(-1)
        if interact_idxs.nelement() > 0:
            object_pred = object_pred.view(
                object_pred.shape[0] * object_pred.shape[1],
                *object_pred.shape[2:])
            object_loss = model_util.obj_classes_loss(
                object_pred, object_gt, interact_idxs)
            losses['object'] = object_loss * self.args.object_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            subgoal_pred = model_out['subgoal'].squeeze(2)
            subgoal_gt = gt_dict['subgoals_completed']
            subgoal_loss = F.mse_loss(subgoal_pred, subgoal_gt, reduction='none')
            subgoal_loss = subgoal_loss.view(-1) * pad_mask.float()
            subgoal_loss = subgoal_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.progress_aux_loss_wt > 0:
            progress_pred = model_out['progress'].squeeze(2)
            progress_gt = gt_dict['goal_progress']
            progress_loss = F.mse_loss(progress_pred, progress_gt, reduction='none')
            progress_loss = progress_loss.view(-1) * pad_mask.float()
            progress_loss = progress_loss.mean()
            losses['progress_aux'] = self.args.progress_aux_loss_wt * progress_loss

        # maximize entropy of the policy if asked
        if self.args.entropy_wt > 0.0:
            policy_entropy = - F.softmax(
                action_pred, dim=1) * F.log_softmax(action_pred, dim=1)
            policy_entropy = policy_entropy.mean(dim=1)
            policy_entropy *= pad_mask.float()
            losses['entropy'] = - policy_entropy.mean() * self.args.entropy_wt

        return losses

    def init_weights(self, init_range=0.1):
        '''
        init embeddings uniformly
        '''
        super().init_weights(init_range)
        self.dec_action.bias.data.zero_()
        self.dec_action.weight.data.uniform_(-init_range, init_range)
        self.emb_action.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose=False):
        '''
        compute exact matching and f1 score for action predictions
        '''
        preds = model_util.extract_action_preds(
            model_out, self.pad, self.vocab_out, lang_only=True)
        stop_token = self.vocab_out.word2index('<<stop>>')
        gt_actions = model_util.tokens_to_lang(
            gt_dict['action'], self.vocab_out, {self.pad, stop_token})
        model_util.compute_f1_and_exact(
            metrics_dict, [p['action'] for p in preds], gt_actions, 'action')
        model_util.compute_obj_class_precision(
            metrics_dict, gt_dict, model_out['object'])