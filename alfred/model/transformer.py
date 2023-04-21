#追加
import re

import torch
from torch import nn
from torch.nn import functional as F
#追加
import clip
from torch.nn.utils.rnn import pad_sequence

from alfred.model import base
from alfred.nn.enc_lang import EncoderLang
from alfred.nn.enc_visual import FeatureFlat
from alfred.nn.enc_vl import EncoderVL
from alfred.nn.encodings import DatasetLearnedEncoding
from alfred.nn.dec_object import ObjectClassifier
from alfred.utils import model_util
from alfred.utils.data_util import tokens_to_lang


class Model(base.Model):
    def __init__(self, args, embs_ann, vocab_out, pad, seg):
        '''
        transformer agent
        '''
        super().__init__(args, embs_ann, vocab_out, pad, seg)

        #追加
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device="cuda")
        for params in self.clip_model.parameters():
            params.requires_grad = False

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
        self.reset()

    #追加
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
    
    #追加
    def preprocess_clip(self, images):
        """
        Preprocess images with the CLIP model.
        """
        images = self.clip_preprocess(images).unsqueeze(0)
        return images
    
    #追加
    def featurize_clip(self, images):
        """
        Featurize images with the CLIP model.
        """
        images = self.preprocess_clip(images)
        image_features = self.clip_model.encode_image(images)
        return image_features
    
    #追加
    def encode_clip(self, sentences, device="cuda:0"):
        """
        Encode two sentences with the CLIP model.
        """
        # if len(sentences) != 2:
        if len(sentences) == 2:
            # Tokenize the text
            tokenized1 = clip.tokenize(sentences[0]).to(device)
            tokenized2 = clip.tokenize(sentences[1]).to(device)
            
            # Encode the text
            text_features1 = self.clip_model.encode_text(tokenized1)
            text_features2 = self.clip_model.encode_text(tokenized2)

            text_features = pad_sequence([text_features1, text_features2], batch_first=True)
            return text_features, torch.tensor([tokenized1.shape[0], tokenized2.shape[0]]).to(device)
        else:
            tokenized = clip.tokenize(sentences[0]).to(device)
            text_features = self.clip_model.encode_text(tokenized)
            return text_features, torch.tensor([tokenized.shape[0]]).to(device)
    #追加
    def concat_embeddings(self, emb_lang, lengths_lang, emb_clip, lengths_clip, device="cuda:0"):
        if len(lengths_lang) == 1:
            temp = torch.cat([emb_lang[0, :lengths_lang[0], :], emb_clip.unsqueeze(1)[0, :lengths_clip[0], :]], dim=0)
            return temp.unsqueeze(0), torch.tensor([temp.shape[0]]).to(device)

        lang_1 = emb_lang[0, :lengths_lang[0], :]
        lang_2 = emb_lang[1, :lengths_lang[1], :]
        clip_1 = emb_clip[0, :lengths_clip[0], :]
        clip_2 = emb_clip[1, :lengths_clip[1], :]

        temp1 = torch.cat([lang_1, clip_1], dim=0)
        temp2 = torch.cat([lang_2, clip_2], dim=0)

        emb_lang = pad_sequence([temp1, temp2], batch_first=True, padding_value=0)

        return emb_lang, torch.tensor([temp1.shape[0], temp2.shape[0]]).to(device)

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        # embed language
        output = {}

        emb_lang, lengths_lang = self.embed_lang(inputs['lang'], vocab)
        #emb_lang:[2, max, 768](2つのデータの大きい方をmaxに入れる), lengths_lang:[2](emb_langの2つのデータの長さを持つ.)
        
        #追加
        # token to sentence
        sentences = self.token_to_sentence_list(inputs['lang'], vocab)
        
        # encode clip
        emb_clip, lengths_clip = self.encode_clip(sentences, device=inputs['lang'].device)

        # concat clip and lang
        emb_lang, lengths_lang = self.concat_embeddings(emb_lang, lengths_lang, emb_clip, lengths_clip, device=inputs['lang'].device)

        emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang

        # print("inputs['frames'].shape", inputs['frames'].shape)
        # print("inputs['frames'] type", type(inputs['frames']))

        # embed frames and actions
        emb_frames, emb_object = self.embed_frames(inputs['frames'])
        lengths_frames = inputs['lengths_frames']

        #emb_frames:[2, max_, 768], lengths_frames:[2],emb_actions:[2,max_,768] (langのmaxとは違う), ex. inputs['frames']: [2, 72, 512, 7, 7]
        emb_actions = self.embed_actions(inputs['action'])
        assert emb_frames.shape == emb_actions.shape
        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs['length_frames_max']

        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang, emb_frames, emb_actions, lengths_lang,
            lengths_frames, lengths_actions, length_frames_max)
        # use outputs corresponding to visual frames for prediction only
        encoder_out_visual = encoder_out[
            :, lengths_lang.max().item():
            lengths_lang.max().item() + length_frames_max]

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
        self.dropout_vis(frames_pad)
        frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).view(
            *frames_pad.shape[:2], -1)
        frames_pad_emb_skip = self.object_feat(
            frames_4d).view(*frames_pad.shape[:2], -1)
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

    def step(self, input_dict, vocab, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''
        frames = input_dict['frames']
        device = frames.device
        if prev_action is not None:
            prev_action_int = vocab['action_low'].word2index(prev_action)
            prev_action_tensor = torch.tensor(prev_action_int)[None, None].to(device)
            self.action_traj = torch.cat(
                (self.action_traj.to(device), prev_action_tensor), dim=1)
        self.frames_traj = torch.cat(
            (self.frames_traj.to(device), frames[None]), dim=1)
        # at timestep t we have t-1 prev actions so we should pad them
        action_traj_pad = torch.cat(
            (self.action_traj.to(device),
             torch.zeros((1, 1)).to(device).long()), dim=1)
        model_out = self.forward(
            vocab=vocab['word'],
            lang=input_dict['lang'],
            lengths_lang=input_dict['lengths_lang'],
            length_lang_max=input_dict['length_lang_max'],
            frames=self.frames_traj.clone(),
            lengths_frames=torch.tensor([self.frames_traj.size(1)]),
            length_frames_max=self.frames_traj.size(1),
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
