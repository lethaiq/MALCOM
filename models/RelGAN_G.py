# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from tqdm import tqdm
from utils.utils import *
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory


class RelGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, 
                vocab_size, max_seq_len, padding_idx, dropout=0.0, gpu=False):
        super(RelGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'relgan'
        self.dropout = nn.Dropout(dropout)
        self.temperature = 1.0  # init value is 1.0
        self.gpu = gpu
        
        # RMC
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.hidden_dim = mem_slots * num_heads * head_size
        self.lstm = RelationalMemory(mem_slots=mem_slots, 
                                     head_size=head_size, 
                                     input_size=embedding_dim*2,
                                     num_heads=num_heads, 
                                     return_all_outputs=True)
        self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        # Prior content feature
        self.content_feature_dim = cfg.content_feature_dim
        self.max_content_len = cfg.max_seq_len
        self.content2feature = nn.Linear(cfg.max_seq_len*embedding_dim, embedding_dim)


        self.downsample = nn.Linear(cfg.max_seq_len * cfg.vocab_size, 512)

        # Prior comment feature
        self.feature_dim = 15
        if not cfg.if_old_version:
            self.emb2feature = nn.Linear(self.embedding_dim, self.feature_dim)
            self.comment2feature = nn.Linear(self.feature_dim*cfg.max_seq_len, self.content_feature_dim)

        self.init_params()
        pass

    def step(self, emb, hidden, mask=None):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        # emb = self.embeddings(inp).unsqueeze(1) # batch_size * 1 * embed_dim

        out, hidden = self.lstm(emb, hidden)
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))

        if self.temperature == 0:
            pred = torch.zeros(len(emb), cfg.vocab_size)
            if np.random.choice([0,1], p=[0.1, 0.9]) == 1:
                rnd = np.random.choice(self.vocab_size, len(emb))
                next_token = torch.from_numpy(rnd).to('cuda')
                pred[:,rnd] = 1
            else:
                next_token = torch.Tensor([cfg.padding_idx]*emb.size()[0]).long().to('cuda')
                pred[:,cfg.padding_idx] = 1
        else:
            next_token = torch.argmax(gumbel_t, dim=1).detach()
            pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
            
        next_token_onehot = None
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o


    def sample(self, num_samples, batch_size, one_hot=False, downsample=False, start_letter=cfg.start_letter, content_iter=None, mask=None):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        # print(batch_size, len(content_iter))
        global all_preds
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.to('cuda')

        if num_batch > 1:
            bar = tqdm(range(num_batch))
        else:
            bar = range(num_batch)

        for b in bar:
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                hidden = hidden.to('cuda')
                inp = inp.to('cuda')

            if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
                batch = content_iter.random_batch(batch_size)
                content = batch['title']
            else:
                content = content_iter['title']
            if self.gpu:
                content = content.to('cuda')

            content = self.embeddings(content).view(len(content), -1)
            emb_content = self.content2feature(content).unsqueeze(1)

            for i in range(self.max_seq_len):
                emb = self.embeddings(inp).unsqueeze(1)
                emb = torch.cat((emb, emb_content), 2)
                pred, hidden, next_token, _, _ = self.step(emb, hidden, mask)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            if downsample:
                all_preds = self.downsample(all_preds.view(batch_size, -1))
            return all_preds  # batch_size * seq_len * vocab_size



        return samples


    def init_hidden_bk(self, batch_size):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size)
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory.to('cuda') if self.gpu else memory


    def init_hidden(self, batch_size):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size, cut_half=False)
        if self.gpu:
            memory = memory.to('cuda')
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory


    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=cfg.CUDA):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())
        if gpu:
            u = u.to('cuda')

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t