# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.helpers import truncated_normal_
from models.discriminator import CNNDiscriminator
import config as cfg
from utils.text_process import build_embedding_matrix

class RelGAN_F(nn.Module):
    def __init__(self, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.65, dropout_content=0.95):
        super(RelGAN_F, self).__init__()
        self.name = "class"
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu
        self.output_dim = 2
        self.content_dim = 512
        self.feature_dim = 15

        # COMMON
        if cfg.if_pretrain_embedding:
            self.embeddings = nn.Linear(vocab_size, self.embedding_dim, bias=False)
            # print("original weight shape:", self.embeddings.weight.size()) original weight shape: torch.Size([200, 10013])
            embedding_matrix = build_embedding_matrix(cfg.dataset, self.embedding_dim) # vocab_size * embedding_size
            self.embeddings.weight = torch.nn.Parameter(embedding_matrix.t())
            self.embeddings.weight.requires_grad = False
        else:
            self.embeddings = nn.Linear(vocab_size, self.embedding_dim, bias=False)

        self.emb2feature = nn.Linear(self.embedding_dim, self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_content = nn.Dropout(dropout_content)

        self.content2feature = nn.Linear(self.content_dim, cfg.max_seq_len*self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim*cfg.max_seq_len, self.output_dim)

        self.init_params()
        self.over_write_hyperparam(cfg)


    def over_write_hyperparam(self, cfg):
        # cfg.F_clf_lr = 0.001
        pass

    def forward(self, inp, comments, num_comments, use_dropout=True):
        all_features = []
        content = self.get_content_feature(inp, use_dropout=use_dropout)
        for i in range(comments[0].size()[0]):
            comment = comments[:,i,:,:].squeeze(1) #batcg_size*1*seq_len*vocab_size
            comment = self.get_feature(comment, use_dropout=use_dropout) # batch_size * 1 * seq_len*feature_dim
            comment = comment.unsqueeze(1)
            all_features.append(comment)
        feature = torch.cat(tuple(all_features), 1) # batch_size * num_comments * (seq_len * feature_dim)

        if cfg.if_flex_com:
            feature = torch.sum(feature, 1) # batch_size * 1 * seq_len * feature_dim
            feature = feature / num_comments.view(num_comments.size()[0], 1, 1)
        else:
            feature = torch.mean(feature, 1)

        feature = feature.view(feature.size()[0], -1) #batch_size * (feature_dim*seq_len)
        feature = (feature + content)/2
        pred = self.feature2out(feature)
        return pred

    def get_content_feature(self, content, use_dropout=True):
        feat = self.content2feature(content)
        if use_dropout:
            feat = self.dropout_content(feat)
        return feat

    def get_feature(self, inp, use_dropout=True):
        emb = self.embeddings(inp)
        pred = self.emb2feature(emb) 
        if use_dropout:
            pred = self.dropout(pred)
        return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)
