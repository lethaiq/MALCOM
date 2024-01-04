# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import config as cfg
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminator import CNNDiscriminator
from utils.helpers import truncated_normal_
from utils.text_process import build_embedding_matrix
f_filter_sizes = [2, 3, 4]
f_num_filters = [15, 15, 15]

class RelGAN_F_CNN(nn.Module):
    def __init__(self, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.65, dropout_content=0.95):
        super(RelGAN_F_CNN, self).__init__()
        self.name = "class"
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu
        self.output_dim = 2
        self.content_dim = 512
        self.feature_dim = sum(f_num_filters)

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

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.embedding_dim)) for (n, f) in zip(f_num_filters, f_filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)

        self.content2feature = nn.Linear(self.content_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, self.output_dim)

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
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim # emb torch.Size([128, 1, 44, 300])
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        # highway = self.highway(pred)
        # pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway
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
