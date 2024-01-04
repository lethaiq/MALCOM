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

f_filter_sizes = [2, 3, 4]
f_num_filters = [50, 50, 50]


class RelGAN_H(nn.Module):
    def __init__(self, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.5, dropout_content=0.85):
        super(RelGAN_H, self).__init__()
        self.name = "class"
        # self.name = "class"
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(f_num_filters)
        self.gpu = gpu

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)
        self.emb2feature = nn.Linear(embed_dim, self.feature_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(f_num_filters, f_filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.downsample = nn.Linear(self.feature_dim, 32)

        output_dim = 2

        self.feature2out = nn.Linear(32*5, 32)
        self.out2logits = nn.Linear(32, output_dim)
        self.content2feature = nn.Linear(512, 32)
        self.dropout_content = nn.Dropout(dropout_content)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp, comments, num_comments=None):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len * vocab
        :param comments: batch_size * 3 * seq_len * vocab
        :return: pred: batch_size * 1
        """
        v1 = self.get_content_feature(inp) # batch_size * 512
        v2 = comments[:,0,:,:].squeeze(1)
        v2 = self.get_feature(v2) # batch_size * 128
        # feature = torch.cat((v1, v2), 1)
        feature = torch.cat((v1,torch.abs(v1 - v2),v2,v1*v2, (v1+v2)/2), 1)
        # feature = v2
        pred = self.feature2out(feature)
        # pred = self.dropout(pred)
        pred = self.out2logits(pred)

        return pred

    def get_content_feature(self, content, dropout=True):
        feat = self.content2feature(content)
        if dropout:
            feat = self.dropout_content(feat)
        return feat

    def get_feature(self, inp, dropout=True):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        pred = self.downsample(pred)
        if dropout:
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
