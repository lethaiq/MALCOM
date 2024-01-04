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

from models.discriminator import CNNDiscriminator

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [150, 150, 150, 150]


class RelGAN_D_TOPIC(CNNDiscriminator):
    def __init__(self, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(RelGAN_D_TOPIC, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx,
                                       gpu, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)
        self.num_rep = num_rep

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(self.emb_dim_single, 1)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim*2, 100)
        self.out2logits = nn.Linear(100, 1)

        self.feature2outB = nn.Linear(self.feature_dim*2, 100)
        self.out2logitsB = nn.Linear(100, 1)

        self.dropout = nn.Dropout(dropout)
        self.content2feature = nn.Linear(512, self.feature_dim)

        self.init_params()


    def forward(self, content, comment, returnA=True):
        content = self.content2feature(content).repeat(self.num_rep,1)

        emb = self.embeddings(comment).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim       # N * C * H * W
        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]

        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim

        pred = torch.cat((pred, content), 1) # batch_size * (feature_dim*2)

        if returnA:
            predA = self.feature2out(self.dropout(pred))
            logits = self.out2logits(predA).squeeze(1)  # [batch_size * num_rep]
        else:
            predB = self.feature2outB(self.dropout(pred))
            logits = self.out2logitsB(predB).squeeze(1)

        return logits
        # return content, pred

    def forward_bk(self, generated, content=None, comment=None):
        """
        Get logits of discriminator
        :param inp: batch_size * 512
        :param comment: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """

        # print("inp", inp.size())
        # print("comment", comment.size())

        # learn feature for comment
        emb = self.embeddings(comment).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim       # N * C * H * W
        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]

        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        # learn feature for content
        content = self.content2feature(content).repeat(self.num_rep,1)
        # print("comment_feature", pred.size()) # 2048 * 1200
        # print("content_feature", content.size()) # 64 * 1200

        pred = torch.cat((pred, content), 1) # batch_size * (feature_dim*2)

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits
