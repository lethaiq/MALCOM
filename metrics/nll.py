# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : nll.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config as cfg
from metrics.basic import Metrics


class NLL(Metrics):
    def __init__(self, name, content_iter=None, if_use=False, gpu=False):
        super(NLL, self).__init__(name)

        self.if_use = if_use
        self.model = None
        self.data_loader = None
        self.label_i = None
        self.leak_dis = None
        self.gpu = gpu
        self.criterion = nn.NLLLoss()
        # self.criterion = nn.CrossEntropyLoss()

    def get_score(self):
        """note that NLL score need the updated model and data loader each time, use reset() before get_score()"""
        if not self.if_use:
            return 0
        assert self.model and self.data_loader, 'Need to reset() before get_score()!'

        if self.leak_dis is not None:  # For LeakGAN
            return self.cal_nll_with_leak_dis(self.model, self.data_loader, self.leak_dis, self.gpu)
        elif self.label_i is not None:  # For category text generation
            return self.cal_nll_with_label(self.model, self.data_loader, self.label_i,
                                           self.criterion, self.gpu)
        else:
            return self.cal_nll(self.model, self.data_loader, self.criterion, self.gpu)

    def reset(self, model=None, data_loader=None, label_i=None, leak_dis=None):
        self.model = model
        self.data_loader = data_loader
        self.label_i = label_i
        self.leak_dis = leak_dis

    # @staticmethod
    # def nll_loss_criterion(pred, target):
    #     # pretrain_loss = -tf.reduce_sum(
    #     # tf.one_hot(tf.to_int32(tf.reshape(x_real, [-1])), vocab_size, 1.0, 0.0) * tf.log(
    #     #     tf.clip_by_value(tf.reshape(g_predictions, [-1, vocab_size]), 1e-20, 1.0)
    #     # )
    #     # ) / (seq_len * batch_size)
    #     # pred = pred.view(inp.size()[0], cfg.max_seq_len)
    #     target_one_hot = torch.clamp(F.one_hot(target, cfg.vocab_size), 1e-20, 1.0)
    #     loss = torch.mean(target_one_hot.float() * F.log_softmax(pred.view(-1, cfg.vocab_size).float()))
    #     loss = loss / (cfg.batch_size * cfg.max_seq_len)
    #     return loss

    # def train_gen_epoch(self, model, data_loader, criterion, optimizer, content_iter=None):
    #     total_loss = 0
    #     for i, data in enumerate(data_loader):
    #         inp, target = data['input'], data['target']
    #         if cfg.CUDA:
    #             inp, target = inp.cuda(), target.cuda()
    #         hidden = model.init_hidden(data_loader.batch_size, content_iter=content_iter)
    #         pred = model.forward(inp, hidden)
    #         loss = criterion(pred, target.view(-1))
    #         self.optimize(optimizer, loss, model)
    #         total_loss += loss.item()
    #     return total_loss / len(data_loader)

    @staticmethod
    def cal_nll(model, data_loader, criterion, gpu=cfg.CUDA):
        """NLL score for general text generation model."""
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target, content = data['input'], data['target'], data['content']
                if gpu:
                    inp, target = inp.cuda(), target.cuda() # batch_size * max_seq_len
                hidden = model.init_hidden(data_loader.batch_size)
                pred = model.forward(inp, hidden, content_iter=data)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)

    @staticmethod
    def cal_nll_with_label(model, data_loader, label_i, criterion, gpu=cfg.CUDA):
        """NLL score for category text generation model."""
        assert type(label_i) == int, 'missing label'
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                label = torch.LongTensor([label_i] * data_loader.batch_size)
                if gpu:
                    inp, target, label = inp.cuda(), target.cuda(), label.cuda()

                hidden = model.init_hidden(data_loader.batch_size)
                if model.name == 'oracle':
                    pred = model.forward(inp, hidden)
                else:
                    pred = model.forward(inp, hidden, label)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)

    @staticmethod
    def cal_nll_with_leak_dis(model, data_loader, leak_dis, gpu=cfg.CUDA):
        """NLL score for LeakGAN."""
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if gpu:
                    inp, target = inp.cuda(), target.cuda()

                loss = model.batchNLLLoss(target, leak_dis)
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)
