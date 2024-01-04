# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
import math

import torch
import torch.nn as nn

import config as cfg
from utils.helpers import truncated_normal_
import torch.nn.functional as F

class LSTMGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(LSTMGenerator, self).__init__()
        self.name = 'vanilla'

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.temperature = 1.0

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_params()
        # print("self.embeddings", self.embeddings)

    def forward(self, inp, hidden, need_hidden=False, return_log_softmax=True, one_hot=None, content_iter=None):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        # print(inp.max())
        # print(inp.min())
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
            
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
            batch = content_iter.random_batch()
            content = batch['title']
        else:
            content = content_iter['title']
            
        if self.gpu:
            content = content.to('cuda:0') # batch_size * 512
        
        content = self.embeddings(content).view(len(content), -1)
        # print("content", content.size()) # torch.Size([8, 640])
        # print("self.content2feature", self.content2feature)
        # print(content)

        content = torch.rand(content.size()).to('cuda')
        
        emb_content = self.content2feature(content)

        emb_content = emb_content.unsqueeze(1) # batch_size * 1 * embedding_dim
        emb_content = emb_content.repeat(1, cfg.max_seq_len, 1)
        emb = torch.cat((emb, emb_content), 2)

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature

        if return_log_softmax:
            pred = self.softmax(out)
        else:
            pred = out

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter, content_iter=None, one_hot=None, use_temperature=False):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.to('cuda')

            for i in range(self.max_seq_len):
                out, hidden = self.forward(inp, hidden, need_hidden=True)  # out: batch_size * vocab_size
                next_token = torch.multinomial(torch.exp(out), 1)  # batch_size * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token.view(-1)
                inp = next_token.view(-1)
        samples = samples[:num_samples]

        return samples

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)

    def init_oracle(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size=cfg.batch_size, content_iter=None):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        if self.gpu:
            return h.to('cuda'), c.to('cuda')
        else:
            return h, c
