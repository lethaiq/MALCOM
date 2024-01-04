import config as cfg
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from models.discriminator import CNNDiscriminator
from utils.helpers import truncated_normal_
from utils.text_process import build_embedding_matrix
f_filter_sizes = [2, 3, 4]
f_num_filters = [15, 15, 15]

class RelGAN_F_CSI(nn.Module):
    def __init__(self, embedding_dim, vocab_size, padding_idx, hidden_dim,
                 gpu=False, dropout=0.2, dropout_content=0.95):
        super(RelGAN_F_CSI, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.gpu = gpu
        self.content_dim = 512

        # self.feature_dim = sum(f_num_filters)
        self.feature_dim = cfg.F_feature_dim

        # COMMON
        if cfg.if_pretrain_embedding:
            self.embeddings = nn.Linear(vocab_size, self.embedding_dim, bias=False)
            embedding_matrix = build_embedding_matrix(cfg.dataset, self.embedding_dim) # vocab_size * embedding_size
            self.embeddings.weight = torch.nn.Parameter(embedding_matrix.t())
            self.embeddings.weight.requires_grad = False
        else:
            self.embeddings = nn.Linear(vocab_size, self.embedding_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.embedding_dim)) for (n, f) in zip(f_num_filters, f_filter_sizes)
        ])

        self.content2feature = nn.Linear(self.content_dim, self.feature_dim)
        self.emb2feature = nn.Linear(self.embedding_dim, self.feature_dim)
        self.dropout_content = nn.Dropout(dropout_content)
        
        self.gru = nn.GRU(self.feature_dim*cfg.max_seq_len, hidden_dim, num_layers=2, bidirectional=True, dropout=0.2)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.init_params()
        self.over_write_hyperparam(cfg)

    def over_write_hyperparam(self, cfg):
        # cfg.F_clf_lr = 0.001
        pass

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h


    def encode_comments(self, comments, use_dropout=True):
        all_comments = []
        for i in range(comments[0].size()[0]):
            comment = comments[:,i,:,:].squeeze(1) 
            comment = self.get_feature(comment, use_dropout=use_dropout) 
            comment = comment.view(comment.size()[0], -1)
            comment = comment.unsqueeze(1)
            all_comments.append(comment)
        comment = torch.cat(tuple(all_comments), 1) # batch_size * num_comments * (seq_len * feature_dim)
        return comment


    def forward(self, inp, comments, num_comments=None, use_dropout=True):
        batch_size = inp.size()[0]
        # content_out = self.get_content_feature(inp)
        comment = self.encode_comments(comments, use_dropout=use_dropout)

        comment = comment.permute(1, 0, 2) # max_num_comment * batch_size * feature_dim
        hidden = self.init_hidden(batch_size)
        _, hidden = self.gru(comment, hidden)  # 4 * batch_size * hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()  # batch_size * 4 * hidden_dim

        comment_out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))  # batch_size * 4 * hidden_dim
        comment_out = torch.tanh(comment_out)  # batch_size * feature_dim
        feature = comment_out
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