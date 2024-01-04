# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : data_loader.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.text_process import *


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class GenDataIterContent:
    def __init__(self, samples, embedding_data, batch_size=64, shuffle=None, drop_last=True, name="TRAIN"):
        self.batch_size = batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        self.embedding_data = embedding_data
        self.drop_last = drop_last
        self.name = name

        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset, cfg.vocab_size)
        self.data = self.__read_data__(samples)
        print("total data", len(self.data))
        self.loaders = {}
        self.loader = DataLoader(
            dataset=GANDataset(self.data),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last)
        self.loaders[self.batch_size] = self.loader
        
    def get_loader(self, batch_size):
        if batch_size not in self.loaders:
            loader = DataLoader(
                dataset=GANDataset(self.data),
                batch_size=batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
            self.loaders[batch_size] = loader
        return self.loaders[batch_size]

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        # global all_data
        all_data = None
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, target = self.prepare(samples, labels)
            all_data = [{'id': i, 'comment':t} for (i, t) in zip(inp, target)]

        elif isinstance(samples, str):  # filename
            inp, target, titles, titles_str, ids = self.load_data(samples)
            all_data = []
            for i in range(len(ids)):
                real_id = ids[i]
                content = torch.from_numpy(self.embedding_data[real_id])
                tmp = {
                        'idx': i,
                        'id':ids[i],
                        'title': titles[i],
                        'title_str': titles_str[i],
                        'content': content, 
                        'input': inp[i],
                        'target': target[i],
                        }
                all_data.append(tmp)
        else:
            all_data = None

        return all_data

    def random_batch(self, batch_size=None):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        if not batch_size:
            batch_size = self.batch_size
        loader = self.get_loader(batch_size)
        idx = random.randint(0, len(loader) - 1)
        return list(loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, gpu=True):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = cfg.start_letter
        inp[:, 1:] = target[:, :cfg.max_seq_len - 1]
        # if gpu:
        #     return inp.cuda(), target.cuda(), labels.cuda()
        return inp, target

    def load_data(self, samples_filename):
        """Load real data from local file"""
        self.tokens, tokens_title, titles_str, ids = get_tokenized_sep_content(samples_filename, sep="::")
        samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
        titles = tokens_to_tensor(tokens_title, self.word2idx_dict)
        inp, target = self.prepare(samples_index)
        return inp, target, titles, titles_str, ids

    def __len__(self):
        return len(self.loader)


class GenDataIter:
    def __init__(self, samples, batch_size, if_test_data=False, shuffle=None):
        self.batch_size = batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        if cfg.if_real_data:
            # print("loading train data", cfg.vocab_size)
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset, cfg.vocab_size)
        if if_test_data:  # used for the classifier
            # print("loading test data", cfg.vocab_size)
            self.word2idx_dict, self.idx2word_dict = load_test_dict(cfg.dataset, cfg.vocab_size)

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        # global all_data
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, target = self.prepare(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        elif isinstance(samples, str):  # filename
            inp, target, length = self.load_data(samples)
            all_data = [{'input': i, 'target': t, 'length': lt} for (i, t, lt) in zip(inp, target, length) if lt>1]
        else:
            all_data = None
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def get_loader(self, batch_size):
        if batch_size not in self.loaders:
            # print("generating new loader for batch_size", batch_size)
            loader = DataLoader(
                dataset=GANDataset(self.data),
                batch_size=batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
            self.loaders[batch_size] = loader
        return self.loaders[batch_size]
        
    def random_batch(self, batch_size=None):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        if not batch_size:
            batch_size = self.batch_size
        loader = self.get_loader(batch_size)
        idx = random.randint(0, len(loader) - 1)
        return list(loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = cfg.start_letter
        inp[:, 1:] = target[:, :cfg.max_seq_len - 1]
        return inp, target

    def load_data(self, filename):
        """Load real data from local file"""
        self.tokens = get_tokenlized(filename)
        samples_index, samples_length = tokens_to_tensor(self.tokens, self.word2idx_dict, return_length=True)
        inp, target = self.prepare(samples_index)
        print("total data loaded is {}".format(len(inp)))
        return inp[:cfg.max_num_data], target[:cfg.max_num_data], samples_length[:cfg.max_num_data]


class DisDataIter:
    def __init__(self, pos_samples, neg_samples, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

    def __read_data__(self, pos_samples, neg_samples):
        """
        input: same as target, but start with start_letter.
        """
        inp, target = self.prepare(pos_samples, neg_samples)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def prepare(self, pos_samples, neg_samples, gpu=False):
        """Build inp and target"""
        inp = torch.cat((pos_samples, neg_samples), dim=0).long().detach()  # !!!need .detach()
        target = torch.ones(inp.size(0)).long()
        target[pos_samples.size(0):] = 0

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target
