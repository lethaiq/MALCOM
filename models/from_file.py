import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import cosine, cdist
import config as cfg
from utils.helpers import truncated_normal_
import numpy as np
from copy import deepcopy
from utils.data_loader_generated import *
from utils.text_process import *


class FromFileGenerator(nn.Module):
    def __init__(self, file, embedding_data, idx2word):
        super(FromFileGenerator, self).__init__()
        self.name = 'from_file_g'
        self.memory_loader = G_DataIterSep(file, embedding_data,
                                            batch_size=64, shuffle=False, 
                                            drop_last=False)
        self.used = []
        self.id2idx = deepcopy(self.memory_loader.id2idx)
        self.idx2word = idx2word

    def sample(self, num_samples, batch_size, one_hot=True, content_iter=None, use_temperature=False): #default sample comments from fake articles
        num = 0
        samples = []
        count_notfound = 0
        while num < num_samples:
            if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
                batch = content_iter.random_batch(batch_size)
            else:
                batch = content_iter

            for i in range(len(batch['content'])):
                _id = batch['id'][i]
                if _id in self.id2idx and len(self.id2idx[_id])>0:
                    best_idx = np.random.choice(self.id2idx[_id])
                    best_data = self.memory_loader.data[best_idx]
                    comment = best_data['single_comment']
                    selected = comment.unsqueeze(0) # 1 * max_seq_len
                    self.id2idx[_id].remove(best_idx)
                else:
                    count_notfound += 1
                    selected = torch.zeros(1, cfg.max_seq_len).long()

                if one_hot:
                    selected = F.one_hot(selected, cfg.vocab_size).float() # 1 * max_seq_len * vocab_size

                samples.append(selected.unsqueeze(0))
                num += 1

        samples = torch.cat(samples, 0)[:num_samples].squeeze(1)
        # print("Generated: {} Not found in this batch: {}".format(samples.size(), count_notfound))

        return samples

