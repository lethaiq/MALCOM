import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils.helpers import truncated_normal_
import numpy as np

class RandomGenerator(nn.Module):
    def __init__(self, train_content_iter, selective=False, sim_matrix=None, id2idx=None, attack_mode="target_real"):
        super(RandomGenerator, self).__init__()
        self.name = 'random_G'
        self.train_content_iter = train_content_iter
        self.sim_matrix = sim_matrix
        self.id2idx = id2idx
        self.idx2id = {}
        if self.id2idx:
            for _id in self.id2idx:
                self.idx2id[self.id2idx[_id]] = _id
        self.selective = selective
        self.attack_mode=attack_mode

    def sample(self, num_samples, batch_size, one_hot=True, content_iter=None, use_temperature=False): #default sample comments from fake articles
        num = 0
        samples = []
        if self.attack_mode:
            if self.attack_mode == "target_fake":
                target_label = 1
            else:
                target_label = 0

        while num < num_samples:
            random_batch = self.train_content_iter.random_batch(batch_size)
            content = random_batch['content'] # article_id
            _id = random_batch['id']
            current_comments = random_batch['comments'] # batch_size * 3 * 20
            label = random_batch['label']

            if self.attack_mode:
                batch_idx = torch.where(label == target_label)[0].detach().cpu().numpy()
            else:
                batch_idx = list(range(len(content)))
                
            comment_idx = np.random.choice(current_comments.size()[1])
            selected = current_comments[batch_idx,comment_idx,:]
            if one_hot:
                selected = F.one_hot(selected, cfg.vocab_size).float()
            samples.append(selected)
            num += selected.size()[0]

        samples = torch.cat(samples, 0)[:num_samples]

        return samples

