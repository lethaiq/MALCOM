import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import cosine, cdist
import config as cfg
from utils.helpers import truncated_normal_
import numpy as np

class RandomBetterGenerator(nn.Module):
    def __init__(self, train_content_iter, embeddings_sim=None, content_embeddings_data=None, id2idx=None, num_article=3, attack_mode="target_real"):
        super(RandomBetterGenerator, self).__init__()
        self.name = 'random_G'
        self.train_content_iter = train_content_iter
        self.embeddings_sim = embeddings_sim
        self.content_embeddings_data = content_embeddings_data
        self.id2idx = id2idx
        self.idx2id = {}
        if self.id2idx:
            for _id in self.id2idx:
                self.idx2id[self.id2idx[_id]] = _id
        self.attack_mode=attack_mode
        self.memory = {}
        self.attacks = ['real', 'fake']
        self.num_article = num_article

        print("processing training set...")
        self.id2data = {}
        self.all_train_data = self.train_content_iter.data
        self.checkedid = {}
        self.train_ids = {}
        for attack in self.attacks:
            self.memory[attack] = {}
            self.train_ids[attack] = []
        for i in range(len(self.all_train_data)):
            _id = self.all_train_data[i]['id']
            _label = self.all_train_data[i]['label']
            if _id not in self.checkedid:
                self.train_ids['fake'].append(_id) if _label == 1 else self.train_ids['real'].append(_id)
            if _id not in self.id2data:
                self.id2data[_id] = []
            self.id2data[_id].append(self.all_train_data[i])

        self.train_embeds = {}
        for attack in self.attacks:
            self.train_embeds[attack] = np.array([self.content_embeddings_data[_id] for _id in self.train_ids[attack]])

    def find_best_match(self, test_content_iter):
        def cos_cdist(matrix, vector):
            v = vector.reshape(1, -1)
            return cdist(matrix, v, 'cosine').reshape(-1)
        print("Memorizing training set...")
        for _, data in tqdm(enumerate(test_content_iter.loader)):
            for i in range(len(data['id'])):
                _id = data['id'][i]
                embed = self.content_embeddings_data[_id]
                for attack in self.attacks:
                    if _id not in self.memory[attack]:
                        sims = cos_cdist(self.train_embeds[attack], embed)
                        best_idx = np.random.choice(np.argsort(sims)[:self.num_article])
                        # best_idx = np.argsort(sims)[0]
                        best_id = self.train_ids[attack][best_idx]
                        best_sample = np.random.choice(self.id2data[best_id])
                        best_match_comments = best_sample['comments'].unsqueeze(0)
                        num_comment = best_sample['num_comment']
                        self.memory[attack][_id] = (best_match_comments, num_comment)

    def find_best_match_article(self, _id, target_label):
        if _id in self.memory[target_label]:
            return self.memory[target_label][_id]
        else:
            print("Cannot find this article in memory....", _id)


    def sample(self, num_samples, batch_size, one_hot=False, content_iter=None, use_temperature=False): #default sample comments from fake articles
        num = 0
        samples = []
        if self.attack_mode == "target_fake":
            target_label = 'fake'
        else:
            target_label = 'real'

        while num < num_samples:
            if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
                batch = content_iter.random_batch(batch_size)
            else:
                batch = content_iter

            for i in range(len(batch['content'])):
                _id = batch['id'][i]
                current_comments, num_comment = self.find_best_match_article(_id, target_label)
                comment_idx = np.random.choice(num_comment)
                selected = current_comments[0, comment_idx, :].unsqueeze(0)
                if one_hot:
                    selected = F.one_hot(selected, cfg.vocab_size).float()
                samples.append(selected)
                num += 1

        samples = torch.cat(samples, 0)[:num_samples]

        return samples

