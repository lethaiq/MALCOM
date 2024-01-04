import config as cfg
import heapq
import nltk
import numpy as np
import os
import pickle
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy import linalg as LA
from operator import itemgetter
from torch.autograd import Variable
from torch.autograd import grad
from tqdm import tqdm
from utils.text_process import build_embedding_matrix
from utils.text_process import tensor_to_tokens
from utils.tweet_utils import *

class UniversalTrigger(nn.Module):
    def __init__(self, clf, train_content_iter, trigger_length, 
                    idx2word, word2idx, random_g, 
                    beam_size=1, n_neighbor=100,
                    attack_mode="target_real",
                    topic_model=None,
                    if_retrain=False):

        super(UniversalTrigger, self).__init__()
        self.name = 'universal_trigger'
        self.train_content_iter = train_content_iter
        self.trigger_length = trigger_length
        self.beam_size = beam_size
        self.topic_model = topic_model
        self.n_neighbor = n_neighbor
        self.if_retrain = if_retrain

        self.extracted_grads = []
        def extract_grad_hook(module, grad_in, grad_out):
            # print("grad out", grad_out[0].size()) grad out torch.Size([128, 26, 16])
            self.extracted_grads.append(grad_out[0])

        if cfg.blackbox_using_path:
            print("(blackbox UniTrigger) loading whitebox classifier at {}".format(cfg.blackbox_using_path))
            from models.RelGAN_F_RNN2 import RelGAN_F_RNN2
            self.clf = RelGAN_F_RNN2(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, extract_grad_hook=extract_grad_hook,
                                                                                            hidden_dim=cfg.rnn_hidden_dim,
                                                                                            dropout=cfg.f_dropout_comment,
                                                                                            dropout_content=cfg.f_dropout_content,
                                                                                            gpu=cfg.CUDA)
            self.clf.load_state_dict(torch.load(cfg.blackbox_using_path))
            self.whitebox = "RNN2"
        else:
            print("(whitebox HotFlip) using same classifier with attack")
            try:
                self.clf = type(clf)(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx) # get a new instance
            except:
                self.clf = type(clf)(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, hidden_dim=cfg.rnn_hidden_dim) # get a new instance
            self.clf.load_state_dict(clf.state_dict()) # copy weights and stuf
            self.whitebox = cfg.f_type

        if cfg.CUDA:
            self.clf = self.clf.cuda()
        for param in self.clf.parameters():
                param.requires_grad = True
        self.clf.train()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.attack_mode=attack_mode
        if self.attack_mode == "target_fake":
            self.target_label = 1
        else:
            self.target_label = 0 
        self.idx2word = idx2word
        self.word2idx = word2idx
        # self.embedding_matrix = build_embedding_matrix(cfg.dataset, 200) # vocab_size * embedding_size
        self.embedding_matrix = self.extract_embedding_matrix() #embedding_dim * vocab
        self.random_g = random_g
        self.memory = {}
        self.init_tensor = {}

        if not self.if_retrain:
            try:
                load_path="{}_UniTrigger_Whitebox-{}_CopyCat-{}_Neighbor-{}".format(cfg.dataset, self.whitebox, self.random_g.num_article, self.n_neighbor)
                self.init_tensor = pickle.load(open(load_path, 'rb'))
                print("UniTrigger loaded from memory file", load_path)
            except Exception as e:
                print("Cannot load UniTrigger trained prefix", e)
        else:
            self.train()


    def extract_embedding_matrix(self):
        return self.clf.embeddings.weight.T # embedding_dim * vocab


    def get_label_for_attack(self, batch_size, attack_mode):
        if attack_mode == "target_real":
            label = torch.ones(batch_size).long()
        elif attack_mode == "target_fake":
            label = torch.zeros(batch_size).long()
        return label
        # if we attack target_real, we want to replace word that is important to fake


    def evaluate_batch(self, batch, init_tensor):
        content = batch['content']
        batch_size = len(content)
        # _title = batch['title']
        _title = self.random_g.sample(batch_size, batch_size, one_hot=False, content_iter=batch)
        _title_trigger = torch.cat((init_tensor.unsqueeze(0).repeat(batch_size,1), _title), 1)
        _title_trigger = _title_trigger[:,:cfg.max_seq_len]
        comment = F.one_hot(_title_trigger, cfg.vocab_size).float()
        label = self.get_label_for_attack(batch_size, self.attack_mode)
        if next(self.clf.parameters()).is_cuda:
            content = content.cuda()
            comment = comment.cuda() # batch_size * 1 * 26 * 10013
            label = label.cuda()
        content = Variable(content, requires_grad=True)
        comment = Variable(comment.unsqueeze(1), requires_grad=True)
        pred = self.clf.forward(content, comment, None, use_dropout=False)
        loss = self.cross_entropy_loss(pred, label) # batch_size
        return loss


    def forward_get_grads_last(self, batch, init_tensor):
        self.extracted_grads = []
        optimizer = optim.Adam(self.clf.parameters())
        optimizer.zero_grad()
        loss = self.evaluate_batch(batch, init_tensor)
        loss.backward(retain_graph=False)
        grads = self.extracted_grads[0].cpu() # 127 * 26 * 16
        grads = torch.sum(grads, dim=0)
        grads = grads[:self.trigger_length] 
        return grads, loss


    def hotflip_attack(self, averaged_grad, embedding_matrix, trigger_token_ids, topic,
                   increase_loss=False, num_candidates=1):
        ### https://github.com/Eric-Wallace/universal-triggers/blob/ed657674862c965b31e0728d71765d0b6fe18f22/attacks.py#L8
        averaged_grad = averaged_grad.cpu()
        embedding_matrix = embedding_matrix.cpu()

        all_word = self.topic_model.get_top_words(topic, self.n_neighbor) ### topic filter
        all_word_idx = np.array([int(self.word2idx[word]) for word in all_word if word in self.word2idx]) ### topic filter
        embedding_matrix = embedding_matrix[all_word_idx,:] ### topic filter

        averaged_grad = averaged_grad.unsqueeze(0) # 1x3x16
        gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",(averaged_grad, embedding_matrix))        
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.
        if num_candidates > 1: # get top k options
            _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2) # 1 x 3 x 10013
            best_k_ids = best_k_ids.detach().cpu().numpy()[0] # 3*num_candidates
            best_k_ids = all_word_idx[best_k_ids]  ### topic filter
            return best_k_ids
        _, best_at_each_step = gradient_dot_embedding_matrix.max(2) # 1 x 3 x 1
        best_at_each_step = best_at_each_step[0].detach().cpu().numpy()
        best_at_each_step = all_word_idx[best_at_each_step] ### topic filter
        return best_at_each_step


    def get_best_candidates(self, batch, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
        ### https://github.com/Eric-Wallace/universal-triggers/blob/ed657674862c965b31e0728d71765d0b6fe18f22/attacks.py#L8
        loss_per_candidate = self.get_loss_per_candidate(0, batch, trigger_token_ids, cand_trigger_token_ids)
        # maximize the loss
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
        for idx in range(1, len(trigger_token_ids)): # for all trigger tokens, skipping the 0th (we did it above)
            loss_per_candidate = []
            for cand, _ in top_candidates: # for all the beams, try all the candidates at idx
                loss_per_candidate.extend(self.get_loss_per_candidate(idx, batch, cand, cand_trigger_token_ids))
            top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
        return max(top_candidates, key=itemgetter(1))[0]


    def get_loss_per_candidate(self, index, batch, trigger_token_ids, cand_trigger_token_ids):
        ## https://github.com/Eric-Wallace/universal-triggers/blob/ed657674862c965b31e0728d71765d0b6fe18f22/utils.py#L67
        if isinstance(cand_trigger_token_ids[0], (np.int64, int)):
            print("Only 1 candidate for index detected, not searching")
            return trigger_token_ids
        loss_per_candidate = []
        curr_loss = self.evaluate_batch(batch, trigger_token_ids)
        curr_loss = curr_loss.cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
        for cand_id in range(len(cand_trigger_token_ids[0])):
            trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
            trigger_token_ids_one_replaced[index] = torch.tensor(cand_trigger_token_ids[index][cand_id]).long() # replace one token
            loss = self.evaluate_batch(batch, trigger_token_ids_one_replaced).cpu().detach().numpy()
            loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
        return loss_per_candidate

    def translate_prefix(self, init_tensor):
        print("index prefix:", init_tensor) # 3 3 3
        print("string prefix", [self.idx2word[str(a)] for a in init_tensor.cpu().numpy()])

    def train(self, save_path="{}_UniTrigger_Whitebox-{}_CopyCat-{}_Neighbor-{}"):
        print("[UniTrigger] Training...")
        for topic in tqdm(range(self.topic_model.get_num_topic())):
            if topic not in self.init_tensor:
                print("Topic", topic)
                init_tokens = ["the"]*self.trigger_length #(1*trigger_length)
                init_tensor = torch.tensor([int(self.word2idx[word]) for word in init_tokens])
                batch_size = cfg.batch_size

                self.translate_prefix(init_tensor)
                for _, batch in tqdm(enumerate(self.train_content_iter.loader)):
                    grads, loss_before = self.forward_get_grads_last(batch, init_tensor)
                    candidates = self.hotflip_attack(grads, self.embedding_matrix, 
                                        init_tensor, topic=topic, increase_loss=True, num_candidates=3) # trigger_length * num_candidates
                    init_tensor = self.get_best_candidates(batch, init_tensor, candidates, beam_size=self.beam_size)
                self.translate_prefix(init_tensor)
                self.init_tensor[topic] = init_tensor
                pickle.dump(self.init_tensor, open(save_path.format(cfg.dataset, self.whitebox, self.random_g.num_article, self.n_neighbor), 'wb'))

    def sample(self, num_samples, batch_size, one_hot=True, content_iter=None, use_temperature=False):
        if len(self.init_tensor) == 0:
            self.train()

        if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
            batch = content_iter.random_batch(batch_size)
        else:
            batch = content_iter

        gen_samples = []
        init_tensor = []
        topics = [self.topic_model.get_topic_from_id(_id) for _id in batch['id']]
        init_tensor = torch.cat([self.init_tensor[topic].unsqueeze(0) for topic in topics], 0)
        # _title = self.random_g.sample(batch_size, batch_size, one_hot=False, content_iter=batch)
        _title = batch['title']
        # _title_trigger = torch.cat((self.init_tensor[topic].unsqueeze(0).repeat(batch_size,1), _title), 1)
        _title_trigger = torch.cat((init_tensor, _title), 1)
        gen_samples = _title_trigger[:,:cfg.max_seq_len]
        
        if one_hot:
            gen_samples = F.one_hot(gen_samples, cfg.vocab_size).float()

        return gen_samples


        


