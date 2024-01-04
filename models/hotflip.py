import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
import config as cfg
from utils.text_process import build_embedding_matrix, tensor_to_tokens
from numpy import linalg as LA
import nltk
from nltk.corpus import stopwords
import string
from utils.tweet_utils import *
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import pickle
import os
from tqdm import tqdm


class HotFlip(nn.Module):
    def __init__(self, clf, idx2word, word2idx, random_g, n_neighbor=20, attack_mode="target_real", threshold=0.8):
        super(HotFlip, self).__init__()
        self.name = 'hotflip'
        
        if cfg.blackbox_using_path:
            print("(blackbox Hotflip) loading whitebox classifier at {}".format(cfg.blackbox_using_path))
            from models.RelGAN_F_RNN2 import RelGAN_F_RNN2
            self.clf = RelGAN_F_RNN2(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, hidden_dim=cfg.rnn_hidden_dim,
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
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.attack_mode=attack_mode
        if self.attack_mode == "target_fake":
            self.target_label = 1
        else:
            self.target_label = 0 
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.n_neighbor = n_neighbor
        self.embedding_matrix = build_embedding_matrix(cfg.dataset, 200) # vocab_size * embedding_size
        self.random_g = random_g
        self.threshold = threshold
        self.stopwords = list(stopwords.words('english'))
        for c in string.punctuation:
            tokens = tweet_ark_tokenize(c.lower())
            self.stopwords.extend(c)
        self.stopwords.extend(['’', '”'])
        self.stopwords_idx = [int(self.word2idx[word]) for word in self.stopwords if word in self.word2idx]
        self.lemmatizer = WordNetLemmatizer()
        self.memory = {}
        if not cfg.if_hotflip_retrain:
            self.load_best_match()


    def find_nearest_words(self, word, idx, at_replace_idx=None, sentences=None):
        """
        contraints:
        (i) cosine similarity between embeddings of two words is larger than 0.8 threshold
        (ii) two words have the same part-of-speech (in a sentence)
        (iii) disallow replacing stop-words or replacing with word having the same lexeme
        """
        word_emb = self.embedding_matrix[idx]
        sims = np.dot(word_emb, self.embedding_matrix.T) #print(sims.shape) 4 * 10013
        best_idx = np.argsort(sims,1)

        rt = []
        for i in range(len(idx)):
            tmp = []
            removed = []
            _best_idx = best_idx[i,:][::-1]
            for j in _best_idx:
                sentence = sentences[i]
                replace_idx = at_replace_idx[i]
                sentence_new = sentence.copy()
                sentence_new[at_replace_idx[i]] = self.idx2word[str(j)]
                tags = pos_tag(sentence)
                tags_new = pos_tag(sentence_new)
                if_same_pos = tags[replace_idx][1] == tags_new[replace_idx][1]
                if_same_lemex = self.lemmatizer.lemmatize(word[i]) == self.lemmatizer.lemmatize(self.idx2word[str(j)])
                if sims[i,j] >= self.threshold and j not in self.stopwords_idx and not if_same_lemex and if_same_pos:
                    tmp.append(j)
                elif sims[i,j] >= self.threshold and j not in self.stopwords_idx:
                    removed.append(j)
                elif sims[i,j] < self.threshold:
                    break
                if len(tmp) >= self.n_neighbor:
                    break

            while len(tmp) < self.n_neighbor:
                tmp.append(tmp[-1]) if len(tmp) > 0 else tmp.append(removed[0])
                    
            rt.append(tmp)

        rt = np.array(rt) 
        # best_idx_sims = sims[threshold_idx]
        # best_idx = best_idx[:,-self.n_neighbor:]
        return rt

    def prepare_data_clf(self, data):
        inp, comments, label = data['content'], data['comments'], data['label']
        num_comments = data['num_comment']
        new_comments = []
        for comment in comments:
            new_comments.append(F.one_hot(comment, cfg.vocab_size).float())
        comments = torch.stack(new_comments)
        return inp, comments, label, num_comments

    def get_label_for_attack(self, batch_size, attack_mode):
        if attack_mode == "target_fake":
            label = torch.ones(batch_size).long()
        elif attack_mode == "target_real":
            label = torch.zeros(batch_size).long()
        return label
        # if we attack target_real, we want to replace word that is important to fake

    def forward_get_grads_last(self, content, comment, label):
        if next(self.clf.parameters()).is_cuda:
            content = content.cuda()
            comment = comment.cuda() # batch_size * 1 * 26 * 10013
            label = label.cuda()
        content = Variable(content, requires_grad=True)
        comment = Variable(comment.unsqueeze(1), requires_grad=True)
        pred = self.clf.forward(content, comment, None, use_dropout=False)
        loss = self.cross_entropy_loss(pred, label) # batch_size
        grads = []
        for i in range(len(loss)):
            loss[i].backward(retain_graph=True)
            invalid = comment[i,0,:,:].view(cfg.max_seq_len, cfg.vocab_size)
            grads.append(comment.grad.data[i]*invalid) # 26 * 10013
        grads = torch.cat(grads, 0)
        grads = torch.sum(grads, 2) # batch_size * max_seq_len
        grads = grads.data.cpu().numpy()

        return grads, loss

    def augument_comment(self, content, comment, at_replace_idx, to_replace_idx):
        rt = []
        batch_size = len(comment)
        content = torch.repeat_interleave(content, self.n_neighbor, 0)
        to_replace_idx = torch.tensor(to_replace_idx).flatten()
        at_replace_idx = torch.repeat_interleave(torch.from_numpy(at_replace_idx), self.n_neighbor, 0)
        comment = torch.repeat_interleave(torch.from_numpy(comment), self.n_neighbor, 0)
        comment[list(range(batch_size*self.n_neighbor)), at_replace_idx] = to_replace_idx

        return content, comment, at_replace_idx


    def filter_grads(self, grads, comment):
        filters_token = [0, 2]
        filters_token.extend([int(self.word2idx[word]) for word in self.stopwords if word in self.word2idx])
        for token in filters_token:
            idx = np.where(comment==token)
            grads[idx] = -np.inf
        return grads


    def load_best_match(self, load_path="{}_HotFlip-{}_CopyCat-{}_EvalNum-{}_Neighbor-{}"):
        try:
            self.memory = pickle.load(open(load_path.format(cfg.dataset, self.whitebox, self.random_g.num_article, cfg.eval_num, self.n_neighbor), 'rb'))
            print("loading memory for HotFlip with {} articles".format(len(self.memory)))
        except Exception as e:
            print("Error HotFlip loading memory...", e)

    def slice_data (self, data, i):
        tmp = {}
        for k in data:
            tmp[k] = data[k][i:i+1]
        return tmp

    def count_comments(self):
        count = 0
        for k in self.memory:
            count += len(self.memory[k])
        return count

    def learn_best_match(self, test_content_iter, save_path="{}_HotFlip-{}_CopyCat-{}_EvalNum-{}_Neighbor-{}", only_unknown=False):
        print("[HotFlip] Matching test set...")
        bar = tqdm(enumerate(test_content_iter.loader))
        for _, data in bar:
            batch_size = len(data['content'])
            gen_samples = None
            if not only_unknown:
                gen_samples = self.sample_(batch_size, batch_size, one_hot=False, content_iter=data)
            for i in range(batch_size):
                _id = data['id'][i]
                if _id not in self.memory:
                    self.memory[_id] = []
                    if only_unknown:
                        gen_samples = self.sample_(batch_size, batch_size, one_hot=False, content_iter=self.slice_data(data, i))
                        self.memory[_id].append(gen_samples[0])
                if not only_unknown:
                    self.memory[_id].append(gen_samples[i])
            bar.set_description("Total {} articles and {} comments".format(len(self.memory), self.count_comments()))
        pickle.dump(self.memory, open(save_path.format(cfg.dataset, self.whitebox, self.random_g.num_article, cfg.eval_num, self.n_neighbor), 'wb'))
        print("SAVED Total {} articles and {} comments".format(len(self.memory), self.count_comments(), cfg.eval_num))


    def sample(self, num_samples, batch_size, one_hot=True, content_iter=None, use_temperature=False):
        if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
            batch = content_iter.random_batch(batch_size)
        else:
            batch = content_iter
        save_samples = []
        ids = batch['id']
        for _id in ids:
            if _id in self.memory:
                idx = np.random.choice(len(self.memory[_id]))
                tmp = self.memory[_id][idx]
                if one_hot:
                    tmp = F.one_hot(tmp, cfg.vocab_size).float()
                save_samples.append(tmp.unsqueeze(0))
            else:
                print("HotFlip cannot find {}".format(_id))
        save_samples = torch.cat(save_samples, 0)
        return save_samples


    def sample_(self, num_samples, batch_size, one_hot=True, content_iter=None, use_temperature=False): #default sample comments from fake articles
        num = 0
        samples = []
        if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
            batch = content_iter.random_batch(batch_size)
        else:
            batch = content_iter

        content = batch['content']
        batch_size = len(content)
        _title_str = batch['title_str']

        # _title = batch['title'] # batch_size * max_seq_len
        _title = self.random_g.sample(batch_size, batch_size, one_hot=False, content_iter=batch)

        title = F.one_hot(_title, cfg.vocab_size).float()
        label = self.get_label_for_attack(batch_size, self.attack_mode)
        grads, loss_before = self.forward_get_grads_last(content, title, label) # batch_size * max_seq_len
        grads = self.filter_grads(grads, _title)

        _title = _title.data.numpy() #(4, 26)
        # print(_title, grads)
        at_replace_idx = np.argmax(grads, 1) #[13 13 13  5] (4,)
        tmp_idx = list(range(len(at_replace_idx)))
        grads_at_replace_idx = grads[tmp_idx,at_replace_idx]
        words_idx = _title[tmp_idx,at_replace_idx]
        # print("words_idx", words_idx)
        words = [self.idx2word[str(word)] for word in words_idx]
        to_replace_idx = self.find_nearest_words(words, words_idx, at_replace_idx, tensor_to_tokens(_title, self.idx2word))
        
        new_content, _new_comments, at_replace_idx_augument = self.augument_comment(content, _title, at_replace_idx, to_replace_idx)
        new_comments = F.one_hot(_new_comments, cfg.vocab_size).float()
        label = self.get_label_for_attack(batch_size*self.n_neighbor, self.attack_mode)
        grads, loss_after = self.forward_get_grads_last(new_content, new_comments, label) # (batch_size*n_neighbor) * max_seq_len
       
        # print(loss_before, loss_after)
        loss_before = torch.repeat_interleave(loss_before, self.n_neighbor)
        loss_delta = loss_before - loss_after
        loss_delta = loss_delta.data.cpu().numpy()
        # print(loss_delta.shape)
        best_delta_idx = np.argmax(np.split(loss_delta,batch_size),1)
        best_word_idx = to_replace_idx[list(range(batch_size)), best_delta_idx]

        if not one_hot:
            new_comments = _new_comments

        new_comments = new_comments.view(batch_size, self.n_neighbor, -1)
        new_comments = new_comments[list(range(batch_size)), best_delta_idx, :]
        if not one_hot:
            # print(tensor_to_tokens(new_comments, self.idx2word))
            pass
        else:
            new_comments = new_comments.view(batch_size, cfg.max_seq_len, -1)



        return new_comments

