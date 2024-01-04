import config as cfg
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from copy import deepcopy
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy import linalg as LA
from operator import itemgetter
from scipy.spatial.distance import cdist, cosine
from torch.autograd import Variable
from torch.autograd import grad
from tqdm import tqdm
from utils.helpers import truncated_normal_
from utils.text_process import build_embedding_matrix
from utils.text_process import tensor_to_tokens


class BugGenerator:
    def __init__(self, idx2word, word2idx, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.notfound = None
        self.array_prox = {}
        self.array_prox['a'] = ['q', 'w', 'z', 'x'];
        self.array_prox['b'] = ['v', 'f', 'g', 'h', 'n'];
        self.array_prox['c'] = ['x', 's', 'd', 'f', 'v'];
        self.array_prox['d'] = ['x', 's', 'w', 'e', 'r', 'f', 'v', 'c'];
        self.array_prox['e'] = ['w', 's', 'd', 'f', 'r'];
        self.array_prox['f'] = ['c', 'd', 'e', 'r', 't', 'g', 'b', 'v'];
        self.array_prox['g'] = ['r', 'f', 'v', 't', 'b', 'y', 'h', 'n'];
        self.array_prox['h'] = ['b', 'g', 't', 'y', 'u', 'j', 'm', 'n'];
        self.array_prox['i'] = ['u', 'j', 'k', 'l', 'o'];
        self.array_prox['j'] = ['n', 'h', 'y', 'u', 'i', 'k', 'm'];
        self.array_prox['k'] = ['u', 'j', 'm', 'l', 'o'];
        self.array_prox['l'] = ['p', 'o', 'i', 'k', 'm'];
        self.array_prox['m'] = ['n', 'h', 'j', 'k', 'l'];
        self.array_prox['n'] = ['b', 'g', 'h', 'j', 'm'];
        self.array_prox['o'] = ['i', 'k', 'l', 'p'];
        self.array_prox['p'] = ['o', 'l'];
        self.array_prox['r'] = ['e', 'd', 'f', 'g', 't'];
        self.array_prox['s'] = ['q', 'w', 'e', 'z', 'x', 'c'];
        self.array_prox['t'] = ['r', 'f', 'g', 'h', 'y'];
        self.array_prox['u'] = ['y', 'h', 'j', 'k', 'i'];
        self.array_prox['v'] = ['', 'c', 'd', 'f', 'g', 'b'];    
        self.array_prox['w'] = ['q', 'a', 's', 'd', 'e'];
        self.array_prox['x'] = ['z', 'a', 's', 'd', 'c'];
        self.array_prox['y'] = ['t', 'g', 'h', 'j', 'u'];
        self.array_prox['z'] = ['x', 's', 'a'];
        self.array_prox['1'] = ['q', 'w'];
        self.array_prox['2'] = ['q', 'w', 'e'];
        self.array_prox['3'] = ['w', 'e', 'r'];
        self.array_prox['4'] = ['e', 'r', 't'];
        self.array_prox['5'] = ['r', 't', 'y'];
        self.array_prox['6'] = ['t', 'y', 'u'];
        self.array_prox['7'] = ['y', 'u', 'i'];
        self.array_prox['8'] = ['u', 'i', 'o'];
        self.array_prox['9'] = ['i', 'o', 'p'];
        self.array_prox['0'] = ['o', 'p'];
    
    def insert(self, w):
        if len(w) >= 6 or len(w)<2:
            return self.notfound
        i = np.random.choice(list(range(1,len(w))))
        w = "{} {}".format(w[:i], w[i:])
        return w.split()
    
    def delete(self, w):
        if len(w) <= 2:
            return self.notfound
        i = np.random.choice(list(range(1,len(w)-1)))
        w = "{}{}".format(w[:i], w[i+1:])
        return [w]
    
    def swap(self, w):
        if len(w) <= 4:
            return self.notfound
        i = np.random.choice(list(range(2,len(w)-1)))
        w = "{}{}{}{}".format(w[:i-1],w[i],w[i-1],w[i+1:])
        return [w]
    
    def subc(self, w):
        from_c = ['o','O','l','1','a','@','m','n']
        to_c = ['O','o','1','l','@','a','n','m']
        i = 0
        rt = ""
        for i in range(len(w)):
            if w[i] in from_c:
                idx = from_c.index(w[i])
                rt += to_c[idx]
            else:
                rt += w[i]
        return [rt]
    
    def subw(self, w, k=5):
        word_emb = self.embedding_matrix[int(self.word2idx[w])]
        sims = np.dot(word_emb, self.embedding_matrix.T)
        words_idx = np.argsort(sims)[-k:]
        words = [[self.idx2word[str(i)]] for i in words_idx]
        return words

    def add_to(self, a, rt):
        if a != None:
            rt.append(a) 
    
    def mispell(self, w):
        rt = ""
        for i in range(len(w)):
            if w[i] in self.array_prox and np.random.choice([0,1], p=[0.95,0.05])==1:
                rt += np.random.choice(self.array_prox[w[i]])
            else:
                rt += w[i]
        if rt == w:
            return self.notfound
        return [rt]

    def generate_all_bugs(self, w):
        rt = []
        self.add_to(self.insert(w), rt)
        self.add_to(self.delete(w), rt)
        self.add_to(self.swap(w), rt)
        self.add_to(self.subc(w), rt)
        self.add_to(self.mispell(w), rt)
        rt.extend(self.subw(w))
        return rt


class TextBugger(nn.Module):
    def __init__(self, random_g, idx2word, word2idx, attack_mode="target_real", if_load_from_file=True):
        super(TextBugger, self).__init__()
        self.name = 'textbugger'
        self.embedding_matrix = build_embedding_matrix(cfg.dataset, 200) # vocab_size * embedding_size
        self.attack_mode=attack_mode
        self.random_g = random_g
        self.memory = {}
        self.idx2word = deepcopy(idx2word)
        self.word2idx =word2idx
        self.attacks = ['real', 'fake']
        for attack in self.attacks:
            self.memory[attack] = {}
        self.bug_generator = BugGenerator(self.idx2word, self.word2idx, self.embedding_matrix)
        # self.return_text_func = return_text_func

        self.extracted_grads = []
        def extract_grad_hook(module, grad_in, grad_out):
            self.extracted_grads.append(grad_out[0])

        if cfg.blackbox_using_path:
            print("(blackbox TextBugger) loading whitebox classifier at {}".format(cfg.blackbox_using_path))
            if 'RNN2' in cfg.blackbox_using_path:
                from models.RelGAN_F_RNN2 import RelGAN_F_RNN2
                self.clf = RelGAN_F_RNN2(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, extract_grad_hook=extract_grad_hook,
                                                                                                hidden_dim=cfg.rnn_hidden_dim,
                                                                                                dropout=cfg.f_dropout_comment,
                                                                                                dropout_content=cfg.f_dropout_content,
                                                                                                gpu=cfg.CUDA)
                self.clf.load_state_dict(torch.load(cfg.blackbox_using_path))
                self.whitebox = "RNN2"
                
            elif 'CNN' in cfg.blackbox_using_path:
                from models.RelGAN_F_CNN import RelGAN_F_CNN
                self.F_clf = RelGAN_F_CNN(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, dropout=cfg.f_dropout_comment, 
                                                                                            dropout_content=cfg.f_dropout_content, 
                                                                                            gpu=cfg.CUDA)
                self.clf.load_state_dict(torch.load(cfg.blackbox_using_path))
                self.whitebox = "CNN"
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
        self.bug_word2idx = {}
        self.bug_idx2word = {}
        self.save_path = "{}TextBugger_Whitebox-{}_CopyCat-{}_EvalNum-{}".format(cfg.dataset, self.whitebox, self.random_g.num_article, cfg.eval_num)
        if if_load_from_file:
            self.load_best_match()

    def count_comments(self):
        count = 0
        for k in self.memory:
            count += len(self.memory[k])
        return count


    def get_label_for_attack(self, batch_size, attack_mode):
        if attack_mode == "target_fake":
            label = torch.ones(batch_size).long()
        elif attack_mode == "target_real":
            label = torch.zeros(batch_size).long()
        return label


    def evaluate_batch(self, content, comment, loss_only=False):
        batch_size = len(content)
        label = self.get_label_for_attack(batch_size, self.attack_mode)
        if next(self.clf.parameters()).is_cuda:
            content = content.cuda()
            comment = comment.cuda() # batch_size * 1 * 26 * 10013
            label = label.cuda()
        content = Variable(content, requires_grad=True)
        comment = Variable(comment.unsqueeze(1), requires_grad=True)
        pred = self.clf.forward(content, comment, None, use_dropout=False)
        loss = self.cross_entropy_loss(pred, label) # batch_size
        if loss_only:
            return pred.argmax(dim=-1), loss
        grads = []
        for i in range(len(loss)):
            loss[i].backward(retain_graph=True)
            invalid = comment[i,0,:,:].view(cfg.max_seq_len, cfg.vocab_size)
            grads.append(comment.grad.data[i]*invalid) # 26 * 10013
        grads = torch.cat(grads, 0)
        grads = torch.sum(grads, 2) # batch_size * max_seq_len
        grads = grads.data.cpu().numpy()

        return grads, loss


    def filter_grads(self, grads, comment):
        filters_token = [0, 2]
        for token in filters_token:
            idx = np.where(comment==token)
            grads[idx] = -np.inf
        return grads


    def token_to_tensor(self, tokens):
        rt = []
        for word in tokens:
            if word in self.word2idx:
                rt.append(int(self.word2idx[word]))
            else:
                key_i = len(self.word2idx)+len(self.bug_word2idx)
                self.idx2word[str(key_i)] = word
                self.bug_word2idx[word] = str(key_i)
                self.bug_idx2word[str(key_i)] = word
                rt.append(cfg.unk_idx)
        return torch.tensor(rt)


    def inject_bug(self, tensor, at_idx, bugs):
        bug_idx = self.token_to_tensor(bugs)
        tensor = torch.cat((tensor[:at_idx], bug_idx, tensor[at_idx+1:]))
        tensor = tensor[:cfg.max_seq_len]
        return tensor


    def bugger(self, loss_before, content, tensor, seq_idx, grad):
        # import time
        # time0 = time.time()
        # given a sentence, try all the bugs at all the positions
        tokens = [self.idx2word[str(i)] for i in tensor.cpu().numpy()]
        best_tensor = tensor
        # print("\ncurrent comment", tensor)
        new_tensors = []
        replace_idx = []
        all_bugs = []
        for i in seq_idx: # at this position
            best_bug_idx = None
            if grad[i] != -np.inf: #if valid token
                bugs = self.bug_generator.generate_all_bugs(tokens[i])
                for bug in bugs:
                    t = self.inject_bug(tensor, i, bug)
                    new_tensors.append(t.unsqueeze(0))
                    replace_idx.append(i)
                    all_bugs.append(bug)

        _new_tensors = torch.cat(new_tensors, 0)
        new_tensors = F.one_hot(_new_tensors, cfg.vocab_size).float()
        # print("new tensors", new_tensors.size()) new tensors torch.Size([8, 26, 10013])
        # print("content", content.size())
        # print("new_tensors", new_tensors.size())
        preds, loss_after = self.evaluate_batch(content.repeat(len(new_tensors), 1), new_tensors, loss_only=True)
        loss_diff = loss_before - loss_after
        best_bug_idx = np.argmax(loss_diff.detach().cpu().numpy())
        best_tensor = _new_tensors[best_bug_idx]

        # print("original text:", " ".join(tokens))
        # print("bugger: {} -> {}".format(tokens[replace_idx[best_bug_idx]], " ".join(all_bugs[best_bug_idx])))
        new_tokens = deepcopy(tokens)
        new_tokens[replace_idx[best_bug_idx]] = " ".join(all_bugs[best_bug_idx])
        best_text = " ".join(new_tokens).replace(cfg.padding_token,'')
        # print("replaced text:", " ".join(new_tokens))
        # print()
        # print(time.time()-time0)
        # print("current best_tensor", best_tensor)
        # print()
        return best_tensor, best_text


    def slice_data (self, data, i):
        tmp = {}
        for k in data:
            tmp[k] = data[k][i:i+1]
        return tmp


    def load_best_match(self):
        try:
            self.memory, self.bug_word2idx, self.bug_idx2word = pickle.load(open(self.save_path, 'rb'))
            for idx in self.bug_idx2word:
                self.idx2word[idx] = self.bug_idx2word[idx]
            print("loading memory for TextBugger with {} articles and {} comments".format(len(self.memory), self.count_comments()))
            print("from ", self.save_path)
        except Exception as e:
            print("Error TextBugger loading memory...", e)


    def learn_best_match(self, test_content_iter, only_unknown=False):
        print("[TextBugger] Matching test set...")
        bar = tqdm(enumerate(test_content_iter.loader))
        for _, data in bar:
            batch_size = len(data['content'])
            gen_samples = None
            if not only_unknown:
                gen_samples, texts = self.sample_(batch_size, batch_size, one_hot=False, content_iter=data)
            for i in range(batch_size):
                _id = data['id'][i]
                if _id not in self.memory:
                    self.memory[_id] = []
                    if only_unknown:
                        gen_samples, texts = self.sample_(batch_size, batch_size, one_hot=False, content_iter=self.slice_data(data, i))
                        self.memory[_id].append([gen_samples[0], texts[0]])
                if not only_unknown:
                    self.memory[_id].append([gen_samples[i], texts[i]])
            bar.set_description("Total {} articles and {} comments".format(len(self.memory), self.count_comments()))
        pickle.dump([self.memory, self.bug_word2idx, self.bug_idx2word], open(self.save_path, 'wb'))
        print("SAVED Total {} articles and {} comments".format(len(self.memory), self.count_comments(), cfg.eval_num))


    def sample(self, num_samples, batch_size, one_hot=True, content_iter=None, use_temperature=False, return_text=False):
        if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
            batch = content_iter.random_batch(batch_size)
        else:
            batch = content_iter

        save_samples = []
        save_texts = []

        ids = batch['id']
        for _id in ids:
            if _id in self.memory:
                idx = np.random.choice(len(self.memory[_id]))
                tmp, text = self.memory[_id][idx]
                save_samples.append(tmp.unsqueeze(0))
                save_texts.append(text.strip())
            else:
                print("TextBugger cannot find {}".format(_id))
        save_samples = torch.cat(save_samples, 0)
        if one_hot:
            save_samples = F.one_hot(tmp, cfg.vocab_size).float()

        if return_text:
            return save_samples, save_texts
        return save_samples

    def sample_(self, num_samples, batch_size, one_hot=True, content_iter=None, use_temperature=False): #default sample comments from fake articles
        if self.attack_mode == "target_fake":
            target_label = 'fake'
        else:
            target_label = 'real'

        samples = []
        texts = []
        while len(samples) < num_samples:
            if 'F_DataIterSep' in str(type(content_iter)) or 'GenDataIterContent' in str(type(content_iter)):
                batch = content_iter.random_batch(batch_size)
            else:
                batch = content_iter

            content = batch['content'] # batch_size * 512
            batch_size = len(content)
            # print("content", content.size())
            # _comment = batch['title'] # batch_size * max_seq_len
            _comment = self.random_g.sample(batch_size, batch_size, one_hot=False, content_iter=batch)
            comment = F.one_hot(_comment, cfg.vocab_size).float()
            grads, loss_before = self.evaluate_batch(content, comment) # batch_size * max_seq_len
            grads = self.filter_grads(grads, _comment) 
            at_replace_idx = np.argsort(grads, 1) # batch_size * max_seq_len

            for i in range(batch_size):
                # print("content[i]", content[i].size())
                # print("title:", batch['title_str'][i])
                # print("Original Comment:", tensor_to_tokens(_comment[i].unsqueeze(0), self.idx2word, external_dict=self.bug_idx2word))
                sample, text = self.bugger(loss_before[i], content[i], _comment[i], at_replace_idx[i], grads[i])
                # print("Changed Comment:", tensor_to_tokens(sample.unsqueeze(0), self.idx2word, external_dict=self.bug_idx2word))
                samples.append(sample.unsqueeze(0))
                texts.append(text)

        samples = torch.cat(samples, 0)[:num_samples]
        if one_hot:
            samples = F.one_hot(samples, cfg.vocab_size).float()

        return samples, texts

