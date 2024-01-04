import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils.helpers import truncated_normal_
import numpy as np
from adversarialchecker.defenses.scRNN.corrector import ScRNNChecker
from utils.text_process import tensor_to_tokens, tokens_to_tensor
import language_check

class RobustFilter(nn.Module):
    def __init__(self, idx2word, word2idx, topic_models, spelling_thres=1, coherency_thres=0.1, unk_thres=0.3):
        super(RobustFilter, self).__init__()
        self.name = 'robust'
        self.spelling_thres = spelling_thres
        self.coherency_thres = coherency_thres
        self.unk_thres=unk_thres
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.topic_models = topic_models
        self.checker = ScRNNChecker()
        self.checker2 = language_check.LanguageTool('en-US')

    def check_coherency(self, s, _id):
        score = np.mean([model.cal_coherency_from_single_id([s], _id)[0] for model in self.topic_models])
        return score

    def correct_spelling_bk(self, token):
        s = " ".join(token)
        matches = self.checker2.check(s)
        s2 = language_check.correct(s, matches)
        return s2, len(matches)

    def correct_spelling(self, s):
        s2 = self.checker.correct_string(s)

        count = 0
        token = s.split()
        # print(token)
        for i in range(len(token)):
            t = token[i]
            if t not in s2.split():
                count +=1
        # print("{}\n=>{}\n".format(s, s2))
        # tensor = tokens_to_tensor(s.split(), self.word2idx)
        return s2, count

    def filter(self, s, t, _id):
        coherency_title = self.check_coherency(t, _id)
        coherency_comment = self.check_coherency(s, _id)
        if_pass_coherency = coherency_comment >= (coherency_title-self.coherency_thres)
        # print(s, coherency_comment)
        # print(t, coherency_title)
        # print("=>{}\n".format(if_pass_coherency))
        
        s, count = self.correct_spelling(s)
        if_pass_mispelling = count <= self.spelling_thres
        return s, int(if_pass_coherency), int(if_pass_mispelling)

    
