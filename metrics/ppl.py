# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : ppl.py
# @Time         : Created at 2019/12/5
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import math
import string

import numpy as np
import os
import random

import config as cfg
from metrics.basic import Metrics
from utils.text_process import write_tokens

kenlm_path = '/home/zhiwei/kenlm'  # specify the kenlm path


class PPL(Metrics):
    def __init__(self, train_data, test_data, n_gram=5, if_use=False):
        super(PPL, self).__init__('[PPL_forward, PPL_reverse]')

        self.n_gram = n_gram
        self.if_use = if_use

        self.gen_tokens = None
        self.train_data = train_data
        self.test_data = test_data

    def get_score(self):
        if not self.if_use:
            return 0
        return self.cal_ppl()

    def reset(self, gen_tokens=None):
        self.gen_tokens = gen_tokens

    def cal_ppl(self):
        save_path = os.path.join("/tmp", ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(6)))

        write_tokens(save_path, self.gen_tokens)  # save to file

        # forward ppl
        for_lm = self.train_ngram_lm(kenlm_path=kenlm_path, data_path=cfg.train_data,
                                     output_path=save_path, n_gram=self.n_gram)
        for_ppl = self.get_ppl(for_lm, self.gen_tokens)

        # reverse ppl
        try:
            rev_lm = self.train_ngram_lm(kenlm_path=kenlm_path, data_path=save_path,
                                         output_path=save_path, n_gram=self.n_gram)

            rev_ppl = self.get_ppl(rev_lm, self.test_data.tokens)
        except:
            # Note: Only after the generator is trained few epochs, the reverse ppl can be calculated.
            print("reverse ppl error: it maybe the generated files aren't valid to obtain an LM")
            rev_ppl = 0

        return [for_ppl, rev_ppl]

    def train_ngram_lm(self, kenlm_path, data_path, output_path, n_gram):
        """
        Trains a modified Kneser-Ney n-gram KenLM from a text file.
        Creates a .arpa file to store n-grams.
        """
        import kenlm
        import subprocess

        # create .arpa and .bin file of n-grams
        curdir = os.path.abspath(os.path.curdir)
        cd_command = "cd " + os.path.join(kenlm_path, 'build')
        command_1 = "bin/lmplz -o {} <{} >{} &".format(str(n_gram), os.path.join(curdir, data_path),
                                                       output_path + ".arpa")
        command_2 = "bin/build_binary -s {} {} &".format(output_path + ".arpa", output_path + ".bin")

        subprocess.getstatusoutput(cd_command + " && " + command_1)  # call without logging output
        subprocess.getstatusoutput(cd_command + " && " + command_2)  # call without logging output

        # create language model
        assert output_path + ".bin"  # captured by try..except block outside
        model = kenlm.Model(output_path + ".bin")

        return model

    def get_ppl(self, lm, tokens):
        """
        Assume sentences is a list of strings (space delimited sentences)
        """
        total_nll = 0
        total_wc = 0
        for words in tokens:
            nll = np.sum([-math.log(math.pow(10.0, score))
                          for score, _, _ in lm.full_scores(' '.join(words), bos=True, eos=False)])
            total_wc += len(words)
            total_nll += nll
        ppl = np.exp(total_nll / total_wc)
        return round(ppl, 4)
