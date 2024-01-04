# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from utils.utils import MMD_loss
import torch.nn.functional as F
from metrics.bleu import BLEU
from metrics.clas_acc import ACC
from metrics.nll import NLL
from metrics.ppl import PPL
from utils.cat_data_loader import CatClasDataIter
from utils.data_loader import GenDataIter, GenDataIterContent
from utils.helpers import Signal, create_logger, get_fixed_temperature
from utils.text_process import load_dict, write_tokens, tensor_to_tokens


class BasicInstructor:
    def __init__(self, opt):
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=cfg.log_filename if cfg.if_test
                                 else [cfg.log_filename, cfg.save_root + 'log.txt'])
        self.sig = Signal(cfg.signal_file)
        self.opt = opt
        # self.show_config()

        self.clas = None

        # load dictionary
        self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset, cfg.vocab_size)
        self.word2idx_dict_all, self.idx2word_dict_all = load_dict(cfg.dataset, -1)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.mse_criterion = nn.MSELoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        self.clas_criterion = nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()
        self.mmd = MMD_loss(cfg.mmd_mean, cfg.mmd_kernel_num)

        # Optimizer
        self.clas_opt = None

        # Metrics
        self.bleu = BLEU('BLEU', gram=[2], if_use=cfg.use_bleu)
        self.nll_gen = NLL('NLL_gen', if_use=cfg.use_nll_gen, gpu=cfg.CUDA)
        self.all_metrics = [self.bleu, self.nll_gen]
        

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def train_gen_epoch(self, model, data_loader, criterion, optimizer, content_iter=None):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target, content = data['input'], data['target'], data['content']
            if cfg.CUDA:
                inp, target, content = inp.cuda(), target.cuda(), content.cuda()
            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden, content_iter=data)
            loss = criterion(pred, target.view(-1))
            self.optimize(optimizer, loss, model)
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def train_dis_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()

            pred = model.forward(inp)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss, model)

            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
            total_num += inp.size(0)

        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc


    def train_classifier(self, epochs):
        """
        Classifier for calculating the classification accuracy metric of category text generation.

        Note: the train and test data for the classifier is opposite to the generator.
        Because the classifier is to calculate the classification accuracy of the generated samples
        where are trained on self.train_samples_list.

        Since there's no test data in synthetic data (oracle data), the synthetic data experiments
        doesn't need a classifier.
        """
        import copy

        # Prepare data for Classifier
        clas_data = CatClasDataIter(self.clas_samples_list)
        eval_clas_data = CatClasDataIter(self.train_samples_list)

        max_acc = 0
        best_clas = None
        for epoch in range(epochs):
            c_loss, c_acc = self.train_dis_epoch(self.clas, clas_data.loader, self.clas_criterion,
                                                 self.clas_opt)
            _, eval_acc = self.eval_dis(self.clas, eval_clas_data.loader, self.clas_criterion)
            if eval_acc > max_acc:
                best_clas = copy.deepcopy(self.clas.state_dict())  # save the best classifier
                max_acc = eval_acc
            self.log.info('[PRE-CLAS] epoch %d: c_loss = %.4f, c_acc = %.4f, eval_acc = %.4f, max_eval_acc = %.4f',
                          epoch, c_loss, c_acc, eval_acc, max_acc)
        self.clas.load_state_dict(copy.deepcopy(best_clas))  # Reload the best classifier

    @staticmethod
    def eval_dis(model, data_loader, criterion):
        total_loss = 0
        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                pred = model.forward(inp)
                loss = criterion(pred, target)
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                total_num += inp.size(0)
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc

    @staticmethod
    def optimize_multi(opts, losses):
        for i, (opt, loss) in enumerate(zip(opts, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=True if i < len(opts) - 1 else False)
            opt.step()

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()

    def show_config(self):
        self.log.info(100 * '=')
        self.log.info('> training arguments:')
        for arg in vars(self.opt):
            self.log.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        self.log.info(100 * '=')

    def cal_metrics(self, train_data, test_data, custom_gen=None):
        """
        Calculate metrics
        :param fmt_str: if return format string for logging
        """
        if custom_gen:
            gen = custom_gen
        else:
            gen = self.gen

        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = gen.sample(cfg.samples_num2, cfg.batch_size*4, content_iter=test_data, one_hot=False)

            # eval_samples, _ = self.generate_best_topic(cfg.samples_num2, cfg.batch_size, 
            #                                                 data_iter=self.test_data, translate=False, eval_num=5)
            # print(gen_tokens)
            # gen_data = GenDataIter(eval_samples, cfg.batch_size)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            # gen_tokens_s = tensor_to_tokens(gen.sample(200, 200), self.idx2word_dict)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=test_data.tokens)
            self.nll_gen.reset(gen, train_data.get_loader(cfg.batch_size))
            # self.nll_div.reset(gen, gen_data.loader)
            # self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            # self.ppl.reset(gen_tokens)

            # self.all_metrics = [self.bleu, self.nll_gen, self.nll_div, self.self_bleu, self.ppl]
            self.all_metrics = [self.bleu, self.nll_gen]


        return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics]), [metric.get_score() for metric in self.all_metrics]


    def _save(self, phase, epoch, content_iter=None):
        """Save model state dict and generator's samples"""
        gen_path = cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch)
        dis_path = cfg.save_model_root + 'dis_{}_{:05d}.pt'.format(phase, epoch)
        gen_optim_path = cfg.save_model_root + 'gen_{}-OPTIMIZER_{:05d}.pt'.format(phase, epoch)
        dis_optim_path = cfg.save_model_root + 'dis_{}-OPTIMIZER_{:05d}.pt'.format(phase, epoch)
        if 'ADV' in phase:
            torch.save(self.gen.state_dict(), gen_path)
            # torch.save(self.dis.state_dict(), dis_path)
            # torch.save(self.gen_adv_opt.state_dict(), gen_optim_path)
            # torch.save(self.dis_opt.state_dict(), dis_optim_path)
        if phase == "MLE":
            torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
            print("Saved MLE model to {}".format(cfg.pretrained_gen_path))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phase, epoch)
        # samples = self.gen.sample(cfg.batch_size, cfg.batch_size, content_iter=content_iter)
        # write_tokens(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))
        return gen_path, dis_path, save_sample_path

    def update_temperature(self, i, N):
        self.gen.temperature.data = torch.Tensor([get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)])
        if cfg.CUDA:
            self.gen.temperature.data = self.gen.temperature.data.cuda()
