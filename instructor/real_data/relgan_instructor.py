# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : relgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import copy
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import config as cfg
from sklearn.metrics import f1_score

from instructor.real_data.instructor import BasicInstructor
from metrics.bleu import BLEU
from metrics.nll import NLL
from models.RelGAN_D import RelGAN_D
from models.RelGAN_D_TOPIC import RelGAN_D_TOPIC
from models.RelGAN_G import RelGAN_G
from models.RelGAN_H import RelGAN_H
from models.RelGAN_TOPIC import RelGAN_TOPIC
from models.generator import LSTMGenerator
from models.hotflip import HotFlip
from models.random import RandomGenerator
from models.random_coherence import RandomBetterGenerator
from models.textbugger import TextBugger
from models.unitrigger import UniversalTrigger
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from utils.data_loader import *
from utils.data_loader_fakenews import *
from utils.data_loader_generated import *
from utils.helpers import cal_cosine
from utils.helpers import get_fixed_temperature
from utils.helpers import get_losses
# from utils.monit import *
from utils.topic_model import *
from utils.utils import *

class RelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(RelGANInstructor, self).__init__(opt)

        # generator, discriminator
        if cfg.model_type == "RMC":
            self.gen = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
            print("Initiate Generator with RMC")
        else:
            self.gen = LSTMGenerator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
            print("Initiate Generator with vanilla LSTM")

        self.dis = RelGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.gen_adv_f_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_f_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.topic_encoder_opt_gen = optim.Adam(self.gen.parameters(), lr=cfg.topic_gen_lr)

        # Initializing CNN
        if "LR" == cfg.f_type:
            from models.RelGAN_F import RelGAN_F
            self.F_clf = RelGAN_F(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, dropout=cfg.f_dropout_comment, 
                                                                                    dropout_content=cfg.f_dropout_content, 
                                                                                    gpu=cfg.CUDA)
            print('initialized LR F')
        elif "CNN" == cfg.f_type:
            from models.RelGAN_F_CNN import RelGAN_F_CNN
            self.F_clf = RelGAN_F_CNN(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, dropout=cfg.f_dropout_comment, 
                                                                                        dropout_content=cfg.f_dropout_content, 
                                                                                        gpu=cfg.CUDA)
            print('initialized CNN F')

        elif "RNN" == cfg.f_type:
            from models.RelGAN_F_RNN import RelGAN_F_RNN
            self.F_clf = RelGAN_F_RNN(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, hidden_dim=cfg.rnn_hidden_dim,
                                                                                        dropout=cfg.f_dropout_comment,
                                                                                        dropout_content=cfg.f_dropout_content,
                                                                                        gpu=cfg.CUDA)
            print('initialized RNN F with CNN Comment Encoder')

        elif "RNN2" == cfg.f_type:
            from models.RelGAN_F_RNN2 import RelGAN_F_RNN2
            self.F_clf = RelGAN_F_RNN2(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, hidden_dim=cfg.rnn_hidden_dim,
                                                                                        dropout=cfg.f_dropout_comment,
                                                                                        dropout_content=cfg.f_dropout_content,
                                                                                        gpu=cfg.CUDA)
            print('initialized RNN2 F with FCN Comment Encoder')

        elif "CSI" == cfg.f_type:
            from models.RelGAN_F_CSI import RelGAN_F_CSI
            self.F_clf = RelGAN_F_CSI(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx, hidden_dim=cfg.rnn_hidden_dim,
                                                                                        dropout=cfg.f_dropout_comment,
                                                                                        dropout_content=cfg.f_dropout_content,
                                                                                        gpu=cfg.CUDA)
            print('initialized CSI F with FCN Comment Encoder')

        elif "TEXTCNN" == cfg.f_type:
            from models.RelGAN_F_TEXTCNN import RelGAN_F_TEXTCNN
            self.F_clf = RelGAN_F_TEXTCNN(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx,
                                                                                        dropout=cfg.f_dropout_comment,
                                                                                        dropout_content=cfg.f_dropout_content,
                                                                                        gpu=cfg.CUDA)
            print('initialized TEXT CNN F with CNN+FCN Comment Encoder')

        elif "TEXTCNN2" == cfg.f_type:
            from models.RelGAN_F_TEXTCNN2 import RelGAN_F_TEXTCNN2
            self.F_clf = RelGAN_F_TEXTCNN2(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx,
                                                                                        dropout=cfg.f_dropout_comment,
                                                                                        dropout_content=cfg.f_dropout_content,
                                                                                        gpu=cfg.CUDA)
            print('initialized TEXT CNN 2 F with CNN+FCN Comment Encoder')
        
        elif "TEXTCNN3" == cfg.f_type:
            from models.RelGAN_F_TEXTCNN3 import RelGAN_F_TEXTCNN3
            self.F_clf = RelGAN_F_TEXTCNN3(cfg.f_embed_dim, cfg.vocab_size, cfg.padding_idx,
                                                                                        dropout=cfg.f_dropout_comment,
                                                                                        dropout_content=cfg.f_dropout_content,
                                                                                        gpu=cfg.CUDA)
            print('initialized TEXT CNN 3 F with CNN+FCN Comment Encoder')

        self.F_clf_opt = optim.Adam(self.F_clf.parameters(), lr=cfg.F_clf_lr)

        # loading embedding data
        print("loading embedding data...")
        self.content_embeddings_data, self.id2idx, self.embeddings_sim = self.load_embeddings(cfg.content_embeddings)
        # self.comment_embeddings_data, _, _ = self.load_embeddings(cfg.comment_embeddings)

        print("loading topic model...")
        self.topic_model = self.load_topic(cfg.topic_file)

        self.train_data = GenDataIterContent(cfg.train_data, self.content_embeddings_data, batch_size=cfg.batch_size, shuffle=True, name="TRAIN")
        self.test_data = GenDataIterContent(cfg.test_data, self.content_embeddings_data, batch_size=cfg.batch_size, shuffle=True, name="TEST")  

        if cfg.train_with_content:
            self.content_iter = F_DataIterSep(cfg.f_train_data_file, cfg.f_train_label_file, 
                            self.content_embeddings_data, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
            self.content_iter_eval = F_DataIterSep(cfg.f_dev_data_file, cfg.f_dev_label_file, 
                            self.content_embeddings_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
            self.content_iter_test = F_DataIterSep(cfg.f_test_data_file, cfg.f_test_label_file, 
                            self.content_embeddings_data, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

            print("OVERLAP ARTICLES: {}".format(len(np.intersect1d(list(self.content_iter.id2idx.keys()), 
                list(self.content_iter_test.id2idx.keys())))))

        else:
            self.content_iter = None
            self.content_iter_eval = None
            self.content_iter_h = None
            self.content_iter_eval_h = None

        if not cfg.if_eval:
            self.init_model()
        else:
            self.bleu = BLEU('BLEU', gram=[2], if_use=cfg.use_bleu)


    def load_topic(self, path):
        topic_model = None
        # try:
        topic_model = TopicCoherency(start_index=cfg.start_index[cfg.dataset])
        topic_model.load(path)
        # except Exception as e:
            # print("ERROR loading topic model ", e)
        return topic_model

    def load_embeddings(self, path, cal_sim=True):
        tmp = np.load(path, allow_pickle=True)
        all_embeddings = tmp['embeddings']
        ids = tmp['ids']
        embeddings_data = {}
        id2idx = {}
        for i in range(len(all_embeddings)):
            embeddings_data[ids[i]] = all_embeddings[i]
            id2idx[ids[i]] = i
        embeddings_data["zero"] = np.zeros(all_embeddings[0].shape)
        if cal_sim:
            embeddings_sim = cal_cosine(all_embeddings)
        else:
            embeddings_sim = None
        return embeddings_data, id2idx, embeddings_sim


    def init_model(self):
        print("Initialization...")
        self.nll_gen = NLL('NLL_gen', if_use=cfg.use_nll_gen, gpu=cfg.CUDA)

        if cfg.CUDA:
            self.gen = self.gen.to('cuda')
            self.dis = self.dis.to('cuda')
            self.F_clf = self.F_clf.to('cuda')

        if cfg.dis_pretrain:
            self.log.info('Load pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path))

        if cfg.load_gen_pretrain:
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))
            self.log.info('Loaded MLE pre-trained generator: {} TEMPERATURE:{}'.format(cfg.pretrained_gen_path, self.gen.temperature))
            self.gen.temperature = 1

        # self.sample_show_sentences(limit=cfg.sample_limit, content_iter=self.train_data)


    def predict(self, content_iter_eval):
        samples = []
        ids = []
        for i, batch in tqdm(enumerate(content_iter_eval)):
            ids.append(batch['id'])
            gen_samples = self.gen.sample(len(batch['id']), min(len(batch['id']), cfg.batch_size), one_hot=False, content_iter=batch)
            samples.append(gen_samples)
        samples = torch.cat(samples, 0)
        ids = np.concatenate(ids, 0)
        sents = tensor_to_tokens(samples, self.idx2word_dict)
        generated = [" ".join(sent) for sent in sents]
        return ids, generated


    def sample_show_sentences(self, limit=5, content_iter=None, attack_mode=None):
        self.log.info('=== Start Sample new sentences... ===')
        contents = None
        titles = None
        gen_samples = None
        acc = 0
        attack_acc = 0
        coherence_acc = 0

        if not content_iter is None:
            batch = content_iter.random_batch()
            contents = batch['content']
            titles = batch['title_str']
            ids = batch['id']

            if attack_mode:
                _, _, original_preds, attack_preds, gen_samples = self.eval_geneartor_with_f_all_single(self.F_clf, batch, cfg.attack_mode, 
                    only_generated_comment, return_all=False, cal_true=True, eval_num=1)
            else:
                gen_samples,_, _ = self.generate_best_topic_batch(batch, custom_model=self.topic_model, 
                                one_hot=False, translate=False, eval_num=5)
        else:
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=False, content_iter=None)
        if 'textbugger' in self.gen.name:
            idx2word = self.gen.idx2word
        else:
            idx2word = self.idx2word_dict
        sents = tensor_to_tokens(gen_samples, idx2word)

        if titles:
            print("Attack Accuracy:{:.3f}".format(attack_acc))
            # print("Coherence Accuracy:{:.3f}".format(coherence_acc))
            for i in range(min(limit, len(gen_samples))):
                sent = sents[i]
                title = titles[i]
                _id = ids[i]

                if attack_mode:
                    original_pred = original_preds[i].cpu().detach().numpy()
                    attack_pred = attack_preds[i].cpu().detach().numpy()
                    text = "Title: {}\n->Comment: {}\nOriginal Pred:{} ==> Attack Pred:{} ".format(title, " ".join(sent), original_pred, attack_pred)
                else:
                    text = "Title: {}\n->Comment: {}\n".format(title, " ".join(sent))
                self.log.info(text)
                # visualize_generated_text(text, content_iter.name)
        else:
            for i in range(limit):
                sent = sents[i]
                text = "Comment: {}".format(" ".join(sent))
                self.log.info(text)
                # visualize_generated_text(text, content_iter.name)
        self.log.info('=== End Sample new sentences... ===\n')


    def _run(self):
        # ===PRE-TRAINING F CLASSIFIER==
        if cfg.flg_train_f:
            self.log.info('\nStarting Misinformation Model F Training...')
            self.pretrain_text_classifier(self.F_clf, self.content_iter, self.content_iter_eval, self.content_iter_test,
                                cfg.f_batch_size*4, # speed-up training
                                cfg.F_train_epoch, self.cross_entropy_loss, self.F_clf_opt, 
                                cfg.f_patience_epoch, cfg.pretrained_f_path, prefix="F")
        else:
            try:
                self.F_clf.load_state_dict(torch.load(cfg.pretrained_f_path))
                test_clf_data = self.content_iter_test
                test_loss, test_acc, preds, labels = self.eval_clf(self.F_clf, test_clf_data.get_loader(cfg.f_batch_size), self.cross_entropy_loss)
                self.log.info('F LOADED from {}; Test Loss: {:.2f}  Test Acc:{:.2f}'.format(cfg.pretrained_f_path, test_loss, test_acc))
                idx = np.where(labels == preds)[0]
                print("Performance on test")
                print(classification_report(labels, preds))
                print("Before Real Attack: {}".format(accuracy_score(np.zeros(len(idx)), preds[idx])))
                print("Before Fake Attack: {}".format(accuracy_score(np.ones(len(idx)), preds[idx])))
            except Exception as e:
                self.log.info('F NOT LOADED . {} NOT FOUND. MSG:{}'.format(cfg.pretrained_f_path, e))


        # ===PRE-TRAINING (GENERATOR)===
        if cfg.repeat_gen_pretrain:
            self.log.info('\nStarting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pretrain_generator: {}'.format(cfg.pretrained_gen_path))

        # metric_str, metric_scores = self.cal_metrics(self.train_data, self.test_data, custom_gen=self.gen)
        # print(metric_str)

        # # ===ADVERSARIAL TRAINING WITH ===
        self.log.info('\nStarting Adversarial Training - ')
        if clg.flg_adv_f:
            print("With F (fake news detector)...")
        if clf.flg_adv_t:
            print("With T (topic)...")
        if clf.flg_adv_d:
            print("With D (text quality/realness)...")

        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            self.sig.update()
            if self.sig.adv_sig:
                if cfg.flg_adv_d:
                    g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
                    d_loss = self.adv_train_discriminator(cfg.ADV_d_step)  # Discriminator

                if cfg.flg_adv_t:
                    g_t_loss = self.topic_train_generator_comment(1)

                if cfg.flg_adv_f:
                    f_alpha = retrieve_alpha(cfg.g_f_alpha)
                    f_alpha = torch.tensor(f_alpha, requires_grad=False).to('cuda').float()
                    f_loss, f_acc = self.adv_train_generator_with_f(self.F_clf, cfg.g_f_step, self.content_iter, criterion=self.cross_entropy_loss, gen_adv_opt=self.gen_adv_f_opt, batch_size=cfg.f_batch_size, alpha=f_alpha, attack_mode=cfg.attack_mode)

                # TEST
                if adv_epoch % cfg.adv_log_step == 0:
                    log_str = ""

                    if cfg.flg_adv_f:
                        f_loss_eval, f_acc_eval, f_acc_eval_true = self.eval_generator_with_f_all(self.F_clf, self.content_iter_test.get_loader(cfg.batch_size), 
                                                                                                multiplier=999999, attack_mode=cfg.attack_mode,
                                                                                                eval_num=cfg.eval_topic_num)
                        # visualize_append_line(f_loss_eval, f_acc_eval, "F-ADV-DEV")
                        # visualize_append_single_line(f_acc_eval_true, "F-ADV-DEV(TRUE)")
                        log_str += "f_loss_eval: %.4f, f_acc_eval: %.4f, f_acc_eval_true: %.4f" % (f_loss_eval, f_acc_eval, f_acc_eval_true)

                    g_t_loss_eval = 0
                    if cfg.flg_adv_t:
                        g_t_loss_eval = self.topic_eval_generator_comment(self.test_data)
                        # visualize_append_single_line(g_t_loss_eval, "G TOPIC EVAL LOSS")
                        
                    topic_score, _ = self.eval_generator_topic(self.test_data)
                    # visualize_append_single_line(topic_score, "TOPIC SCORE")
                    log_str += "|| g_t_loss_eval: %.4f, topic_score: %.4f" % (g_t_loss_eval, topic_score)

                    if cfg.if_eval_score:
                        metric_str, metric_scores = self.cal_metrics(self.train_data, self.test_data, custom_gen=self.gen)
                        # visualize_scores(metric_scores)
                        log_str += "|| {}".format(metric_str)

                    self.log.info(log_str)
                    self.sample_show_sentences(limit=10, content_iter=self.train_data)
                    self.sample_show_sentences(limit=20, content_iter=self.test_data)

                    if cfg.if_save and not cfg.if_test:
                        gen_path, dis_path, save_sample_path = self._save('ADV-F', adv_epoch, self.content_iter)
                        print('SAVED G: {}\n D:{}\n Samples:{}\n'.format(gen_path, dis_path, save_sample_path))
                
                self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature
                
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                progress.close()
                break


    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass


    def generate_best_topic_batch(self, batch, custom_model=None, one_hot=False, translate=False, eval_num=5, return_text=False):
        topic_model = self.topic_model if not custom_model else custom_model
        batch_size = len(batch['content'])
        _ids = batch['id']
        id2best = {}
        id2generated = {}
        id2text = {}

        for _ in range(eval_num):
            if not return_text:
                gen_samples = self.gen.sample(batch_size, batch_size, one_hot=False, content_iter=batch)
                texts = torch.zeros(len(gen_samples))
            else:
                gen_samples, texts = self.gen.sample(batch_size, batch_size, one_hot=False, content_iter=batch, return_text=True)

            comments_token, comments_translated = tensor_to_tokens(gen_samples, self.idx2word_dict, joined=True)
            comments = comments_token if translate else gen_samples
            scores = topic_model.cal_coherency_from_ids(comments_translated, batch['id'], mean=False)
            for j in range(batch_size):
                if _ids[j] not in id2best:
                    id2generated[_ids[j]] = [comments[j]]
                    id2best[_ids[j]] = [scores[j]]
                    id2text[_ids[j]] = [texts[j]]
                else:
                    id2generated[_ids[j]].append(comments[j])
                    id2best[_ids[j]].append(scores[j])
                    id2text[_ids[j]].append(texts[j])
        generated = []
        scores = []
        texts = []

        for i in range(batch_size):
            _id = _ids[i]
            idx = np.argsort(id2best[_id])[::-1]
            generated.append(id2generated[_id][idx[0]].unsqueeze(0))
            scores.append(id2best[_id][idx[0]])
            texts.append(id2text[_id][idx[0]])
            id2best[_id].pop(idx[0])
            id2generated[_id].pop(idx[0])
            id2text[_id].pop(idx[0])
        generated = torch.cat(generated, 0)
        if one_hot:
            generated = F.one_hot(generated, cfg.vocab_size).float()

        return generated, scores, texts


    def generate_best_topic(self, num_samples, batch_size, data_iter, custom_model=None, eval_num=5, translate=False):
        ids = []
        generated = []
        coherency_scores = []
        texts = []
        with torch.no_grad():
            while len(generated) < num_samples:
                batch = data_iter.random_batch(batch_size)
                contents = batch['content']
                ids = batch['id']
                tmp, coherency, text = self.generate_best_topic_batch(batch, custom_model=custom_model, 
                            one_hot=False, translate=translate, eval_num=eval_num)
                # tmp: batch_size * max_seq_len
                generated.append(tmp)
                coherency_scores.append(coherency)
                texts.append(text)
        generated = torch.cat(generated, 0)
        generated = generated[:num_samples]
        coherency_scores = np.concatenate(coherency_scores,0)[:num_samples]
        texts = np.concatenate(texts, 0)[:num_samples]
        return generated, coherency_scores, texts


    def eval_generator_topic(self, data_iter, custom_model=None, eval_num=5):
        ids = []
        id2best = {}
        id2generated = {}
        if not custom_model:
            topic_model = self.topic_model
        else:
            topic_model = custom_model

        if not cfg.if_eval:
            self.gen.eval()

        for _, batch in enumerate(data_iter.loader):
            contents = batch['content']
            _ids = batch['id']
            idx = batch['idx'].numpy()
            ids.append(_ids)
            for i in range(eval_num):
                gen_samples = self.gen.sample(len(contents), len(contents), one_hot=False, content_iter=batch)
                _comments = tensor_to_tokens(gen_samples, self.idx2word_dict)
                _comments = [" ".join(c) for c in _comments]
                scores = topic_model.cal_coherency_from_ids(_comments, _ids, mean=False)
                for j in range(len(_ids)):
                    if _ids[j] not in id2best:
                        id2best[_ids[j]] = scores[j]
                        id2generated[_ids[j]] = _comments[j]
                    if id2best[_ids[j]] < scores[j]:
                        id2best[_ids[j]] = scores[j]
                        id2generated[_ids[j]] = _comments[j] # batch_size * max_seq_len
        score = np.mean([id2best[k] for k in id2best])
        generated = [id2generated[k] for k in id2generated]
        if not cfg.if_eval:
            self.gen.train()

        return score, generated

    def evaluate_gen(self):
        scores = {}

        if cfg.flg_adv_f:
            f_loss_eval, f_acc_eval, f_acc_eval_true = self.eval_generator_with_f_all(self.F_clf, self.content_iter_test.get_loader(cfg.batch_size), 
                attack_mode=cfg.attack_mode, multiplier=999999, eval_num=cfg.eval_topic_num, show_progress=True, show_sentence=cfg.if_show_sentence)
            scores['f_loss'] = f_loss_eval
            scores['f_acc'] = f_acc_eval
            scores['f_acc_true'] = f_acc_eval_true

        if cfg.flg_adv_t:
            topic_score, _ = self.eval_generator_topic(self.test_data)
            scores['topic'] = topic_score

        if cfg.if_eval_score:
            metric_str, metric_scores = self.cal_metrics(self.train_data, self.test_data, custom_gen=self.gen)
            scores['Others'] = metric_str
        
        return scores

    def load_f_pretrain_evaluate(self):
        self.F_clf.load_state_dict(torch.load(cfg.pretrained_f_path))
        test_loss, test_acc, preds, labels = self.eval_clf(self.F_clf, self.content_iter_test.loader, self.cross_entropy_loss)
        self.log.info('F LOADED from {}; Test Loss: {:.2f}  Test Acc:{:.2f}'.format(cfg.pretrained_f_path, test_loss, test_acc))
        idx = np.where(labels == preds)[0]
        print("Performance on test")
        print(classification_report(labels, preds, digits=3))
        print("Before Real Attack: {}".format(accuracy_score(np.zeros(len(idx)), preds[idx])))
        print("Before Fake Attack: {}".format(accuracy_score(np.ones(len(idx)), preds[idx])))


    def load_g_evaluate(self, path, temperature=1, show_sentence=False):
        self.gen.load_state_dict(torch.load(path))
        self.gen.temperature = temperature # very important here
        scores = self.evaluate_gen()
        self.log.info('TEMPERATURE:{}||| Loaded Generator from: {}'.format(temperature, path))
        if show_sentence:
            self.sample_show_sentences(limit=cfg.batch_size, content_iter=self.test_data, attack_mode=cfg.attack_mode)
            # self.sample_show_sentences(limit=cfg.batch_size, content_iter=self.train_data, attack_mode=None)

        return scores

    def evaluate_random(self, random_g, show_sentence=False):
        self.gen = random_g
        if show_sentence:
            self.sample_show_sentences(limit=cfg.batch_size, content_iter=self.test_data, attack_mode=None)
        scores = self.evaluate_gen()
        return scores

    def evaluate_baseline(self, g, show_sentence=False):
        self.gen = g
        if show_sentence:
            self.sample_show_sentences(limit=cfg.batch_size, content_iter=self.test_data, attack_mode=None)
        scores = self.evaluate_gen()
        return scores

    def evaluate_lstm(self, show_sentence=False):
        lstm_g = LSTMGenerator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.gen = lstm_g
        if show_sentence:
            self.sample_show_sentences(limit=cfg.batch_size, content_iter=self.test_data, attack_mode=None)
        scores = self.evaluate_gen()
        return scores

    def _predict(self, to_file=None):
        import pandas as pd
        if cfg.CUDA:
            self.gen = self.gen.to('cuda')
        self.gen.load_state_dict(torch.load(cfg.eval_g_path))
        self.gen.temperature = 1
        ids, generated = self.predict(self.content_iter_test.loader)

        data = []
        for i in range(len(ids)):
            tmp = {}
            tmp['id'] = ids[i]
            tmp['comment'] = generated[i]
            data.append(tmp)
        df = pd.DataFrame.from_dict(data)
        file = '{}_PREDICT_{}.csv'.format(cfg.eval_g_path, time.time())
        df.to_csv(file)
        print("generated", len(generated))
        print("saved prediction at {}".format(file))


    # def _eval_combine(self):
    #     generators = [cfg.eval_g_path, cfg.pretrained_gen_path, "copycat"]
    #     types = ["malcom", "malcom", "copycat"]

    def attack_strategy(self, existing, ratio):
        attack = int(existing*ratio)
        attack = max(attack, 1)
        return attack

    def initialize_gen(self):
        self.gen = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        if cfg.CUDA:
            self.gen = self.gen.to('cuda')

    # def _robust(self):
        # 

    def gen_save(self, gen, custom_text=False):
        self.gen = gen
        data = []
        # all_scores = []

        for q in range(1):
            save_path = "{}_{}-{}.csv".format(gen.name, q, cfg.atk_num_comment)
            save_path = os.path.join("./gen_{}".format(cfg.dataset), save_path)
            if not os.path.exists(save_path):
                print("generating for...", self.gen.name)
                bar = tqdm(enumerate(self.content_iter_test.loader))
                for _, batch in bar:
                    batch_size = len(batch['content'])
                    gen_samples = []
                    texts = []
                    # coherency_scores = []
                    for i in range(cfg.atk_num_comment):
                        gen_sample, coherency, text = self.generate_best_topic_batch(batch, custom_model=self.topic_model, 
                                                one_hot=False, translate=False, eval_num=cfg.eval_topic_num, return_text=custom_text)

                        gen_samples.append(gen_sample.unsqueeze(1))
                        texts.append(text)
                        # coherency_scores.append(np.array(coherency).reshape(-1, 1))
                        # all_scores.append(np.mean(coherency))
                    gen_samples = torch.cat(gen_samples, 1) #batch_size * atk_num_comment * max_seq_len
                    # coherency_scores = np.concatenate(coherency_scores, 1)

                    print("texts", len(texts))

                    for i in range(batch_size):
                        _id = batch['id'][i]
                        title = batch['title_str'][i]
                        label = batch['label'][i]
                        # label = None
                        gen_sample = gen_samples[i] # atk_num_comment * seq_len
                        gen_sample = tensor_to_tokens(gen_sample, self.idx2word_dict)
                        if custom_text:
                            _comments = [t[i] for t in texts]
                        else:
                            _comments = [" ".join(c) for c in gen_sample]
                        comments = "::".join(_comments)
                        tmp = "{}::{}::{}::{}".format(title, _id, comments, label)
                        data.append(tmp)

                        # coherency2 = self.topic_model.cal_coherency_from_single_id(_comments, _id, mean=False)
                        # print(_comments)
                        # print("Coherency", (coherency_scores[i], coherency2))

                with open(save_path,'w') as f:
                    for x in data:
                        f.write("{}\n".format(x))
                print("saved to file", save_path)
                # print("coherency:", (np.array(coherency_scores).flatten(), np.mean(np.array(all_scores).flatten())))
            else:
                print(save_path, "already exists")

    def _gen(self):
        # pre_style_atk
        self.gen.load_state_dict(torch.load(cfg.eval_g_path))
        self.gen.temperature = 1 # very important here
        self.gen.name = "Pretrain+Style+Attack"
        self.gen_save(self.gen)

        #pre
        self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))
        self.gen.temperature = 1 # very important here
        self.gen.name = "Pretrain"
        self.gen_save(self.gen)

        #pre_style
        self.gen.load_state_dict(torch.load(cfg.eval_style_path))
        self.gen.temperature = 1 # very important here
        self.gen.name = "Pretrain+Style"
        self.gen_save(self.gen)

        #pre_attack
        self.gen.load_state_dict(torch.load(cfg.eval_pre_attack_path))
        self.gen.temperature = 1 # very important here
        self.gen.name = "Pretrain+Attack"
        self.gen_save(self.gen)

        #copycat
        random_g = RandomBetterGenerator(self.content_iter, self.embeddings_sim, 
                                        self.content_embeddings_data, self.id2idx, 
                                        num_article=cfg.random_num_article, 
                                        attack_mode=cfg.attack_mode)
        random_g.find_best_match(self.content_iter_test)
        random_g.find_best_match(self.test_data)
        random_g.find_best_match(self.content_iter_eval)
        random_g.name = "CopyCat"
        self.gen_save(random_g)

        #textbugger
        textbugger_g = TextBugger(random_g, self.idx2word_dict, self.word2idx_dict, 
                                    attack_mode=cfg.attack_mode, if_load_from_file=not cfg.if_textbugger_retrain) 
        textbugger_g.name = "TextBugger"
        self.gen_save(textbugger_g, custom_text=True)

        #hotflip
        hotflip_g = HotFlip(self.F_clf, self.idx2word_dict, self.word2idx_dict, 
                                        random_g, n_neighbor=20, attack_mode=cfg.attack_mode)
        hotflip_g.name = "HotFlip"
        self.gen_save(hotflip_g)

        #unitrigger
        unitrigger_g = UniversalTrigger(self.F_clf, self.content_iter_eval, cfg.trigger_length, 
                                                    self.idx2word_dict, self.word2idx_dict, 
                                                    random_g, 
                                                    n_neighbor=cfg.trigger_neighbor,
                                                    beam_size=cfg.trigger_beam_size, 
                                                    attack_mode="target_real",
                                                    topic_model=self.topic_model,
                                                    if_retrain=cfg.if_trigger_retrain)
        unitrigger_g.name = "UniTrigger"
        self.gen_save(unitrigger_g)


    def _eval(self):
        show_sentence = False
        results = []
        save_to_file = None

        if cfg.CUDA:
            self.gen = self.gen.to('cuda')
            self.F_clf = self.F_clf.to('cuda')

        if cfg.flg_adv_f:
            self.load_f_pretrain_evaluate()

        random_g = None
        if cfg.flg_eval_copycat or cfg.flg_eval_hotflip or cfg.flg_eval_unitrigger or cfg.flg_eval_textbugger:
            random_g = RandomBetterGenerator(self.content_iter, self.embeddings_sim, self.content_embeddings_data, self.id2idx, num_article=cfg.random_num_article, attack_mode=cfg.attack_mode)
            random_g.find_best_match(self.content_iter_test)
            random_g.find_best_match(self.test_data)
            random_g.find_best_match(self.content_iter_eval)

        if cfg.if_eval_robust:
            save_to_file = "{}_RobustAttackStats_{}.csv".format(cfg.dataset, cfg.atk_num_comment)
            from models.from_file import FromFileGenerator
            import glob
            import os
            from pathlib import Path    
            print("loading embedding data...")
            self.content_embeddings_data, self.id2idx, self.embeddings_sim = self.load_embeddings(cfg.content_embeddings)

            files = glob.glob('./gen_{}/*{}.csv'.format(cfg.dataset, cfg.atk_num_comment))
            # files = glob.glob('./gen/*{}.csv')
            print(files)

            for file in files:
                filtered_file = "{}_filtered_S{}_H{}".format(file, cfg.spelling_thres, cfg.coherency_thres)
                if os.path.exists(file):
                    g = FromFileGenerator(file, self.content_embeddings_data, self.idx2word_dict)
                    g_scores = self.evaluate_baseline(g, show_sentence=False)
                    g_scores['model'] = Path(file).name
                    g_scores['mode'] ="before"
                    results.append(g_scores)
                    print(g_scores)
                else:
                    print("file {} does not exist".format(file))

                if os.path.exists(filtered_file):
                    g = FromFileGenerator(filtered_file, self.content_embeddings_data, self.idx2word_dict)
                    g_scores = self.evaluate_baseline(g, show_sentence=False)
                    g_scores['model'] = Path(file).name
                    g_scores['mode'] ="after"
                    results.append(g_scores)
                    print(g_scores)
                else:
                    print("filtered_file {} does not exist".format(filtered_file))

        if "{}" in cfg.eval_g_path:
            for i in tqdm(range(0, 5000, 100)):
                g_scores = self.load_g_evaluate(cfg.eval_g_path.format(str(i).zfill(4)), show_sentence=show_sentence)
                g_scores['model'] = 'AtkCom-{}'.format(i)
                results.append(g_scores)
                print(g_scores)

        if cfg.max_num_comment_test < 0:
            ratio = cfg.attack_ratio
            save_to_file = "Robustness_{}_{}.csv".format(cfg.f_type, ratio)
            if not os.path.exists(save_to_file):
                for existing_comment in range(cfg.max_num_comment):
                    cfg.atk_num_comment = self.attack_strategy(existing_comment, ratio)
                    cfg.max_num_comment_test = cfg.atk_num_comment + existing_comment

                    self.initialize_gen()
                    for i in range(cfg.eval_num):
                        g_scores = self.load_g_evaluate(cfg.eval_g_path, show_sentence=show_sentence)
                        g_scores['model'] = 'PreStyleAttack-{}-{}'.format(existing_comment, cfg.atk_num_comment)
                        g_scores['existing'] = existing_comment
                        g_scores['atk'] = cfg.atk_num_comment
                        results.append(g_scores)
                        print(g_scores)

                    for i in range(cfg.eval_num):
                        g_scores = self.load_g_evaluate(cfg.pretrained_gen_path, show_sentence=show_sentence)
                        g_scores['model'] = 'Pre-{}-{}'.format(existing_comment, cfg.atk_num_comment)
                        g_scores['existing'] = existing_comment
                        g_scores['atk'] = cfg.atk_num_comment
                        results.append(g_scores)
                        print(g_scores)

                    for i in range(cfg.eval_num):
                        g_scores = self.load_g_evaluate(cfg.eval_style_path, show_sentence=show_sentence)
                        g_scores['model'] = 'PreStyle-{}-{}'.format(existing_comment, cfg.atk_num_comment)
                        g_scores['existing'] = existing_comment
                        g_scores['atk'] = cfg.atk_num_comment
                        results.append(g_scores)
                        print(g_scores)

                    for i in range(cfg.eval_num):
                        g_scores = self.evaluate_random(random_g, show_sentence=show_sentence)
                        g_scores['model'] = 'CopyCat-{}-{}'.format(existing_comment, cfg.atk_num_comment)
                        g_scores['existing'] = existing_comment
                        g_scores['atk'] = cfg.atk_num_comment
                        results.append(g_scores)
                        print(g_scores)

                    for i in range(cfg.eval_num):
                        hotflip_g = HotFlip(self.F_clf, self.idx2word_dict, self.word2idx_dict, 
                                        random_g, n_neighbor=20, attack_mode=cfg.attack_mode)
                        g_scores = self.evaluate_baseline(hotflip_g, show_sentence=show_sentence)
                        g_scores['model'] = 'HotFlip-{}-{}'.format(existing_comment, cfg.atk_num_comment)
                        g_scores['existing'] = existing_comment
                        g_scores['atk'] = cfg.atk_num_comment
                        results.append(g_scores)
                        print(g_scores)

                    for i in range(cfg.eval_num):
                        unitrigger_g = UniversalTrigger(self.F_clf, self.content_iter_eval, cfg.trigger_length, 
                                                    self.idx2word_dict, self.word2idx_dict, 
                                                    random_g, n_neighbor=cfg.trigger_neighbor,
                                                    beam_size=cfg.trigger_beam_size, 
                                                    attack_mode="target_real",
                                                    topic_model=self.topic_model,
                                                    if_retrain=cfg.if_trigger_retrain)
                        g_scores = self.evaluate_baseline(unitrigger_g, show_sentence=show_sentence)
                        g_scores['model'] = 'UniTrigger-{}-{}'.format(existing_comment, cfg.atk_num_comment)
                        g_scores['existing'] = existing_comment
                        g_scores['atk'] = cfg.atk_num_comment
                        results.append(g_scores)
                        print(g_scores)

                    for i in range(cfg.eval_num):
                        textbugger_g = TextBugger(random_g, self.idx2word_dict, self.word2idx_dict, 
                                    attack_mode=cfg.attack_mode, if_load_from_file=not cfg.if_textbugger_retrain) 
                        g_scores = self.evaluate_baseline(textbugger_g, show_sentence=show_sentence)
                        g_scores['model'] = 'TextBugger-{}-{}'.format(existing_comment, cfg.atk_num_comment)
                        g_scores['existing'] = existing_comment
                        g_scores['atk'] = cfg.atk_num_comment
                        results.append(g_scores)
                        print(g_scores)
                

        if cfg.num_topics < 0:
            min_topic = 5
            max_topic = 20

            if cfg.flg_eval_g:
                for i in tqdm(range(min_topic, max_topic)):
                    topic_model = self.load_topic('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(i))
                    self.gen.load_state_dict(torch.load(cfg.eval_g_path))
                    self.log.info('Loaded Generator from: {}'.format(cfg.eval_g_path))
                    self.gen.temperature = 1
                    topic_score, generated = self.eval_generator_topic(self.test_data, custom_model=topic_model)
                    g_scores = {}
                    g_scores['num_topics'] = i
                    g_scores['topic'] = topic_score
                    g_scores['model'] = 'AtkCom'
                    print(g_scores)
                    results.append(g_scores)

            if cfg.flg_eval_pre:
                for i in tqdm(range(min_topic, max_topic)):
                    topic_model = self.load_topic('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(i))
                    self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))
                    self.log.info('Loaded Generator from: {}'.format(cfg.pretrained_gen_path))
                    self.gen.temperature = 1
                    topic_score, generated = self.eval_generator_topic(self.test_data, custom_model=topic_model)
                    g_scores = {}
                    g_scores['num_topics'] = i
                    g_scores['topic'] = topic_score
                    g_scores['model'] = 'AtkCom-Pre'
                    results.append(g_scores)


            if cfg.flg_eval_copycat:
                for i in tqdm(range(min_topic, max_topic)):
                    topic_model = self.load_topic('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(i))
                    self.gen = random_g
                    topic_score, generated = self.eval_generator_topic(self.test_data, custom_model=topic_model)
                    g_scores = {}
                    g_scores['num_topics'] = i
                    g_scores['topic'] = topic_score
                    g_scores['model'] = 'CopyCat'
                    results.append(g_scores)

            if cfg.flg_eval_hotflip:
                hotflip_g = HotFlip(self.F_clf, self.idx2word_dict, self.word2idx_dict, 
                                    random_g, n_neighbor=20, attack_mode=cfg.attack_mode)

                for i in tqdm(range(min_topic, max_topic)):
                    topic_model = self.load_topic('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(i))
                    self.gen = hotflip_g
                    topic_score, generated = self.eval_generator_topic(self.test_data, custom_model=topic_model)
                    g_scores = {}
                    g_scores['num_topics'] = i
                    g_scores['topic'] = topic_score
                    g_scores['model'] = 'HotFlip'
                    results.append(g_scores)

            if cfg.flg_eval_unitrigger:
                topic_model = self.load_topic('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(7))
                unitrigger_g = UniversalTrigger(self.F_clf, self.content_iter_eval, cfg.trigger_length, 
                                                self.idx2word_dict, self.word2idx_dict, 
                                                random_g, 
                                                n_neighbor=cfg.trigger_neighbor,
                                                beam_size=cfg.trigger_beam_size, 
                                                attack_mode="target_real",
                                                topic_model=topic_model,
                                                if_retrain=cfg.if_trigger_retrain)

                for i in tqdm(range(min_topic, max_topic)):
                    topic_model = self.load_topic('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(i))
                    self.gen = unitrigger_g
                    topic_score, generated = self.eval_generator_topic(self.test_data, custom_model=topic_model)
                    g_scores = {}
                    g_scores['num_topics'] = i
                    g_scores['topic'] = topic_score
                    g_scores['model'] = 'UniTrigger'
                    results.append(g_scores)

            if cfg.flg_eval_textbugger:
                textbugger_g = TextBugger(random_g, self.idx2word_dict, self.word2idx_dict, 
                                        attack_mode=cfg.attack_mode, if_load_from_file=not cfg.if_textbugger_retrain)
                for i in tqdm(range(min_topic, max_topic)):
                    topic_model = self.load_topic('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(i))
                    self.gen = textbugger_g
                    topic_score, generated = self.eval_generator_topic(self.test_data, custom_model=topic_model)
                    g_scores = {}
                    g_scores['num_topics'] = i
                    g_scores['topic'] = topic_score
                    g_scores['model'] = 'TextBugger'
                    results.append(g_scores)

        else:
            if cfg.flg_eval_g:
                for i in range(cfg.eval_num):
                    g_scores = self.load_g_evaluate(cfg.eval_g_path, show_sentence=show_sentence)
                    g_scores['model'] = 'AtkCom'
                    results.append(g_scores)
                    print(g_scores)

            if cfg.flg_eval_pre:
                for i in range(cfg.eval_num):
                    mle_scores = self.load_g_evaluate(cfg.pretrained_gen_path, show_sentence=show_sentence)
                    mle_scores['model'] = 'AtkCom-Pre'
                    results.append(mle_scores)

            if cfg.flg_eval_copycat:
                for i in range(cfg.eval_num):
                    copycat_scores = self.evaluate_random(random_g, show_sentence=show_sentence)
                    copycat_scores['model'] = 'CopyCat'
                    results.append(copycat_scores)

            if cfg.flg_eval_hotflip:
                hotflip_g = HotFlip(self.F_clf, self.idx2word_dict, self.word2idx_dict, 
                                    random_g, n_neighbor=20, attack_mode=cfg.attack_mode)
                if cfg.if_hotflip_retrain:
                    for i in range(cfg.eval_num):
                        hotflip_g.learn_best_match(self.content_iter_test)
                    for i in range(cfg.eval_num):
                        hotflip_g.learn_best_match(self.test_data, only_unknown=True)
                for i in range(cfg.eval_num):
                    hotflip_scores = self.evaluate_baseline(hotflip_g, show_sentence=show_sentence)
                    hotflip_scores['model'] = 'HotFlip'
                    results.append(hotflip_scores)
            
            if cfg.flg_eval_unitrigger:
                unitrigger_g = UniversalTrigger(self.F_clf, self.content_iter_eval, cfg.trigger_length, 
                                                self.idx2word_dict, self.word2idx_dict, 
                                                random_g, n_neighbor=cfg.trigger_neighbor,
                                                beam_size=cfg.trigger_beam_size, 
                                                attack_mode="target_real",
                                                topic_model=self.topic_model,
                                                if_retrain=cfg.if_trigger_retrain)
                for i in range(cfg.eval_num):
                    unitrigger_scores = self.evaluate_baseline(unitrigger_g, show_sentence=show_sentence)
                    unitrigger_scores['model'] = 'UniTrigger'
                    results.append(unitrigger_scores)

            if cfg.flg_eval_textbugger:
                textbugger_g = TextBugger(random_g, self.idx2word_dict, self.word2idx_dict, 
                                attack_mode=cfg.attack_mode, if_load_from_file=not cfg.if_textbugger_retrain) 
                if cfg.if_textbugger_retrain:   
                    for i in range(cfg.eval_num):
                        textbugger_g.learn_best_match(self.content_iter_test)
                    textbugger_g.learn_best_match(self.test_data, only_unknown=True)
                for i in range(cfg.eval_num):
                    textbugger_scores = self.evaluate_baseline(textbugger_g, show_sentence=show_sentence)
                    textbugger_scores['model'] = 'TextBugger'
                    results.append(textbugger_scores)

        df = pd.DataFrame.from_dict(results)
        print("============RESULTS============")
        print(df.to_json())
        print(df)
        print("AVG", df.groupby('model').mean())
        
        if "{}" in cfg.eval_g_path:
            file = "{}_{}.csv".format(cfg.eval_g_path, time.time())
            df.to_csv(file)
            print("saved to ", file)

        if cfg.num_topics < 0:
            file = "{}_{}.csv".format("topic_evaluation", time.time())
            df.to_csv(file)
            print("saved to ", file)

        if save_to_file:
            file = save_to_file
            df.to_csv(file)
            print("saved to ", file)

    def eval_clf(self, model, data_loader, criterion, target_class=None):
        total_loss = 0
        total_acc = 0
        total_num = 0
        total_batch = 0
        preds = []
        labels = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, comments, label, num_comments = self.prepare_data_clf(data)
                if cfg.CUDA:
                    inp, comments, label = inp.to('cuda'), comments.to('cuda'), label.to('cuda')
                    num_comments = num_comments.to('cuda')

                idx = range(len(label))
                if target_class:
                    idx = torch.where(label == target_class)

                pred = model.forward(inp[idx], comments[idx], num_comments[idx])
                loss = criterion(pred, label)
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == label)).item()
                total_num += inp.size(0)
                total_batch += 1

                if len(pred.size()) > 1:
                    preds.append(pred.argmax(dim=-1).cpu().detach().numpy())
                else: #regression
                    preds.append(pred.cpu().deteach.numpy())

                labels.append(label.cpu().detach().numpy())
            total_loss /= total_batch if total_batch > 0 else 0
            total_acc /= total_num
        preds = np.concatenate(preds, 0)
        labels = np.concatenate(labels, 0)
        return total_loss, total_acc, preds, labels

    def train_clf_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, comments, label, num_comments = self.prepare_data_clf(data)
            if cfg.CUDA:
                inp, comments, label = inp.to('cuda'), comments.to('cuda'), label.to('cuda')
                num_comments = num_comments.to('cuda')
                    
            pred = model.forward(inp, comments, num_comments)
            loss = criterion(pred, label)
            self.optimize(optimizer, loss, model, clip_grad=False)
            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == label)).item()
            total_num += inp.size(0)

        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc


    def pretrain_text_classifier(self, clf, content_iter, content_iter_eval, content_iter_test, 
                                batch_size, epochs, criterion, clf_opt, 
                                patience_epoch, pretrained_path, prefix="CLF", save=True):
        import copy

        # Prepare data for Classifier
        # clf_data = F_DataIterSep(data_file, label_file, self.content_embeddings_data, batch_size=batch_size, shuffle=True)
        # eval_clf_data = F_DataIterSep(eval_data_file, eval_label_file, self.content_embeddings_data, batch_size=batch_size, shuffle=True)
        clf_data = content_iter
        eval_clf_data = content_iter_eval
        test_clf_data = content_iter_test

        class_weight = clf_data.get_class_weight()
        if len(class_weight) == 2:
            class_weight = torch.from_numpy(np.array([class_weight[1], class_weight[0]])).float().to('cuda')
            print("class_weight", class_weight)
        else:
            class_weight = None

        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        
        self.log.info('Loaded classifier training and testing data: \
                    train:{}, test:{}'.format(len(clf_data.loader.dataset), len(eval_clf_data.loader.dataset)))

        best_loss = 9999
        best_clf = None
        not_improve_num = 0
        stop = False
        for epoch in range(epochs):
            try:
                c_loss, c_acc = self.train_clf_epoch(clf, clf_data.get_loader(batch_size), criterion, clf_opt)
            except KeyboardInterrupt:
                print("End training. KeyboardInterrupt!")
                stop = True

            if cfg.flg_adv_f:
                f_loss_eval, f_acc_eval, f_acc_eval_true = self.eval_generator_with_f_all(self.F_clf, self.content_iter_test.get_loader(cfg.batch_size), multiplier=999999, attack_mode=cfg.attack_mode)
                print("Attack Performance:{} | {} | {}".format(f_loss_eval, f_acc_eval, f_acc_eval_true))

            eval_loss, eval_acc, _, _ = self.eval_clf(clf, eval_clf_data.get_loader(batch_size), criterion)
            if eval_loss < best_loss:
                best_clf = copy.deepcopy(clf.state_dict())  # save the best classifier
                best_loss = eval_loss
                not_improve_num = 0
            else:
                not_improve_num += 1

            self.log.info('[PRE-CLAS] epoch %d: c_loss = %.4f, c_acc = %.4f, eval_loss = %.4f, eval_acc = %.4f, min_eval_loss = %.4f',
                          epoch, c_loss, c_acc, eval_loss, eval_acc, best_loss)
            # visualize_append_line(eval_loss, eval_acc, "{}-PRETRAIN-DEV".format(prefix))

            # test_loss, test_acc, preds, labels = self.eval_clf(clf, test_clf_data.get_loader(batch_size), criterion)
            # print(classification_report(labels, preds))

            if not_improve_num >= patience_epoch or stop:
                self.log.info('[PRE-CLAS] Early Stopped')
                break

        clf.load_state_dict(copy.deepcopy(best_clf))  # Reload the best classifier
        
        test_loss, test_acc, _, _ = self.eval_clf(clf, test_clf_data.get_loader(cfg.f_batch_size), criterion)
        self.log.info('[PRE-CLAS] test_loss = %.4f, test_acc = %.4f', test_loss, test_acc)

        if save:
            save_path = "{}.{:.4f}B".format(pretrained_path, best_loss)
            torch.save(best_clf, save_path)
            self.log.info('[PRE=CLAS] SAVED classifier to: {}'.format(save_path))


    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        total_time = []

        # mle_train_data = GenDataIter(cfg.train_data, batch_size=cfg.mle_batch_size)
        # mle_test_data = GenDataIter(cfg.test_data, batch_size=cfg.mle_batch_size, if_test_data=True)  

        # mle_train_data = GenDataIterContent(cfg.train_data_content, self.content_embeddings_data, batch_size=cfg.mle_batch_size, shuffle=True)
        # mle_test_data = GenDataIterContent(cfg.test_data_content, batch_size=cfg.mle_batch_size)  

        progress = tqdm(range(epochs))
        for epoch in progress:
            self.sig.update()
            if self.sig.pre_sig:
                # ===Train===
                start_time = time.time()
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, 
                                                self.mle_criterion, self.gen_opt,
                                                content_iter=self.content_iter)
                end_time = time.time()
                total_time.append(end_time - start_time)

                progress.set_description("epoch %d : pre_loss = %.4f" % (epoch, pre_loss))
                # ===Test===
                if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                    metric_str, metric_scores = self.cal_metrics(self.train_data, self.test_data)
                    self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f, %s | AvgTime: %.4f' % (
                        epoch, pre_loss, metric_str, np.mean(total_time)))

                    # visualize_scores(metric_scores)

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch, self.test_data)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break


    def adv_train_generator_topic(self, g_step):
        total_loss = 0
        for step in range(g_step):
            batch = self.train_data.random_batch()
            contents = batch['content']
            real_samples = batch['target']
            gen_samples = self.gen.sample(len(real_samples), len(real_samples), one_hot=True, content_iter=batch)
            if cfg.CUDA:
                real_samples, gen_samples, contents = real_samples.to('cuda'), gen_samples.to('cuda'), contents.to('cuda')
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.dis_topic(contents, real_samples)
            d_out_fake = self.dis_topic(contents, gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0


    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            batch = self.train_data.random_batch()
            real_samples = batch['target']
            gen_samples = self.gen.sample(len(real_samples), len(real_samples), one_hot=True, content_iter=batch)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.to('cuda'), gen_samples.to('cuda')
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0


    def prepare_data_clf(self, data):
        inp, comments, label = data['content'], data['comments'], data['label']
        num_comments = data['num_comment']
        new_comments = []
        for comment in comments:
            new_comments.append(F.one_hot(comment, cfg.vocab_size).float())
        comments = torch.stack(new_comments)
        return inp, comments, label, num_comments


    def get_label_for_attack(self, batch_size, attack_mode):
        # label = torch.zeros(batch_size).long() #default label is to target_real
        if attack_mode == "target_fake":
            label = torch.ones(batch_size).long()
        elif attack_mode == "target_real":
            label = torch.zeros(batch_size).long()
        return label

    def get_idx_of_label(self, label, attack_mode):
        if attack_mode == "target_fake":
            idx = torch.where(label == 0) #reverse with get_label_for_attack
        elif attack_mode == "target_real":
            idx = torch.where(label == 1) #reverse with get_label_for_attack
        return idx

    def eval_geneartor_with_f_all_single(self, model, random_batch, attack_mode, only_generated_comment, 
                                        return_all=False, cal_true=True, eval_num=1, show_sentence=False):
        content = random_batch['content']
        current_comments = random_batch['comments'] # batch_size * 3 * 20
        num_comment = random_batch['num_comment'] # 64, 3, 20
        current_labels = random_batch['label']

        gen_samples = []
        for i in range(cfg.atk_num_comment):
            if eval_num>1:
                gen_sample, _, _ = self.generate_best_topic_batch(random_batch, custom_model=self.topic_model, 
                                one_hot=True, translate=False, eval_num=eval_num)
            else:
                gen_sample = self.gen.sample(len(content), len(content), one_hot=True, content_iter=random_batch)
            gen_samples.append(gen_sample.unsqueeze(1))
        gen_samples = torch.cat(gen_samples, 1) #batch_size * atk_num_comment * (max_seq_len*vocab_size)
        # print("gen_samples", gen_samples.size())

        label = self.get_label_for_attack(gen_samples.size()[0], attack_mode)

        if cfg.CUDA:
            gen_samples, label, content = gen_samples.to('cuda'), label.to('cuda'), content.to('cuda')
            num_comment = num_comment.to('cuda')

        if not only_generated_comment:
            all_comments = self.prepare_data_adv_clf(current_comments, num_comment, gen_samples, max_comment=cfg.max_num_comment_test)
            # print("all_comments", all_comments.size())
            pred = model.forward(content, all_comments, num_comment)   
        else:
            pred = model.forward(content, gen_samples.unsqueeze(1))

        loss = self.cross_entropy_loss(pred, label).item()
        acc = torch.sum((pred.argmax(dim=-1) == label)).item()/gen_samples.size()[0]

        acc_true = 0.0
        original_preds = None
        if cal_true:
            current_comments = F.one_hot(current_comments, cfg.vocab_size).float()
            original_preds = self.predict_clf(model, content, current_comments, num_comment).cpu()
            idx = torch.where(current_labels == original_preds.argmax(dim=-1))[0]
            if len(idx) > 0:
                acc_true = torch.sum((pred.argmax(dim=-1)[idx] == label[idx])).item()/label[idx].size()[0]
            else:
                acc_true = torch.tensor(0)
            original_preds = original_preds.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()

        if show_sentence:
            out = ""
            current_comments = random_batch['comments']
            gen_samples = gen_samples.argmax(3)
            # print(current_comments[0].size())
            # print(gen_samples.size())
            # print(gen_samples[0].size())
            for i in range(len(content)):
                current_comments_str = tensor_to_tokens(current_comments[i], self.idx2word_dict)
                # print(current_comments_str[0])
                generated_comments_str = tensor_to_tokens(gen_samples[i], self.idx2word_dict)
                # print(generated_comments_str[0])
                out += "\nTitle:  {}\n".format(random_batch['title_str'][i])
                out += "ID:   {}\n".format(random_batch['id'][i])
                for j in range(len(current_comments_str)):
                    out += "User: {}\n".format(" ".join(current_comments_str[j]))
                for j in range(len(generated_comments_str)):
                    out += "Malcom: {}\n".format(" ".join(generated_comments_str[j]))
                out += "True Label:{}\n".format(current_labels[i])
                out += "Before Attack: {}\n".format("Real" if np.argmax(original_preds[i]) == 0 else "Fake")
                out += "After Attack: {}\n".format("Real" if np.argmax(pred[i]) == 0 else "Fake")
            print(out)

        if not return_all:
            return loss, acc, acc_true
        else:
            return loss, acc, original_preds, pred, gen_samples


    def eval_generator_with_f_all(self, model, content_iter_eval_loader, 
                                    attack_mode="target_real", only_generated_comment=False, 
                                    multiplier=99999, return_all=False, cal_true=True, 
                                    eval_num=1, show_progress=False, show_sentence=False):
        total_loss = []
        total_acc = []
        total_acc_true = []
        total_loss_success = []
        total_acc_success = []

        model.eval()

        bar = tqdm(enumerate(content_iter_eval_loader)) if show_progress else enumerate(content_iter_eval_loader)

        for i, random_batch in bar:
            loss, acc, acc_true = self.eval_geneartor_with_f_all_single(model, random_batch, attack_mode, only_generated_comment, 
                                                                        cal_true=cal_true, eval_num=eval_num, show_sentence=show_sentence)
            total_loss_success.append(loss)
            total_acc_success.append(acc)
            total_acc_true.append(acc_true)

            multiplier -= 1
            if multiplier < 0:
                break

        model.train()

        return np.mean(total_loss_success), np.mean(total_acc_success), np.mean(total_acc_true)

    def eval_generator_with_f(self, model, content_iter_eval, multiplier=5, attack_mode="target_real", only_generated_comment=False):
        loss = 0
        acc = 0
        with torch.no_grad():
            contents = []
            current_comments_s = []
            num_comments = []
            current_labels = []
            gen_samples = []
            for i in range(multiplier):
                random_batch = content_iter_eval.random_batch(cfg.f_batch_size)
                content = random_batch['content']
                current_comments = random_batch['comments'] # batch_size * 3 * 20
                num_comment = random_batch['num_comment'] # 64, 3, 20
                contents.append(content)
                num_comments.append(num_comment)
                current_comments_s.append(current_comments)
                gen_samples.append(self.gen.sample(len(content), 
                                        len(content), 
                                        one_hot=False, content_iter=random_batch).float())
            content = torch.cat(contents, 0)
            num_comment = torch.cat(num_comments, 0)
            gen_samples = torch.cat(gen_samples, 0)
            current_comments = torch.cat(current_comments_s, 0)
            
            # content = F.one_hot(content, cfg.vocab_size).float()
            label = self.get_label_for_attack(gen_samples.size()[0], attack_mode)

            if cfg.CUDA:
                gen_samples, label = gen_samples.to('cuda'), label.to('cuda')
                content = content.to('cuda')
                num_comment = num_comment.to('cuda')

            if not only_generated_comment:
                all_comments = self.prepare_data_adv_clf(current_comments, num_comment, gen_samples, max_comment=cfg.max_num_comment_test)
                pred = model.forward(content, all_comments, num_comment)
            else:
                pred = model.forward(content, gen_samples.unsqueeze(1))
            loss = self.cross_entropy_loss(pred, label).item() # we want this to as big as possible
            acc = torch.sum((pred.argmax(dim=-1) == label)).item()/gen_samples.size()[0]

        return loss, acc

    def prepare_data_adv_clf(self, current_comments, num_comment, gen_samples, max_comment):
        # iterate through each sample (news), add generated comment to the last, 
        # keeping up to max_num_comment_test -1 previous comments
        # gen_samples: batch_size * max_seq_len * vocab_size
        # max_comment cfg.max_num_comment_test

        # current_comments = batch_size * max_num_comment * max_seq_len
        if not cfg.flg_eval_baseline:
            all_comments = []
            for i in range(gen_samples.size()[0]):
                total_comments = []
                for j in range(min(num_comment[i].item(), max_comment-cfg.atk_num_comment)):
                    comment = current_comments[i:i+1,j:j+1,:].squeeze(1) # 1 * 20
                    comment = F.one_hot(comment, cfg.vocab_size).float() # 1 * 20 * vocab_size
                    if gen_samples.is_cuda:
                        comment = comment.to('cuda')
                    total_comments.append(comment)

                if cfg.atk_num_comment>0:
                    for j in range(cfg.atk_num_comment):
                        total_comments.append(gen_samples[i:i+1,j,:,:])

                # pad zero comment if total # of comments is less than max_num_comment_test
                for k in range(max_comment - len(total_comments)):
                    zero_comment = torch.zeros((1, cfg.max_seq_len)).long()
                    zero_comment = F.one_hot(zero_comment, cfg.vocab_size).float() # 1 * 26 * vocab_size
                    if gen_samples.is_cuda:
                        zero_comment = zero_comment.to('cuda')
                    total_comments.append(zero_comment)

                total_comments = torch.stack(total_comments)
                all_comments.append(total_comments)
            all_comments = torch.stack(all_comments).squeeze(2)

        else:
            all_comments = []
            for i in range(gen_samples.size()[0]):
                total_comments = []
                for j in range(min(num_comment[i].item(), max_comment)):
                    comment = current_comments[i:i+1,j:j+1,:].squeeze(1) # 1 * 20
                    comment = F.one_hot(comment, cfg.vocab_size).float() # 1 * 20 * vocab_size
                    if gen_samples.is_cuda:
                        comment = comment.to('cuda')
                    total_comments.append(comment) 

                # pad zero comment if total # of comments is less than max_num_comment
                for k in range(max_comment - len(total_comments)):
                    zero_comment = torch.zeros(total_comments[0].size()).float()
                    if gen_samples.is_cuda:
                        zero_comment = zero_comment.to('cuda')
                    total_comments.append(zero_comment)
                all_comments.append(torch.stack(total_comments))
            all_comments = torch.stack(all_comments).squeeze(2)

        # new_comments = []
        # for comment in current_comments:
        #     new_comments.append(F.one_hot(comment, cfg.vocab_size).float())
        # all_comments = torch.stack(new_comments)

        return all_comments

    def reverse_label(self, label):
        label = 1 - label
        return label

    def predict_clf(self, model, inp, comments, num_comments):
        if next(model.parameters()).is_cuda:
            inp = inp.to('cuda')
            comments = comments.to('cuda')
            num_comments = num_comments.to('cuda')
        pred = F.softmax(model.forward(inp, comments, num_comments), dim=-1)
        # pred = pred.argmax(dim=-1)
        return pred


    def adv_train_generator_with_f(self, model, steps, content_iter, alpha, batch_size, criterion, gen_adv_opt, attack_mode="target_real", only_generated_comment=False):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for step in range(steps):
            random_batch = content_iter.random_batch(batch_size)
            content = random_batch['content']
            current_comments = random_batch['comments'] # batch_size * 3 * 20
            num_comment = random_batch['num_comment'] # 64, 3, 20
            gen_samples = self.gen.sample(len(content), len(content), one_hot=True, content_iter=random_batch).float() # 64, 20, 5001
            gen_samples = gen_samples.unsqueeze(1)

            label = self.get_label_for_attack(gen_samples.size()[0], attack_mode)
            if cfg.CUDA:
                gen_samples, label, content = gen_samples.to('cuda'), label.to('cuda'), content.to('cuda')
                num_comment = num_comment.to('cuda')

            model.train()
            if not only_generated_comment:
                all_comments = self.prepare_data_adv_clf(current_comments, num_comment, gen_samples, max_comment=cfg.max_num_comment_test)
                pred = model.forward(content, all_comments, num_comment)
            else:
                gen_samples = gen_samples.unsqueeze(1) # batch_size * 1 * seq_len * vocab_size
                pred = model.forward(content, gen_samples, num_comment)

            clf_loss = criterion(pred, label)
            total_loss += clf_loss.item() 
            clf_loss = alpha*clf_loss
            self.optimize(gen_adv_opt, clf_loss, self.gen, clip_grad=True, show_grad=False)
            total_acc += torch.sum((pred.argmax(dim=-1) == label)).item()
            total_num += gen_samples.size(0)

        avg_loss = total_loss / steps if steps != 0 else 0
        avg_acc = total_acc / total_num
        return avg_loss, avg_acc


    def topic_eval_discriminator(self, data_iter):
        total_loss = []
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                contents = batch['content']
                real_samples = batch['target']
                if cfg.CUDA:
                    contents, real_samples = contents.to('cuda'), real_samples.to('cuda')
                real_samples = F.one_hot(real_samples, cfg.vocab_size).float()
                batch_size = len(contents)

                 # ===Train===
                # contents_emb, emb = self.topic_encoder(contents, real_samples)
                emb = self.topic_encoder(real_samples)
                loss = self.mmd(contents.view(batch_size, -1), emb.view(batch_size, -1))
                total_loss.append(loss.item())

        return np.mean(total_loss)


    def topic_train_discriminator(self, steps):
        total_loss = 0
        for step in range(steps):
            batch = self.train_data.random_batch()
            contents = batch['content']
            real_samples = batch['target']
            if cfg.CUDA:
                contents, real_samples = contents.to('cuda'), real_samples.to('cuda')
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()
            batch_size = len(contents)

             # ===Train===
            # contents_emb, emb = self.topic_encoder(contents, real_samples)
            emb = self.topic_encoder(real_samples)
            loss = self.mmd(contents.view(batch_size, -1), emb.view(batch_size, -1))
            self.optimize(self.topic_encoder_opt, loss, self.topic_encoder)
            total_loss += loss.item()

        return total_loss / steps if steps != 0 else 0



    def topic_eval_generator_comment(self, data_iter):
        total_loss = []
        with torch.no_grad():
            for i, batch in enumerate(data_iter.get_loader(cfg.h_batch_size)):
                contents = batch['content']
                titles = batch['title']
                titles = F.one_hot(titles, cfg.vocab_size).float()
                gen_samples = self.gen.sample(len(contents), len(contents), one_hot=True, content_iter=batch)
                if cfg.CUDA:
                    gen_samples = gen_samples.to('cuda')
                    titles = titles.to('cuda')
                batch_size = len(contents)

                 # ===Train===
                loss = self.mmd(titles.view(batch_size, -1), gen_samples.view(batch_size, -1))
                total_loss.append(loss.item())

        return np.mean(total_loss)


    def topic_train_generator_comment(self, steps):
        total_loss = 0
        for step in range(steps):
            batch = self.train_data.random_batch(cfg.h_batch_size)
            contents = batch['content']
            titles = batch['title']
            titles = F.one_hot(titles, cfg.vocab_size).float()
            gen_samples = self.gen.sample(len(contents), len(contents), one_hot=True, content_iter=batch)
            if cfg.CUDA:
                gen_samples = gen_samples.to('cuda')
                titles = titles.to('cuda')
            batch_size = len(contents)

             # ===Train===
            loss = self.mmd(titles.view(batch_size, -1), gen_samples.view(batch_size, -1))
            self.optimize(self.topic_encoder_opt_gen, loss, self.gen)
            total_loss += loss.item()

        return total_loss / steps if steps != 0 else 0

    def topic_train_generator(self, steps):
        total_loss = 0
        for step in range(steps):
            batch = self.train_data.random_batch()
            contents = batch['content']
            gen_samples = self.gen.sample(len(contents), len(contents), one_hot=True, content_iter=batch)
            if cfg.CUDA:
                contents = contents.to('cuda')
                gen_samples = gen_samples.to('cuda')
            batch_size = len(contents)

             # ===Train===
            # contents_emb, emb = self.topic_encoder(contents, gen_samples)
            emb = self.topic_encoder(gen_samples)
            loss = self.mmd(contents.view(batch_size, -1), emb.view(batch_size, -1))
            self.optimize(self.topic_encoder_opt_gen, loss, self.gen)
            total_loss += loss.item()

        return total_loss / steps if steps != 0 else 0


    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            batch = self.train_data.random_batch()
            real_samples = batch['target']
            gen_samples = self.gen.sample(len(real_samples), len(real_samples), one_hot=True, content_iter=batch)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.to('cuda'), gen_samples.to('cuda')
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)
        # visualize_temperature(self.gen.temperature, update_dashboard=True)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False, clip_grad=True, show_grad=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if show_grad:
            all_grads = []
            for param in model.parameters():
                if param.requires_grad:
                    all_grads.append(torch.sum(param.grad))
            print(torch.sum(all_grads))
        if model is not None and clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()