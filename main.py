# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
from __future__ import print_function

import argparse

import config as cfg

from utils.text_process import load_dict
from utils.text_process import load_test_dict
from utils.text_process import text_process
from utils.text_process import text_process_seq_len

def program_config(parser):
    # Program
    parser.add_argument('--if_train_defend', default=cfg.if_train_defend, type=int)
    parser.add_argument('--if_eval', default=cfg.if_eval, type=int)
    parser.add_argument('--if_gen', default=cfg.if_gen, type=int)
    parser.add_argument('--if_robust', default=cfg.if_robust, type=int)
    parser.add_argument('--if_save', default=cfg.if_save, type=int)
    parser.add_argument('--if_train_val', default=cfg.if_train_val, type=int)
    parser.add_argument('--if_show_sentence', default=cfg.if_show_sentence, type=int)
    parser.add_argument('--if_trigger_retrain', default=cfg.if_trigger_retrain, type=int)
    parser.add_argument('--if_textbugger_retrain', default=cfg.if_textbugger_retrain, type=int)
    parser.add_argument('--if_hotflip_retrain', default=cfg.if_hotflip_retrain, type=int)
    parser.add_argument('--flg_adv_d', default=cfg.flg_adv_d, type=int)
    parser.add_argument('--flg_eval_g', default=cfg.flg_eval_g, type=int)
    parser.add_argument('--flg_eval_pre', default=cfg.flg_eval_pre, type=int)
    parser.add_argument('--flg_eval_copycat', default=cfg.flg_eval_copycat, type=int)
    parser.add_argument('--flg_eval_hotflip', default=cfg.flg_eval_hotflip, type=int)
    parser.add_argument('--flg_eval_unitrigger', default=cfg.flg_eval_unitrigger, type=int)
    parser.add_argument('--flg_eval_textbugger', default=cfg.flg_eval_textbugger, type=int)
    parser.add_argument('--flg_eval_baseline', default=cfg.flg_eval_baseline, type=int)
    parser.add_argument('--if_eval_robust', default=cfg.if_eval_robust, type=int)
    parser.add_argument('--trigger_neighbor', default=cfg.trigger_length, type=int)
    parser.add_argument('--trigger_length', default=cfg.trigger_neighbor, type=int)
    parser.add_argument('--trigger_beam_size', default=cfg.trigger_beam_size, type=int)
    parser.add_argument('--attack_ratio', default=cfg.attack_ratio, type=float)
    parser.add_argument('--spelling_thres', default=cfg.spelling_thres, type=float)
    parser.add_argument('--coherency_thres', default=cfg.coherency_thres, type=float)

    parser.add_argument('--if_predict', default=cfg.if_predict, type=int)
    parser.add_argument('--if_monit', default=cfg.if_monit, type=int)
    parser.add_argument('--if_old_version', default=cfg.if_old_version, type=int)
    parser.add_argument('--if_eval_score', default=cfg.if_eval_score, type=int)
    parser.add_argument('--if_pretrain_embedding', default=cfg.if_pretrain_embedding, type=int)
    parser.add_argument('--eval_g_path', default=cfg.eval_g_path, type=str)
    parser.add_argument('--eval_style_path', default=cfg.eval_style_path, type=str)
    parser.add_argument('--eval_pre_attack_path', default=cfg.eval_pre_attack_path, type=str)
    parser.add_argument('--attack_mode', default=cfg.attack_mode, type=str)
    parser.add_argument('--attack_f_path', default="", type=str)
    parser.add_argument('--attack_h_path', default="", type=str)
    parser.add_argument('--pretrained_gen_path', default="", type=str)
    parser.add_argument('--blackbox_using_path', default=cfg.blackbox_using_path, type=str)
    parser.add_argument('--eval_outfile', default=cfg.eval_outfile, type=str)
    parser.add_argument('--eval_num', default=cfg.eval_num, type=int)
    parser.add_argument('--random_num_article', default=cfg.random_num_article, type=int)
    parser.add_argument('--max_num_eval', default=cfg.samples_num2, type=int)
    parser.add_argument('--num_topics', default=cfg.num_topics, type=int)
    parser.add_argument('--eval_topic_num', default=cfg.eval_topic_num, type=int)
    parser.add_argument('--atk_num_comment', default=cfg.atk_num_comment, type=int)
    parser.add_argument('--F_train_epoch', default=cfg.F_train_epoch, type=int)
    parser.add_argument('--F_patience_epoch', default=cfg.f_patience_epoch, type=int)


    parser.add_argument('--topic_epochs', default=cfg.topic_epochs, type=int)
    parser.add_argument('--adv_topic_step', default=cfg.ADV_topic_step, type=int)
    parser.add_argument('--mmd_mean', default=cfg.mmd_mean, type=int)
    parser.add_argument('--mmd_kernel_num', default=cfg.mmd_kernel_num, type=int)
    parser.add_argument('--topic_lr', default=cfg.topic_lr, type=float)
    parser.add_argument('--topic_gen_lr', default=cfg.topic_gen_lr, type=float)
    parser.add_argument('--flg_adv_t', default=cfg.flg_adv_t, type=int)

    parser.add_argument('--if_test', default=cfg.if_test, type=int)
    parser.add_argument('--run_model', default=cfg.run_model, type=str)
    parser.add_argument('--root_folder', default=cfg.root_folder, type=str)
    parser.add_argument('--k_label', default=cfg.k_label, type=int)
    parser.add_argument('--dataset', default=cfg.dataset, type=str)
    parser.add_argument('--model_type', default=cfg.model_type, type=str)
    parser.add_argument('--loss_type', default=cfg.loss_type, type=str)
    parser.add_argument('--if_real_data', default=cfg.if_real_data, type=int)
    parser.add_argument('--cuda', default=cfg.CUDA, type=int)
    parser.add_argument('--device', default=cfg.device, type=int)
    parser.add_argument('--shuffle', default=cfg.data_shuffle, type=int)
    parser.add_argument('--gen_init', default=cfg.gen_init, type=str)
    parser.add_argument('--dis_init', default=cfg.dis_init, type=str)
    parser.add_argument('--max_num_data', default=cfg.max_num_data, type=int)
    parser.add_argument('--max_num_comment', default=cfg.max_num_comment, type=int)
    parser.add_argument('--max_num_comment_test', default=cfg.max_num_comment_test, type=int)
    parser.add_argument('--max_test_comment', default=cfg.max_test_comment, type=int)

    # Basic Train
    parser.add_argument('--samples_num', default=cfg.samples_num, type=int)
    parser.add_argument('--vocab_size', default=cfg.vocab_size, type=int)
    parser.add_argument('--mle_epoch', default=cfg.MLE_train_epoch, type=int)
    parser.add_argument('--clas_pre_epoch', default=cfg.PRE_clas_epoch, type=int)
    parser.add_argument('--adv_epoch', default=cfg.ADV_train_epoch, type=int)
    parser.add_argument('--inter_epoch', default=cfg.inter_epoch, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--t_batch_size', default=cfg.t_batch_size, type=int)
    parser.add_argument('--mle_batch_size', default=cfg.mle_batch_size, type=int)
    parser.add_argument('--max_seq_len', default=cfg.max_seq_len, type=int)
    parser.add_argument('--start_letter', default=cfg.start_letter, type=int)
    parser.add_argument('--padding_idx', default=cfg.padding_idx, type=int)
    parser.add_argument('--gen_lr', default=cfg.gen_lr, type=float)
    parser.add_argument('--gen_adv_lr', default=cfg.gen_adv_lr, type=float)
    parser.add_argument('--dis_lr', default=cfg.dis_lr, type=float)
    parser.add_argument('--clip_norm', default=cfg.clip_norm, type=float)
    parser.add_argument('--pre_log_step', default=cfg.pre_log_step, type=int)
    parser.add_argument('--adv_log_step', default=cfg.adv_log_step, type=int)
    parser.add_argument('--train_data', default=cfg.train_data, type=str)
    parser.add_argument('--test_data', default=cfg.test_data, type=str)
    parser.add_argument('--temp_adpt', default=cfg.temp_adpt, type=str)
    parser.add_argument('--temperature', default=cfg.temperature, type=int)
    parser.add_argument('--ora_pretrain', default=cfg.oracle_pretrain, type=int)
    parser.add_argument('--gen_pretrain', default=cfg.repeat_gen_pretrain, type=int)
    parser.add_argument('--load_gen_pretrain', default=cfg.load_gen_pretrain, type=int)
    parser.add_argument('--dis_pretrain', default=cfg.dis_pretrain, type=int)
    parser.add_argument('--flg_train_f', default=cfg.flg_train_f, type=int)
    parser.add_argument('--flg_train_h', default=cfg.flg_train_h, type=int)
    parser.add_argument('--flg_adv_f', default=cfg.flg_adv_f, type=int)
    parser.add_argument('--flg_adv_h', default=cfg.flg_adv_h, type=int)
    parser.add_argument('--train_with_content', default=cfg.train_with_content, type=int)


    # Generator
    parser.add_argument('--adv_g_step', default=cfg.ADV_g_step, type=int)
    parser.add_argument('--rollout_num', default=cfg.rollout_num, type=int)
    parser.add_argument('--gen_embed_dim', default=cfg.gen_embed_dim, type=int)
    parser.add_argument('--gen_hidden_dim', default=cfg.gen_hidden_dim, type=int)
    parser.add_argument('--goal_size', default=cfg.goal_size, type=int)
    parser.add_argument('--step_size', default=cfg.step_size, type=int)
    parser.add_argument('--mem_slots', default=cfg.mem_slots, type=int)
    parser.add_argument('--num_heads', default=cfg.num_heads, type=int)
    parser.add_argument('--head_size', default=cfg.head_size, type=int)

    parser.add_argument('--g_f_step', default=cfg.g_f_step, type=int)
    parser.add_argument('--g_f_alpha', default=cfg.g_f_alpha, type=float)
    parser.add_argument('--gen_adv_f_lr', default=cfg.gen_adv_f_lr, type=float)
    parser.add_argument('--g_h_step', default=cfg.g_h_step, type=int)
    parser.add_argument('--g_h_alpha', default=cfg.g_h_alpha, type=float)
    parser.add_argument('--if_adapt_alpha', default=cfg.if_adapt_alpha, type=int)
    parser.add_argument('--f_batch_size', default=cfg.f_batch_size, type=int)
    parser.add_argument('--f_embed_dim', default=cfg.f_embed_dim, type=int)
    parser.add_argument('--f_dropout_comment', default=cfg.f_dropout_comment, type=float)
    parser.add_argument('--f_dropout_content', default=cfg.f_dropout_content, type=float)
    parser.add_argument('--F_clf_lr', default=cfg.F_clf_lr, type=float)
    parser.add_argument('--F_feature_dim', default=cfg.F_feature_dim, type=int)
    

    parser.add_argument('--h_batch_size', default=cfg.h_batch_size, type=int)

    parser.add_argument('--f_type', default=cfg.f_type, type=str)


    # Discriminator
    parser.add_argument('--d_step', default=cfg.d_step, type=int)
    parser.add_argument('--d_epoch', default=cfg.d_epoch, type=int)
    parser.add_argument('--adv_d_step', default=cfg.ADV_d_step, type=int)
    parser.add_argument('--adv_d_epoch', default=cfg.ADV_d_epoch, type=int)
    parser.add_argument('--dis_embed_dim', default=cfg.dis_embed_dim, type=int)
    parser.add_argument('--dis_hidden_dim', default=cfg.dis_hidden_dim, type=int)
    parser.add_argument('--num_rep', default=cfg.num_rep, type=int)

    # Metrics
    parser.add_argument('--use_nll_oracle', default=cfg.use_nll_oracle, type=int)
    parser.add_argument('--use_nll_gen', default=cfg.use_nll_gen, type=int)
    parser.add_argument('--use_nll_div', default=cfg.use_nll_div, type=int)
    parser.add_argument('--use_bleu', default=cfg.use_bleu, type=int)
    parser.add_argument('--use_self_bleu', default=cfg.use_self_bleu, type=int)
    parser.add_argument('--use_clas_acc', default=cfg.use_clas_acc, type=int)
    parser.add_argument('--use_ppl', default=cfg.use_ppl, type=int)

    # Log
    parser.add_argument('--log_file', default=cfg.log_filename, type=str)
    parser.add_argument('--save_root', default=cfg.save_root, type=str)
    parser.add_argument('--signal_file', default=cfg.signal_file, type=str)
    parser.add_argument('--tips', default=cfg.tips, type=str)
    parser.add_argument('--visdom_ip', default="207.148.26.49", type=str)
    parser.add_argument('--user', default=cfg.user, type=str)

    return parser


# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser = program_config(parser)
    opt = parser.parse_args()
    word2idx = None

    # opt.max_seq_len, opt.vocab_size = text_process('dataset/' + opt.dataset + '.txt', opt.vocab_size)
    word2idx, idx2word = load_dict(opt.dataset, opt.vocab_size, use_external_file="{}_all".format(opt.dataset))
    opt.vocab_size = len(word2idx)+1
    print("!vocab_size", opt.vocab_size)
    opt.max_seq_len = text_process_seq_len("{}_all".format(opt.dataset))
    # opt.max_seq_len = opt.max_seq_len*2 # changed, double the size of potential max len
    word2idx_all, idx2word_all = load_test_dict(opt.dataset, opt.vocab_size)
    cfg.extend_vocab_size = len(word2idx_all)
    # opt.vocab_size = cfg.extend_vocab_size
    print("!extend_vocab_size", cfg.extend_vocab_size)

    cfg.init_param(opt)
    opt.save_root = cfg.save_root

    # ===Dict===
    from instructor.real_data.relgan_instructor import RelGANInstructor

    if not cfg.if_robust:
        inst = RelGANInstructor(cfg)

    if cfg.if_eval:
        print("[[Evaluation Mode]]")
        inst._eval()
    elif cfg.if_predict:
        print("[[Prediction Mode]]")
        inst._predict()
    elif cfg.if_gen:
        print("[[Generation Mode]]")
        inst._gen()
    elif cfg.if_robust:
        print("[[Evaluating with Robust Mode]]")
        from instructor.real_data.relgan_robust import RelGANRobust
        topic_files = []
        if cfg.num_topics < 0:
            for i in range(cfg.min_topic[cfg.dataset], cfg.max_topic[cfg.dataset]+1):
                topic_files.append('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(i))
        else:
            topic_files.append('dataset/' + cfg.dataset + '_topic_{}.pkl'.format(cfg.num_topics))

        robust_runner = RelGANRobust(cfg.dataset, topic_files, idx2word, word2idx)
        robust_runner._run()
    else:
        print("[[training Mode]]")
        inst._run()
        