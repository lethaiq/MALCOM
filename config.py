# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
import time
from time import strftime, localtime

import os
import re
import torch

# ===Program===
if_test = False
if_eval = False
if_gen = False
if_robust = False
if_monit = False
if_old_version = False
if_eval_score = True
if_flex_com = False
if_train_defend = False
if_predict = False
if_train_val = False
if_eval_robust = False
if_show_sentence = False
if_accept_nocomment = True
if_hotflip_retrain = False

min_topic = {}
max_topic = {}
min_topic['gossipcopWithContent20x'] = 7
max_topic['gossipcopWithContent20x'] = 16
min_topic['rumors20x'] = 9
max_topic['rumors20x'] = 12

flg_eval_g = False
flg_eval_pre = False
flg_eval_copycat = False
flg_eval_hotflip = False
flg_eval_unitrigger = False
trigger_length = 3
trigger_beam_size = 2
trigger_neighbor = 30
if_trigger_retrain = False
if_textbugger_retrain = False
flg_eval_textbugger = False
flg_eval_baseline = False

attack_ratio = 1.0

eval_outfile = "Quality"
eval_num = 1
load_gen_adv_path = None
load_dis_adv_path = None
eval_topic_num = 1
atk_num_comment = 1

start_index = {}
start_index['gossipcopWithContent20x'] = 1
start_index['rumors20x'] = 0

CUDA = True
if_save = True
data_shuffle = False  # False
oracle_pretrain = True  # True
# gen_pretrain = True
dis_pretrain = False
clas_pretrain = False
visdom_ip = "207.148.26.49"
eval_g_path = ""
eval_style_path = ""
eval_pre_attack_path = ""
attack_mode = "target_real"
user = "DefaultUser"
blackbox_using_path = ""

run_model = 'relgan'  # seqgan, leakgan, maligan, jsdgan, relgan, sentigan
k_label = 2  # num of labels, >=2
gen_init = 'truncated_normal'  # normal, uniform, truncated_normal
dis_init = 'truncated_normal'  # normal, uniform, truncated_normal

# ===Oracle or Real, type===
if_real_data = True  # if use real data
dataset = 'gossipcopWithContent20x'  # oracle, image_coco, emnlp_news, amazon_app_book, mr15
model_type = 'RMC'  # vanilla, RMC (custom)
f_type = 'LR' #can be LR, CNN, RNN
h_type = 'CNN' #can be LR, CNN, RNN
loss_type = 'rsgan'  # standard, JS, KL, hinge, tv, LS, rsgan (for RelGAN)
vocab_size = 10000  # oracle: 5000, coco: 6613, emnlp: 5255, amazon_app_book: 6418, mr15: 6289
max_seq_len = 26  # oracle: 20, coco: 37, emnlp: 51, amazon_app_book: 40
ADV_train_epoch = 5000  # SeqGAN, LeakGAN-200, RelGAN-3000
extend_vocab_size = 0  # plus test data, only used for Classifier

temp_adpt = 'exp'  # no, lin, exp, log, sigmoid, quad, sqrt
temperature = 1000 # largest temperature


# ===Special===
sample_limit = 10


F_clf_lr = 0.001
F_train_epoch = 50
F_feature_dim = 45
f_embed_dim = 16
f_patience_epoch = 3
f_batch_size = 64
f_dropout_comment = 0.5
f_dropout_content = 0.95

rnn_hidden_dim = 32

num_topics = 14

H_clf_lr = 0.001
H_train_epoch = 50
h_embed_dim = 16
h_patience_epoch = 2
h_batch_size = 64

t_batch_size = 16

gen_adv_f_lr = 1e-4
g_f_step = 1
g_f_alpha = 1.0 # alpha multiplier for loss backprobagation from F to optimize G
if_adapt_alpha = False

gen_adv_h_lr = 5e-4
g_h_step = 1
g_h_alpha = 0.0


max_num_data = 50000
min_input_sent_length = 3
max_num_comment = 10
max_num_comment_test = 1
max_test_comment = 2
unk_threshold = 2

spelling_thres = 1
coherency_thres = 0.1

# max_content_len = 150
repeat_gen_pretrain = int(True)
load_gen_pretrain = int(True)

train_with_content = True
flg_train_f = False
flg_train_h = False
flg_adv_f = False
flg_adv_h = False
if_pretrain_embedding = False
random_num_article = 3

# ===Basic Train===
samples_num = 10000  # 10000, mr15: 2000,
samples_num2 = 128
MLE_train_epoch = 150  # SeqGAN-80, LeakGAN-8, RelGAN-150
PRE_clas_epoch = 10
inter_epoch = 15  # LeakGAN-10
batch_size = 256  # 64
mle_batch_size = 256
start_letter = 1
padding_idx = 0
unk_idx = 2
start_token = 'BOS'
end_token = 'EOS'
padding_token = '_'
unk_token = '<unk>'
gen_lr = 0.01  # 0.01
gen_adv_lr = 1e-4  # RelGAN-1e-4
dis_lr = 1e-4  # SeqGAN,LeakGAN-1e-2, RelGAN-1e-4
clas_lr = 1e-3
clip_norm = 5.0

pre_log_step = 50
adv_log_step = 100

train_data = 'dataset/' + dataset + '.txt'
test_data = 'dataset/testdata/' + dataset + '_test.txt'
cat_train_data = 'dataset/' + dataset + '_cat{}.txt'
cat_test_data = 'dataset/testdata/' + dataset + '_cat{}_test.txt'
content_embeddings = 'dataset/' + dataset + '_content_embeddings.npz'
comment_embeddings = 'dataset/' + dataset + '_comment_embeddings.npz'
h_train_data_file = 'dataset/' + dataset + '_coherence_train_data.txt'
h_train_label_file = 'dataset/' + dataset + '_coherence_train_label.txt'
h_dev_data_file = 'dataset/' + dataset + '_coherence_val_data.txt'
h_dev_label_file = 'dataset/' + dataset + '_coherence_val_label.txt'
h_test_data_file = 'dataset/' + dataset + '_coherence_test_data.txt'
h_test_label_file = 'dataset/' + dataset + '_coherence_test_label.txt'
f_train_data_file = 'dataset/' + dataset + '_detect_train_data.txt'
f_train_label_file = 'dataset/' + dataset + '_detect_train_label.txt'
f_train_val_data_file = 'dataset/' + dataset + '_detect_train_val_data.txt'
f_train_val_label_file = 'dataset/' + dataset + '_detect_train_val_label.txt'
f_dev_data_file = 'dataset/' + dataset + '_detect_val_data.txt'
f_dev_label_file = 'dataset/' + dataset + '_detect_val_label.txt'
f_test_data_file = 'dataset/' + dataset + '_detect_test_data.txt'
f_test_label_file = 'dataset/' + dataset + '_detect_test_label.txt'
topic_file = 'dataset/' + dataset + '_topic.pkl'

# ===Metrics===
use_nll_oracle = False
use_nll_gen = True
use_nll_div = False
use_bleu = True
use_self_bleu = False
use_clas_acc = False
use_ppl = False

# ===Generator===
ADV_g_step = 1  # 1
rollout_num = 16  # 4
gen_embed_dim = 32  # 32
gen_hidden_dim = 32  # 32
goal_size = 16  # LeakGAN-16
step_size = 4  # LeakGAN-4

mem_slots = 1  # RelGAN-1
num_heads = 2  # RelGAN-2
head_size = 256  # RelGAN-256 #changed from 64 # needs to change content_feature_dim
content_feature_dim = head_size #need to change head_size


mmd_mean = 2
mmd_kernel_num = 2
topic_epochs = 500
topic_lr = 0.0001
topic_gen_lr = 0.0001
ADV_topic_step = 1
flg_adv_t = True
flg_adv_d = True

# ===Discriminator===
d_step = 5  # SeqGAN-50, LeakGAN-5
d_epoch = 3  # SeqGAN,LeakGAN-3
ADV_d_step = 5  # SeqGAN,LeakGAN,RelGAN-5
ADV_d_epoch = 3  # SeqGAN,LeakGAN-3
dis_embed_dim = 64 #changed
dis_hidden_dim = 64 #changed
num_rep = 64  # RelGAN


device = 0
# ===log===
log_time_str = strftime("%m%d_%H%M_%S", localtime())
log_filename = strftime("log/log_%s" % log_time_str)
if os.path.exists(log_filename + '.txt'):
    i = 2
    while True:
        if not os.path.exists(log_filename + '_%d' % i + '.txt'):
            log_filename = log_filename + '_%d' % i
            break
        i += 1
log_filename = log_filename + '.txt'

# # Automatically choose GPU or CPU
# if torch.cuda.is_available() and torch.cuda.device_count() > 0:
#     os.system('nvidia-smi -q -d Utilization > gpu')
#     with open('gpu', 'r') as _tmpfile:
#         util_gpu = list(map(int, re.findall(r'Gpu\s+:\s*(\d+)\s*%', _tmpfile.read())))
#     os.remove('gpu')
#     if len(util_gpu):
#         device = util_gpu.index(min(util_gpu))
#     else:
#         device = 0
# else:
#     device = -1
# # device=1
# # print('device: ', device)
# torch.cuda.set_device(device)


# ===Save Model and samples===
root_folder = "./"
save_root = root_folder + '/save/{}/{}/{}_{}_lt-{}_sl{}_temp{}_T{}/'.format(time.strftime("%Y%m%d"),
                                                             dataset, run_model, model_type,
                                                             loss_type, max_seq_len,
                                                             temperature,
                                                             log_time_str)
save_samples_root = save_root + 'samples/'
save_model_root = save_root + 'models/'

oracle_state_dict_path = 'pretrain/oracle_data/oracle_lstm.pt'
oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}.pt'
multi_oracle_state_dict_path = 'pretrain/oracle_data/oracle{}_lstm.pt'
multi_oracle_samples_path = 'pretrain/oracle_data/oracle{}_lstm_samples_{}.pt'

pretrain_root = root_folder+'/pretrain/{}/'.format(dataset if if_real_data else 'oracle_data')
pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_{}_{}_{}_sl{}_sn{}_hs{}_maxsn{}_vs{}_{}cmt.pt'.format(dataset, run_model, model_type,
                                                                                       max_seq_len, samples_num, head_size, max_num_data, vocab_size, max_num_comment)
pretrained_f_path = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment)
pretrained_f_path_CNN = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_CNN.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment)
pretrained_f_path_RNN = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_RNN.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
pretrained_h_path = pretrain_root + 'H_pretrain_{}_{}_sl{}_sn{}_vs{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                   samples_num, vocab_size)
pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}_sl{}_sn{}_vs{}.pt'.format(run_model, model_type, max_seq_len,
                                                                               samples_num, vocab_size)
pretrained_clas_path = pretrain_root + 'clas_pretrain_{}_{}_sl{}_sn{}_vs{}.pt'.format(run_model, model_type, max_seq_len,
                                                                               samples_num, vocab_size)
topic_encoder_path = pretrain_root + 'topic_encoder_pretrained_MMD{}-{}'.format(mmd_mean, mmd_kernel_num)

signal_file = 'run_signal.txt'

tips = ''


# Init settings according to parser
def init_param(opt):
    global run_model, model_type, loss_type, CUDA, device, data_shuffle, samples_num, vocab_size, \
        MLE_train_epoch, ADV_train_epoch, inter_epoch, mle_batch_size, batch_size, max_seq_len, start_letter, padding_idx, \
        gen_lr, gen_adv_lr, dis_lr, clip_norm, pre_log_step, adv_log_step, train_data, test_data, temp_adpt, \
        temperature, oracle_pretrain, load_gen_pretrain, repeat_gen_pretrain,  dis_pretrain, ADV_g_step, rollout_num, gen_embed_dim, \
        gen_hidden_dim, goal_size, step_size, mem_slots, num_heads, head_size, d_step, d_epoch, \
        ADV_d_step, ADV_d_epoch, dis_embed_dim, dis_hidden_dim, num_rep, log_filename, save_root, \
        signal_file, tips, save_samples_root, save_model_root, if_real_data, pretrained_gen_path, \
        pretrained_dis_path, pretrain_root, if_test, dataset, PRE_clas_epoch, oracle_samples_path, \
        pretrained_clas_path, gen_init, dis_init, multi_oracle_samples_path, k_label, cat_train_data, cat_test_data, \
        use_nll_oracle, use_nll_gen, use_nll_div, use_bleu, use_self_bleu, use_clas_acc, use_ppl, \
        max_num_data, flg_train_f, flg_train_h, flg_adv_f, flg_adv_h, f_batch_size, h_batch_size, \
        g_f_step, g_f_alpha, train_with_content, if_eval, eval_g_path, attack_mode, if_monit, \
        h_train_data_file, h_train_label_file, h_dev_data_file, h_dev_label_file, h_test_data_file, h_test_label_file, \
        f_train_data_file, f_train_label_file, f_dev_data_file, f_dev_label_file, content_embeddings, comment_embeddings, \
        pretrained_f_path, pretrained_h_path, user, max_num_comment, if_eval_score, if_adapt_alpha,\
        g_h_step, g_h_alpha, f_type, max_test_comment, content_feature_dim, if_pretrain_embedding, f_embed_dim, \
        f_dropout_comment, f_dropout_content, pretrained_f_path_RNN, if_old_version, eval_num, eval_outfile, \
        if_train_defend, if_predict, f_test_data_file, f_test_label_file, \
        mmd_mean, mmd_kernel_num, topic_epochs, topic_lr, topic_gen_lr, ADV_topic_step, flg_adv_t, flg_adv_d, \
        topic_file, samples_num2, num_topics, t_batch_size, if_save, max_num_comment_test, F_train_epoch, \
        if_train_val, f_train_val_data_file, f_train_val_label_file, f_patience_epoch, root_folder, eval_topic_num, \
        atk_num_comment, blackbox_using_path, flg_eval_g, flg_eval_pre, flg_eval_hotflip, flg_eval_copycat, \
        if_show_sentence, random_num_article, flg_eval_unitrigger, trigger_length, trigger_beam_size, \
        trigger_neighbor, if_trigger_retrain, flg_eval_textbugger, if_textbugger_retrain, attack_ratio, \
        eval_style_path, if_gen, if_robust, eval_pre_attack_path, \
        spelling_thres, coherency_thres, if_eval_robust, F_clf_lr, gen_adv_f_lr, flg_eval_baseline, if_hotflip_retrain, F_feature_dim

    if_test = True if opt.if_test == 1 else False
    if_eval = True if opt.if_eval == 1 else False
    if_monit = True if opt.if_monit == 1 else False
    if_predict = True if opt.if_predict == 1 else False
    if_eval_score = True if opt.if_eval_score == 1 else False
    if_train_defend = True if opt.if_train_defend ==1 else False
    if_train_val = True if opt.if_train_val == 1 else False
    flg_adv_d = True if opt.flg_adv_d == 1 else False
    flg_eval_g = True if opt.flg_eval_g == 1 else False
    flg_eval_pre = True if opt.flg_eval_pre == 1 else False
    flg_eval_hotflip = True if opt.flg_eval_hotflip == 1 else False
    flg_eval_copycat = True if opt.flg_eval_copycat == 1 else False
    if_show_sentence = True if opt.if_show_sentence == 1 else False
    flg_eval_unitrigger = True if opt.flg_eval_unitrigger == 1 else False 
    if_trigger_retrain = True if opt.if_trigger_retrain == 1 else False
    flg_eval_textbugger = True if opt.flg_eval_textbugger == 1 else False
    if_textbugger_retrain = True if opt.if_textbugger_retrain == 1 else False
    if_gen = True if opt.if_gen == 1 else False
    if_robust = True if opt.if_robust == 1 else False
    if_eval_robust = True if opt.if_eval_robust == 1 else False
    flg_eval_baseline = True if opt.flg_eval_baseline == 1 else False
    if_hotflip_retrain = True if opt.if_hotflip_retrain == 1 else False
    
    trigger_length = opt.trigger_length
    trigger_neighbor = opt.trigger_neighbor
    F_clf_lr = opt.F_clf_lr
    eval_g_path = opt.eval_g_path
    eval_style_path = opt.eval_style_path
    eval_pre_attack_path = opt.eval_pre_attack_path
    run_model = opt.run_model
    k_label = opt.k_label
    dataset = opt.dataset
    model_type = opt.model_type
    loss_type = opt.loss_type
    if_real_data = True if opt.if_real_data == 1 else False
    if_old_version = True if opt.if_old_version == 1 else False
    CUDA = True if opt.cuda == 1 else False
    device = opt.device
    data_shuffle = opt.shuffle
    gen_init = opt.gen_init
    dis_init = opt.dis_init
    eval_num = opt.eval_num
    eval_outfile = opt.eval_outfile
    samples_num2 = opt.max_num_eval
    num_topics = opt.num_topics
    if_save = opt.if_save
    max_num_comment_test = opt.max_num_comment_test
    F_train_epoch = opt.F_train_epoch
    f_patience_epoch = opt.F_patience_epoch
    eval_topic_num = opt.eval_topic_num
    atk_num_comment = opt.atk_num_comment
    blackbox_using_path = opt.blackbox_using_path
    random_num_article = opt.random_num_article
    trigger_beam_size = opt.trigger_beam_size
    attack_ratio = opt.attack_ratio
    spelling_thres = opt.spelling_thres
    coherency_thres = opt.coherency_thres
    gen_adv_f_lr = opt.gen_adv_f_lr
    F_feature_dim = opt.F_feature_dim

    samples_num = opt.samples_num
    vocab_size = opt.vocab_size
    MLE_train_epoch = opt.mle_epoch
    PRE_clas_epoch = opt.clas_pre_epoch
    ADV_train_epoch = opt.adv_epoch
    inter_epoch = opt.inter_epoch
    batch_size = opt.batch_size
    mle_batch_size = opt.mle_batch_size
    max_seq_len = opt.max_seq_len
    start_letter = opt.start_letter
    padding_idx = opt.padding_idx
    gen_lr = opt.gen_lr
    gen_adv_lr = opt.gen_adv_lr
    dis_lr = opt.dis_lr
    clip_norm = opt.clip_norm
    pre_log_step = opt.pre_log_step
    adv_log_step = opt.adv_log_step
    train_data = opt.train_data
    test_data = opt.test_data
    temp_adpt = opt.temp_adpt
    temperature = opt.temperature
    g_f_step = opt.g_f_step
    g_f_alpha = opt.g_f_alpha
    f_embed_dim = opt.f_embed_dim
    g_h_step = opt.g_h_step
    g_h_alpha = opt.g_h_alpha
    t_batch_size = opt.t_batch_size
    if_adapt_alpha = opt.if_adapt_alpha
    attack_mode = opt.attack_mode
    user = opt.user
    max_num_comment = opt.max_num_comment
    max_test_comment = opt.max_test_comment
    f_type = opt.f_type
    f_dropout_comment = opt.f_dropout_comment
    f_dropout_content = opt.f_dropout_content

    mmd_mean = opt.mmd_mean
    mmd_kernel_num = opt.mmd_kernel_num
    topic_epochs = opt.topic_epochs
    topic_lr = opt.topic_lr
    topic_gen_lr = opt.topic_gen_lr
    ADV_topic_step = opt.adv_topic_step

    oracle_pretrain = True if opt.ora_pretrain == 1 else False
    load_gen_pretrain = True if opt.load_gen_pretrain == 1 else False
    dis_pretrain = True if opt.dis_pretrain == 1 else False
    repeat_gen_pretrain = True if opt.gen_pretrain == 1 else False
    flg_adv_f = True if opt.flg_adv_f == 1 else False
    flg_adv_h = True if opt.flg_adv_h == 1 else False
    flg_train_f = True if opt.flg_train_f == 1 else False
    flg_train_h = True if opt.flg_train_h == 1 else False
    train_with_content = True if opt.train_with_content == 1 else False
    if_pretrain_embedding = True if opt.if_pretrain_embedding == 1 else False

    max_num_data = opt.max_num_data

    ADV_g_step = opt.adv_g_step
    rollout_num = opt.rollout_num
    gen_embed_dim = opt.gen_embed_dim
    gen_hidden_dim = opt.gen_hidden_dim
    goal_size = opt.goal_size
    step_size = opt.step_size
    mem_slots = opt.mem_slots
    num_heads = opt.num_heads
    head_size = opt.head_size
    content_feature_dim = head_size
    f_batch_size = opt.f_batch_size
    h_batch_size = opt.h_batch_size

    d_step = opt.d_step
    d_epoch = opt.d_epoch
    ADV_d_step = opt.adv_d_step
    ADV_d_epoch = opt.adv_d_epoch
    dis_embed_dim = opt.dis_embed_dim
    dis_hidden_dim = opt.dis_hidden_dim
    num_rep = opt.num_rep
    flg_adv_t = True if opt.flg_adv_t == 1 else False

    use_nll_oracle = True if opt.use_nll_oracle == 1 else False
    use_nll_gen = True if opt.use_nll_gen == 1 else False
    use_nll_div = True if opt.use_nll_div == 1 else False
    use_bleu = True if opt.use_bleu == 1 else False
    use_self_bleu = True if opt.use_self_bleu == 1 else False
    use_clas_acc = True if opt.use_clas_acc == 1 else False
    use_ppl = True if opt.use_ppl == 1 else False

    log_filename = opt.log_file
    signal_file = opt.signal_file
    tips = opt.tips
    root_folder = opt.root_folder

    # CUDA device
    # torch.cuda.set_device(device)

    # Save path
    save_root = root_folder + '/save/{}/{}/{}_{}_lt-{}_sl{}_temp{}_T{}/'.format(time.strftime("%Y%m%d"),
                                                                 dataset, run_model, model_type,
                                                                 loss_type, max_seq_len,
                                                                 temperature,
                                                                 log_time_str)
    save_samples_root = save_root + 'samples/'
    save_model_root = save_root + 'models/'

    train_data = 'dataset/' + dataset + '.txt'
    test_data = 'dataset/testdata/' + dataset + '_test.txt'
    train_data_content = 'dataset/' + dataset + '.txt'
    test_data_content = 'dataset/testdata/' + dataset + '_test.txt'
    cat_train_data = 'dataset/' + dataset + '_cat{}.txt'
    cat_test_data = 'dataset/testdata/' + dataset + '_cat{}_test.txt'


    # if max_seq_len == 40:
    #     oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}_sl40.pt'
    #     multi_oracle_samples_path = 'pretrain/oracle_data/oracle{}_lstm_samples_{}_sl40.pt'

    pretrain_root = root_folder+'/pretrain/{}/'.format(dataset if if_real_data else 'oracle_data')
    pretrained_gen_path = pretrain_root + 'gen_MLENEW_pretrain_{}_{}_{}_sl{}_sn{}_hs{}_maxsn{}_vs{}.pt'.format(dataset, run_model, model_type,
                                                                                       max_seq_len, samples_num, head_size, max_num_data, vocab_size)
    pretrain_root = root_folder+'/pretrain/{}/'.format(dataset if if_real_data else 'oracle_data')
    pretrained_f_path = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_f_path_CNN = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_CNN.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_f_path_RNN = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_RNN.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_f_path_RNN2 = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_RNN2.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_f_path_CSI = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_CSI.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_f_path_TEXTCNN = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_TEXTCNN.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_f_path_TEXTCNN2 = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_TEXTCNN2.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_f_path_TEXTCNN3 = pretrain_root + 'F_pretrain_{}_{}_sl{}_sn{}_vs{}_{}cmt_Glove{}_TEXTCNN3.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size, max_num_comment, f_embed_dim if if_pretrain_embedding else False)
    pretrained_h_path = pretrain_root + 'H_pretrain_{}_{}_sl{}_sn{}_vs{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                       samples_num, vocab_size)
    pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}_sl{}_sn{}_vs{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                   samples_num, vocab_size)
    pretrained_clas_path = pretrain_root + 'clas_pretrain_{}_{}_sl{}_sn{}_vs{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                     samples_num, vocab_size)
    topic_encoder_path = pretrain_root + 'topic_encoder_pretrained_MMD{}-{}'.format(mmd_mean, mmd_kernel_num)

    if f_type == "CNN":
        pretrained_f_path = pretrained_f_path_CNN 
    elif f_type == "RNN":
        pretrained_f_path = pretrained_f_path_RNN
    elif f_type == "RNN2":
        pretrained_f_path = pretrained_f_path_RNN2
    elif f_type == "CSI":
        pretrained_f_path = pretrained_f_path_CSI
    elif f_type == "TEXTCNN":
        pretrained_f_path = pretrained_f_path_TEXTCNN
    elif f_type == "TEXTCNN2":
        pretrained_f_path = pretrained_f_path_TEXTCNN2
    elif f_type == "TEXTCNN3":
        pretrained_f_path = pretrained_f_path_TEXTCNN3

    content_embeddings = 'dataset/' + dataset + '_content_embeddings.npz'
    comment_embeddings = 'dataset/' + dataset + '_comment_embeddings.npz'
    h_train_data_file = 'dataset/' + dataset + '_coherence_train_data.txt'
    h_train_label_file = 'dataset/' + dataset + '_coherence_train_label.txt'
    h_dev_data_file = 'dataset/' + dataset + '_coherence_val_data.txt'
    h_dev_label_file = 'dataset/' + dataset + '_coherence_val_label.txt'
    h_test_data_file = 'dataset/' + dataset + '_coherence_test_data.txt'
    h_test_label_file = 'dataset/' + dataset + '_coherence_test_label.txt'
    f_train_data_file = 'dataset/' + dataset + '_detect_train_data.txt'
    f_train_label_file = 'dataset/' + dataset + '_detect_train_label.txt'
    f_train_val_data_file = 'dataset/' + dataset + '_detect_train_val_data.txt'
    f_train_val_label_file = 'dataset/' + dataset + '_detect_train_val_label.txt'
    f_dev_data_file = 'dataset/' + dataset + '_detect_val_data.txt'
    f_dev_label_file = 'dataset/' + dataset + '_detect_val_label.txt'
    f_test_data_file = 'dataset/' + dataset + '_detect_test_data.txt'
    f_test_label_file = 'dataset/' + dataset + '_detect_test_label.txt'
    topic_file = 'dataset/' + dataset + '_topic_{}.pkl'.format(num_topics)

    pretrained_f_path = opt.attack_f_path if opt.attack_f_path != "" else pretrained_f_path
    pretrained_h_path = opt.attack_h_path if opt.attack_h_path != "" else pretrained_h_path
    pretrained_gen_path = opt.pretrained_gen_path if opt.pretrained_gen_path != "" else pretrained_gen_path

    if if_train_val:
        f_train_data_file = f_train_val_data_file
        f_train_label_file = f_train_val_label_file
    # Assertion
    assert k_label >= 2, 'Error: k_label = {}, which should be >=2!'.format(k_label)

    # Create Directory
    dir_list = ['save', 'savefig', 'log', 'pretrain', 'dataset',
                'pretrain/{}'.format(dataset if if_real_data else 'oracle_data')]
    if not if_test:
        dir_list.extend([save_root, save_samples_root, save_model_root])
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
  # Assertion
    assert k_label >= 2, 'Error: k_label = {}, which should be >=2!'.format(k_label)

    # Create Directory
    dir_list = ['save', 'savefig', 'log', 'pretrain', 'dataset',
                'pretrain/{}'.format(dataset if if_real_data else 'oracle_data')]
    if not if_test:
        dir_list.extend([save_root, save_samples_root, save_model_root])
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
