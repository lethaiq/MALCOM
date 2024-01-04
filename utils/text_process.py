# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : text_process.py
# @Time         : Created at 2019-05-14
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import nltk
import numpy as np
import os
import torch

from tqdm import tqdm

import config as cfg

from utils.tweet_utils import *


def get_tokenlized(file):
    """tokenlize the file"""
    tokenlized = list()
    with open(file, encoding='utf-8') as raw:
        for text in raw:
            # text_tokens= nltk.word_tokenize(text.lower())
            text_token = tweet_ark_tokenize(text.lower())
            tokenlized.append(text_token)
            # if 'i have a feeling' in text:
            #     print(text_tokens)
    return tokenlized

def get_tokenized_sep_content(file, sep="::"):
    tokenlized = list()
    tokenlized_title = list()
    titles_str = list()
    ids = []
    with open(file, encoding='utf-8') as raw:
        for text in raw:
            tmp = {}
            all_str = text.split(sep)
            _id = all_str[1].replace(':','')
            title = all_str[0].lower().replace('\n','')
            text_token = tweet_ark_tokenize(all_str[2].lower().replace('\n',''))
            title_token = tweet_ark_tokenize(title)
            tokenlized.append(text_token)
            tokenlized_title.append(title_token)
            ids.append(_id)
            titles_str.append(title)
    return tokenlized, tokenlized_title, titles_str, ids

def get_tokenized_sep(file, sep="::"):
    tokenlized = list()
    tokenlized_title = list()
    data = []
    with open(file, encoding='utf-8') as raw:
        for text in tqdm(raw):
            tmp = {}
            all_str = text.split(sep)
            title_token= tweet_ark_tokenize(all_str[0].lower().replace('\n',''))
            tokenlized_title.append(title_token)
            tmp['title'] = all_str[0]
            tmp['id'] = all_str[1].replace(':','')
            tmp['comments'] = []
            for s in all_str[2:]:
                text_token= tweet_ark_tokenize(s.lower().replace('\n',''))
                if len(text_token) > 0:
                    tokenlized.append(text_token)
                    tmp['comments'].append(len(tokenlized)-1)
            data.append(tmp)
    return tokenlized, tokenlized_title, data

def get_word_list(tokens, vocab_size=-1):
    """get word set"""
    word_set = list()
    punctChars = r"['\"“”‘’.?!…,:;]"
    
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    words, counts = np.unique(word_set, return_counts=True)
    freq_idx = np.argsort(counts)[::-1]

    if vocab_size == -1:
        word_list = list(set(word_set))
    else:
        word_list =  list(words[freq_idx[:vocab_size]])

    for c in punctChars:
        if c not in word_list:
            word_list.append(c)

    return word_list


def get_dict(word_set):
    """get word2idx_dict and idx2word_dict"""
    word2idx_dict = dict()
    idx2word_dict = dict()

    index = 3
    word2idx_dict[cfg.padding_token] = str(cfg.padding_idx)  # padding token
    idx2word_dict[str(cfg.padding_idx)] = cfg.padding_token
    word2idx_dict[cfg.start_token] = str(cfg.start_letter)  # start token
    idx2word_dict[str(cfg.start_letter)] = cfg.start_token
    word2idx_dict[cfg.unk_token] = str(cfg.unk_idx)  # unknown token
    idx2word_dict[str(cfg.unk_idx)] = cfg.unk_token

    for word in word_set:
        word2idx_dict[word] = str(index)
        idx2word_dict[str(index)] = word
        index += 1
    return word2idx_dict, idx2word_dict


def text_process_seq_len(train_text_loc):
    train_text_loc = 'dataset/{}.txt'.format(train_text_loc)
    train_tokens = get_tokenlized(train_text_loc)
    max_len = 0
    for token in train_tokens:
        if max_len < len(token):
            max_len = len(token)
    return max_len
    
def text_process(train_text_loc, vocab_size=-1, test_text_loc=None):
    """get sequence length and dict size"""
    train_text_loc = 'dataset/{}.txt'.format(train_text_loc)
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens, vocab_size)
    word2idx_dict, idx2word_dict = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    # return sequence_len, len(word2idx_dict)
    return sequence_len


# ============================================
def init_dict(dataset, vocab_size, use_external_file=None):
    """
    Initialize dictionaries of dataset, please note that '0': padding_idx, '1': start_letter.
    Finally save dictionary files locally.
    """
    if use_external_file:
        tokens = get_tokenlized('dataset/{}.txt'.format(use_external_file))
    else:
        tokens = get_tokenlized('dataset/{}.txt'.format(dataset))

    word_set = get_word_list(tokens, vocab_size)
    word2idx_dict, idx2word_dict = get_dict(word_set)
    print("initialize dictionary", len(word2idx_dict))

    with open('dataset/{}_wi_dict.txt'.format(dataset), 'w', encoding='utf-8') as dictout:
        dictout.write(str(word2idx_dict))
    with open('dataset/{}_iw_dict.txt'.format(dataset), 'w', encoding='utf-8') as dictout:
        dictout.write(str(idx2word_dict))

    print('total tokens: ', len(word2idx_dict))

    word2idx_dict, idx2word_dict = load_test_dict(dataset)
    print("updating with test set vocab...")
    with open('dataset/{}_wi_dict.txt'.format(dataset), 'w', encoding='utf-8') as dictout:
        dictout.write(str(word2idx_dict))
    with open('dataset/{}_iw_dict.txt'.format(dataset), 'w', encoding='utf-8') as dictout:
        dictout.write(str(idx2word_dict))

def load_dict(dataset, vocab_size=-1, use_external_file=None):
    # print("loading dict with vocab_size", vocab_size)
    """Load dictionary from local files"""
    iw_path = 'dataset/{}_iw_dict.txt'.format(dataset)
    wi_path = 'dataset/{}_wi_dict.txt'.format(dataset)

    if not os.path.exists(iw_path) or not os.path.exists(iw_path):  # initialize dictionaries
        init_dict(dataset, vocab_size, use_external_file)

    with open(iw_path, 'r', encoding="utf-8") as dictin:
        idx2word_dict = eval(dictin.read().strip())
    with open(wi_path, 'r', encoding="utf-8") as dictin:
        word2idx_dict = eval(dictin.read().strip())
    
    return word2idx_dict, idx2word_dict


def load_test_dict(dataset, vocab_size=-1):
    # print("loading test dict with vocab_size", vocab_size)

    """Build test data dictionary, extend from train data. For the classifier."""
    word2idx_dict, idx2word_dict = load_dict(dataset, vocab_size)  # train dict
    # tokens = get_tokenlized('dataset/testdata/{}_clas_test.txt'.format(dataset))
    tokens = get_tokenlized('dataset/testdata/{}_test.txt'.format(dataset))
    word_set = get_word_list(tokens, vocab_size)
    index = len(word2idx_dict)  # current index

    # extend dict with test data
    for word in word_set:
        if word not in word2idx_dict:
            word2idx_dict[word] = str(index)
            idx2word_dict[str(index)] = word
            index += 1

    # with open('dataset/{}_wi_dict.txt'.format(dataset), 'w') as dictout:
    #     dictout.write(str(word2idx_dict))
    # with open('dataset/{}_iw_dict.txt'.format(dataset), 'w') as dictout:
    #     dictout.write(str(idx2word_dict))

    return word2idx_dict, idx2word_dict


def tensor_to_tokens(tensor, dictionary, external_dict={}, joined=False):
    """transform Tensor to word tokens"""
    tokens = []
    text = []
    for sent in tensor:
        sent_token = []
        sent_token_str = ""
        for word in sent.tolist(): #reverse
            if word == cfg.padding_idx:
                break
            # if word == cfg.unk_idx:
            #     sent_token.append(cfg.unk_token)
            if str(word) in dictionary:
                x = dictionary[str(word)]
                sent_token.append(x)
                sent_token_str = sent_token_str + " " + x
            else:
                if str(word) in external_dict:
                    x = external_dict[str(word)]
                    sent_token.append(x)
                    sent_token_str = sent_token_str + " " + x
                else:
                    sent_token.append(cfg.unk_token)
                    sent_token_str = sent_token_str + " " + cfg.unk_token
        tokens.append(sent_token)
        text.append(sent_token_str)
    if joined:
        return tokens, text
    return tokens


def tokens_to_tensor(tokens, dictionary, return_length=False):
    """transform word tokens to Tensor"""
    global i
    tensor = []
    lengths = []
    for sent in tokens:
        sent_ten = []
        filtered_length = 0
        sent_str = " ".join(sent)
        i = -1
        for i, word in enumerate(sent):
            # if word == cfg.end_token:
            #     break
            if str(word) in dictionary:
                sent_ten.append(int(dictionary[str(word)]))
                filtered_length += 1
            else:
                sent_ten.append(cfg.unk_idx)
                
        lengths.append(filtered_length)
        while i < cfg.max_seq_len - 1:
            sent_ten.append(cfg.padding_idx)
            i += 1

        total_unk = np.where(sent_ten == cfg.unk_idx)[0]
        if len(total_unk) < cfg.unk_threshold:
            tensor.append(sent_ten[:cfg.max_seq_len])

    if not return_length:
        return torch.LongTensor(tensor)
    return torch.LongTensor(tensor), torch.LongTensor(lengths)


def padding_token(tokens):
    """pad sentences with padding_token"""
    global i
    pad_tokens = []
    for sent in tokens:
        sent_token = []
        for i, word in enumerate(sent):
            if word == cfg.padding_token:
                break
            sent_token.append(word)
        while i < cfg.max_seq_len - 1:
            sent_token.append(cfg.padding_token)
            i += 1
        pad_tokens.append(sent_token)
    return pad_tokens


def write_tokens(filename, tokens):
    """Write word tokens to a local file (For Real data)"""
    with open(filename, 'w') as fout:
        for sent in tokens:
            fout.write(' '.join(sent))
            fout.write('\n')


def write_tensor(filename, tensor):
    """Write Tensor to a local file (For Oracle data)"""
    with open(filename, 'w') as fout:
        for sent in tensor:
            fout.write(' '.join([str(i) for i in sent.tolist()]))
            fout.write('\n')


def process_cat_text():
    import random

    dataset = 'mr'

    test_ratio = 0.3
    seq_len = 15

    pos_file = 'dataset/{}/{}{}_cat1.txt'.format(dataset, dataset, seq_len)
    neg_file = 'dataset/{}/{}{}_cat0.txt'.format(dataset, dataset, seq_len)
    pos_sent = open(pos_file, 'r').readlines()
    neg_sent = open(neg_file, 'r').readlines()

    pos_len = int(test_ratio * len(pos_sent))
    neg_len = int(test_ratio * len(neg_sent))

    random.shuffle(pos_sent)
    random.shuffle(neg_sent)

    all_sent_test = pos_sent[:pos_len] + neg_sent[:neg_len]
    all_sent_train = pos_sent[pos_len:] + neg_sent[neg_len:]
    random.shuffle(all_sent_test)
    random.shuffle(all_sent_train)

    f_pos_train = open('dataset/{}{}_cat1.txt'.format(dataset, seq_len), 'w')
    f_neg_train = open('dataset/{}{}_cat0.txt'.format(dataset, seq_len), 'w')
    f_pos_test = open('dataset/testdata/{}{}_cat1_test.txt'.format(dataset, seq_len), 'w')
    f_neg_test = open('dataset/testdata/{}{}_cat0_test.txt'.format(dataset, seq_len), 'w')

    for p_s in pos_sent[:pos_len]:
        f_pos_test.write(p_s)
    for n_s in neg_sent[:neg_len]:
        f_neg_test.write(n_s)
    for p_s in pos_sent[pos_len:]:
        f_pos_train.write(p_s)
    for n_s in neg_sent[neg_len:]:
        f_neg_train.write(n_s)

    with open('dataset/testdata/{}{}_test.txt'.format(dataset, seq_len), 'w') as fout:
        for sent in all_sent_test:
            fout.write(sent)
    with open('dataset/{}{}.txt'.format(dataset, seq_len), 'w') as fout:
        for sent in all_sent_train:
            fout.write(sent)

    f_pos_train.close()
    f_neg_train.close()
    f_pos_test.close()
    f_neg_test.close()


def combine_amazon_text():
    cat0_name = 'app'
    cat1_name = 'book'
    root_path = 'dataset/'
    cat0_train = open(root_path + cat0_name + '.txt', 'r').readlines()
    cat0_test = open(root_path + cat0_name + '_test.txt', 'r').readlines()
    cat1_train = open(root_path + cat1_name + '.txt', 'r').readlines()
    cat1_test = open(root_path + cat1_name + '_test.txt', 'r').readlines()

    with open(root_path + 'amazon_{}_{}.txt'.format(cat0_name, cat1_name), 'w') as fout:
        for sent in cat0_train:
            fout.write(sent)
        for sent in cat1_train:
            fout.write(sent)
    with open(root_path + 'testdata/amazon_{}_{}_test.txt'.format(cat0_name, cat1_name), 'w') as fout:
        for sent in cat0_test:
            fout.write(sent)
        for sent in cat1_test:
            fout.write(sent)


def extend_clas_train_data():
    data_name = 'mr'
    dataset = 'mr20'
    neg_filter_file = 'dataset/{}/{}_cat0.txt'.format(data_name, dataset)  # include train and test for generator
    pos_filter_file = 'dataset/{}/{}_cat1.txt'.format(data_name, dataset)
    neg_test_file = 'dataset/testdata/{}_cat0_test.txt'.format(dataset)
    pos_test_file = 'dataset/testdata/{}_cat1_test.txt'.format(dataset)
    neg_all_file = 'dataset/{}/{}_cat0.txt'.format(data_name, data_name)
    pos_all_file = 'dataset/{}/{}_cat1.txt'.format(data_name, data_name)

    neg_filter = open(neg_filter_file, 'r').readlines()
    pos_filter = open(pos_filter_file, 'r').readlines()
    neg_test = open(neg_test_file, 'r').readlines()
    pos_test = open(pos_test_file, 'r').readlines()
    neg_all = open(neg_all_file, 'r').readlines()
    pos_all = open(pos_all_file, 'r').readlines()

    # print('neg filter:', len(neg_filter))
    # print('neg test:', len(neg_test))
    # print('neg all:', len(neg_all))
    # print('pos filter:', len(pos_filter))
    # print('pos test:', len(pos_test))
    # print('pos all:', len(pos_all))

    print('neg before:', len(neg_test))
    for line in neg_all:
        if line not in neg_filter:
            neg_test.append(line)
    print('neg after:', len(neg_test))

    print('pos before:', len(pos_test))
    for line in pos_all:
        if line not in pos_filter:
            pos_test.append(line)
    print('pos after:', len(pos_test))

    with open('dataset/testdata/{}_cat0_clas_test.txt'.format(dataset), 'w') as fout:
        for line in neg_test:
            fout.write(line)
    with open('dataset/testdata/{}_cat1_clas_test.txt'.format(dataset), 'w') as fout:
        for line in pos_test:
            fout.write(line)
    with open('dataset/testdata/{}_clas_test.txt'.format(dataset), 'w') as fout:
        for line in neg_test:
            fout.write(line)
        for line in pos_test:
            fout.write(line)


def load_word_vec(path, word2idx_dict=None, type='glove'):
    """Load word embedding from local file"""
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    if type == 'glove':
        word2vec_dict = {}
        for line in fin:
            tokens = line.rstrip().split()
            if word2idx_dict is None or tokens[0] in word2idx_dict.keys():
                word2vec_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    elif type == 'word2vec':
        import gensim
        word2vec_dict = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        raise NotImplementedError('No such type: %s' % type)
    return word2vec_dict



def build_embedding_matrix(dataset, size=200, tensor=True):
    """Load or build Glove embedding matrix."""
    embed_filename = 'dataset/glove_embedding_{}d_{}.pt'.format(size, dataset)
    if os.path.exists(embed_filename):
        print('Loading embedding:', embed_filename)
        embedding_matrix = torch.load(embed_filename)
        if not tensor:
            embedding_matrix = embedding_matrix.data.numpy()
    else:
        print('Loading Glove word vectors...')
        # word2idx_dict, _ = load_dict(dataset)
        word2idx_dict, _ = load_dict(dataset, cfg.vocab_size)
        embedding_matrix = np.random.random((len(word2idx_dict), size))  # 2 for padding token and start token and unknown token
        fname = 'glove/glove.twitter.27B.{}d.txt'.format(size)  # Glove file
        # fname = '../GoogleNews-vectors-negative300.bin' # Google Word2Vec file
        word2vec_dict = load_word_vec(fname, word2idx_dict=word2idx_dict, type='glove')
        print('Building embedding matrix:', embed_filename)
        oovs = []
        for word, i in word2idx_dict.items():
            if word in word2vec_dict:
                # words not found in embedding index will be randomly initialized such as adding, start or unknown token
                embedding_matrix[int(i)] = word2vec_dict[word]
            else:
                oovs.append(word)
        print("# Out of Vocabulary:{}".format(len(oovs)))
        if tensor:
            embedding_matrix = torch.FloatTensor(embedding_matrix)
            torch.save(embedding_matrix, embed_filename)
    return embedding_matrix


if __name__ == '__main__':
    os.chdir('../')
    build_embedding_matrix('gossipcopWithContent40x')
    # process_cat_text()
    # load_test_dict('mr15')
    # extend_clas_train_data()
    pass
