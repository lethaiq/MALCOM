
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory

def retrieve_alpha(alpha):
    if not cfg.if_adapt_alpha:
        rt = alpha
    else:
        rt = alpha - get_fixed_temperature(alpha, adv_epoch, cfg.ADV_train_epoch, "log")
    return rt
    
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
        
def prepare_data_clf(self, data):
    inp, comments, label = data['target'], data['comments_target'], data['label']
    new_comments = []
    for comment in comments:
        new_comments.append(F.one_hot(comment, cfg.vocab_size).float())
    comments = torch.stack(new_comments)
    return inp, comments, label

def remove_last_comment(data, is_cuda=True, one_hot=False):
    content = data['target']
    current_comments = data['comments_target'] # batch_size * 3 * 20
    num_comment = data['num_comment'] # 64, 3, 20
    label = data['label']
    all_comments = []
    batch_size = content.size()[0]
    if cfg.max_num_comment > 1:
        for i in range(batch_size):
            total_comments = []
            for j in range(min(num_comment[i].item(), cfg.max_num_comment-1)):
                comment = current_comments[i:i+1,j:j+1,:].squeeze(1) # batch_size * 20
                if one_hot:
                    comment = F.one_hot(comment, cfg.vocab_size).float() # 1 * 20 * vocab_size
                else:
                    comment = comment.long()
                if is_cuda:
                    comment = comment.cuda()
                total_comments.append(comment) 
            for j in range(cfg.max_num_comment-1-len(total_comments)):
                if one_hot:
                    zero_comment = torch.zeros((1, cfg.max_seq_len, cfg.vocab_size)).float()
                else:
                    zero_comment = torch.zeros((1, cfg.max_seq_len)).long()
                if is_cuda:
                    zero_comment = zero_comment.cuda()
                total_comments.append(zero_comment)
            all_comments.append(torch.stack(total_comments))
        all_comments = torch.stack(all_comments)
        all_comments = all_comments.squeeze(2)
    else:
        if one_hot:
            all_comments = torch.zeros((batch_size, 1, cfg.max_seq_len, cfg.vocab_size)).float()
        else:
            all_comments = torch.zeros((batch_size, 1, cfg.max_seq_len)).long()

    # print(all_comments.size())
    return content, all_comments, label, num_comment 