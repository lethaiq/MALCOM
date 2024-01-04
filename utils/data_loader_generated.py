import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.text_process import *

class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class G_DataIterSep:
    def __init__(self, samples, embedding_data,
                    batch_size=64, shuffle=None, 
                    drop_last=False):
        self.batch_size = batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        self.max_num_comment = cfg.max_num_comment
        self.embedding_data = embedding_data
        self.drop_last = drop_last

        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset, cfg.vocab_size)
        self.data, self.id2idx = self.__read_data__(samples)

        self.loaders = {}
        self.loader = DataLoader(
            dataset=GANDataset(self.data),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last)
        self.loaders[self.batch_size] = self.loader

    def get_class_weight(self):
        all_labels = [a['label'] for a in self.data]
        counts = np.bincount(all_labels)
        weight = 1.0*counts/counts.sum()
        return weight
        
    def get_loader(self, batch_size):
        if batch_size not in self.loaders:
            # print("generating new loader for batch_size", batch_size)
            loader = DataLoader(
                dataset=GANDataset(self.data),
                batch_size=batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
            self.loaders[batch_size] = loader
        return self.loaders[batch_size]

    def add_attribute(self, preds, key_name="prediction"):
        for i in range(len(self.data)):
            self.data[i][key_name] = preds[self.data[i]['id']]
        print("added key {} into data_loader".format(key_name))

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        # global all_data
        id2idx = {}
        all_data = None
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, target, label = self.prepare(samples)
            all_data = [{'input': i, 'target':t, 'lable': l} for (i, t, l) in zip(inp, target, label)]
        elif isinstance(samples[0], str):  # filename
            inp, target, titles_tensor, index_data = self.load_data(samples)
            all_data = []
            for i in range(len(index_data)):
                instance = index_data[i]
                title_str = instance['title']
                real_id = instance['id']
                comments_idx = instance['comments']
                title = titles_tensor[i]
                label = instance['label']
                
                total_comments = []
                for j in range(len(comments_idx)):
                    total_comments.append(target[comments_idx[j]])
                original_num_comment = len(total_comments)

                content = torch.from_numpy(self.embedding_data[real_id])

                for comment in total_comments:
                    comment_tensor = comment
                    tmp = {
                            'content': content, 
                            'id': real_id,
                            'title': title,
                            'title_str': title_str,
                            'single_comment': comment_tensor,
                            'label': label,
                            'num_comment': original_num_comment
                            }
                    if real_id not in id2idx:
                        id2idx[real_id] = []
                    id2idx[real_id].append(len(all_data))
                    all_data.append(tmp)
        else:
            all_data = None

        print("LOADED {} DATA WITH TOTAL {} ARTICLES".format(len(all_data), len(id2idx)))
        return all_data, id2idx

    def random_batch(self, batch_size=None):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        if not batch_size:
            batch_size = self.batch_size
        loader = self.get_loader(batch_size)
        idx = random.randint(0, len(loader) - 1)
        return list(loader)[idx]

    def sample_from_id(self, _id):
        return self.data[self.id2idx[_id]]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, gpu=True):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = cfg.start_letter
        inp[:, 1:] = target[:, :cfg.max_seq_len - 1]
        return inp, target

    def get_tokenized_sep(self, file, sep="::"):
        tokenlized = []
        tokenlized_title = []
        data = []
        self.pass_coherency = 0
        self.pass_misspelling = 0
        self.original_count = 0
        with open(file) as raw:
            for text in raw:
                if len(text)>5:
                    tmp = {}
                    all_str = text.split(sep)
                    title_token= tweet_ark_tokenize(all_str[0].lower().replace('\n',''))
                    _id = all_str[1].replace(':','')
                    tokenlized_title.append(title_token)
                    tmp['title'] = all_str[0]
                    tmp['id'] = _id
                    tmp['comments'] = []
                    tmp['label'] = int(all_str[-1].replace('\n','').replace(':',''))
                    for s in all_str[2:-1]:
                        c = s.lower().replace('\n','')
                        c_token= tweet_ark_tokenize(c)
                        if len(c_token) > 0:
                            tokenlized.append(c_token)
                            tmp['comments'].append(len(tokenlized)-1)
                    data.append(tmp)
        return tokenlized, tokenlized_title, data

    def load_data(self, samples_filename):
        """Load real data from local file"""
        self.tokens, tokens_title, index_data = self.get_tokenized_sep(samples_filename, sep="::")
        samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
        titles_tensor = tokens_to_tensor(tokens_title, self.word2idx_dict)
        inp, target = self.prepare(samples_index)
        return inp, target, titles_tensor, index_data

    def __len__(self):
        return len(self.loader)