import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg
from utils.text_process import *


class GANDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

class F_DataIterSep:
	def __init__(self, samples, labels, embedding_data, 
					comment_embedding_data=None, 
					batch_size=64, shuffle=None, 
					drop_last=False):
	
		self.batch_size = batch_size
		self.max_seq_len = cfg.max_seq_len
		self.start_letter = cfg.start_letter
		self.shuffle = cfg.data_shuffle if not shuffle else shuffle
		self.max_num_comment = cfg.max_num_comment
		self.embedding_data = embedding_data
		self.comment_embedding_data = comment_embedding_data
		self.drop_last = drop_last

		if cfg.if_real_data:
			self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset, cfg.vocab_size)
			print("fake_news_loader vocab_size", len(self.word2idx_dict))
		self.data, self.id2idx = self.__read_data__(samples, labels)

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

	def __read_data__(self, samples, labels):
		"""
		input: same as target, but start with start_letter.
		"""
		# global all_data
		id2idx = {}
		all_data = None
		if isinstance(samples, torch.Tensor):  # Tensor
			inp, target, label = self.prepare(samples, labels)
			all_data = [{'input': i, 'target':t, 'lable': l} for (i, t, l) in zip(inp, target, label)]
		elif isinstance(samples[0], str):  # filename
			inp, target, label, labels_idx, titles_tensor, index_data = self.load_data(samples, labels)
			all_data = []
			for i in range(len(index_data)):
				instance = index_data[i]
				title_str = instance['title']
				real_id = instance['id']
				comments_idx = instance['comments']
				title = titles_tensor[i]
				
				total_comments = []
				for j in range(len(comments_idx)):
					total_comments.append(target[comments_idx[j]].unsqueeze(0))
					if len(total_comments) >= self.max_num_comment:
						break

				original_num_comment = len(total_comments)
				for _ in range(cfg.max_num_comment - original_num_comment):
					zero_comment = torch.zeros(target[0].size()).long().unsqueeze(0)
					total_comments.append(zero_comment)

				# retrieve comment 512 size vector
				total_comments_vecs = []
				for j in range(len(comments_idx)):
					com_id = "{}-{}".format(real_id, j)

				if title_str == "":
					print("sample with empty title found. continued.")
					continue

				# target_vector = self.load_vector_from_id(real_id)
				content = torch.from_numpy(self.embedding_data[real_id])
				tmp = {
						'idx': i,
						'content': content, 
						'id': real_id,
						'title': title,
						'title_str': title_str,
						'comments': torch.cat(total_comments, 0).long(),
						'label': label[i],
						'num_comment': original_num_comment
						}
				id2idx[real_id] = len(all_data)
				
				if original_num_comment < 1:
					if cfg.if_accept_nocomment:
						all_data.append(tmp)
					else:
						print("remove ", real_id)
				else:
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
	def prepare(samples, labels, gpu=True):
		"""Add start_letter to samples as inp, target same as samples"""
		inp = torch.zeros(samples.size()).long()
		target = samples
		inp[:, 0] = cfg.start_letter
		inp[:, 1:] = target[:, :cfg.max_seq_len - 1]

		# if gpu:
		#     return inp.cuda(), target.cuda(), labels.cuda()
		return inp, target, labels

	@staticmethod
	def load_label_from_file(file):
		labels = list()
		labels_idx = {}
		i = 0
		label_is_int = True
		with open(file) as raw:
			for label in raw:
				label = label.replace('\n','')
				try: # preparing for both classification and regression
					label = int(label)
				except:
					label = float(label)
					label_is_int = False
				labels.append(label)
				if label not in labels_idx:
					labels_idx[label] = []
				labels_idx[label].append(i)
				i += 1

		if label_is_int:
			labels = torch.LongTensor(labels)
		else:
			labels = torch.FloatTensor(labels)
		return labels, labels_idx

	def load_data(self, samples_filename, labels_filename):
		"""Load real data from local file"""
		self.tokens, tokens_title, index_data = get_tokenized_sep(samples_filename, sep="::")
		samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
		titles_tensor = tokens_to_tensor(tokens_title, self.word2idx_dict)
		labels, labels_idx = self.load_label_from_file(labels_filename)
		inp, target, label = self.prepare(samples_index, labels)
		return inp, target, label, labels_idx, titles_tensor, index_data

	def __len__(self):
		return len(self.loader)