import torch
from torch.utils.data import Dataset
import numpy as np

from mild_hri.utils import *

class YumiDataset(Dataset):
	def __init__(self, datafile, train=True, downsample=1):
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				self.traj_data = data['train_data']
				self.labels = data['train_labels']
				self.actidx = np.array([[0,8],[8,16],[16,24],[24,32]]) # Human-robot trajs
			else:
				self.traj_data = data['test_data']
				self.labels = data['test_labels']
				self.actidx = np.array([[0,2],[2,4],[4,6],[6,9]]) # Human-robot trajs
			
			for i in range(len(self.traj_data)):
				seq_len, dims = self.traj_data[i].shape
				traj_h = self.traj_data[i][:,:-7]
				traj_r = self.traj_data[i][:,-7:]
				if downsample < 1:
					assert downsample != 0
					self.traj_data[i] = np.concatenate(downsample_trajs([traj_h[:,None], traj_r[:,None]], int(downsample*seq_len), device),axis=-1)[:, 0, :]
			self.len = len(self.traj_data)
			self.labels = np.zeros(self.len)
			for idx in range(len(self.actidx)):
				self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)

class YumiWindowDataset(Dataset):
	def __init__(self, datafile, train=True, window_length=40, downsample=1):
		dataset = YumiDataset(datafile, train, downsample)
		self.actidx = dataset.actidx
		self.traj_data = window_concat(dataset.traj_data, window_length, 'yumi')
		self.len = len(self.traj_data)
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)