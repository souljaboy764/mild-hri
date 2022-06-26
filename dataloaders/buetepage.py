import torch
from torch.utils.data import Dataset
import numpy as np

# class SkeletonDataset(Dataset):
# 	def __init__(self, datafile, train=True):
# 		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		with np.load(datafile, allow_pickle=True) as data:
# 			if train:
# 				traj_data = np.array(data['train_data'])
# 				self.actidx = np.array([[0,24],[24,54],[54,110],[110,149]])
# 			else:
# 				traj_data = np.array(data['test_data'])
# 				self.actidx = np.array([[0,7],[7,15],[15,29],[29,39]])
# 			self.labels = traj_data[:, -1]
# 			self.idx = traj_data[:, -2]
# 			self.traj_data = traj_data[:, :-2]
# 			self.len = self.traj_data.shape[0]
# 			starts = np.where(self.idx==0)[0]
# 			ends = np.array(starts[1:].tolist() + [traj_data.shape[0]])
# 			self.traj_lens = np.zeros_like(self.idx)
# 			for i in range(len(starts)):
# 				self.traj_lens[starts[i]:ends[i]] = ends[i] - starts[i]


# 	def __len__(self):
# 		return self.len

# 	def __getitem__(self, index):
# 		return self.traj_data[index], self.idx[index], self.labels[index], self.traj_lens[index]

class SequenceDataset(Dataset):
	def __init__(self, datafile, train=True):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				self.traj_data = data['train_data']
				self.labels = data['train_labels']
				self.actidx = np.array([[0,24],[24,54],[54,110],[110,149]])
			else:
				self.traj_data = data['test_data']
				self.labels = data['test_labels']
				self.actidx = np.array([[0,7],[7,15],[15,29],[29,39]])
			
			self.len = len(self.traj_data)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), np.arange(self.traj_data[index].shape[0]), self.labels[index].astype(np.int32)