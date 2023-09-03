import torch
from torch.utils.data import Dataset
import numpy as np

from mild_hri.utils import *

class HHDataset(Dataset):
	def __init__(self, datafile, train=True, downsample=None): # downsample only needed for compatibility
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				self.traj_data = data['train_data']
				self.labels = data['train_labels']
				self.actidx = np.array([[0, 9], [9, 17], [17, 26], [26, 33]])

			else:
				self.traj_data = data['test_data']
				self.labels = data['test_labels']
				self.actidx = np.array([[0, 3], [3, 6], [6, 9], [9, 11]])
			
			self.len = len(self.traj_data)
			self.labels = np.zeros(self.len)
			for idx in range(len(self.actidx)):
				self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

			for i in range(len(self.traj_data)):
				seq_len, dims = self.traj_data[i].shape
				traj_1 = self.traj_data[i][:, :dims//2]
				traj_2 = self.traj_data[i][:, dims//2:]
				# traj_1 = self.traj_data[i][:, dims//2-3:dims//2]
				# traj_2 = self.traj_data[i][:, -3:]

				vel_1 = np.diff(traj_1, axis=0, prepend=traj_1[0:1,:])
				vel_2 = np.diff(traj_2, axis=0, prepend=traj_2[0:1,:])

				traj_1 = np.concatenate([traj_1, vel_1],axis=-1)
				traj_2 = np.concatenate([traj_2, vel_2],axis=-1)

				self.traj_data[i] = np.concatenate([traj_1, traj_2], axis=-1)
			

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)

class HHWindowDataset(Dataset):
	def __init__(self, datafile, train=True, window_length=5, downsample=None): # downsample only needed for compatibility
		dataset = HHDataset(datafile, train)
		self.actidx = dataset.actidx
		self.traj_data = window_concat(dataset.traj_data, window_length)
		self.len = len(self.traj_data)
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)

	
class PepperDataset(HHDataset):
	def __init__(self, datafile, train=True, downsample=None):
		super(PepperDataset, self).__init__(datafile, train, downsample)
		
		self.joints_min = np.ones(4)*10
		self.joints_max = np.ones(4)*-10
		for i in range(len(self.traj_data)):
			seq_len, dims = self.traj_data[i].shape
			traj_r = []

			for frame in self.traj_data[i][:, dims//2:].reshape((seq_len, dims//6, 3)):
				joints = joint_angle_extraction(frame)
				traj_r.append(joints)

			traj_r = np.array(traj_r) # seq_len, 4

			traj_min = traj_r.min(0)
			traj_max = traj_r.max(0) 
			self.joints_min = np.where(self.joints_min>traj_min, traj_min, self.joints_min)
			self.joints_max = np.where(self.joints_max<traj_max, traj_max, self.joints_max)
			
			self.traj_data[i] = np.concatenate([self.traj_data[i][:, :dims//2], traj_r], axis=-1) # seq_len, dims//2 + 4
			# self.traj_data[i] = np.concatenate([self.traj_data[i][:, dims//4-3:dims//4], self.traj_data[i][:, dims//2-3:dims//2], traj_r], axis=-1) # seq_len, dims//2 + 4
		# print(self.joints_min)
		# print(self.joints_max)
	
class PepperWindowDataset(HHWindowDataset):
	def __init__(self, datafile, train=True, window_length=5, downsample = 1):
		self._dataset = PepperDataset(datafile, train, downsample)
		self.actidx = self._dataset.actidx
		self.traj_data = window_concat(self._dataset.traj_data, window_length, 'pepper')
		self.len = len(self.traj_data)
		self.labels = np.zeros(self.len)
		self.joints_max = np.tile(self._dataset.joints_max, window_length)
		self.joints_min = np.tile(self._dataset.joints_min, window_length)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

