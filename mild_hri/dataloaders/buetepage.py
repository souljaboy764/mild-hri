import torch
from torch.utils.data import Dataset
import numpy as np

from mild_hri.utils import *

class HHDataset(Dataset):
	def __init__(self, datafile, train=True, downsample=1):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		with np.load(datafile, allow_pickle=True) as data:
			if train:
				traj_data = data['train_data']
				labels = data['train_labels']
				self.actidx = np.array([[0,24],[24,54],[54,110],[110,149]])*2

			else:
				traj_data = data['test_data']
				labels = data['test_labels']
				self.actidx = np.array([[0,7],[7,15],[15,29],[29,39]])*2

			self.traj_data = []
			for i in range(len(traj_data)):
				traj_data[i] = traj_data[i][:, 1:, :] # Ignoring the first shoulder/body joint as it is almost static
				seq_len, njoints, dims = traj_data[i].shape
				traj_1 = traj_data[i][..., :3].reshape((seq_len, (njoints)*3))
				traj_2 = traj_data[i][..., 3:].reshape((seq_len, (njoints)*3))

				if downsample < 1:
					traj = np.array(downsample_trajs([np.concatenate([traj_1[:, None], traj_2[:, None]], axis=-1)], int(downsample*seq_len), device))[0, :, 0, :]
					seq_len, dims = traj.shape
					traj_1 = traj[:, :dims//2]
					traj_2 = traj[:, dims//2:]

				vel_1 = np.diff(traj_1, axis=0, prepend=traj_1[0:1,:])
				vel_2 = np.diff(traj_2, axis=0, prepend=traj_2[0:1,:])
				# traj_1 = np.concatenate([traj_1, vel_1],axis=-1)
				# traj_2 = np.concatenate([traj_2, vel_2],axis=-1)

				self.traj_data.append(np.concatenate([traj_1, vel_1, traj_2, vel_2], axis=-1))
				self.traj_data.append(np.concatenate([traj_2, vel_2, traj_1, vel_1], axis=-1))
			
			self.len = len(self.traj_data)
			self.labels = np.zeros(self.len)
			for idx in range(len(self.actidx)):
				self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)
class HHWindowDataset(Dataset):
	def __init__(self, datafile, train=True, window_length=40, downsample = 1):
		dataset = HHDataset(datafile, train, downsample)
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
	def __init__(self, datafile, train=True, downsample=1):
		super(PepperDataset, self).__init__(datafile, train, downsample)
		
		for i in range(len(self.traj_data)):
			seq_len, dims = self.traj_data[i].shape
			# input_slice = slice(int(2*dims/3),dims)
			input_slice = slice(int(dims//2), int(3*dims//4))
			traj_r = []

			for frame in self.traj_data[i][:, input_slice].reshape((seq_len, -1, 3)):
				joints = joint_angle_extraction(frame)
				traj_r.append(joints)

			traj_r = np.array(traj_r) # seq_len, 4
			
			self.traj_data[i] = np.concatenate([self.traj_data[i][:, :input_slice.start], traj_r], axis=-1) # seq_len, input_dim + 4
	
class PepperWindowDataset(HHWindowDataset):
	def __init__(self, datafile, train=True, window_length=40, downsample = 1):
		self._dataset = PepperDataset(datafile, train, downsample)
		self.actidx = self._dataset.actidx
		self.traj_data = window_concat(self._dataset.traj_data, window_length, 'pepper')
		self.len = len(self.traj_data)
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

