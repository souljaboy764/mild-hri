import numpy as np
import os
from torch.utils.data import Dataset

class HHWindowDataset(Dataset):
	# mode: 1 - unimanual, 2 - bimanual, 3 - both
	# def __init__(self, train, mode=3, window_length=5):
	def __init__(self, datafile, train=True, window_length=5, downsample = 1, mode=3):
		with np.load(os.path.join(datafile), allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, _, _ = data['train_data']
				self.actidx = np.array([[0,105], [105, 168]])
			else:
				p1_trajs, p2_trajs, _, _ = data['test_data']
				self.actidx = np.array([[0,12], [12, 24]])
			joints = data['joints']
		
		joints_dic = {joints[i]:i for i in range(len(joints))}
		joints_idx = [joints_dic[i] for i in ['LUArm', 'LFArm', 'LHand', 'RUArm', 'RFArm', 'RHand']]
		self.input_data = []
		self.output_data = []
		self.traj_data = []
		for i in range(len(p1_trajs)):
			p1_pos = p1_trajs[i][::4, joints_idx]
			p1_pos = p1_pos.reshape((p1_pos.shape[0], 3*len(joints_idx)))
			p2_pos = p2_trajs[i][::4, joints_idx]
			p2_pos = p2_pos.reshape((p2_pos.shape[0], 3*len(joints_idx)))

			p1_vel = np.diff(p1_pos, axis=0, prepend=p1_pos[0:1])
			p2_vel = np.diff(p2_pos, axis=0, prepend=p2_pos[0:1])

			p1_traj = np.hstack([p1_pos, p1_vel])
			p2_traj = np.hstack([p2_pos, p2_vel])
			
			input_dim = output_dim = p1_traj.shape[-1]
			seq_len = p1_traj.shape[0]
			idx = np.array([np.arange(i,i+window_length) for i in range(seq_len + 1 - 2*window_length)])
			self.input_data.append(p2_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*input_dim)))
			idx = np.array([np.arange(i,i+window_length) for i in range(window_length, seq_len + 1 - window_length)])
			self.output_data.append(p1_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*output_dim)))

			self.traj_data.append(np.concatenate([self.input_data[-1], self.output_data[-1]], axis=-1))

		self.input_data = np.array(self.input_data, dtype=object)
		self.output_data = np.array(self.output_data, dtype=object)
		self.traj_data = np.array(self.traj_data, dtype=object)
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]

		self.len = len(self.input_data)
		
		# For mild_hri
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

		# # For buetepage_hri
		# self.labels = []
		# for idx in range(len(self.actidx)):
		# 	for i in range(self.actidx[idx][0],self.actidx[idx][1]):
		# 		label = np.zeros((self.traj_data[i].shape[0],len(self.actidx)))
		# 		label[:, idx] = 1
		# 		self.labels.append(label)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return np.hstack([self.input_data[index], self.output_data[index]]).astype(np.float32), self.labels[index].astype(np.int32)
