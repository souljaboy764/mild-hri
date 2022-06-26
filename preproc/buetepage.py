import torch
from torch.nn. functional import grid_sample, affine_grid

import numpy as np
import os
import argparse

from read_buetepage_data import read_data, joints_dic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def windowed_preproc(trajectories, labels, window_length=20):
	hsmmvae_inputs = []
	sequences = []
	seq_lens = []
	for i in range(len(trajectories)):
		trajs_concat = []
		seq_len, num_joints, dim = trajectories[i].shape
		new_len = seq_len + 1 - window_length
		idx = np.array([np.arange(j,j+window_length) for j in range(new_len)])

		for traj in [trajectories[i][:,:,:3], trajectories[i][:,:,3:]]:
			trajs_concat.append(traj.reshape(seq_len, -1)[idx].reshape((new_len, window_length*num_joints*dim//2)))
		
		idx_list = np.arange(new_len).reshape(-1, 1)
		labels_list = labels[i][:new_len].argmax(1).reshape(-1, 1)
		trajs_concat = np.concatenate(trajs_concat+[idx_list,labels_list],axis=-1)
		if i == 0:
			hsmmvae_inputs = trajs_concat
		else:
			hsmmvae_inputs = np.vstack([hsmmvae_inputs, trajs_concat])
		
	return np.array(hsmmvae_inputs)

def preproc(src_dir, downsample_len=250, augment=False):
	
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	
	# idx_list = np.array([joints_dic[joint] for joint in ['Hips', 'Spine1', 'Neck', 'Head', 
	# 													'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
	# 													'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']])
	idx_list = np.array([joints_dic[joint] for joint in ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand']])
	# idx_list = np.array([joints_dic[joint] for joint in ['RightHand']])
	theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device)
	theta = theta.repeat(idx_list.shape[0],1,1)
	actions = ['hand_wave', 'hand_shake', 'rocket', 'parachute']
	
	for a in range(len(actions)):
		action = actions[a]
		traj_labels = []
		trajectories = []
		for trial in ['1','2']:
			data_file_p1 = os.path.join(src_dir, 'hh','p1',action+'_s1_'+trial+'.csv')
			data_p1, data_q, names, times = read_data(data_file_p1)

			data_file_p2 = os.path.join(src_dir, 'hh','p2',action+'_s2_'+trial+'.csv')
			data_p2, data_q, names, times = read_data(data_file_p2)
		
			segment_file = os.path.join(src_dir, 'hh', 'segmentation', action+'_'+trial+'.npy')
			segments = np.load(segment_file)
			for s in segments:
				traj1 = data_p1[s[0]:s[1], idx_list] # seq_len, n_joints, 3
				traj2 = data_p2[s[0]:s[1], idx_list] # seq_len, n_joints, 3
				traj = np.concatenate([traj1, traj2], axis=-1) # seq_len, n_joints, 6
				traj = traj - traj[0,0]
				# downsample = int((s[1] - s[0])*0.4)
				if downsample_len > 0:
				# if True:
					traj = traj.transpose(1,2,0) # n_joints, 6, seq_len
					traj = torch.Tensor(traj).to(device).unsqueeze(2) # n_joints, 6, 1 seq_len
					traj = torch.concat([traj, torch.zeros_like(traj)], dim=2) # n_joints, 6, 2 seq_len
					
					grid = affine_grid(theta, torch.Size([len(idx_list), 3, 2, downsample_len]), align_corners=True)
					traj = grid_sample(traj.type(torch.float32), grid, align_corners=True) # n_joints, 6, 2 new_length
					traj = traj[:, :, 0].cpu().detach().numpy() # n_joints, 6, new_length
					traj = traj.transpose(2,0,1) # new_length, n_joints, 6
				
				# Avoiding reshaping (n_joints, 6) at once since order gets messed up
				# Now [..., :(n_joints*3)] is the first actor and [..., (n_joints*3):] is the second actor 
				shape = traj.shape
				trajectories.append(np.concatenate([traj[:,:,:3].reshape((shape[0],-1)), traj[:,:,3:].reshape((shape[0],-1))], axis=-1))
				
				labels = np.ones((traj.shape[0]))*a
				
				# the indices where no movement occurs at the end are annotated as "not active". (Sec. 4.3.1 of the paper)
				# notactive_idx = np.where(np.sqrt(np.power(np.diff(traj, axis=0),2).sum((2))).mean(1) > 1e-3)[0]
				# labels[notactive_idx[-1]:] = action_onehot[-1]
				
				traj_labels.append(labels)
		
		# the first 80% are for training and the last 20% are for testing (Sec. 4.3.2)
		split_idx = int(0.8*len(trajectories))
		train_data += trajectories[:split_idx]
		test_data += trajectories[split_idx:]
		train_labels += traj_labels[:split_idx]
		test_labels += traj_labels[split_idx:]
	
	train_data = np.array(train_data)
	test_data = np.array(test_data)
	train_labels = np.array(train_labels)
	test_labels = np.array(test_labels)
	print('Sequences: Training',train_data.shape, 'Testing', test_data.shape)
	print('Labels: Training',train_labels.shape, 'Testing', test_labels.shape)
	
	if augment: # Augment only if downsampling the trajectories
		M = np.eye(downsample_len)*2
		for i in range(1,downsample_len):
			M[i][i-1] = M[i-1][i] = -1

		B = torch.Tensor(np.linalg.pinv(M) * 5e-5).to(device)
		L = torch.linalg.cholesky(B) # faster/momry-friendly to directly give cholesky
		num_trajs = len(train_data)
		n_augments = 70
		for i in range(num_trajs):
			augments = torch.distributions.MultivariateNormal(torch.Tensor(train_data[i]).to(device).reshape(len(idx_list)*6, downsample_len), scale_tril=L).sample((n_augments,)).cpu().numpy()
			train_data = np.concatenate([train_data,augments.reshape(n_augments, downsample_len, len(idx_list), 6)], 0)
			train_labels = np.concatenate([train_labels,np.repeat(train_labels[i:i+1], augments.shape[0], axis=0)], 0)
		print('Augmented Sequences: Training',train_data.shape, 'Testing', test_data.shape)
		print('Augmented Labels: Training',train_labels.shape, 'Testing', test_labels.shape)
	return train_data, train_labels, test_data, test_labels

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Data preprocessing for Right arm trajectories of Buetepage et al. (2020).')
	parser.add_argument('--src-dir', type=str, default='/home/vignesh/playground/human_robot_interaction_data', metavar='SRC',
						help='Path where https://github.com/souljaboy764/human_robot_interaction_data is extracted to read csv files (default: ./human_robot_interaction_data).')
	parser.add_argument('--dst-dir', type=str, default='./data/buetepage/', metavar='DST',
						help='Path to save the processed trajectories to (default: ./data/buetepage/).')
	parser.add_argument('--downsample-len', type=int, default=0, metavar='NEW_LEN',
						help='Length to downsample trajectories to. If 0, no downsampling is performed (default: 0).')
	parser.add_argument('--augment', action="store_true",
						help='Whether to skip the trajectory augmentation or not. (default: False).')
	args = parser.parse_args()
	
	train_data, train_labels, test_data, test_labels = preproc(args.src_dir, args.downsample_len, args.augment)
	if args.dst_dir is not None:
		if not os.path.exists(args.dst_dir):
			os.mkdir(args.dst_dir)
		np.savez_compressed(os.path.join(args.dst_dir, 'traj_data.npz'), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)

		# windowed_train_data = windowed_preproc(train_data, train_labels)
		# windowed_test_data = windowed_preproc(test_data, test_labels)
		# np.savez_compressed(os.path.join(args.dst_dir,'windowed_data.npz'), train_data=windowed_train_data, test_data=windowed_test_data)