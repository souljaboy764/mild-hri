import torch
from torch.nn. functional import grid_sample, affine_grid

import numpy as np
import os
import argparse

from read_buetepage_data import read_data, joints_dic

def preproc(src_dir):
	
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	
	idx_list = np.array([joints_dic[joint] for joint in [
														'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
														# 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
														# 'Hips', 'Spine1', 'Neck', 'Head', 
														]])
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
				
				# readjusting coordinates to have x axis twoards front, y axis towards left, and z axis upwards
				traj1 = traj1[..., np.array([2,0,1])]
				traj1[..., 1] *= -1
				traj2 = traj2[..., np.array([2,0,1])]
				traj2[..., 1] *= -1
				
				# Avoiding reshaping (n_joints, 6) at once since order gets messed up
				# Now [..., :3] is the first actor and [..., 3:] is the second actor 
				traj = np.concatenate([traj1, traj2], axis=-1) # seq_len, n_joints, 6
				traj = traj - traj[0,0]
				trajectories.append(traj)
				
				labels = np.ones((traj.shape[0]))*a				
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
	
	return train_data, train_labels, test_data, test_labels

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Data preprocessing for Right arm trajectories of Buetepage et al. (2020).')
	parser.add_argument('--src-dir', type=str, default='~/playground/human_robot_interaction_data', metavar='SRC',
						help='Path where https://github.com/souljaboy764/human_robot_interaction_data is extracted to read csv files (default: ./human_robot_interaction_data).')
	parser.add_argument('--dst-dir', type=str, default='./data/buetepage/', metavar='DST',
						help='Path to save the processed trajectories to (default: ./data/buetepage/).')
	args = parser.parse_args()
	
	train_data, train_labels, test_data, test_labels = preproc(args.src_dir)
	if not os.path.exists(args.dst_dir):
		os.mkdir(args.dst_dir)
	np.savez_compressed(os.path.join(args.dst_dir, 'buetepage_hh_dataset.npz'), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
