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
	actions = ['Hand_Over', 'High_Five', 'Pull_Up', 'Shake_Hands', 'Wave_Hands']
	action_onehot = np.eye(len(actions))
	
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	
	# idx_list = np.array([joints_dic[joint] for joint in ['RightHand']])
	# theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(len(idx_list),1,1)

	
	for a in range(len(actions)):
		num_trajs = 0
		num_rejected = 0
		action = actions[a]
		action_dir = os.path.join(src_dir, action)
		for sc in sorted(os.listdir(action_dir)):
			annotation_dir = os.path.join(action_dir, sc, 'Annotation_'+action)
			skeleton_dir = os.path.join(action_dir, sc, 'Skeletons_'+action)			
			for trial in sorted(os.listdir(skeleton_dir)):
				trial_dir = os.path.join(skeleton_dir,trial)
				if sc=='sc1':
					annotation_file = open(os.path.join(annotation_dir,'Annotation_'+trial+'.txt'))
				else:
					annotation_file = open(os.path.join(annotation_dir,trial+'.txt'))
				annotations = []
				for line in annotation_file.read().splitlines():
					annotation = list(map(int,line.split()))
					if annotation !=[]:
						annotations.append(annotation) # ID, starting frame, ending frame, role of agent 1, role of agent 2
				skeleton_trajs = {}
				skeleton_timestamps = {}
				demo_num = 0
				for skeleton_file in sorted(os.listdir(trial_dir)):
					# print(skeleton_file)
					_, timestep, skelid = skeleton_file.split('.')[0].split('_')
					# timestep = int(timestep)
					# if timestep < annotations[demo_num][1]:
					# 	continue
					# if timestep == annotations[demo_num][2]:
					# 	demo_num += 1
					# 	if demo_num>=len(annotations):
					# 		break
					
					skeleton = np.array([list(map(float,i.split()[1:4])) for i in open(os.path.join(trial_dir,skeleton_file)).read().splitlines()[1:26]])
					if skelid in skeleton_trajs:
						skeleton_trajs[skelid].append(skeleton)
						skeleton_timestamps[skelid].append(int(timestep))
					else:
						skeleton_trajs[skelid] = [skeleton]
						skeleton_timestamps[skelid] = [int(timestep)]

				lens = []
				idx = []
				for skelid in skeleton_trajs:
					lens.append(len(skeleton_trajs[skelid]))
					idx.append(skelid)
				if len(set(lens))==1:
					# print(action,sc,trial,lens,idx,len(annotations))
					num_trajs += len(annotations)
				else:
				# 	print('Mismatch in',a,sc,trial,lens,idx)
				# 	print('')
					num_rejected += len(annotations)
		print('Num Trajs and Rejected for Action',action,':',num_trajs, num_rejected)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Data preprocessing for skeleton trajectories of the HHOI dataset')
	parser.add_argument('--src-dir', type=str, default='/home/vignesh/playground/HHOI', metavar='SRC',
						help='Path where the HHOI dataset is extracted to read csv files (default: /home/vignesh/playground/HHOI).')
	parser.add_argument('--dst-dir', type=str, default='./data/HHOI/', metavar='DST',
						help='Path to save the processed trajectories to (default: ./data/HHOI/).')
	parser.add_argument('--downsample-len', type=int, default=0, metavar='NEW_LEN',
						help='Length to downsample trajectories to. If 0, no downsampling is performed (default: 0).')
	parser.add_argument('--augment', action="store_true",
						help='Whether to skip the trajectory augmentation or not. (default: False).')
	args = parser.parse_args()
	
	# train_data, train_labels, test_data, test_labels = 
	preproc(args.src_dir, args.downsample_len, args.augment)
	# np.savez_compressed(os.path.join(args.dst_dir, 'raw_data.npz'), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
	# if args.dst_dir is not None:
	# 	if not os.path.exists(args.dst_dir):
	# 		os.mkdir(args.dst_dir)

		# windowed_train_data = windowed_preproc(train_data, train_labels)
		# windowed_test_data = windowed_preproc(test_data, test_labels)
		# np.savez_compressed(os.path.join(args.dst_dir,'windowed_data.npz'), train_data=windowed_train_data, test_data=windowed_test_data)