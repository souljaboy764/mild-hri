import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pbdlib as pbd

from datetime import datetime

import networks
import config
from utils import *
import dataloaders

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_types = "ablation_vae"
z_dims = [10]
z_trajs = []
z_trajs_cov = []

z_dim = 3
ckpt = f'logs/2023/buetepage_downsampled/mild_sophia_ablation_5hsmm/z3/trial0/models/final.pth' # input()
dirname = os.path.dirname(ckpt)
hyperparams = np.load(os.path.join(dirname,'hyperparams.npz'), allow_pickle=True)
args = hyperparams['args'].item()
ckpt = torch.load(ckpt)

model = getattr(networks, args.model)(**(hyperparams['ae_config'].item().__dict__)).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
z_dim = 3
nb_dim = 4*z_dim
# dataset = getattr(dataloaders, args.dataset)
train_iterator = dataloaders.buetepage_downsampled.SequenceWindowDataset('./data/buetepage/traj_data.npz', train=True, window_length=40)
test_iterator = dataloaders.buetepage_downsampled.SequenceWindowDataset('./data/buetepage/traj_data.npz', train=False, window_length=40)
# for a in range(len(train_iterator.actidx)):
# 	s = train_iterator.actidx[a]
# 	z_encoded = []
# 	z_encoded_cov = []
# 	for j in range(s[0], s[1]):
# 	# for j in np.random.randint(s[0], s[1], 12):
# 		x, label = train_iterator[j]
# 		assert np.all(label == a)
# 		x = torch.Tensor(x).to(device)
# 		_, seq_len, dims = x.shape
# 		# x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		
# 		# zpost_samples = model(x, encode_only=True)
# 		# if model.window_size == 1:
# 		# 	z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
# 		# 	z2_vel = torch.diff(zpost_samples[1], prepend=zpost_samples[1][0:1], dim=0)
# 		# 	z_encoded.append(torch.concat([zpost_samples[0], z1_vel, zpost_samples[1], z2_vel], dim=-1).detach().cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
# 		# else:
# 		# 	z_encoded.append(torch.concat([zpost_samples[0], zpost_samples[1]], dim=-1).detach().cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)

# 		x_gen, zpost_samples, zpost_dist = model(x)
# 		z_encoded.append(torch.concat([zpost_dist.mean[0], zpost_dist.mean[1]], dim=-1).detach().cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
# 		z_encoded_cov.append(torch.concat([zpost_dist.covariance_matrix[0], zpost_dist.covariance_matrix[1]], dim=-1).detach().cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
# 	z_encoded = np.array(z_encoded)
# 	z_encoded_cov = np.array(z_encoded_cov)
# 	z_trajs.append(z_encoded)
# 	z_trajs_cov.append(z_encoded_cov)

# np.savez_compressed('z_trajs.npz', mu=z_trajs, covs=z_trajs_cov)

data = np.load('z_trajs.npz', allow_pickle=True)
z_trajs = data['mu']
z_trajs_cov = data['covs']
nb_states = 6
num_train = 5

print('Conditioning with Cov')
for nb_states in [4,5,6,10,12]:
	hsmm = [pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states) for a in range(len(train_iterator.actidx))]
	
	mse_error = []
	mse_error_cov = []
	for a in range(len(train_iterator.actidx)):
		hsmm[a].init_hmm_kbins(z_trajs[a])
		hsmm[a].em(z_trajs[a], nb_max_steps=100)
		
		for i in range(len(z_trajs[a])):
			z_traj = np.array(z_trajs[a][i])
			z_traj_cov = np.array(z_trajs_cov[a][i])
			z2_pred, _ = hsmm[a].condition(z_traj[:, :z_dim], z_traj_cov[:, :, :z_dim], dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			z1_pred, _ = hsmm[a].condition(z_traj[:, z_dim:], z_traj_cov[:, :, z_dim:], dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
			
			mse_i = ((z_traj[:, z_dim:] - z2_pred)**2).sum(-1)
			mse_error_cov += mse_i.tolist()

			z2_pred, _ = hsmm[a].condition(z_traj[:, :z_dim], None, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			z1_pred, _ = hsmm[a].condition(z_traj[:, z_dim:], None, dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
			
			mse_i = ((z_traj[:, z_dim:] - z2_pred)**2).sum(-1)
			mse_error += mse_i.tolist()
			# recon_error += recon_i.tolist()
	print(f"| W/o Cov | {a} | {nb_states} | {np.mean(mse_error):.4e} ± {np.std(mse_error):.4e} |")
	print(f"| W/  Cov | {a} | {nb_states} | {np.mean(mse_error_cov):.4e} ± {np.std(mse_error_cov):.4e} |")
			# print(f'MSE: {np.sum(mse_error)}')
	print('')

# for i in range(len(z_trajs)):
# 	for j in range(len(z_trajs[i])):
# 		z_vel = np.diff(z_trajs[i][j], axis=0, prepend=z_trajs[i][j][0:1])
# 		z_trajs[i][j] = np.concatenate([z_trajs[i][j][:, :z_dim], z_vel[:, :z_dim], z_trajs[i][j][:, z_dim:], z_vel[:, z_dim:]], axis=1)
# print('with velocity')
# for nb_states in [4,5,6,10,12]:
# 	hsmm = [pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states) for a in range(len(train_iterator.actidx))]
	
# 	for a in range(len(train_iterator.actidx)):
# 		# train_idx = np.random.choice(np.arange(len(z_trajs[a])),num_train,False).astype(int)
# 		z_trajs_train = []
# 		for z_traj in z_trajs[a]:
# 			z_trajs_train.append(z_traj[::5])
# 		train_idx = np.arange(len(z_trajs[a])).astype(int)
# 		hsmm[a].init_hmm_kbins(z_trajs_train)
# 		hsmm[a].em(z_trajs_train, nb_max_steps=100)
	
# 		mse_error = []
# 		for z_traj in z_trajs[a]:
# 			z_traj = np.array(z_traj)
# 			z2_pred, sigma2 = hsmm[a].condition(z_traj[:, :2*z_dim], None, dim_in=slice(0, 2*z_dim), dim_out=slice(2*z_dim, 3*z_dim))
# 			# z2 = hsmm[label].condition(zpost_dist.mean[0].detach().cpu().numpy(), dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			
# 			mse_i = ((z_traj[:, 2*z_dim:3*z_dim] - z2_pred)**2).sum(-1)
# 			mse_error += mse_i.tolist()
# 			# recon_error += recon_i.tolist()
# 		print(f"| {a} | {nb_states} | {np.mean(mse_error):.4e} ± {np.std(mse_error):.4e} |")
# 			# print(f'MSE: {np.sum(mse_error)}')
# 	print('')

