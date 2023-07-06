import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Sophia import SophiaG

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pbdlib as pbd
import pbdlib_torch as pbd_torch

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

z_dim = 3
# ckpt = f'logs/2023/buetepage_downsampled/mild_sophia_ablation_5hsmm/z3/trial0/models/final.pth' # input()
# dirname = os.path.dirname(ckpt)
# hyperparams = np.load(os.path.join(dirname,'hyperparams.npz'), allow_pickle=True)
# args = hyperparams['args'].item()
# ckpt = torch.load(ckpt)

# model = getattr(networks, args.model)(**(hyperparams['ae_config'].item().__dict__)).to(device)
# model.load_state_dict(ckpt['model'])
# model.eval()
z_dim = 3
nb_dim = 2*z_dim
# dataset = getattr(dataloaders, args.dataset)
train_iterator = dataloaders.buetepage_downsampled.SequenceWindowDataset('./data/buetepage/traj_data.npz', train=True, window_length=40)
test_iterator = dataloaders.buetepage_downsampled.SequenceWindowDataset('./data/buetepage/traj_data.npz', train=False, window_length=40)
# for a in range(len(train_iterator.actidx)):
# 	s = train_iterator.actidx[a]
# 	z_encoded = []
# 	for j in range(s[0], s[1]):
# 	# for j in np.random.randint(s[0], s[1], 12):
# 		x, label = train_iterator[j]
# 		assert np.all(label == a)
# 		x = nn.Parameter(torch.Tensor(x).to(device),requires_grad=True)
# 		seq_len, dims = x.shape
# 		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		
# 		zpost_samples = model(x, encode_only=True)
# 		if model.window_size == 1:
# 			z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
# 			z2_vel = torch.diff(zpost_samples[1], prepend=zpost_samples[1][0:1], dim=0)
# 			z_encoded.append(torch.concat([zpost_samples[0], z1_vel, zpost_samples[1], z2_vel], dim=-1).detach().cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
# 		else:
# 			z_encoded.append(torch.concat([zpost_samples[0], zpost_samples[1]], dim=-1).detach().cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
# 	z_encoded = np.array(z_encoded)
# 	z_trajs.append(z_encoded)

# np.savez_compressed('z_trajs.npz', z_trajs)

z_trajs = np.load('z_trajs.npz', allow_pickle=True)['arr_0']
nb_states = 5
num_train = 5

hsmm = [pbd.HMM(nb_dim=nb_dim, nb_states=nb_states) for a in range(len(train_iterator.actidx))]

mse_error_orig = []
mse_error_adapted = []
times = []
for a in range(len(train_iterator.actidx)):
	hsmm = pbd.HMM(nb_dim=nb_dim, nb_states=nb_states)
	hsmm.init_hmm_kbins(z_trajs[a])
	hsmm.em(z_trajs[a])
	
	for z_traj in z_trajs[a]:
		z_traj = np.array(z_traj)
		z2_pred, sigma2 = hsmm.condition(z_traj[:, :z_dim], None, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
		z1_pred, sigma1 = hsmm.condition(z_traj[:, z_dim:], None, dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
		
		mse_i_1 = ((z_traj[:, :z_dim] - z1_pred)**2).sum(-1)
		mse_i_2 = ((z_traj[:, z_dim:] - z2_pred)**2).sum(-1)
		mse_error_orig += mse_i_1.tolist()
		mse_error_orig += mse_i_2.tolist()

	tril_indices = torch.tril_indices(row=z_dim, col=z_dim, offset=0)
	hsmm_torch = pbd_torch.HMM(nb_dim=nb_dim, nb_states=nb_states)
	hsmm_torch._mu = nn.Parameter(torch.Tensor(hsmm.mu).to(device),requires_grad=True)
	hsmm_torch._sigma_chol_ = nn.Parameter(torch.Tensor(hsmm.sigma_chol).to(device)[..., tril_indices[0], tril_indices[1]],requires_grad=True)
	hsmm_torch._trans_logits = nn.Parameter(torch.Tensor(hsmm.trans).to(device),requires_grad=True)
	hsmm_torch._init_priors_logits = nn.Parameter(torch.Tensor(hsmm.init_priors).to(device),requires_grad=True)
	hsmm_torch._reg = torch.Tensor(hsmm.reg).to(device)
	hsmm_torch._sigma_chol = torch.Tensor(hsmm.sigma_chol).to(device)
	# hsmm_torch.mu_d = nn.Parameter(torch.Tensor(hsmm.mu_d).to(device),requires_grad=True)
	# hsmm_torch.sigma_d = nn.Parameter(torch.Tensor(hsmm.sigma_d).to(device),requires_grad=True)
	# hsmm_torch.trans_d = nn.Parameter(torch.Tensor(hsmm.trans_d).to(device),requires_grad=True)
	# hsmm_torch.Mu_Pd = nn.Parameter(torch.Tensor(hsmm.Mu_Pd).to(device),requires_grad=True)
	# hsmm_torch.Sigma_Pd = nn.Parameter(torch.Tensor(hsmm.Sigma_Pd).to(device),requires_grad=True)
	# hsmm_torch.Trans_Pd = nn.Parameter(torch.Tensor(hsmm.Trans_Pd).to(device),requires_grad=True)

	params = [hsmm_torch._mu, hsmm_torch._sigma_chol_, hsmm_torch._trans_logits, hsmm_torch._init_priors_logits,
	   		#  hsmm_torch.mu_d, hsmm_torch.sigma_d, hsmm_torch.trans_d, hsmm_torch.Mu_Pd, hsmm_torch.Sigma_Pd, hsmm_torch.Trans_Pd
			 ]
	
	chol_mask = torch.tril(torch.ones_like(hsmm_torch._sigma_chol_))
	chol_mask.requires_grad = False

	# optimizer = SophiaG(params, lr=1e-3, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
	optimizer = torch.optim.SGD(params, lr=1e35)
	for i in range(len(z_trajs[a])):
		z_trajs[a][i] = torch.Tensor(np.array(z_trajs[a][i])).to(device)

	lowest = torch.finfo(torch.float32).max
	best_model = None
	for epoch in range(30):
		total_loss = 0
		for i in np.random.choice(np.arange(len(z_trajs[a])),len(z_trajs[a]),False):
			optimizer.zero_grad()
			hsmm_torch._sigma = None
			hsmm_torch._lambda = None
			hsmm_torch._init_priors = hsmm_torch._init_priors_logits / (torch.sum(hsmm_torch._init_priors_logits) + pbd_torch.realmin)
			hsmm_torch._trans = hsmm_torch._trans_logits / (torch.sum(hsmm_torch._trans_logits, dim=1) + pbd_torch.realmin)
			hsmm_torch._sigma_chol[..., tril_indices[0], tril_indices[1]].copy_(hsmm_torch._sigma_chol_)

			assert torch.all(torch.diagonal(hsmm_torch._sigma_chol, dim1=-1, dim2=-2) >=0)
			hsmm_torch._set_sigmas()
			z_traj = z_trajs[a][i]
			z2_pred = hsmm_torch.condition(z_traj[:, :z_dim], dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), return_cov=False)
			# z1_pred = hsmm_torch.condition(z_traj[:, z_dim:], dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim), return_cov=False)
			# loss = F.mse_loss(torch.concat([z1_pred,z2_pred], dim=-1), z_traj)
			loss = F.mse_loss(z2_pred, z_traj[:, z_dim:], reduction='sum')
			loss.backward()
			# torch.nn.utils.clip_grad_value_(params, 0.1)
			optimizer.step()
			total_loss += loss

		if lowest > total_loss:
			lowest = total_loss
			best_model = []
			for p in params:
				best_model.append(p[:])
			print('saving best model')
		print(f'Epoch {epoch}: Loss: {total_loss.detach().cpu().item()}')

	hsmm_torch._sigma = None
	hsmm_torch._lambda = None
	hsmm_torch._sigma_chol[..., tril_indices[0], tril_indices[1]].copy_(hsmm_torch._sigma_chol_)
	hsmm_torch._set_sigmas()
		
	hsmm.mu = best_model[0].detach().cpu().numpy()
	hsmm.sigma = best_model[1].matmul(best_model[1].transpose(-1,-2)).detach().cpu().numpy()
	hsmm.trans = best_model[2].detach().cpu().numpy()
	hsmm.trans /= (np.sum(hsmm.trans, axis=1) + pbd.realmin)
	hsmm.init_priors = best_model[3].detach().cpu().numpy() 
	hsmm.init_priors /= (np.sum(hsmm.init_priors) + pbd.realmin)
	# hsmm.priors = best_model[4].detach().cpu().numpy()
	# hsmm.mu_d = hsmm_torch.mu_d.detach().cpu().numpy()
	# hsmm.sigma_d = hsmm_torch.sigma_d.detach().cpu().numpy()
	# hsmm.trans_d = hsmm_torch.trans_d.detach().cpu().numpy()
	# hsmm.Mu_Pd = hsmm_torch.Mu_Pd.detach().cpu().numpy()
	# hsmm.Sigma_Pd = hsmm_torch.Sigma_Pd.detach().cpu().numpy()
	# hsmm.Trans_Pd = hsmm_torch.Trans_Pd.detach().cpu().numpy()
	for z_traj in z_trajs[a]:
		z_traj = z_traj.cpu().numpy()
		z2_pred, sigma2 = hsmm.condition(z_traj[:, :z_dim], None, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
		z1_pred, sigma1 = hsmm.condition(z_traj[:, z_dim:], None, dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
		
		mse_i_1 = ((z_traj[:, :z_dim] - z1_pred)**2).sum(-1)
		mse_i_2 = ((z_traj[:, z_dim:] - z2_pred)**2).sum(-1)
		mse_error_adapted += mse_i_1.tolist()
		mse_error_adapted += mse_i_2.tolist()

		
print(f"MSE Without Adaptation: {np.mean(mse_error_orig):.4e} ± {np.std(mse_error_orig):.4e}")
print(f"MSE With Adaptation: {np.mean(mse_error_adapted):.4e} ± {np.std(mse_error_adapted):.4e}")