import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from mild_hri import vae
from mild_hri.dataloaders import *

import pbdlib_torch as pbd_torch

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = '../../logs/2023/bp_hh_20hz/v1_0/diaghmm_z3h9/trial1/models/0070.pth'
ckpt = torch.load(ckpt_path)
args_ckpt = ckpt['args']
ssm = ckpt['ssm']

model = getattr(vae, args_ckpt.model)(**(args_ckpt.__dict__)).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
z_dim = model.latent_dim

# if args_ckpt.dataset == 'buetepage_pepper':
# 	dataset = dataloaders.buetepage.PepperWindowDataset
# elif args_ckpt.dataset == 'buetepage':
dataset = buetepage.HHWindowDataset
# TODO: Nuitrack

test_dataset = dataset(os.path.join('../../',args_ckpt.src), train=False, window_length=args_ckpt.window_size, downsample=args_ckpt.downsample)
actions = ['Hand Wave', 'Hand Shake', 'Rocket Fistbump', 'Parachute Fistbump']

fig = plt.figure()
plt.ion()
ax_skel = fig.add_subplot(3, 1, 1, projection='3d')
ax_alpha = fig.add_subplot(3, 1, 2)
ax_latent = fig.add_subplot(3, 1, 3, projection='3d')
# plt.show()
actidx = np.hstack(test_dataset.actidx - np.array([0,1]))
for a in actidx:
	x, label = test_dataset[a]
	seq_len = x.shape[0]
	dims_h = model.input_dim
	x = torch.Tensor(x).to(device)
	x_h = x[:, :dims_h]
	x_r = x[:, dims_h:]
	
	xh_gen, _, zh_post = model(x_h)
	xr_gen, _, zr_post = model(x_r)
	
	alpha = ssm[label].forward_variable(marginal=[], sample_size=seq_len).T.detach().cpu().numpy()
	alpha_h = ssm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim)).T.detach().cpu().numpy()
	
	alpha_idx = np.linspace(0, 1, seq_len)
	for i in range(0, seq_len, 10):
		ax_alpha.cla()
		ax_alpha.set_xlim(0,1)
		ax_alpha.set_ylim(0,1)
		ax_alpha.plot(alpha_idx[:i+1], alpha_h[:i+1, :])
			
		ax_latent.clear()
		ax_latent.scatter3D(zh_post.mean[:i+1:10, 0].detach().cpu().numpy(), zh_post.mean[:i+1:10, 1].detach().cpu().numpy(), zh_post.mean[:i+1:10, 2].detach().cpu().numpy(), 'r.', alpha=0.1)
		ax_latent.scatter3D(zr_post.mean[:i+1:10, 0].detach().cpu().numpy(), zr_post.mean[:i+1:10, 1].detach().cpu().numpy(), zr_post.mean[:i+1:10, 2].detach().cpu().numpy(), 'b.', alpha=0.1)
		for k in range(ssm[label].nb_states):
			pbd_torch.plot_gauss3d(ax_latent, ssm[label].mu[k, :3].detach().cpu().numpy(), ssm[label].sigma[k, :3, :3].detach().cpu().numpy(),
						color='red', alpha=max(0.05, alpha_h[i,k]))
			pbd_torch.plot_gauss3d(ax_latent, ssm[label].mu[k, z_dim:z_dim+3].detach().cpu().numpy(), ssm[label].sigma[k, z_dim:z_dim+3, z_dim:z_dim+3].detach().cpu().numpy(),
						color='blue', alpha=max(0.05, alpha_h[i,k]))
		plt.pause(0.001)

	# break
