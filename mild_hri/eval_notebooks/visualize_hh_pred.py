import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from mild_hri import vae
from mild_hri.utils import mypause
from mild_hri.dataloaders import *

import pbdlib_torch as pbd_torch

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='HSMM VAE Training')
parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
args = parser.parse_args()

ckpt = torch.load(args.ckpt)
args_ckpt = ckpt['args']
ssm = ckpt['ssm']

model = getattr(vae, args_ckpt.model)(**(args_ckpt.__dict__)).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
z_dim = model.latent_dim

if args_ckpt.dataset == 'buetepage':
	dataset = buetepage.HHWindowDataset
	actions = ['Hand Wave', 'Hand Shake', 'Rocket Fistbump', 'Parachute Fistbump']
elif args_ckpt.dataset == 'nuisi':
	dataset = nuisi.HHWindowDataset
	actions = ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']
# TODO: Nuitrack

test_dataset = dataset(os.path.join(args_ckpt.src), train=False, window_length=args_ckpt.window_size, downsample=args_ckpt.downsample)

fig = plt.figure()
spec = fig.add_gridspec(2, 2)
plt.ion()
ax_skel = fig.add_subplot(spec[0,0], projection='3d')
ax_latent = fig.add_subplot(spec[0,1], projection='3d')
ax_alpha = fig.add_subplot(spec[1,:])
plt.show(block=False)
actidx = np.hstack(test_dataset.actidx - np.array([0,1]))

for a in actidx[::2]:
	x, label = test_dataset[a]
	seq_len = x.shape[0]
	dims_h = model.input_dim
	x = torch.Tensor(x).to(device)
	x_h = x[:, :dims_h]
	x_r = x[:, dims_h:]
	
	xh_gen, _, zh_post = model(x_h)
	xr_gen, _, zr_post = model(x_r)

	zr_cond, _ = ssm[label].condition(zh_post.mean, data_Sigma_in=None, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
	zh_cond, _ = ssm[label].condition(zh_post.mean, data_Sigma_in=None, dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
	x_cond = model._output(model._decoder(torch.concat([zh_cond[None], zr_cond[None]])))
	
	
	alpha = ssm[label].forward_variable(marginal=[], sample_size=seq_len).T.detach().cpu().numpy()
	alpha_h = ssm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim)).T.detach().cpu().numpy()

	x_h = x_h.reshape(seq_len, model.window_size, model.num_joints, model.joint_dims).detach().cpu().numpy()
	x_r = x_r.reshape(seq_len, model.window_size, model.num_joints, model.joint_dims).detach().cpu().numpy()
	xh_gen = xh_gen.reshape(seq_len, model.window_size, model.num_joints, model.joint_dims).detach().cpu().numpy()
	xr_gen = xr_gen.reshape(seq_len, model.window_size, model.num_joints, model.joint_dims).detach().cpu().numpy()
	xh_cond = x_cond[0].reshape(seq_len, model.window_size, model.num_joints, model.joint_dims).detach().cpu().numpy()
	xr_cond = x_cond[1].reshape(seq_len, model.window_size, model.num_joints, model.joint_dims).detach().cpu().numpy()

	z_h = zh_post.mean.detach().cpu().numpy()
	z_r = zr_post.mean.detach().cpu().numpy()
	zh_cond = zh_cond.detach().cpu().numpy()
	zr_cond = zr_cond.detach().cpu().numpy()

	x_r[:,:,:,0] = 0.7 - x_r[:,:,:,0]
	x_r[:,:,:,1] *= -1
	xr_gen[:,:,:,0] = 0.7 - xr_gen[:,:,:,0]
	xr_gen[:,:,:,1] *= -1
	xr_cond[:,:,:,0] = 0.7 - xr_cond[:,:,:,0]
	xr_cond[:,:,:,1] *= -1

	mu_h = ssm[label].mu[:, :3].detach().cpu().numpy()
	mu_r = ssm[label].mu[:, z_dim:z_dim+3].detach().cpu().numpy()
	sigma_h = ssm[label].sigma[:, :3, :3].detach().cpu().numpy()
	sigma_r = ssm[label].sigma[:, z_dim:z_dim+3, z_dim:z_dim+3].detach().cpu().numpy()
	
	alpha_idx = np.linspace(0, 1, seq_len)
	for i in range(seq_len):
		ax_skel.cla()
		ax_skel.set_xlabel('X')
		ax_skel.set_ylabel('Y')
		ax_skel.set_zlabel('Z')
		ax_skel.set_xlim(-0.05,0.75)
		ax_skel.set_ylim(-0.4,0.4)
		ax_skel.set_zlim(-0.6,0.2)
		for w in range(model.window_size):
			ax_skel.plot(x_h[i,w,:,0], x_h[i,w,:,1], x_h[i,w,:,2], 'k-', markerfacecolor='r', marker='o', alpha=(w+1)/model.window_size)
			ax_skel.plot(x_r[i,w,:,0], x_r[i,w,:,1], x_r[i,w,:,2], 'k-', markerfacecolor='b', marker='o', alpha=(w+1)/model.window_size)
			ax_skel.plot(xh_cond[i,w,:,0], xh_cond[i,w,:,1], xh_cond[i,w,:,2], 'k--', markerfacecolor='m', marker='o', alpha=(w+1)/model.window_size)
			ax_skel.plot(xr_cond[i,w,:,0], xr_cond[i,w,:,1], xr_cond[i,w,:,2], 'k--', markerfacecolor='c', marker='o', alpha=(w+1)/model.window_size)

		ax_alpha.cla()
		ax_alpha.set_xlim(0,1)
		ax_alpha.set_ylim(-0.01,1.01)
		ax_alpha.plot(alpha_idx[:i+1], alpha_h[:i+1, :])
			
		ax_latent.cla()
		for k in range(ssm[label].nb_states):
			pbd_torch.plot_gauss3d(ax_latent, mu_h[k], sigma_h[k], color='r', alpha=max(0.1,alpha[i,k])*0.8)
			pbd_torch.plot_gauss3d(ax_latent, mu_r[k], sigma_r[k], color='b', alpha=max(0.1,alpha[i,k])*0.8)
		ax_latent.scatter3D(z_h[i, 0], z_h[i, 1], z_h[i, 2], c='r')
		ax_latent.scatter3D(z_r[i, 0], z_r[i, 1], z_r[i, 2], c='b')
		ax_latent.scatter3D(zh_cond[i, 0], zh_cond[i, 1], zh_cond[i, 2], c='m')
		ax_latent.scatter3D(zr_cond[i, 0], zr_cond[i, 1], zr_cond[i, 2], c='c')
		mypause(0.01)
		if not plt.fignum_exists(fig.number):
			break
	if not plt.fignum_exists(fig.number):
		break
	# break
