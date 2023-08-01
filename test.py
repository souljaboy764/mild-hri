import torch
from torch.utils.data import DataLoader

import numpy as np
import os, datetime, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import config
from utils import *
from vae import *
import dataloaders

import pbdlib as pbd
import pbdlib_torch as pbd_torch
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_ckpt(ckpt_path, use_cov):
	ckpt = torch.load(ckpt_path)
	hyperparams = np.load(os.path.join(os.path.dirname(ckpt_path),'hyperparams.npz'), allow_pickle=True)
	args_ckpt = hyperparams['args'].item()
	ae_config = hyperparams['ae_config'].item()
	robot_vae_config = hyperparams['robot_vae_config'].item()
	if args_ckpt.dataset == 'buetepage_pepper':
		dataset = dataloaders.buetepage.PepperWindowDataset
	elif args_ckpt.dataset == 'buetepage':
		dataset = dataloaders.buetepage.HHWindowDataset
	# TODO: Nuitrack

	test_iterator = DataLoader(dataset(args_ckpt.src, train=False, window_length=args_ckpt.window_size, downsample=args_ckpt.downsample), batch_size=1, shuffle=False)

	model_h = VAE(**(ae_config.__dict__)).to(device)
	model_h.load_state_dict(ckpt['model_h'])
	model_h.eval()
	model_r = VAE(**(robot_vae_config.__dict__)).to(device)
	model_r.load_state_dict(ckpt['model_r'])
	model_r.eval()
	hsmm = ckpt['hsmm']

	pred_mse = []
	vae_mse = []
	with torch.no_grad():
		for i, x in enumerate(test_iterator):
			# if i<7:
			# 	continue
			x, label = x
			x = x[0]
			label = label[0]
			x = torch.Tensor(x).to(device)
			x_h = x[:, :model_h.input_dim]
			x_r = x[:, model_h.input_dim:]
			z_dim = hsmm[label].nb_dim//2
			
			zh_post = model_h(x_h, dist_only=True)
			xr_gen, _, _ = model_r(x_r)
			if use_cov:
				data_Sigma_in = zh_post.covariance_matrix
			else: 
				data_Sigma_in = None
			zr_cond = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), 
											data_Sigma_in=data_Sigma_in,
											return_cov=False) 
			xr_cond = model_r._output(model_r._decoder(zr_cond))
			# pred_mse += ((xr_cond - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints)).mean(-1).mean(-1).detach().cpu().numpy().tolist()
			# vae_mse += ((xr_gen - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints)).mean(-1).mean(-1).detach().cpu().numpy().tolist()
			pred_mse += ((xr_cond - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints, model_r.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()
			vae_mse += ((xr_gen - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints, model_r.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()

	return pred_mse, vae_mse


if __name__=='__main__':
	# parser = argparse.ArgumentParser(description='HSMM VAE Training')
	# parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
	# 					help='Checkpoint to resume training from (default: None)')
	# args = parser.parse_args()
	# torch.autograd.set_detect_anomaly(True)

	for model_type, use_cov in [
							('v1_0', False),
							('v2_1', False),
							('v2_2', True),
							('v3_1', False),
							('v3_2', True),
							('v4_1', False),
							('v4_2', True),
						]:
		for ckpt_name in [
							'final_100.pth', 
							# 'final_100_finetuning.pth', 
							# 'final_199.pth', 
							# 'final_199_finetuning.pth',
						]:
			pred_mse = []
			vae_mse = []
			for trial in range(4):
				# ckpt_path = f'logs/2023/bp_pepper_downsampled_klweight/{model_type}/models/{ckpt_name}'
				ckpt_path = f'logs/2023/bp_pepper_downsampled_robotfuture_ablation/z05/{model_type}/vanilla/trial{trial}/models/{ckpt_name}'
				pred_mse_ckpt, vae_mse_ckpt = evaluate_ckpt(ckpt_path, use_cov)
				pred_mse += pred_mse_ckpt
				vae_mse += vae_mse_ckpt
			if ckpt_name[10:] == 'finetuning.pth':
				ckpt_name = 'tuned_'+ckpt_name[6:9]
			# elif ckpt_name == 'final_100.pth':
				# ckpt_name = 'final_100'
			else:
				ckpt_name = ckpt_name[:-4]
			model_name = f'{model_type:<40}' + str(use_cov)[0] + f' {ckpt_name}'
			print(f'{model_name}\t{np.mean(pred_mse):.4e} ± {np.std(pred_mse):.4e} \t{np.mean(vae_mse):.4e} ± {np.std(vae_mse):.4e}')