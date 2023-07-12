import torch
from torch.utils.data import DataLoader

import numpy as np
import os, datetime, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd
import pbdlib_torch as pbd_torch
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description='HSMM VAE Training')
# parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
# 					help='Checkpoint to resume training from (default: None)')
# args = parser.parse_args()
# torch.autograd.set_detect_anomaly(True)

for model_type, use_cov, ckpt_name in [
					('mild_vanilla', False, 'final.pth'),
					('mild_vanilla', False, 'final_250.pth'),
					('mild_vanilla', False, 'final_100_finetuning.pth'),
					# ('mild_crossrecon_nocovcond', False, 'final_250.pth'),
					# ('mild_crossrecon_covcond', True, 'final.pth'),
					# ('mild_crossrecon_condsampling', True, 'final.pth'),
					# ('mild_crossrecon_samplecond', False, 'final.pth'),
					# ('vae_vanilla', False, 'final.pth'),
					# ('vae_crossrecon_nocovcond', False, 'final.pth'),
					# ('vae_crossrecon_covcond', True, 'final.pth'),
					# ('vae_crossrecon_condsampling', True, 'final.pth'),
					# ('vae_crossrecon_samplecond', False, 'final.pth'),
					]:
	pred_mse = []
	vae_mse = []
	for trial in range(4):
		ckpt_path = f'logs/2023/downsampled/bp_pepper/{model_type}/z5/trial{trial}/models/{ckpt_name}'
		ckpt = torch.load(ckpt_path)
		hyperparams = np.load(os.path.join(os.path.dirname(ckpt_path),'hyperparams.npz'), allow_pickle=True)
		args_ckpt = hyperparams['args'].item()
		global_config = hyperparams['global_config'].item()
		ae_config = hyperparams['ae_config'].item()
		robot_vae_config = hyperparams['robot_vae_config'].item()
		if args_ckpt.dataset == 'buetepage_pepper':
			dataset = dataloaders.buetepage.PepperWindowDataset
		elif args_ckpt.dataset == 'buetepage':
			dataset = dataloaders.buetepage.HHWindowDataset
		# TODO: Nuitrack

		test_iterator = DataLoader(dataset(args_ckpt.src, train=False, window_length=global_config.window_size, downsample=global_config.downsample), batch_size=1, shuffle=False)

		model_h = getattr(networks, args_ckpt.model)(**(ae_config.__dict__)).to(device)
		model_h.load_state_dict(ckpt['model_h'])
		model_h.eval()
		model_r = getattr(networks, args_ckpt.model)(**(robot_vae_config.__dict__)).to(device)
		model_r.load_state_dict(ckpt['model_r'])
		model_r.eval()
		hsmm = ckpt['hsmm']
		# print(ckpt['epoch'])
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
				pred_mse += ((xr_cond - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints)).sum(-1).mean(-1).detach().cpu().numpy().tolist()
				vae_mse += ((xr_gen - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints)).sum(-1).mean(-1).detach().cpu().numpy().tolist()
			
	print(f'{model_type} {use_cov} \t{np.mean(pred_mse):.4e} ± {np.std(pred_mse):.4e} \t{np.mean(vae_mse):.4e} ± {np.std(vae_mse):.4e}')