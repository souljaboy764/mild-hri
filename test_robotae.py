import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
# from config.buetepage import robot_vae_config
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, model_r, optimizer):
	iters = 0
	total_recon = []
	total_reg = []
	total_loss = []
	zr_dim = model_r.latent_dim
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		# idx = idx[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		# print(label)
		label = label[0]
		xr = x[:, -model_r.input_dim:] # x[0] = Agent 1, x[1] = Agent 2

		if model_r.training:
			optimizer.zero_grad()
 
		# xh_gen, zh_post_samples, zh_post_mean, zh_post_var = model_h(xh)
		# xr_gen, zr_post_samples, zr_post_mean, zr_post_var = model_r(xr)
		
		xr_gen, zr_post_samples, zr_post_dist = model_r(xr)
		x_gen = xr_gen
		
		if model_r.training and isinstance(model_r, networks.VAE):
			recon_loss_r = F.mse_loss(xr.repeat(11,1,1), xr_gen, reduction='sum')
		else:
			recon_loss_r = F.mse_loss(xr, xr_gen, reduction='sum')

		reg_loss = 0.

		loss = recon_loss_r + reg_loss

		total_recon.append(recon_loss_r)
		total_reg.append(reg_loss)
		total_loss.append(loss)

		if model_r.training:
			loss.backward()
			optimizer.step()
		iters += 1
	
	return total_recon, total_reg, total_loss, x_gen, zr_post_samples, x, iters

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Robot AE Testing')
	parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Path to read checkpoint')			
						 
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	MODELS_FOLDER = os.path.dirname(args.ckpt)
	
	hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	saved_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	global_config = hyperparams['global_config'].item()
	# config_r = hyperparams['ae_config'].item()
	config_r = config.buetepage.robot_vae_config()

	HUMAN_MODELS_FOLDER = os.path.dirname(saved_args.human_ckpt)
	hyperparams_h = np.load(os.path.join(HUMAN_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	saved_args_h = hyperparams_h['args'].item() # overwrite args if loading from checkpoint
	# vae_config = getattr(config, saved_args.dataset).ae_config()
	vae_h_config = hyperparams_h['ae_config'].item()

	print("Creating Model and Optimizer from previous CKPT")
	model_r = networks.AE(**(config_r.__dict__)).to(device)
	model_h = getattr(networks, saved_args_h.model)(**(vae_h_config.__dict__)).to(device)
	print("Loading Checkpoints",saved_args.human_ckpt)
	ckpt = torch.load(saved_args.human_ckpt)
	model_h.load_state_dict(ckpt['model'])
	model_h.eval()
	ckpt = torch.load(args.ckpt)
	model_r.load_state_dict(ckpt['model_r'])
	model_r.eval()
	
	print("Reading Data")
	# dataset = getattr(dataloaders, args.dataset)
	if model_r.window_size ==1:
		dataset = dataloaders.buetepage_hr.SequenceDataset(saved_args.src, train=False)
	else:
		dataset = dataloaders.buetepage_hr.SequenceWindowDataset(saved_args.src, train=False, window_length=model_r.window_size)
	NUM_ACTIONS = len(dataset.actidx)
	
	actions = ['Waving', 'Handshaking', 'Rocket Fistbump', 'Parachute Fistbump']
	reconstruction_error, gt_data, gen_data, lens = [], [], [], []
	for i in range(NUM_ACTIONS):
		s = dataset.actidx[i]
		for j in range(s[0], s[1]):
			x, label = dataset[j]
			gt_data.append(x)
			x = torch.Tensor(x).to(device)
			seq_len, dims = x.shape
			lens.append(seq_len)
			x1_gt = x[:, :model_h.input_dim]
			x2_gt = x[:, -model_r.input_dim:]

			with torch.no_grad():
				x1_gen, _, _ = model_h(x1_gt)
				x2_gen, _, _ = model_r(x2_gt)
			
			reconstruction_error.append(F.mse_loss(x2_gt, x2_gen,reduction='none').detach().cpu().numpy())
			gen_data.append(torch.concat([x1_gen, x2_gen], dim=-1).detach().cpu().numpy())
	np.savez_compressed('robotae_test.npz', x_gen=np.array(gen_data), test_data=np.array(gt_data), lens=lens)