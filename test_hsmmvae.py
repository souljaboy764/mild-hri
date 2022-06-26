import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
from config import global_config, human_vae_config
from utils import *
import dataloaders

import pbdlib as pbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, hsmm, model, optimizer):
	iters = 0
	total_recon = []
	total_reg = []
	total_loss = []
	z_dim = model.latent_dim
	for i, x in enumerate(iterator):
		x, idx, label = x
		x = x[0]
		idx = idx[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		label = label[0]
		if not isinstance(model, networks.AE):
			alpha_hsmm, _, _, _, _ = hsmm[label].compute_messages(marginal=[], sample_size=seq_len)
			if np.any(np.isnan(alpha_hsmm)):
				alpha_hsmm = forward_variable(hsmm[label], seq_len)
			mu_prior = torch.Tensor([hsmm[label].mu[:,:z_dim], hsmm[label].mu[:,z_dim:]]).to(device)
			Sigma_prior = torch.Tensor([hsmm[label].sigma[:, :z_dim, :z_dim], hsmm[label].sigma[:, z_dim:, z_dim:]]).to(device)
			alpha_hsmm = torch.Tensor(alpha_hsmm).to(device)
			# seq_alpha = alpha_hsmm.argmax(-1, keepdim=True).cpu().numpy()

		if model.training:
			optimizer.zero_grad()

		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		x_gen, zpost_samples, zpost_dist = model(x)
		if model.training and not isinstance(model, networks.AE):
			recon_loss = F.mse_loss(x[None].repeat(11,1,1,1), x_gen, reduction='sum')
		else:
			recon_loss = F.mse_loss(x, x_gen, reduction='sum')

		reg_loss = 0.
		if not isinstance(model, networks.AE):
			for c in range(hsmm[label].nb_states):
				z_prior = torch.distributions.MultivariateNormal(mu_prior[:, c:c+1].repeat(1,seq_len,1), Sigma_prior[:, c:c+1].repeat(1,seq_len,1,1))
				kld = torch.distributions.kl_divergence(zpost_dist, z_prior).mean(0)
				reg_loss += (alpha_hsmm[c]*kld).mean()
			# z_prior = torch.distributions.MultivariateNormal(mu_prior[:, seq_alpha], Sigma_prior[:, seq_alpha])
			# reg_loss = torch.distributions.kl_divergence(zpost_dist, z_prior).mean()
		loss = recon_loss/model.beta + reg_loss

		total_recon.append(recon_loss)
		total_reg.append(reg_loss)
		total_loss.append(loss)

		if model.training:
			loss.backward()
			optimizer.step()
		iters += 1
	
	return total_recon, total_reg, total_loss, x_gen, zpost_samples, x, iters

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--ckpt', type=str, default='./logs/results/'+datetime.datetime.now().strftime("%m%d%H%M"), metavar='RES',
						help='Path for saving results (default: ./logs/results/MMDDHHmm).')
	parser.add_argument('--src', type=str, default='/home/vignesh/playground/buetepage-phri/data/hsmmvae_downsampled/hsmmvae_data.npz', metavar='RES',
						help='Path to read training and testin data (default: ./data/orig/vae/data.npz).')
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	MODELS_FOLDER = os.path.dirname(args.ckpt)
	hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	saved_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	config = hyperparams['global_config'].item()
	vae_config = hyperparams['vae_config'].item()

	print("Creating Model and Optimizer")
	model = getattr(networks, saved_args.model)(**(vae_config.__dict__)).to(device)
	z_dim = model.latent_dim
	
	print("Loading Checkpoints")
	ckpt = torch.load(args.ckpt)
	model.load_state_dict(ckpt['model'])
	hsmm = ckpt['hsmm']
	model.eval()
	
	print("Reading Data")
	dataset = getattr(dataloaders, saved_args.dataset)(saved_args.src, train=False)
	NUM_ACTIONS = len(test_iterator.actidx)

	print("Starting Epochs")
	reconstruction_error, gt_data, gen_data = [], [], []
	for i in range(NUM_ACTIONS):
		s = dataset.actidx[i]
		for j in range(s[0], s[1]):
			x, idx, label = dataset[j]
			gt_data.append(x)
			assert np.all(label == i)
			x = torch.Tensor(x).to(device)
			seq_len, dims = x.shape
			x2_gt = x[:, dims//2:]
			z1 = model(x[:, :dims//2], encode_only=True).detach().cpu().numpy()
			z2, _ = hsmm[i].condition(z1, dim_in=slice(0, dims//2), dim_out=slice(dims//2, dims))
			x2_gen = model._output(model._decoder(torch.Tensor(z2).to(device)))
			reconstruction_error.append(F.mse_loss(x2_gt, x2_gen).detach().cpu().numpy())
			gen_data.append(x2_gen.detach().cpu().numpy())

	