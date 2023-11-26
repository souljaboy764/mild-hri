import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

from vae import VAE
from utils import *
from phd_utils.dataloaders import *

import pbdlib_torch as pbd_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_iteration(iterator, ssm, model_h, model_r, optimizer, args, epoch, priors):
	iters = 0
	total_recon, total_reg, total_loss = [], [], []
	mu_prior, Sigma_chol_prior, alpha_prior, alpha_argmax_prior = priors
	z_dim = args.latent_dim
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		x_h = x[:, :model_h.input_dim]
		x_r = x[:, model_h.input_dim:]
		
		if model_r.training:
			optimizer.zero_grad()

		# with torch.cuda.amp.autocast(dtype=torch.bfloat16):
		if True: #
			xr_gen, zr_samples, zr_post = model_r(x_r)
			# xr_gen = xr_gen * args_r.joints_range + args_r.joints_min
			
			if args.variant != 1 or not model_r.training:
				zh_post = model_h(x_h, dist_only=True)
				if args.cov_cond:
					data_Sigma_in = zh_post.covariance_matrix
				else:
					data_Sigma_in = None
			if model_r.training:
				recon_loss = F.mse_loss(x_r[None].repeat(args.mce_samples+1,1,1), xr_gen, reduction='sum')

				if args.variant == 2:
					zh_samples = torch.concat([zh_post.rsample((model_r.mce_samples,)), zh_post.mean[None]])
					# Sample Conditioning: Sampling from Posterior and then Conditioning the HMM
					zr_cond_mean = []
					for zh in zh_samples:
						zr_cond = ssm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=alpha_prior[label][:, :seq_len], 
														return_cov=False, data_Sigma_in=data_Sigma_in)
						zr_cond_mean.append(zr_cond[None])

					zr_cond_mean = torch.concat(zr_cond_mean)
					xr_cond = model_r._output(model_r._decoder(zr_cond_mean))# * args_r.joints_range + args_r.joints_min
					recon_loss = recon_loss + F.mse_loss(x_r[None].repeat(args.mce_samples+1,1,1), xr_cond, reduction='sum')
				
				if args.variant == 3 or args.variant == 4:
					# Conditional Sampling: Conditioning the HMM with the Posterior and Sampling from this Conditional distribution
					zr_cond_mean, zr_cond_sigma = ssm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=alpha_prior[label][:, :seq_len], 
													return_cov=True, data_Sigma_in=data_Sigma_in)
					if args.variant == 3:
						zr_cond_L = batchNearestPDCholesky(zr_cond_sigma)
						zr_cond_samples = zr_cond_mean + (zr_cond_L@torch.randn((model_r.mce_samples, seq_len, model_r.latent_dim, 1), device=device))[..., 0]
					else:
						zr_cond_stddev = torch.sqrt(torch.diagonal(zr_cond_sigma, dim2=-2, dim1=-1)) + pbd_torch.realmin
						zr_cond_samples = zr_cond_mean + zr_cond_stddev*torch.randn((model_r.mce_samples, seq_len, model_r.latent_dim), device=device)

					zr_cond_samples = torch.concat([zr_cond_samples, zr_cond_mean[None]])
					xr_cond = model_r._output(model_r._decoder(zr_cond_samples))# * args_r.joints_range + args_r.joints_min
					recon_loss = recon_loss + F.mse_loss(x_r.repeat(args.mce_samples+1,1,1), xr_cond, reduction='sum')
				
			else:
				zr_cond = ssm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim),
												return_cov=False, data_Sigma_in=data_Sigma_in)
				
				xr_cond = model_r._output(model_r._decoder(zr_cond))# * args_r.joints_range + args_r.joints_min
				recon_loss = F.mse_loss(x_r, xr_cond, reduction='sum')
			
			if model_r.training and epoch!=0:	
				seq_alpha = alpha_argmax_prior[label][:seq_len]
				with torch.no_grad():
					zr_prior = torch.distributions.MultivariateNormal(mu_prior[label][1, seq_alpha], scale_tril=Sigma_chol_prior[label][1, seq_alpha])
				reg_loss = torch.distributions.kl_divergence(zr_post, zr_prior).sum()
				total_reg.append(reg_loss)
				loss = recon_loss + args.beta*reg_loss
			else:
				loss = recon_loss
				total_reg.append(0)

			total_recon.append(recon_loss)
			total_loss.append(loss)

		if model_r.training:
			loss.backward()
			optimizer.step()
		iters += 1
	
	return total_recon, total_reg, total_loss, iters

if __name__=='__main__':
	args_r = training_hr_argparse()
	ckpt_h = torch.load(args_r.ckpt_h)
	args_h = ckpt_h['args']
	args_r.latent_dim = args_h.latent_dim
	args_r.activation = args_h.activation
	args_r.joint_dims = 1
	print('Random Seed',args_r.seed)
	torch.manual_seed(args_r.seed)
	np.random.seed(args_r.seed)
	torch.autograd.set_detect_anomaly(True)

	if args_r.dataset == 'buetepage_pepper':
		dataset = buetepage.PepperWindowDataset
	elif args_r.dataset == 'buetepage_yumi':
		dataset = buetepage_hr.YumiWindowDataset
	elif args_r.dataset == 'nuisi_pepper':
		dataset = nuisi.PepperWindowDataset
	
	print("Reading Data")
	train_iterator = DataLoader(dataset(train=True, window_length=args_r.window_size, downsample=args_r.downsample), batch_size=1, shuffle=True)
	test_iterator = DataLoader(dataset(train=False, window_length=args_r.window_size, downsample=args_r.downsample), batch_size=1, shuffle=False)

	print("Creating Model and Optimizer")
	ssm = ckpt_h['ssm']
	model_h = VAE(**(args_h.__dict__)).to(device)
	model_h.load_state_dict(ckpt_h['model'])
	model_r = VAE(**{**(args_h.__dict__), **(args_r.__dict__)})
	model_r = model_r.to(device)
	params = model_r.parameters()
	named_params = model_r.named_parameters()
	optimizer = torch.optim.AdamW(params, lr=args_r.lr, fused=True)

	MODELS_FOLDER = os.path.join(args_r.results, "models")
	SUMMARIES_FOLDER = os.path.join(args_r.results, "summary")
	if not os.path.exists(args_r.results):
		print("Creating Result Directory")
		os.makedirs(args_r.results)
	if not os.path.exists(MODELS_FOLDER):
		print("Creating Model Directory")
		os.makedirs(MODELS_FOLDER)
	if not os.path.exists(SUMMARIES_FOLDER):
		print("Creating Model Directory")
		os.makedirs(SUMMARIES_FOLDER)
	global_step = 0
	global_epochs = 0

	if args_r.variant == 3:
		cov_reg = torch.diag_embed(torch.tile(torch.linspace(0.91, 1.0, args_h.latent_dim),(2,)))*args_h.cov_reg
	else:
		cov_reg = args_h.cov_reg
	
	NUM_ACTIONS = len(test_iterator.dataset.actidx)
	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)

	if args_r.ckpt is not None:
		ckpt = torch.load(args_r.ckpt)
		model_r.load_state_dict(ckpt['model_r'])
		optimizer.load_state_dict(ckpt['optimizer'])
		global_epochs = ckpt['epoch']

	torch.compile(model_h)
	torch.compile(model_r)
	model_h.eval()
	mu_prior, Sigma_chol_prior, alpha_prior, alpha_argmax_prior = [], [], [], []
	z_dim = args_r.latent_dim
	if ssm!=[] and model_r.training:
		with torch.no_grad():
			for i in range(len(ssm)):
				mu_prior.append(torch.concat([ssm[i].mu[None,:,:z_dim], ssm[i].mu[None, :,z_dim:]]))
				Sigma_chol_prior.append(torch.concat([batchNearestPDCholesky(ssm[i].sigma[None, :, :z_dim, :z_dim]), batchNearestPDCholesky(ssm[i].sigma[None, :, z_dim:, z_dim:])]))
				alpha_prior.append(ssm[i].forward_variable(marginal=[], sample_size=1000))
				alpha_argmax_prior.append(alpha_prior[-1].argmax(0))
	priors = mu_prior, Sigma_chol_prior, alpha_prior, alpha_argmax_prior
	for epoch in range(global_epochs, args_r.epochs):
		model_r.train()
		train_recon, train_kl, train_loss, iters = run_iteration(train_iterator, ssm, model_h, model_r, optimizer, args_r, epoch, priors)

		if epoch % 10 == 0 or epoch==args_r.epochs-1:
			model_r.eval()
			with torch.no_grad():
				test_recon, test_kl, test_loss, iters = run_iteration(test_iterator, ssm, model_h, model_r, optimizer, args_r, epoch, priors)
			write_summaries_vae(writer, train_recon, train_kl, epoch, 'train')
			write_summaries_vae(writer, test_recon, test_kl, epoch, 'test')
			params = []
			grads = []
			
			for name, param in model_r.named_parameters():
				if param.grad is None:
					continue
				writer.add_histogram('grads_r/'+name, param.grad.reshape(-1), epoch)
				writer.add_histogram('param_r/'+name, param.reshape(-1), epoch)
				if torch.allclose(param.grad, torch.zeros_like(param.grad)):
					print('zero grad for',name)
			
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.3d.pth'%(epoch))
			torch.save({'model_r': model_r.state_dict(), 'model_h': model_h.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args_r':args_r, 'args_h':args_h, 'ssm':ssm}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch}.pth')
	torch.save({'model_r': model_r.state_dict(), 'model_h': model_h.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args_r':args_r, 'args_h':args_h, 'ssm':ssm}, checkpoint_file)
