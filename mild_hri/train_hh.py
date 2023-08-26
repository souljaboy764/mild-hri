import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse

import mild_hri.vae
from mild_hri.utils import *
from mild_hri.dataloaders import *

import pbdlib as pbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, ssm, model, optimizer, args, epoch):
	iters = 0
	total_recon, total_reg, total_loss = [], [], []
	mu_prior, Sigma_chol_prior, alpha_prior, alpha_argmax_prior = [], [], [], []
	z_dim = args.latent_dim
	if ssm!=[] and model.training:
		with torch.no_grad():
			for i in range(len(ssm)):
				mu_prior.append(torch.concat([ssm[i].mu[None,:,:z_dim], ssm[i].mu[None, :,z_dim:]]))
				Sigma_chol_prior.append(batchNearestPDCholesky(torch.concat([ssm[i].sigma[None, :, :z_dim, :z_dim], ssm[i].sigma[None, :, z_dim:, z_dim:]])))
				alpha_prior.append(ssm[i].forward_variable(marginal=[], sample_size=1000))
				alpha_argmax_prior.append(alpha_prior[-1].argmax(0))
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # (2, seq_len, dims) x[0] = Agent 1, x[1] = Agent 2

		if model.training:
			optimizer.zero_grad()

		# with torch.cuda.amp.autocast(dtype=torch.bfloat16):
		if True: #
			xh_gen, zh_samples, zh_post = model(x[0])
			xr_gen, zr_samples, zr_post = model(x[1])
			if args.cov_cond:
				data_Sigma_in = zh_post.covariance_matrix
			else:
				data_Sigma_in = None
			
			if model.training:
				x_gen = torch.concat([xh_gen[:, None], xr_gen[:, None]], dim=1) # (mce_samples, 2, seq_len, dims)
				recon_loss = F.mse_loss(x[None].repeat(args.mce_samples+1,1,1,1), x_gen, reduction='sum')

				if args.variant == 1:
					variant_loss = torch.zeros_like(recon_loss)
				if args.variant == 2:
					# Sample Conditioning: Sampling from Posterior and then Conditioning the HMM
					zr_cond_mean = []
					for zh in zh_samples:
						zr_cond = ssm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=alpha_prior[label][:, :seq_len], 
														return_cov=False, data_Sigma_in=data_Sigma_in)
						zr_cond_mean.append(zr_cond[None])

					zr_cond_mean = torch.concat(zr_cond_mean)
					xr_cond = model._output(model._decoder(zr_cond_mean))
					variant_loss = F.mse_loss(x[None,1].repeat(args.mce_samples+1,1,1), xr_cond, reduction='mean')
				
				if args.variant == 3 or args.variant == 4:
					# Conditional Sampling: Conditioning the HMM with the Posterior and Sampling from this Conditional distribution
					zr_cond_mean, zr_cond_sigma = ssm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=alpha_prior[label][:, :seq_len], 
													return_cov=True, data_Sigma_in=data_Sigma_in)
					if args.variant == 3:
						zr_cond_L = batchNearestPDCholesky(zr_cond_sigma)
						zr_cond_samples = zr_cond_mean + (zr_cond_L@torch.randn((model.mce_samples, seq_len, model.latent_dim, 1), device=device))[..., 0]
					else:
						zr_cond_stddev = torch.sqrt(torch.diagonal(zr_cond_sigma, dim2=-2, dim1=-1))
						zr_cond_samples = zr_cond_mean + zr_cond_stddev*torch.randn((model.mce_samples, seq_len, model.latent_dim), device=device)

					zr_cond_samples = torch.concat([zr_cond_samples, zr_cond_mean[None]])
					xr_cond = model._output(model._decoder(zr_cond_samples))
					variant_loss = F.mse_loss(x[None,1].repeat(args.mce_samples+1,1,1), xr_cond, reduction='mean')
				
				# gamma = 0.99**epoch
				# recon_loss = gamma*recon_loss + (1-gamma)*variant_loss
			else:
				# x_gen = torch.concat([xh_gen[None], xr_gen[None]]) # (2, seq_len, dims)
				# recon_loss = F.mse_loss(x, x_gen, reduction='sum')
				zr_cond = ssm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim),
												return_cov=False, data_Sigma_in=data_Sigma_in)
				
				xr_cond = model._output(model._decoder(zr_cond))
				recon_loss = F.mse_loss(x[1], xr_cond, reduction='sum')
			
			if model.training and epoch!=0:	
				seq_alpha = alpha_argmax_prior[label][:seq_len]
				with torch.no_grad():
					zh_prior = torch.distributions.MultivariateNormal(mu_prior[label][0, seq_alpha], scale_tril=Sigma_chol_prior[label][0, seq_alpha])
					zr_prior = torch.distributions.MultivariateNormal(mu_prior[label][1, seq_alpha], scale_tril=Sigma_chol_prior[label][1, seq_alpha])
				reg_loss = torch.distributions.kl_divergence(zh_post, zh_prior).mean() + torch.distributions.kl_divergence(zr_post, zr_prior).mean()
				total_reg.append(reg_loss)
				loss = recon_loss + args.beta*reg_loss
			else:
				loss = recon_loss
				total_reg.append(0)

			total_recon.append(recon_loss)
			total_loss.append(loss)

		if model.training:
			loss.backward()
			optimizer.step()
		iters += 1
	
	return total_recon, total_reg, total_loss, iters

if __name__=='__main__':
	args = training_hh_argparse()
	print('Random Seed',args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.autograd.set_detect_anomaly(True)

	if args.dataset == 'buetepage':
		dataset = buetepage.HHWindowDataset
	elif args.dataset == 'nuisi':
		dataset = nuisi.HHWindowDataset
	# TODO: Nuitrack
	
	print("Reading Data")
	train_iterator = DataLoader(dataset(args.src, train=True, window_length=args.window_size, downsample=args.downsample), batch_size=1, shuffle=True)
	test_iterator = DataLoader(dataset(args.src, train=False, window_length=args.window_size, downsample=args.downsample), batch_size=1, shuffle=False)
	print("Creating Model and Optimizer")

	DEFAULT_RESULTS_FOLDER = args.results
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
	global_step = 0
	global_epochs = 0

	print("Creating Model and Optimizer")
	model = getattr(mild_hri.vae, args.model)(**(args.__dict__)).to(device)
	params = model.parameters()
	named_params = model.named_parameters()
	optimizer = torch.optim.AdamW(params, lr=args.lr, fused=True)

	MODELS_FOLDER = os.path.join(args.results, "models")
	SUMMARIES_FOLDER = os.path.join(args.results, "summary")
	global_step = 0
	global_epochs = 0

	NUM_ACTIONS = len(test_iterator.dataset.actidx)
	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)

	if not os.path.exists(args.results):
		print("Creating Result Directory")
		os.makedirs(args.results)
	if not os.path.exists(MODELS_FOLDER):
		print("Creating Model Directory")
		os.makedirs(MODELS_FOLDER)
	if not os.path.exists(SUMMARIES_FOLDER):
		print("Creating Model Directory")
		os.makedirs(SUMMARIES_FOLDER)

	if args.ckpt is not None:
		ckpt = torch.load(args.ckpt)
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		ssm = ckpt['ssm']
		# global_epochs = ckpt['epoch']

	print("Starting Epochs")
	ssm = init_ssm_torch(2*args.latent_dim, args.ssm_components, args.ssm, NUM_ACTIONS, device)

	torch.compile(model)
	for epoch in range(global_epochs, args.epochs):
		model.train()
		train_recon, train_kl, train_loss, iters = run_iteration(train_iterator, ssm, model, optimizer, args, epoch)
		model.eval()
		with torch.no_grad():
			# Updating Prior
			for a in range(len(train_iterator.dataset.actidx)):
				s = train_iterator.dataset.actidx[a]
				z_encoded = []
				lens = []
				for j in range(s[0], s[1]):
				# for j in np.random.randint(s[0], s[1], 12):
					x, label = train_iterator.dataset[j]
					x = torch.Tensor(x).to(device)
					seq_len, dims = x.shape
					lens.append(seq_len)
					x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
					
					zh = model(x[0], encode_only=True)
					zr = model(x[1], encode_only=True)
					# zh = model(x[0], encode_only=True)
					# zr = model(x[1], encode_only=True)
					z_encoded.append(torch.concat([zh, zr], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				ssm_np = getattr(pbd, args.ssm)(nb_dim=2*args.latent_dim, nb_states=args.ssm_components)
				ssm_np.init_hmm_kbins(z_encoded)
				ssm_np.em(z_encoded, reg=args.cov_reg, reg_finish=args.cov_reg)
				# ssm[a].init_hmm_kbins(z_encoded)
				# ssm[a].em(z_encoded)
				for k in vars(ssm_np).keys():
					if isinstance(ssm_np.__getattribute__(k), np.ndarray):
						ssm[a].__setattr__(k, torch.Tensor(ssm_np.__getattribute__(k)).to(device).requires_grad_(False))
					else:
						ssm[a].__setattr__(k, ssm_np.__getattribute__(k))
				# if args.variant == 3:
				# 	ssm[a].reg = cov_reg.to(device).requires_grad_(False)
				# else:
				# 	ssm[a].reg = torch.Tensor(ssm_np.reg).to(device).requires_grad_(False)

				# z_encoded = torch.concat(z_encoded)
				# z_encoded = np.concatenate(z_encoded)
				# for zdim in range(args.latent_dim):
				# 	writer.add_histogram(f'z_h/{a}_{zdim}', z_encoded[:,zdim], epoch)
				# 	writer.add_histogram(f'z_r/{a}_{zdim}', z_encoded[:,args.latent_dim+zdim], epoch)
				# writer.add_image(f'hmm_{a}_trans', ssm[a].Trans, epoch, dataformats='HW')
				# alpha_ssm = ssm[a].forward_variable(marginal=[], sample_size=np.mean(lens).astype(int))
				# writer.add_histogram(f'alpha/{a}', alpha_ssm.argmax(0), epoch)

		if epoch % 10 == 0 or epoch==args.epochs-1:
			with torch.no_grad():
				test_recon, test_kl, test_loss, iters = run_iteration(test_iterator, ssm, model, optimizer, args, epoch)
			write_summaries_vae(writer, train_recon, train_kl, epoch, 'train')
			write_summaries_vae(writer, test_recon, test_kl, epoch, 'test')
			params = []
			grads = []
			for name, param in list(model.named_parameters())+list(model.named_parameters()):
			# for name, param in model.named_parameters():
				if param.grad is None:
					continue
				writer.add_histogram('grads/'+name, param.grad.reshape(-1), epoch)
				writer.add_histogram('param/'+name, param.reshape(-1), epoch)
				if torch.allclose(param.grad, torch.zeros_like(param.grad)):
					print('zero grad for',name)

			for a in range(len(ssm)):
				alpha_ssm = ssm[a].forward_variable(marginal=[], sample_size=np.mean(lens).astype(int))
				writer.add_histogram(f'alpha/{a}', alpha_ssm.argmax(0), epoch)
		
			
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.3d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ssm':ssm, 'args':args}, checkpoint_file)
			# torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ssm':ssm, 'args':args}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch}.pth')
	torch.save({'model': model.state_dict(), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ssm':ssm, 'args':args}, checkpoint_file)
	# torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ssm':ssm, 'args':args}, checkpoint_file)
