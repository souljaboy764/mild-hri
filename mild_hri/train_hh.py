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
	if ssm is not None and model.training:
		with torch.no_grad():
			mu_prior = torch.concat([ssm.mu[None,:,:z_dim], ssm.mu[None, :,-z_dim:]])
			Sigma_chol_prior = batchNearestPDCholesky(torch.concat([ssm.sigma[None, :, :z_dim, :z_dim], ssm.sigma[None, :, -z_dim:, -z_dim:]]))

			alpha_prior = ssm.forward_variable(marginal=[], sample_size=1000)
			alpha_argmax_prior = alpha_prior.argmax(0)
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape # dims = 2*(pos_dim+vel_dim) if  cartesian velocity is used
		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # (2, seq_len, (pos_dim+vel_dim)) x[0] = Agent 1, x[1] = Agent 2

		if model.training:
			optimizer.zero_grad()

		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
			xh_gen, zh_samples, zh_post = model(x[0])
			xr_gen, zr_samples, zr_post = model(x[1])
			
			if model.training:
				x_gen = torch.concat([xh_gen[:, None], xr_gen[:, None]], dim=1) # (mce_samples, 2, seq_len, (pos_dim+vel_dim))
				recon_loss = F.mse_loss(x[None].repeat(args.mce_samples+1,1,1,1), x_gen, reduction='sum')
				total_recon.append(recon_loss)
			else:

				zr_cond = ssm.condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), return_cov=False, data_Sigma_in=None)
				xr_cond = model._output(model._decoder(zr_cond))
				recon_loss = ((x[1] - xr_cond)**2).reshape((xr_cond.shape[0], model.window_size, model.num_joints, model.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()
				total_recon += recon_loss
			
			if model.training and epoch!=0:	
				seq_alpha = alpha_argmax_prior[:seq_len]
				with torch.no_grad():
					zh_prior = torch.distributions.MultivariateNormal(mu_prior[0, seq_alpha], scale_tril=Sigma_chol_prior[0, seq_alpha])
					zr_prior = torch.distributions.MultivariateNormal(mu_prior[1, seq_alpha], scale_tril=Sigma_chol_prior[1, seq_alpha])
				reg_loss = torch.distributions.kl_divergence(zh_post, zh_prior).mean() + torch.distributions.kl_divergence(zr_post, zr_prior).mean()
				total_reg.append(reg_loss)
				loss = recon_loss + args.beta*reg_loss
			else:
				loss = recon_loss
				total_reg.append(0)

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
	elif args.dataset == 'alap':
		dataset = alap.HHWindowDataset
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
	args.num_joints *= 2
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
	ssm = init_ssm_torch(args.latent_dim*2, args.ssm_components, args.ssm, 1, device)[0]

	print("Starting Epochs")

	torch.compile(model)
	for epoch in range(global_epochs, args.epochs):
		model.train()
		train_recon, train_kl, train_loss, iters = run_iteration(train_iterator, ssm, model, optimizer, args, epoch)
		model.eval()
		with torch.no_grad():
			# Updating Prior
			z_encoded = []
			for a in range(len(train_iterator.dataset.actidx)):
				s = train_iterator.dataset.actidx[a]
				lens = []
				for j in range(s[0], s[1]):
				# for j in np.random.randint(s[0], s[1], 12):
					x, label = train_iterator.dataset[j]
					x = torch.Tensor(x).to(device)
					seq_len, dims = x.shape
					lens.append(seq_len)
					x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
					
					zh = model(x[0], encode_only=True)
					# zh_diff = torch.diff(zh,dim=0,prepend=zh[0:1,:])
					zr = model(x[1], encode_only=True)
					z_encoded.append(torch.concat([zh, zr], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
			ssm_np = getattr(pbd, args.ssm)(nb_dim=ssm.nb_dim, nb_states=ssm.nb_states)
			ssm_np.init_hmm_kbins(z_encoded)
			ssm_np.em(z_encoded, reg=args.cov_reg, reg_finish=args.cov_reg)

			for k in vars(ssm_np).keys():
				if isinstance(ssm_np.__getattribute__(k), np.ndarray):
					ssm.__setattr__(k, torch.Tensor(ssm_np.__getattribute__(k)).to(device).requires_grad_(False))
				else:
					ssm.__setattr__(k, ssm_np.__getattribute__(k))
				
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

			# for a in range(len(ssm)):
			# 	alpha_ssm = ssm[a].forward_variable(marginal=[], sample_size=np.mean(lens).astype(int))
			# 	writer.add_histogram(f'alpha/{a}', alpha_ssm.argmax(0), epoch)
			alpha_ssm = ssm.forward_variable(marginal=[], sample_size=np.mean(lens).astype(int))
			writer.add_histogram(f'alpha', alpha_ssm.argmax(0), epoch)
		
			
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.3d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ssm':ssm, 'args':args}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch}.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ssm':ssm, 'args':args}, checkpoint_file)
