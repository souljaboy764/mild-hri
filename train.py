import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from vae import *
import config
from utils import *
import dataloaders

import pbdlib as pbd
import pbdlib_torch as pbd_torch

from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(
					iterator:DataLoader, 
		  			hsmm:List[pbd_torch.HMM], 
					model_h:VAE, 
					model_r:VAE, 
					optimizer:torch.optim.Optimizer,
					args,
					epoch,
				):
	total_recon, total_reg, total_loss, mu_prior, Sigma_prior, alpha_prior = [], [], [], [], [], []
	z_dim = model_h.latent_dim
	num_alpha = 1500
	with torch.no_grad():
		for label in range(len(hsmm)):
			mu_prior.append(torch.concat([hsmm[label].mu[None,:,:z_dim], hsmm[label].mu[None, :,z_dim:]]).to(device))
			Sigma_prior.append(torch.concat([hsmm[label].sigma[None, :, :z_dim, :z_dim], hsmm[label].sigma[None, :, z_dim:, z_dim:]]).to(device) + torch.eye(z_dim, device=device)*1e-4)
			alpha_prior.append(hsmm[label].forward_variable(marginal=[], sample_size=num_alpha))

	for i, x in enumerate(iterator):
		# start = datetime.datetime.now()
		if model_h.training:
			optimizer.zero_grad()

		x, label = x
		x = x[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len = x.shape[0]
		x_h = x[:, :model_h.input_dim]
		x_r = x[:, model_h.input_dim:]
		
		xh_gen, zh_samples, zh_post = model_h(x_h)
		xr_gen, zr_samples, zr_post = model_r(x_r)

		if model_h.training:
			fwd_h = alpha_prior[label][:, torch.linspace(0, num_alpha-1, seq_len, dtype=int)]
		else:
			fwd_h = hsmm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim))
			# assert not torch.any(torch.isnan(fwd_h))
			# assert not torch.any(torch.isinf(fwd_h))

		if args.cov_cond:
			data_Sigma_in = zh_post.covariance_matrix
		else:
			data_Sigma_in = None

		if args.variant==2:
			# Sample Conditioning: Sampling from Posterior and then Conditioning the HMM
			xr_cond = []
			zr_cond_mean = []
			for zh in zh_samples:
				zr_cond = hsmm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
												return_cov=False, data_Sigma_in=data_Sigma_in)
				zr_cond_mean.append(zr_cond[None])

			zr_cond_mean = torch.concat(zr_cond_mean)
			# assert not torch.any(torch.isnan(zr_cond_mean))
			# assert not torch.any(torch.isinf(zr_cond_mean))
			xr_cond = model_r._output(model_r._decoder(zr_cond_mean))

		elif args.variant==3 or args.variant==4:
			# Conditioned Sampling: Conditioning on the Posterior and then Sampling from the conditional distribution
			if model_h.training and model_r.mce_samples>0:
				zr_cond_mean, zr_cond_sigma = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
											return_cov=True, data_Sigma_in=data_Sigma_in)
				if args.variant==3:
					zr_cond = torch.distributions.MultivariateNormal(zr_cond_mean, scale_tril=batchNearestPDCholesky(zr_cond_sigma))#, eps=hsmm[label].reg[:z_dim, :z_dim])
				else:# args.variant==4:
					zr_cond = torch.distributions.Normal(zr_cond_mean, torch.sqrt(torch.diagonal(zr_cond_sigma,dim1=-1,dim2=-2)))
				
				# eps = torch.randn((model_r.mce_samples,)+zr_cond_mean.shape, device=zr_cond_mean.device)
				# zr_samples = zr_cond_mean + (zr_cond_L @ eps[..., None])[..., 0]
				xr_cond = model_r._output(model_r._decoder(torch.concat([zr_cond.rsample((model_r.mce_samples,)), zr_cond_mean[None]], dim=0)))
			else:
				zr_cond_mean = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
											return_cov=False, data_Sigma_in=data_Sigma_in)
				xr_cond = model_r._output(model_r._decoder(zr_cond_mean))

		if model_h.training:
			if args.variant==1 or epoch==0:
				recon_loss = ((xh_gen - x_h[None])**2).mean() + ((xr_gen - x_r[None])**2).mean()
				# for m in range(model_r.mce_samples+1):
				# 	recon_loss = recon_loss + torch.norm(PepperFK(xr_gen[m].reshape(-1,4)) @ PepperFK(x_r.reshape(-1,4)).inverse() - torch.eye(4,device=device), dim=(-1,-2)).mean()
			else:
				if args.gamma==1:
					recon_loss = ((xh_gen - x_h[None])**2).mean() + ((xr_gen - x_r[None])**2).mean() + ((xr_cond - x_r[None])**2).mean()
				else:
					factor = 1 - epoch/args.epochs
					recon_loss = ((xh_gen - x_h[None])**2).mean() + 2*(((xr_gen - x_r[None])**2).mean()*factor + ((xr_cond - x_r[None])**2).mean()*(1-factor))
		else:
			if args.variant==1:
				# Simple conditioninal reconstruction
				zr_cond_mean = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
												return_cov=False, data_Sigma_in=data_Sigma_in)
				xr_cond = model_r._output(model_r._decoder(zr_cond_mean))
			# xr_cond = xr_cond*args.joints_range + args.joints_min
			# x_r = x_r*args.joints_range + args.joints_min
			recon_loss = ((xr_cond - x_r)**2).sum()
	
		loss = recon_loss

		if args.beta!=0 and epoch!=0:
			with torch.no_grad():
				alpha_argmax = fwd_h.argmax(0)
				zh_prior = torch.distributions.MultivariateNormal(mu_prior[label][0, alpha_argmax], Sigma_prior[label][0, alpha_argmax])
				zr_prior = torch.distributions.MultivariateNormal(mu_prior[label][1, alpha_argmax], Sigma_prior[label][1, alpha_argmax])
			reg_loss = args.beta * ((torch.distributions.kl_divergence(zh_post, zh_prior) + \
					torch.distributions.kl_divergence(zr_post, zr_prior))).mean()
			loss += reg_loss
			total_reg.append(reg_loss.mean())
		else:
			total_reg.append(0)

		total_recon.append(recon_loss.mean())
		total_loss.append(loss.mean())
		if model_h.training:
			if args.grad_clip!=0:
				torch.nn.utils.clip_grad_norm_(model_h.parameters(), args.grad_clip)
				torch.nn.utils.clip_grad_norm_(model_r.parameters(), args.grad_clip)
			loss.backward()
			optimizer.step()
	return total_recon, total_reg, total_loss, i

if __name__=='__main__':
	args = training_argparse()
	print('Random Seed',args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.autograd.set_detect_anomaly(True)
	
	ae_config = config.buetepage.ae_config()
	if args.dataset == 'buetepage_pepper':
		robot_vae_config = config.buetepage.robot_vae_config()
		robot_vae_config.num_joints = 4
		dataset = dataloaders.buetepage.PepperWindowDataset
	elif args.dataset == 'buetepage':
		robot_vae_config = config.buetepage.ae_config()
		dataset = dataloaders.buetepage.HHWindowDataset
	# TODO: Nuitrack

	robot_vae_config.latent_dim = ae_config.latent_dim = args.latent_dim
	robot_vae_config.window_size = ae_config.window_size = args.window_size
	robot_vae_config.mce_samples = ae_config.mce_samples = args.mce_samples

	print("Reading Data")
	train_iterator = DataLoader(dataset(args.src, train=True, window_length=args.window_size, downsample=args.downsample), batch_size=1, shuffle=True)
	test_iterator = DataLoader(dataset(args.src, train=False, window_length=args.window_size, downsample=args.downsample), batch_size=1, shuffle=False)
	print("Creating Model and Optimizer")

	model_h = VAE(**(ae_config.__dict__)).to(device)
	model_r = VAE(**(robot_vae_config.__dict__)).to(device)
	params = list(model_h.parameters()) + list(model_r.parameters())
	named_params = list(model_h.named_parameters()) + list(model_r.named_parameters())
	optimizer = torch.optim.AdamW(params, lr=args.lr)

	MODELS_FOLDER = os.path.join(args.results, "models")
	SUMMARIES_FOLDER = os.path.join(args.results, "summary")
	global_step = 0
	global_epochs = 0

	if args.variant == 3:
		cov_reg = torch.diag_embed(torch.tile(torch.linspace(0.91, 1.0, args.latent_dim),(2,)))*args.cov_reg
	else:
		cov_reg = args.cov_reg
	
	NUM_ACTIONS = len(test_iterator.dataset.actidx)
	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	
	s = ''
	for k in ae_config.__dict__:
		s += str(k) + ' : ' + str(ae_config.__dict__[k]) + '\n'
	writer.add_text('human_ae_config', s)

	s = ''
	for k in robot_vae_config.__dict__:
		s += str(k) + ' : ' + str(robot_vae_config.__dict__[k]) + '\n'
	writer.add_text('robot_ae_config', s)

	s = ''
	for k in args.__dict__:
		s += str(k) + ' : ' + str(args.__dict__[k]) + '\n'
	writer.add_text('args', s)

	writer.flush()

	if not os.path.exists(args.results):
		print("Creating Result Directory")
		os.makedirs(args.results)
	if not os.path.exists(MODELS_FOLDER):
		print("Creating Model Directory")
		os.makedirs(MODELS_FOLDER)
	if not os.path.exists(SUMMARIES_FOLDER):
		print("Creating Model Directory")
		os.makedirs(SUMMARIES_FOLDER)

	hsmm = []
	nb_dim = 2*args.latent_dim
	nb_states = args.hsmm_components
	with torch.no_grad():
		for i in range(NUM_ACTIONS):
			hsmm_i = pbd_torch.HMM(nb_dim=nb_dim, nb_states=nb_states)
			hsmm_i.init_zeros(device)
			hsmm_i.init_priors = torch.ones(nb_states, device=device) / nb_states
			hsmm_i.Trans = torch.ones((nb_states, nb_states), device=device)/nb_states
			# hsmm_i.Mu_Pd = torch.zeros(nb_states, device=device)
			# hsmm_i.Sigma_Pd = torch.ones(nb_states, device=device)
			# hsmm_i.Trans_Pd = torch.ones((nb_states, nb_states), device=device)/nb_states
			hsmm.append(hsmm_i)
	
	if args.ckpt is not None:
		ckpt = torch.load(args.ckpt)
		model_h.load_state_dict(ckpt['model_h'])
		model_r.load_state_dict(ckpt['model_r'])
		optimizer.load_state_dict(ckpt['optimizer'])
		hsmm = ckpt['hsmm']
		# hyperparams = np.load(os.path.join(os.path.dirname(args.ckpt),'hyperparams.npz'), allow_pickle=True)
		# seed = hyperparams['args'].item().seed
		# torch.manual_seed(seed)
		# np.random.seed(seed)
		global_epochs = ckpt['epoch']
	else:
		np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, ae_config=ae_config, robot_vae_config=robot_vae_config)
		checkpoint_file = os.path.join(MODELS_FOLDER, 'init_ckpt.pth')
		torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0, 'hsmm':[]}, checkpoint_file)
	
	print("Starting Epochs")
	epoch = global_epochs
	for epoch in range(global_epochs, args.epochs):# + global_epochs):
		model_h.train()
		model_r.train()
		train_recon, train_kl, train_loss, iters = run_iteration(train_iterator, hsmm, model_h, model_r, optimizer, args, epoch)
		steps_done = (epoch+1)*iters
		write_summaries_vae(writer, train_recon, train_kl, steps_done, 'train')
		for name, param in named_params:
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		model_h.eval()
		model_r.eval()
		with torch.no_grad():
			entropy = 0
			# Updating Prior
			for a in range(len(train_iterator.dataset.actidx)):
				s = train_iterator.dataset.actidx[a]
				z_encoded = []
				for j in range(s[0], s[1]):
				# for j in np.random.randint(s[0], s[1], 12):
					x, label = train_iterator.dataset[j]
					x = torch.Tensor(x).to(device)
					x_h = x[:, :model_h.input_dim]
					x_r = x[:, model_h.input_dim:]
					
					z_h = model_h(x_h, encode_only=True)
					z_r = model_r(x_r, encode_only=True)
					z_encoded.append(torch.concat([z_h, z_r], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				hsmm_np = pbd.HMM(nb_dim=nb_dim, nb_states=nb_states)
				# hsmm_np.init_hmm_kbins(z_encoded)
				hsmm_np.init_params_scikit(np.concatenate(z_encoded))
				if args.variant == 3:
					hsmm_np.em(z_encoded, reg=cov_reg.cpu().detach().numpy(), reg_finish=cov_reg.cpu().detach().numpy())
				else:
					hsmm_np.em(z_encoded, reg=cov_reg, reg_finish=cov_reg)
				z_encoded = np.concatenate(z_encoded)
				for zdim in range(args.latent_dim):
					writer.add_histogram(f'z_h/{a}_{zdim}', z_encoded[:,zdim], steps_done)
					writer.add_histogram(f'z_r/{a}_{zdim}', z_encoded[:,args.latent_dim+zdim], steps_done)
				writer.add_image(f'hmm_{a}_trans', hsmm_np.Trans*255, steps_done, dataformats='HW')
				alpha = np.zeros((nb_states*10, 100))
				alpha_hsmm = hsmm_np.forward_variable(marginal=[], sample_size=100)
				for n in range(nb_states):
					alpha[n*10:(n+1)*10, :] = alpha_hsmm[n]
				writer.add_image(f'hmm_{a}_alpha', alpha*255, steps_done, dataformats='HW')
				
				writer.add_histogram(f'alpha/{a}', alpha_hsmm.argmax(0), steps_done)

				if args.variant == 3:
					hsmm[a].reg = cov_reg.to(device).requires_grad_(False)
				else:
					hsmm[a].reg = torch.Tensor(hsmm_np.reg).to(device).requires_grad_(False)
				hsmm[a].mu = torch.Tensor(hsmm_np.mu).to(device).requires_grad_(False)
				hsmm[a].sigma = torch.Tensor(hsmm_np.sigma).to(device).requires_grad_(False)
				hsmm[a].priors = torch.Tensor(hsmm_np.priors).to(device).requires_grad_(False)
				hsmm[a].trans = torch.Tensor(hsmm_np.trans).to(device).requires_grad_(False)
				hsmm[a].Trans = torch.Tensor(hsmm_np.Trans).to(device).requires_grad_(False)
				hsmm[a].init_priors = torch.Tensor(hsmm_np.init_priors).to(device).requires_grad_(False)
				# hsmm[a].mu_d = torch.Tensor(hsmm_np.mu_d).to(device).requires_grad_(False)
				# hsmm[a].sigma_d = torch.Tensor(hsmm_np.sigma_d).to(device).requires_grad_(False)
				# hsmm[a].trans_d = torch.Tensor(hsmm_np.trans_d).to(device).requires_grad_(False)
				# hsmm[a].Mu_Pd = torch.Tensor(hsmm_np.Mu_Pd).to(device).requires_grad_(False)
				# hsmm[a].Sigma_Pd = torch.Tensor(hsmm_np.Sigma_Pd).to(device).requires_grad_(False)
				# hsmm[a].Trans_Pd = torch.Tensor(hsmm_np.Trans_Pd).to(device).requires_grad_(False)
			test_recon, test_kl, test_loss, iters = run_iteration(test_iterator, hsmm, model_h, model_r, optimizer, args, epoch)
			write_summaries_vae(writer, test_recon, test_kl, steps_done, 'test')
		
		if (epoch+1) % 10 == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, f'{epoch:04d}.pth')
			torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)
		print(epoch,'epochs done')
	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch+1}.pth')
	torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

	writer.flush()

