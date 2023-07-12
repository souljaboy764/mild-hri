import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd
import pbdlib_torch as pbd_torch
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator:DataLoader, hsmm:List[pbd_torch.HMM], model_h:networks.VAE, model_r:networks.VAE, optimizer:torch.optim.Optimizer):
	total_recon, total_reg, total_loss, mu_prior, Sigma_prior, alpha_prior = [], [], [], [], [], []
	z_dim = model_h.latent_dim
	num_alpha = 1500
	I = torch.eye(z_dim, device=device)*1e-8
	with torch.no_grad():
		for label in range(len(hsmm)):
			mu_prior.append(torch.concat([hsmm[label].mu[None,:,:z_dim], hsmm[label].mu[None, :,z_dim:]]).to(device))
			Sigma_prior.append(torch.concat([hsmm[label].sigma[None, :, :z_dim, :z_dim], hsmm[label].sigma[None, :, z_dim:, z_dim:]]).to(device) + I)
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
		alpha = alpha_prior[label][:, torch.linspace(0, num_alpha-1, seq_len, dtype=int)]
		
		xh_gen, zh_samples, zh_post = model_h(x_h)
		xr_gen, zr_samples, zr_post = model_r(x_r)

		# zh_post = model_h(x_h, dist_only=True)
		# zr_post = model_r(x_r, dist_only=True)

		if model_h.training:
			fwd_h = alpha
		else:
			fwd_h = hsmm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim))
			# zh_samples = [zh_post.mean]
		
		# # Sample Conditioning: Sampling from Posterior and then Conditioning the HMM
		# xr_cond = []
		# for zh in zh_samples:
		# 	zr_cond = hsmm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
		# 									return_cov=False)#, data_Sigma_in=zh_post.covariance_matrix)
		# 	xr_cond.append(model_r._output(model_r._decoder(zr_cond))[None])
		# xr_cond = torch.concat(xr_cond)

		# # Conditioned Sampling: Conditioning on the Posterior and then Sampling from the conditional distribution
		zr_cond_mean, zr_cond_sigma = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
										return_cov=True, data_Sigma_in=zh_post.covariance_matrix)
		if model_h.training:
			try:
				zr_cond = torch.distributions.MultivariateNormal(zr_cond_mean, zr_cond_sigma)
			except Exception as e:
				eigvals, eigvecs = torch.linalg.eigh(zr_cond_sigma)
				D = torch.diag_embed(torch.nn.ReLU()(eigvals) + 1e-2)
				new_sigma = eigvecs @ D @ eigvecs.transpose(-1,-2)
				zr_cond = torch.distributions.MultivariateNormal(zr_cond_mean, new_sigma)
			xr_cond = model_r._output(model_r._decoder(torch.concat([zr_cond.rsample((model_r.mce_samples,)), zr_cond_mean[None]], dim=0)))
		else:
			xr_cond = model_r._output(model_r._decoder(zr_cond_mean[None]))

		# Simple conditioninal reconstruction
		# zr_cond_mean = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
		# 								return_cov=False)#, data_Sigma_in=zh_post.covariance_matrix)
		# xr_cond = model_r._output(model_r._decoder(zr_cond_mean[None]))

		
		if model_h.training:
			# recon_loss = ((xh_gen - x_h[None])**2).mean() + ((xr_gen - x_r[None])**2).mean() + ((xr_cond - x_r[None])**2).mean()
			
			xh_gen = model_h._output(model_h._decoder(zh_post.mean))
			xr_gen = model_r._output(model_r._decoder(zr_post.mean))
			recon_loss = ((xh_gen - x_h)**2).mean() + ((xr_gen - x_r)**2).mean() + ((xr_cond - x_r)**2).mean()
		else:
			recon_loss = ((xr_cond - x_r)**2).sum()
		
		loss = recon_loss

		if model_h.beta!=0:
			with torch.no_grad():
				seq_alpha = alpha.argmax(0)
				zh_prior = torch.distributions.MultivariateNormal(mu_prior[label][0, seq_alpha], Sigma_prior[label][0, seq_alpha])
				zr_prior = torch.distributions.MultivariateNormal(mu_prior[label][1, seq_alpha], Sigma_prior[label][1, seq_alpha])
			reg_loss = model_h.beta * torch.distributions.kl_divergence(zh_post, zh_prior).mean() + \
						model_r.beta * torch.distributions.kl_divergence(zr_post, zr_prior).mean()
				
		else:
			reg_loss = 0.

		loss += reg_loss

		total_recon.append(recon_loss)
		total_reg.append(reg_loss)
		total_loss.append(loss)
		if model_h.training:
			loss.backward()
			optimizer.step()
		# print((datetime.datetime.now() - start).total_seconds()/seq_len)
	return total_recon, total_reg, total_loss, i

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/debug',#+datetime.datetime.now().strftime("%m%d%H%M"),
						help='Path for saving results (default: ./logs/results/MMDDHHmm).', metavar='RES')
	parser.add_argument('--src', type=str, default='./data/buetepage/traj_data.npz', metavar='SRC',
						help='Path to read training and testing data (default: ./data/buetepage/traj_data.npz).')
	parser.add_argument('--hsmm-components', type=int, default=5, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 5).')
	parser.add_argument('--model', type=str, default='VAE', metavar='ARCH', choices=['AE', 'VAE', 'FullCovVAE'],
						help='Model to use: AE, VAE or FullCovVAE (default: VAE).')
	parser.add_argument('--dataset', type=str, default='buetepage_pepper', metavar='DATASET', choices=['buetepage', 'buetepage_pepper'],
						help='Dataset to use: buetepage, buetepage_pepper or nuitrack (default: buetepage_pepper).')
	parser.add_argument('--seed', type=int, default=np.random.randint(0,np.iinfo(np.int32).max), metavar='SEED',
						help='Random seed for training (randomized by default).')
	parser.add_argument('--latent-dim', type=int, default=3, metavar='Z',
						help='Latent space dimension (default: 3)')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	args = parser.parse_args()
	print('Random Seed',args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.autograd.set_detect_anomaly(True)
	

	global_config = config.buetepage.global_config()
	ae_config = config.buetepage.ae_config()
	if args.dataset == 'buetepage_pepper':
		robot_vae_config = config.buetepage.robot_vae_config()
		global_config.robot_joints = 4
		robot_vae_config.num_joints = 4
		dataset = dataloaders.buetepage.PepperWindowDataset
	elif args.dataset == 'buetepage':
		robot_vae_config = config.buetepage.ae_config()
		dataset = dataloaders.buetepage.HHWindowDataset
	# TODO: Nuitrack

	print("Reading Data")
	train_iterator = DataLoader(dataset(args.src, train=True, window_length=global_config.window_size, downsample=global_config.downsample), batch_size=1, shuffle=True)
	test_iterator = DataLoader(dataset(args.src, train=False, window_length=global_config.window_size, downsample=global_config.downsample), batch_size=1, shuffle=False)
	ae_config.latent_dim = args.latent_dim
	robot_vae_config.latent_dim = args.latent_dim

	MODELS_FOLDER = os.path.join(args.results, "models")
	SUMMARIES_FOLDER = os.path.join(args.results, "summary")
	global_step = 0
	global_epochs = 0

	print("Creating Model and Optimizer")

	model_h = getattr(networks, args.model)(**(ae_config.__dict__)).to(device)
	model_r = getattr(networks, args.model)(**(robot_vae_config.__dict__)).to(device)
	params = list(model_h.parameters()) + list(model_r.parameters())
	names_params = list(model_h.named_parameters()) + list(model_r.named_parameters())
	optimizer = getattr(torch.optim, global_config.optimizer)(params, lr=global_config.lr)
	
	NUM_ACTIONS = len(test_iterator.dataset.actidx)
	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	
	s = ''
	for k in global_config.__dict__:
		s += str(k) + ' : ' + str(global_config.__dict__[k]) + '\n'
	writer.add_text('global_config', s)

	s = ''
	for k in ae_config.__dict__:
		s += str(k) + ' : ' + str(ae_config.__dict__[k]) + '\n'
	writer.add_text('human_ae_config', s)

	s = ''
	for k in robot_vae_config.__dict__:
		s += str(k) + ' : ' + str(robot_vae_config.__dict__[k]) + '\n'
	writer.add_text('robot_ae_config', s)

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
		hyperparams = np.load(os.path.join(os.path.dirname(args.ckpt),'hyperparams.npz'), allow_pickle=True)
		seed = hyperparams['args'].item().seed
		torch.manual_seed(seed)
		np.random.seed(seed)
		global_epochs = ckpt['epoch']
	else:
		np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, global_config=global_config, ae_config=ae_config, robot_vae_config=robot_vae_config)
		checkpoint_file = os.path.join(MODELS_FOLDER, 'init_ckpt.pth')
		torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0, 'hsmm':[]}, checkpoint_file)
		checkpoint_file = os.path.join(MODELS_FOLDER, 'last_ckpt.pth')
		torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': 0, 'hsmm':[]}, checkpoint_file)
	print("Starting Epochs")

	for epoch in range(global_epochs, global_config.EPOCHS + global_epochs):
		model_h.train()
		model_r.train()
		train_recon, train_kl, train_loss, iters = run_iteration(train_iterator, hsmm, model_h, model_r, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_vae(writer, train_recon, train_kl, steps_done, 'train')
		params = []
		grads = []
		for name, param in names_params:
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		model_h.eval()
		model_r.eval()
		with torch.no_grad():
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
				hsmm_np.init_hmm_kbins(z_encoded)
				hsmm_np.em(z_encoded)
				# hsmm_np.em(np.concatenate(z_encoded))

				hsmm[a].mu = torch.Tensor(hsmm_np.mu).to(device).requires_grad_(False)
				hsmm[a].sigma = torch.Tensor(hsmm_np.sigma).to(device).requires_grad_(False)
				hsmm[a].priors = torch.Tensor(hsmm_np.priors).to(device).requires_grad_(False)
				hsmm[a].reg = torch.Tensor(hsmm_np.reg).to(device).requires_grad_(False)
				hsmm[a].trans = torch.Tensor(hsmm_np.trans).to(device).requires_grad_(False)
				hsmm[a].Trans = torch.Tensor(hsmm_np.Trans).to(device).requires_grad_(False)
				hsmm[a].init_priors = torch.Tensor(hsmm_np.init_priors).to(device).requires_grad_(False)
				# hsmm[a].mu_d = torch.Tensor(hsmm_np.mu_d).to(device)
				# hsmm[a].sigma_d = torch.Tensor(hsmm_np.sigma_d).to(device)
				# hsmm[a].trans_d = torch.Tensor(hsmm_np.trans_d).to(device)
				# hsmm[a].Mu_Pd = torch.Tensor(hsmm_np.Mu_Pd).to(device)
				# hsmm[a].Sigma_Pd = torch.Tensor(hsmm_np.Sigma_Pd).to(device)
				# hsmm[a].Trans_Pd = torch.Tensor(hsmm_np.Trans_Pd).to(device)
						
			test_recon, test_kl, test_loss, iters = run_iteration(test_iterator, hsmm, model_h, model_r, optimizer)
			write_summaries_vae(writer, test_recon, test_kl, steps_done, 'test')

		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			os.rename(os.path.join(MODELS_FOLDER, 'last_ckpt.pth'), os.path.join(MODELS_FOLDER, '2ndlast_ckpt.pth'))
			checkpoint_file = os.path.join(MODELS_FOLDER, 'last_ckpt.pth')
			torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch+1}.pth')
	torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)
