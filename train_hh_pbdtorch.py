import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd
import pbdlib_torch as pbd_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, hsmm, model, optimizer):
	total_recon, total_reg, total_loss, mu_prior, Sigma_prior = [], [], [], [], []
	z_dim = model.latent_dim
	I = torch.eye(z_dim, device=device)*1e-8
	if isinstance(model, networks.VAE) or hsmm!=[]:
		for label in range(len(hsmm)):
			mu_prior.append(torch.concat([hsmm[label].mu[None,:,:z_dim], hsmm[label].mu[None, :,z_dim:]]).to(device))
			Sigma_prior.append(torch.concat([hsmm[label].sigma[None, :, :z_dim, :z_dim], hsmm[label].sigma[None, :, z_dim:, z_dim:]]).to(device) + I)
		
	for i, x in enumerate(iterator):
		if model.training:
			optimizer.zero_grad()

		x, label = x
		x = x[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		
		if isinstance(model, networks.VAE) or hsmm!=[]:
			alpha_hsmm = hsmm[label].forward_variable_ts(seq_len)
			# if torch.any(torch.isnan(alpha_hsmm)):
			# 	print('Alpha Nan')
			# 	alpha_hsmm = forward_variable(hsmm[label], n_step=seq_len)

			seq_alpha = alpha_hsmm.argmax(0)

		x_gen, zpost_samples, zpost_dist = model(x)
		if model.training and isinstance(model, networks.VAE):
			recon_loss = ((x[None].repeat(model.mce_samples+1,1,1,1) - x_gen)**2).mean(0).sum()
			# recon_loss = F.mse_loss(x[None].repeat(model.mce_samples+1,1,1,1), x_gen, reduction='sum')
		else:
			recon_loss = F.mse_loss(x, x_gen, reduction='sum')

		# reg_loss = 0.
		if isinstance(model, networks.VAE):
			z_prior = torch.distributions.MultivariateNormal(mu_prior[label][:, seq_alpha], Sigma_prior[label][:, seq_alpha])
			reg_loss = torch.distributions.kl_divergence(zpost_dist, z_prior).mean() 

			z1_cond = hsmm[label].condition(zpost_dist.mean[1], dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
			z2_cond = hsmm[label].condition(zpost_dist.mean[0], dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			mu_prior_cond = torch.concat([z1_cond[None], z2_cond[None]])
			# Sigma_prior_cond = torch.concat([sigma_z1_cond[None], sigma_z2_cond[None]]) + I
			
			# z_prior_cond = torch.distributions.MultivariateNormal(mu_prior_cond, Sigma_prior_cond)
			
		
			# # Cross training (z1|z2) and (z2|z1) 	
			# reg_loss += torch.distributions.kl_divergence(zpost_dist, z_prior_cond).mean()
		
			# reg_loss += F.mse_loss(zpost_dist.mean, mu_prior_cond, reduction='sum')
			
		# if not model.training:
			x_gen_cond = model._output(model._decoder(mu_prior_cond))

			recon_loss += ((x - x_gen_cond)**2).sum()
		

		# recon_loss += F.mse_loss(x, x_gen_cond, reduction='sum')
		
		loss = recon_loss + model.beta*reg_loss

		total_recon.append(recon_loss)
		total_reg.append(reg_loss)
		total_loss.append(loss)
		print(i)
		if model.training:
			loss.backward()
			optimizer.step()
	return total_recon, total_reg, total_loss, x_gen, zpost_samples, x, i

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/2023/bp_hands_downsample/vaehsmm_torch',#+datetime.datetime.now().strftime("%m%d%H%M"), metavar='RES',
						help='Path for saving results (default: ./logs/results/MMDDHHmm).')
	parser.add_argument('--src', type=str, default='./data/buetepage/traj_data.npz', metavar='SRC',
						help='Path to read training and testing data (default: ./data/buetepage/traj_data.npz).')
	parser.add_argument('--hsmm-components', type=int, default=10, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 10).')
	parser.add_argument('--model', type=str, default='VAE', metavar='ARCH', choices=['AE', 'VAE', 'WAE', 'FullCovVAE'],
						help='Model to use: AE, VAE or WAE (default: VAE).')
	parser.add_argument('--dataset', type=str, default='buetepage', metavar='DATASET', choices=['buetepage', 'nuitrack'],
						help='Dataset to use: buetepage, hhoi or shakefive (default: buetepage).')
	parser.add_argument('--seed', type=int, default=np.random.randint(0,np.iinfo(np.int32).max), metavar='SEED',
						help='Random seed for training (randomized by default).')
	parser.add_argument('--latent-dim', type=int, default=3, metavar='Z',
						help='Latent space dimension (default: 3)')
	args = parser.parse_args()
	print('Random Seed',args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.autograd.set_detect_anomaly(True)
	args.dataset = 'bp_hands_downsample'
	global_config = getattr(config, args.dataset).global_config()
	ae_config = getattr(config, args.dataset).ae_config()
	ae_config.latent_dim = args.latent_dim

	MODELS_FOLDER = os.path.join(args.results, "models")
	SUMMARIES_FOLDER = os.path.join(args.results, "summary")
	global_step = 0
	global_epochs = 0

	print("Creating Model and Optimizer")

	model = getattr(networks, args.model)(**(ae_config.__dict__)).to(device)
	optimizer = getattr(torch.optim, global_config.optimizer)(model.parameters(), lr=global_config.lr)
	
	print("Reading Data")
	dataset = getattr(dataloaders, args.dataset)
	train_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=True, window_length=model.window_size), batch_size=1, shuffle=True)
	test_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=False, window_length=model.window_size), batch_size=1, shuffle=True)
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
	np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, global_config=global_config, ae_config=ae_config)

	print("Starting Epochs")
	hsmm = []
	nb_dim = 2*model.latent_dim
	nb_states = args.hsmm_components
	with torch.no_grad():
		for i in range(NUM_ACTIONS):
			hsmm.append(pbd_torch.HSMM(nb_dim=nb_dim, nb_states=nb_states))
			hsmm[-1].init_zeros()
			hsmm[-1].trans = torch.ones((nb_states, nb_states)).to(device) / nb_states
			hsmm[-1].init_priors = torch.ones(nb_states).to(device) / nb_states
			hsmm[-1].Mu_Pd = torch.Tensor(np.zeros(nb_states)).to(device)
			hsmm[-1].Sigma_Pd = torch.Tensor(np.ones(nb_states)).to(device)
			hsmm[-1].Trans_Pd = torch.Tensor(np.ones((nb_states, nb_states))).to(device)/nb_states

	for epoch in range(global_epochs,global_epochs+global_config.EPOCHS):
		model.train()
		train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iteration(train_iterator, hsmm if args.model!='AE' else [], model, optimizer)
		steps_done = (epoch+1)*iters
		write_summaries_vae(writer, train_recon, train_kl, train_loss, x_gen, zx_samples, x, steps_done, 'train', model)
		params = []
		grads = []
		for name, param in model.named_parameters():
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		model.eval()
		with torch.no_grad():
			# Updating Prior
			for a in range(len(train_iterator.dataset.actidx)):
				s = train_iterator.dataset.actidx[a]
				z_encoded = []
				for j in range(s[0], s[1]):
				# for j in np.random.randint(s[0], s[1], 12):
					x, label = train_iterator.dataset[j]
					assert np.all(label == a)
					x = torch.Tensor(x).to(device)
					seq_len, dims = x.shape
					x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
					
					zpost_samples = model(x, encode_only=True)
					z_encoded.append(torch.concat([zpost_samples[0], zpost_samples[1]], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				hsmm_np = pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states)
				hsmm_np.init_hmm_kbins(z_encoded)
				hsmm_np.em(z_encoded)

				hsmm[a].mu = torch.Tensor(hsmm_np.mu).to(device)
				hsmm[a].sigma = torch.Tensor(hsmm_np.sigma).to(device)
				hsmm[a].trans = torch.Tensor(hsmm_np.trans).to(device)
				hsmm[a].Trans = torch.Tensor(hsmm_np.Trans).to(device)
				hsmm[a].init_priors = torch.Tensor(hsmm_np.init_priors).to(device)
				hsmm[a].priors = torch.Tensor(hsmm_np.priors).to(device)
				hsmm[a].reg = torch.Tensor(hsmm_np.reg).to(device)
				hsmm[a].mu_d = torch.Tensor(hsmm_np.mu_d).to(device)
				hsmm[a].sigma_d = torch.Tensor(hsmm_np.sigma_d).to(device)
				hsmm[a].trans_d = torch.Tensor(hsmm_np.trans_d).to(device)
				hsmm[a].Mu_Pd = torch.Tensor(hsmm_np.Mu_Pd).to(device)
				hsmm[a].Sigma_Pd = torch.Tensor(hsmm_np.Sigma_Pd).to(device)
				hsmm[a].Trans_Pd = torch.Tensor(hsmm_np.Trans_Pd).to(device)
				

			
			test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iteration(test_iterator, hsmm, model, optimizer)
			write_summaries_vae(writer, test_recon, test_kl, test_loss, x_gen, zx_samples, x, steps_done, 'test', model)
			print(np.any(np.isnan(x_gen.detach().cpu().numpy())))

		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step, 'hsmm':hsmm}, checkpoint_file)
