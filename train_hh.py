import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Sophia import SophiaG

import numpy as np
import os, datetime, argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, hsmm, model, optimizer):
	total_recon, total_reg, total_loss, mu_prior, Sigma_prior = [], [], [], [], []
	z_dim = model.latent_dim
	I = torch.eye(z_dim, device=device)*1e-6
	# if isinstance(model, networks.VAE) or hsmm!=[]:
	# 	for label in range(len(hsmm)):
	# 		mu_prior.append(torch.Tensor([hsmm[label].mu[:,:z_dim], hsmm[label].mu[:,z_dim:]]).to(device))
	# 		Sigma_prior.append(torch.Tensor([hsmm[label].sigma[:, :z_dim, :z_dim], hsmm[label].sigma[:, z_dim:, z_dim:]]).to(device) + I)
		
	for i, x in enumerate(iterator):
		if model.training:
			optimizer.zero_grad()

		x, label = x
		x = x[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		
		# if isinstance(model, networks.VAE) or hsmm!=[]:
		# 	alpha_hsmm, _, _, _, _ = hsmm[label].compute_messages(marginal=[], sample_size=seq_len)
		# 	if np.any(np.isnan(alpha_hsmm)):
		# 		print('Alpha Nan', i)
		# 		alpha_hsmm = forward_variable(hsmm[label], n_step=seq_len)

		# 	seq_alpha = alpha_hsmm.argmax(0)

		x_gen, zpost_samples, zpost_dist = model(x)

		reg_loss = 0.
		if isinstance(model, networks.VAE):
			with torch.no_grad():
				# z_prior = torch.distributions.MultivariateNormal(mu_prior[label][:, seq_alpha], Sigma_prior[label][:, seq_alpha])

				z1_cond, sigma_z1_cond = hsmm[label].condition(zpost_dist.mean[1].detach().cpu().numpy(), zpost_dist.covariance_matrix[1].detach().cpu().numpy(), dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
				z2_cond, sigma_z2_cond = hsmm[label].condition(zpost_dist.mean[0].detach().cpu().numpy(), zpost_dist.covariance_matrix[0].detach().cpu().numpy(), dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
				mu_prior_cond = torch.Tensor(np.array([z1_cond, z2_cond])).to(device)
				Sigma_prior_cond = torch.Tensor(np.array([sigma_z1_cond, sigma_z2_cond])).to(device) + I

				# # z1_cond, sigma_z1_cond = hsmm[label].condition(zpost_dist.mean[1], zpost_dist.covariance_matrix[1], dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
				# # z2_cond, sigma_z2_cond = hsmm[label].condition(zpost_dist.mean[0], zpost_dist.covariance_matrix[0], dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
				# # mu_prior_cond = torch.concat([z1_cond[None], z2_cond[None]])
				# # Sigma_prior_cond = torch.concat([sigma_z1_cond[None], sigma_z2_cond[None]]) + I
				
				z_prior_cond = torch.distributions.MultivariateNormal(mu_prior_cond, Sigma_prior_cond, validate_args=False)
			
			# reg_loss += torch.distributions.kl_divergence(zpost_dist, z_prior).mean()
		
			# Cross training (z1|z2) and (z2|z1) 	
			reg_loss += torch.distributions.kl_divergence(zpost_dist, z_prior_cond).mean()
		
			# reg_loss += F.mse_loss(zpost_dist.mean, mu_prior_cond, reduction='sum')
	
		if not model.training:
			# z1_cond, _ = hsmm[label].condition(zpost_dist.mean[1].detach().cpu().numpy(), zpost_dist.covariance_matrix[1].detach().cpu().numpy(), dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
			# z2_cond, _ = hsmm[label].condition(zpost_dist.mean[0].detach().cpu().numpy(), zpost_dist.covariance_matrix[0].detach().cpu().numpy(), dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			# z1_cond, sigma_z1_cond = hsmm[label].condition(zpost_dist.mean[1].detach().cpu().numpy(), None, dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
			# z2_cond, sigma_z2_cond = hsmm[label].condition(zpost_dist.mean[0].detach().cpu().numpy(), None, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			# mu_prior_cond = torch.Tensor(np.array([z1_cond, z2_cond])).to(device)
			x_gen = model._output(model._decoder(mu_prior_cond))
		
		if model.training and isinstance(model, networks.VAE):
			recon_loss = F.mse_loss(x[None].repeat(model.mce_samples+1,1,1,1), x_gen, reduction='sum')
		else:
			recon_loss = F.mse_loss(x, x_gen, reduction='sum')

		# recon_loss += F.mse_loss(x, x_gen_cond, reduction='sum')
		
		loss = recon_loss + model.beta*reg_loss

		total_recon.append(recon_loss)
		total_reg.append(reg_loss)
		total_loss.append(loss)

		if model.training:
			loss.backward()
			optimizer.step()

	return total_recon, total_reg, total_loss, x_gen, zpost_samples, x, i

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/results/'+datetime.datetime.now().strftime("%m%d%H%M"), 
						metavar='RES', help='Path for saving results (default: ./logs/results/MMDDHHmm).')
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
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	args = parser.parse_args()
	print('Random Seed',args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.autograd.set_detect_anomaly(True)
	args.dataset = 'buetepage_downsampled'
	if args.ckpt is None:
		global_config = getattr(config, args.dataset).global_config()
		ae_config = getattr(config, args.dataset).ae_config()
		ae_config.latent_dim = args.latent_dim

		global_step = 0
		global_epochs = 0
		ckpt = None


	else:
		dirname = os.path.dirname(args.ckpt)
		ckpt = torch.load(args.ckpt)
		hyperparams = np.load(os.path.join(dirname,'hyperparams.npz'), allow_pickle=True)
		args = hyperparams['args'].item()
		global_config = hyperparams['global_config'].item()
		ae_config = hyperparams['ae_config'].item()
		global_epochs = ckpt['epoch']

	MODELS_FOLDER = os.path.join(args.results, "models")
	SUMMARIES_FOLDER = os.path.join(args.results, "summary")
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

	print("Creating Model and Optimizer")
	model = getattr(networks, args.model)(**(ae_config.__dict__)).to(device)
	# optimizer = getattr(torch.optim, global_config.optimizer)(model.parameters(), lr=global_config.lr)
	optimizer = SophiaG(model.parameters(), lr=global_config.lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
	
	if ckpt is not None:
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
	
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


	print("Starting Epochs")
	hsmm = []
	nb_dim = 2*model.latent_dim
	nb_states = args.hsmm_components
	for i in range(NUM_ACTIONS):
		hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
		hsmm[-1].init_zeros()
		hsmm[-1].init_priors = np.ones(nb_states) / nb_states
		hsmm[-1].Trans = np.ones((nb_states, nb_states))/nb_states
		hsmm[-1].Mu_Pd = np.zeros(nb_states)
		hsmm[-1].Sigma_Pd = np.ones(nb_states)
		hsmm[-1].Trans_Pd = np.ones((nb_states, nb_states))/nb_states

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
				hsmm[a].init_hmm_kbins(z_encoded)
				hsmm[a].em(z_encoded, nb_max_steps=60)
			
			test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iteration(test_iterator, hsmm, model, optimizer)
			write_summaries_vae(writer, test_recon, test_kl, test_loss, x_gen, zx_samples, x, steps_done, 'test', model)
			print(np.any(np.isnan(x_gen.detach().cpu().numpy())))

		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)
