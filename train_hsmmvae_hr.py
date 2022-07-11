import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
from config.buetepage import robot_vae_config
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, hsmm, model_h, model_r, optimizer):
	assert type(model_r) == type(model_h)
	iters = 0
	total_recon = []
	total_reg = []
	total_loss = []
	zh_dim = model_h.latent_dim
	zr_dim = model_r.latent_dim
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		# idx = idx[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		# print(label)
		label = label[0]
		# if model_h.window_len:
		# 	xh = x[:, :model_h.input_dim]
		# else:
		# 	xh = x[:, :12]
		# if model_r.window_len:
		# 	xr = x[:, -model_r.input_dim:] # x[0] = Agent 1, x[1] = Agent 2
		# else:
		# 	xr = x[:, 12:] # x[0] = Agent 1, x[1] = Agent 2

		xh = x[:, :model_h.input_dim]
		xr = x[:, -model_r.input_dim:] # x[0] = Agent 1, x[1] = Agent 2
		# if model.window_size>1:
		# 	bp_idx = np.zeros((seq_len-model.window_size, model.window_size))
		# 	for i in range(model.window_size):
		# 		bp_idx[:,i] = idx[i:seq_len-model.window_size+i]
		# 	x = x[:, bp_idx].flatten(2)
		if hsmm != []:
			mu = hsmm[label].mu
			mu_h = torch.Tensor(hsmm[label].mu[:,:zh_dim]).to(device)
			mu_r = torch.Tensor(hsmm[label].mu[:,2*zh_dim:2*zh_dim+zr_dim]).to(device)
			# mu_prior = torch.Tensor([hsmm[label].mu[:,:zh_dim], hsmm[label].mu[:,2*zh_dim:2*zh_dim+zr_dim]]).to(device)
			# Sigma_prior = torch.Tensor([hsmm[label].sigma[:, :zh_dim, :zh_dim], hsmm[label].sigma[:, 2*zh_dim:2*zh_dim+zr_dim, 2*zh_dim:2*zh_dim+zr_dim]]).to(device)
			Sigma_h = torch.Tensor(hsmm[label].sigma[:, :zh_dim, :zh_dim]).to(device)
			Sigma_r = torch.Tensor(hsmm[label].sigma[:, 2*zh_dim:2*zh_dim+zr_dim, 2*zh_dim:2*zh_dim+zr_dim]).to(device)
			alpha_hsmm, _, _, _, _ = hsmm[label].compute_messages(marginal=[], sample_size=seq_len)
			if np.any(np.isnan(alpha_hsmm)):
				alpha_hsmm = forward_variable(hsmm[label], n_step=seq_len)
			# alpha_hsmm = torch.Tensor(alpha_hsmm).to(device)
			seq_alpha = alpha_hsmm.argmax(0)
			

		if model_h.training or model_r.training:
			optimizer.zero_grad()
 
		# xh_gen, zh_post_samples, zh_post_mean, zh_post_var = model_h(xh)
		# xr_gen, zr_post_samples, zr_post_mean, zr_post_var = model_r(xr)

		xh_gen, zh_post_samples, zh_post_dist = model_h(xh)
		xr_gen, zr_post_samples, zr_post_dist = model_r(xr)
		if model_h.training and isinstance(model_h, networks.VAE):
			recon_loss_h = F.mse_loss(xh.repeat(11,1,1), xh_gen, reduction='sum')
		else:
			recon_loss_h = F.mse_loss(xh, xh_gen, reduction='sum')
		
		if model_r.training and isinstance(model_r, networks.VAE):
			recon_loss_r = F.mse_loss(xr.repeat(11,1,1), xr_gen, reduction='sum')
		else:
			recon_loss_r = F.mse_loss(xr, xr_gen, reduction='sum')

		reg_loss = 0.
		if hsmm != []:
			# reg_loss = model.latent_loss(zpost_dist, zpost_samples)
			zh_prior = torch.distributions.MultivariateNormal(mu_h[seq_alpha], Sigma_h[seq_alpha])
			kld_h = torch.distributions.kl_divergence(zh_post_dist, zh_prior).mean(0)
			# kld_h = kl_div(zh_post_mean, zh_post_var, mu_prior[0, c:c+1].repeat(seq_len,1), Sigma_prior[0, c:c+1].repeat(seq_len,1,1)).mean(0)
			# reg_loss += (alpha_hsmm[c]*kld_h).mean()
			reg_loss += model_h.beta*kld_h.mean()

			zr_prior = torch.distributions.MultivariateNormal(mu_r[seq_alpha], Sigma_r[seq_alpha])
			kld_r = torch.distributions.kl_divergence(zr_post_dist, zr_prior).mean(0)
			# kld_r = kl_div(zr_post_mean, zr_post_var, mu_prior[1, c:c+1].repeat(seq_len,1), Sigma_prior[1, c:c+1].repeat(seq_len,1,1)).mean(0)
			reg_loss += model_r.beta*kld_r.mean()
			# if torch.any(torch.isnan(kld_h)) or torch.any(torch.isnan(kld_r)):
			# 	print('NaN', torch.any(torch.isnan(kld_h)), torch.any(torch.isnan(kld_r)), torch.any(torch.isnan(torch.logdet(zh_post_var))), torch.any(torch.isnan(torch.logdet(zr_post_var))))
			# 	for t in [zh_post_mean, zh_post_var, mu_prior[0, c:c+1].repeat(seq_len,1), Sigma_prior[0, c:c+1].repeat(seq_len,1,1)]:
			# 		print(t.shape, torch.any(torch.isnan(t)))
			# if torch.any(torch.isinf(kld_h)) or torch.any(torch.isinf(kld_r)):
			# 	print('inf', torch.any(torch.isinf(kld_h)), torch.any(torch.isinf(kld_r)))
			# z_prior = torch.distributions.MultivariateNormal(mu_prior[:, seq_alpha], Sigma_prior[:, seq_alpha])
			# reg_loss = torch.distributions.kl_divergence(zpost_dist, z_prior).mean()
		loss = recon_loss_h + recon_loss_r + reg_loss

		total_recon.append(recon_loss_h + recon_loss_r)
		total_reg.append(reg_loss)
		total_loss.append(loss)

		if model_h.training or model_r.training:
			loss.backward()
			optimizer.step()
		iters += 1
	
	return total_recon, total_reg, total_loss, xh_gen, xr_gen, zh_post_samples, zr_post_samples, x, iters

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/vae_hr_07080033/', metavar='RES',
						help='Path for saving results (default: ./logs/results/MMDDHHmm).')
	parser.add_argument('--src', type=str, default='/home/vignesh/playground/hsmmvae/data/buetepage_hr/labelled_sequences.npz', metavar='RES',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/data/buetepage_hr/labelled_sequences.npz).')
	parser.add_argument('--prior', type=str, default='HSMM', metavar='P(Z)', choices=['None', 'RNN', 'BIP', 'HSMM'],
						help='Prior to use for the VAE (default: None')
	parser.add_argument('--hsmm-components', type=int, default=10, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 10).')
	parser.add_argument('--model', type=str, default='FullCovVAE', metavar='ARCH', choices=['AE', 'VAE', 'WAE', 'FullCovVAE'],
						help='Model to use: AE, VAE or WAE (default: VAE).')
	parser.add_argument('--dataset', type=str, default='buetepage', metavar='DATASET', choices=['buetepage', 'hhoi', 'shakefive'],
						help='Dataset to use: buetepage, hhoi or shakefive (default: buetepage).')
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	global_config = getattr(config, args.dataset).global_config()
	ae_config = getattr(config, args.dataset).ae_config()

	DEFAULT_RESULTS_FOLDER = args.results
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
	global_step = 0
	if not os.path.exists(DEFAULT_RESULTS_FOLDER):
		print("Creating Result Directories")
		os.makedirs(DEFAULT_RESULTS_FOLDER)
		os.makedirs(MODELS_FOLDER)
		os.makedirs(SUMMARIES_FOLDER)
		np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, global_config=global_config, ae_config=ae_config)

	# elif os.path.exists(os.path.join(MODELS_FOLDER,'hyperparams.npz')):
	# 	hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	# 	args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	# 	config = hyperparams['global_config'].item()
	# 	ae_config = hyperparams['ae_config'].item()

	print("Creating Model and Optimizer")
	model_h = getattr(networks, args.model)(**(ae_config.__dict__)).to(device)
	model_r = getattr(networks, args.model)(**(robot_vae_config().__dict__)).to(device)
	optimizer = getattr(torch.optim, global_config.optimizer)(list(model_h.parameters()) + list(model_r.parameters()), lr=global_config.lr)

	if os.path.exists(os.path.join(MODELS_FOLDER, 'final.pth')):
		print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(MODELS_FOLDER, 'final.pth'))
		model_h.load_state_dict(ckpt['model_h'])
		model_r.load_state_dict(ckpt['model_r'])
		optimizer.load_state_dict(ckpt['optimizer'])
		global_step = ckpt['epoch']

	print("Reading Data")
	# dataset = getattr(dataloaders, args.dataset)
	dataset = dataloaders.buetepage_hr
	if global_config.WINDOW_LEN ==1:
		train_iterator = DataLoader(dataset.SequenceDataset(args.src, train=True), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceDataset(args.src, train=False), batch_size=1, shuffle=True)
	else:
		train_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=True, window_length=global_config.WINDOW_LEN), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=False, window_length=global_config.WINDOW_LEN), batch_size=1, shuffle=True)
	NUM_ACTIONS = len(test_iterator.dataset.actidx)
	print("Building Writer")
	writer = SummaryWriter(SUMMARIES_FOLDER)
	# model.eval()
	# writer.add_graph(model, torch.Tensor(test_data[:10]).to(device))
	# model.train()
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
	if args.model != 'AE':
		nb_dim = 2*model_h.latent_dim + 2*model_r.latent_dim
		nb_states = args.hsmm_components
		for i in range(NUM_ACTIONS):
			hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
			hsmm[-1].mu = np.zeros((nb_states, nb_dim))
			hsmm[-1].sigma = np.eye(nb_dim)[None].repeat(nb_states,0)
			hsmm[-1].Mu_Pd = np.zeros(nb_states)
			hsmm[-1].Sigma_Pd = np.ones(nb_states)
			hsmm[-1].Trans_Pd = np.ones((nb_states, nb_states))/nb_states

	for epoch in range(global_config.EPOCHS):
		model_h.train()
		model_r.train()
		print('training model')
		train_recon, train_kl, train_loss, xh_gen, xr_gen, zh_post_samples, zr_post_samples, x, iters = run_iteration(train_iterator, hsmm, model_h, model_r, optimizer)
		steps_done = (epoch+1)*iters
		print('writing training summaries')
		write_summaries_hr(writer, train_recon, train_kl, train_loss, xh_gen, xr_gen, zh_post_samples, zr_post_samples, x, steps_done, 'train')
		params = []
		grads = []
		print('writing training grad robot summaries')
		for name, param in model_r.named_parameters():
			if param.grad is None:
				continue
			writer.add_histogram('robot/grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('robot/param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		print('writing training grad human summaries')
		for name, param in model_h.named_parameters():
			if param.grad is None:
				continue
			writer.add_histogram('human/grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('human/param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		model_h.eval()
		model_r.eval()
		with torch.no_grad():
			print('testing model')
			test_recon, test_kl, test_loss, xh_gen, xr_gen, zh_post_samples, zr_post_samples, x, iters = run_iteration(test_iterator, hsmm, model_h, model_r, optimizer)
			print('writing testing summaries')
			write_summaries_hr(writer, test_recon, test_kl, test_loss, xh_gen, xr_gen, zh_post_samples, zr_post_samples, x, steps_done, 'test')
		
			if hsmm != []:
				# Updating Prior
				print('Training HSMM')
				for a in range(len(train_iterator.dataset.actidx)):
					s = train_iterator.dataset.actidx[a]
					z_encoded = []
					for j in range(s[0], s[1]):
					# for j in np.random.randint(s[0], s[1], 20):
						x, label = train_iterator.dataset[j]
						assert np.all(label == a)
						x = torch.Tensor(x).to(device)
						seq_len, dims = x.shape
						# xh = x[:, :12]
						# xr = x[:, 12:] # x[0] = Agent 1, x[1] = Agent 2
						xh = x[:, :model_h.input_dim]
						xr = x[:, -model_r.input_dim:] # x[0] = Agent 1, x[1] = Agent 2
						
						zh_post_samples = model_h(xh, encode_only=True)
						zr_post_samples = model_r(xr, encode_only=True)
						z1_vel = torch.diff(zh_post_samples, prepend=zh_post_samples[0:1], dim=0)
						z2_vel = torch.diff(zr_post_samples, prepend=zr_post_samples[0:1], dim=0)
						z_encoded.append(torch.concat([zh_post_samples, z1_vel, zr_post_samples, z2_vel], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
					print('learning HSMM',a)
					hsmm[a].init_hmm_kbins(z_encoded)
					hsmm[a].em(z_encoded)
					print('')

		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step, 'hsmm':hsmm}, checkpoint_file)
