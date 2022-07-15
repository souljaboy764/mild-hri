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

def run_iteration(iterator, model_r, optimizer, hsmm=None, model_h=None):
	iters = 0
	total_recon = []
	total_reg = []
	total_loss = []
	zr_dim = model_r.latent_dim
	if model_h is not None:
		zh_dim = model_h.latent_dim
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		label = label[0]
		# idx = idx[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		# print(label)
		xr = x[:, -model_r.input_dim:] # x[0] = Agent 1, x[1] = Agent 2

		if model_r.training:
			optimizer.zero_grad()
		if hsmm is not None:
			if model_r.window_size > 1 :
				mu_r = torch.Tensor(hsmm[label].mu[:,-zr_dim:]).to(device)
				Sigma_r = torch.Tensor(hsmm[label].sigma[:, -zr_dim:, -zr_dim:]).to(device)
			else:
				mu_r = torch.Tensor(hsmm[label].mu[:,-2*zr_dim:-zr_dim]).to(device)
				Sigma_r = torch.Tensor(hsmm[label].sigma[:, -2*zr_dim:-zr_dim, -2*zr_dim:-zr_dim]).to(device)
			alpha_hsmm, _, _, _, _ = hsmm[label].compute_messages(marginal=[], sample_size=seq_len)
			if np.any(np.isnan(alpha_hsmm)):
				alpha_hsmm = forward_variable(hsmm[label], n_step=seq_len)
			# alpha_hsmm = torch.Tensor(alpha_hsmm).to(device)
			seq_alpha = alpha_hsmm.argmax(0)
			
 
		# xh_gen, zh_post_samples, zh_post_mean, zh_post_var = model_h(xh)
		# xr_gen, zr_post_samples, zr_post_mean, zr_post_var = model_r(xr)
		
		xr_gen, zr_post_samples, zr_post_dist = model_r(xr)
		x_gen = xr_gen
		
		if model_r.training and isinstance(model_r, networks.VAE):
			recon_loss_r = F.mse_loss(xr.repeat(11,1,1), xr_gen, reduction='sum')
		else:
			recon_loss_r = F.mse_loss(xr, xr_gen, reduction='sum')

		reg_loss = 0.
		if hsmm is not None:
			zr_prior = torch.distributions.MultivariateNormal(mu_r[seq_alpha], Sigma_r[seq_alpha])
			kld_r = torch.distributions.kl_divergence(zr_post_dist, zr_prior).mean(0)
			# kld_r = kl_div(zr_post_mean, zr_post_var, mu_prior[1, c:c+1].repeat(seq_len,1), Sigma_prior[1, c:c+1].repeat(seq_len,1,1)).mean(0)
			reg_loss += model_r.beta*kld_r.mean()

			if not model_r.training:
				with torch.no_grad():
					xh = x[:, :model_h.input_dim]
					xh_gen, z1, _ = model_h(xh)
					if model_r.window_size >1:
						z2, _ = hsmm[label].condition(z1.cpu().numpy(), dim_in=slice(0, zh_dim), dim_out=slice(zh_dim, zh_dim+zr_dim))
					else:
						z1_vel = torch.diff(z1, prepend=z1[0:1], dim=0)
						z2, _ = hsmm[label].condition(torch.concat([z1, z1_vel], dim=-1).cpu().numpy(), dim_in=slice(0, 2*zh_dim), dim_out=slice(2*zh_dim, 2*zh_dim+zr_dim))
					xr_gen = model_r._output(model_r._decoder(torch.Tensor(z2).to(device)))
					x_gen = torch.concat([xh_gen, xr_gen], dim=-1)
			
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
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/results/robot_'+datetime.datetime.now().strftime("%m%d%H%M"), metavar='RES',
						help='Path for saving results (default: ./logs/results/robot_MMDDHHmm).')
	parser.add_argument('--src', type=str, default='/home/vignesh/playground/hsmmvae/data/buetepage_hr/labelled_sequences.npz', metavar='RES',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/data/buetepage_hr/labelled_sequences.npz).')
	parser.add_argument('--human-ckpt', type=str, default='/home/vignesh/playground/hsmmvae/logs/fullvae_rarm_window_alphaargmaxnocond_07132220/models/0570.pth', metavar='CKPT',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/logs/fullvae_rarm_window_alphaargmaxnocond_07132220/models/0570.pth).')			
	parser.add_argument('--hsmm-components', type=int, default=10, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 10).')			 
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	global_config = config.buetepage.global_config()
	config_r = config.buetepage.robot_vae_config()

	HUMAN_MODELS_FOLDER = os.path.dirname(args.human_ckpt)
	hyperparams_h = np.load(os.path.join(HUMAN_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	saved_args_h = hyperparams_h['args'].item() # overwrite args if loading from checkpoint
	vae_config_h = hyperparams_h['ae_config'].item()
	
	model_h = getattr(networks, saved_args_h.model)(**(vae_config_h.__dict__)).to(device)
	ckpt = torch.load(args.human_ckpt)
	model_h.load_state_dict(ckpt['model'])
	model_h.eval()

	DEFAULT_RESULTS_FOLDER = args.results
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
	global_step = 0
	if not os.path.exists(DEFAULT_RESULTS_FOLDER):
		print("Creating Result Directories")
		os.makedirs(DEFAULT_RESULTS_FOLDER)
		os.makedirs(MODELS_FOLDER)
		os.makedirs(SUMMARIES_FOLDER)
		np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, global_config=global_config, ae_config=config_r)

	# elif os.path.exists(os.path.join(MODELS_FOLDER,'hyperparams.npz')):
	

	if os.path.exists(os.path.join(MODELS_FOLDER, 'final.pth')):
		print("Creating Model and Optimizer from previous CKPT")
		hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
		args = hyperparams['args'].item() # overwrite args if loading from checkpoint
		config = hyperparams['global_config'].item()
		ae_config = hyperparams['ae_config'].item()
		model_r = networks.AE(**(config_r.__dict__)).to(device)
		optimizer = getattr(torch.optim, global_config.optimizer)(model_r.parameters(), lr=global_config.lr)
		print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(MODELS_FOLDER, 'final.pth'))
		model_r.load_state_dict(ckpt['model_r'])
		optimizer.load_state_dict(ckpt['optimizer'])
		global_step = ckpt['epoch']
	else:
		print("Creating Model and Optimizer from Human VAE")
		ckpt = torch.load(args.human_ckpt)
		if 'post_mean.weight' in ckpt['model']:
			config_r.latent_dim = ckpt['model']['post_mean.bias'].shape[0]
		else:
			config_r.latent_dim = ckpt['model']['latent.bias'].shape[0]

		if ckpt['model']['_output.bias'].shape[0] == 12:
			config_r.window_size = 1
		elif ckpt['model']['_output.bias'].shape[0] == 480:
			config_r.window_size = 40

		model_r = networks.FullCovVAE(**(config_r.__dict__)).to(device)
		optimizer = getattr(torch.optim, global_config.optimizer)(model_r.parameters(), lr=global_config.lr)
		print("Loading Checkpoints")
		with torch.no_grad():
			model_r._encoder[2].weight.copy_(ckpt['model']['_encoder.2.weight'])
			model_r._encoder[2].bias.copy_(ckpt['model']['_encoder.2.bias'])
			if 'post_mean.weight' in ckpt['model']:
				if isinstance(model_r, networks.AE):
					model_r.latent.weight.copy_(ckpt['model']['post_mean.weight'])
					model_r.latent.bias.copy_(ckpt['model']['post_mean.bias'])
				else:
					model_r.post_mean.weight.copy_(ckpt['model']['post_mean.weight'])
					model_r.post_mean.bias.copy_(ckpt['model']['post_mean.bias'])
					model_r.post_logstd.weight.copy_(ckpt['model']['post_logstd.weight'])
					model_r.post_logstd.bias.copy_(ckpt['model']['post_logstd.bias'])
					if 'post_cholesky.weight' in ckpt['model'] and isinstance(model_r, networks.FullCovVAE):
						model_r.post_cholesky.weight.copy_(ckpt['model']['post_cholesky.weight'])
						model_r.post_cholesky.bias.copy_(ckpt['model']['post_cholesky.bias'])
			else:
				model_r.latent.weight.copy_(ckpt['model']['latent.weight'])
				model_r.latent.bias.copy_(ckpt['model']['latent.bias'])
			model_r._decoder[0].weight.copy_(ckpt['model']['_decoder.0.weight'])
			model_r._decoder[0].bias.copy_(ckpt['model']['_decoder.0.bias'])
			model_r._decoder[2].weight.copy_(ckpt['model']['_decoder.2.weight'])
			model_r._decoder[2].bias.copy_(ckpt['model']['_decoder.2.bias'])
	
	print("Reading Data")
	# dataset = getattr(dataloaders, args.dataset)
	dataset = dataloaders.buetepage_hr
	if model_r.window_size ==1:
		train_iterator = DataLoader(dataset.SequenceDataset(args.src, train=True), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceDataset(args.src, train=False), batch_size=1, shuffle=True)
	else:
		train_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=True, window_length=model_r.window_size), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=False, window_length=model_r.window_size), batch_size=1, shuffle=True)
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
	for k in config_r.__dict__:
		s += str(k) + ' : ' + str(config_r.__dict__[k]) + '\n'
	writer.add_text('robot_config', s)

	writer.flush()

	print("Starting Epochs")
	hsmm = []
	if model_h.window_size > 1:
		nb_dim = model_h.latent_dim + model_r.latent_dim
	else:
		nb_dim = 2*model_h.latent_dim + 2*model_r.latent_dim
	nb_states = args.hsmm_components
	for i in range(NUM_ACTIONS):
		hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
		hsmm[-1].mu = np.zeros((nb_states, nb_dim))
		hsmm[-1].sigma = np.eye(nb_dim)[None].repeat(nb_states,0)
		hsmm[-1].Mu_Pd = np.zeros(nb_states)
		hsmm[-1].Sigma_Pd = np.ones(nb_states)
		hsmm[-1].Trans_Pd = np.ones((nb_states, nb_states))/nb_states

	for epoch in range(global_step, global_step+global_config.EPOCHS):
		model_r.train()
		print('training model')
		train_recon, train_kl, train_loss, xr_gen, zr_post_samples, x, iters = run_iteration(train_iterator, model_r, optimizer, hsmm, model_h)
		steps_done = (epoch+1)*iters
		print('writing training summaries')
		# write_summaries_robot(writer, train_recon, train_kl, train_loss, xr_gen, zr_post_samples, x, steps_done, 'train', model_r)
		# write_summaries_hr(writer, train_recon, train_kl, train_loss, xr_gen, zr_post_samples, x, steps_done, 'train', model_r)
		writer.add_scalar('train/loss', sum(train_loss), steps_done)
		writer.add_scalar('train/kl_div', sum(train_kl), steps_done)
		writer.add_scalar('train/recon_loss', sum(train_recon), steps_done)
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
		
		model_r.eval()
		with torch.no_grad():

			print('Training HSMM')
			for a in range(len(train_iterator.dataset.actidx)):
				s = train_iterator.dataset.actidx[a]
				z_encoded = []
				for j in range(s[0], s[1]):
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

					if model_h.window_size == 1:
						z1_vel = torch.diff(zh_post_samples, prepend=zh_post_samples[0:1], dim=0)
						z2_vel = torch.diff(zr_post_samples, prepend=zr_post_samples[0:1], dim=0)
						z_encoded.append(torch.concat([zh_post_samples, z1_vel, zr_post_samples, z2_vel], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
					else:
						z_encoded.append(torch.concat([zh_post_samples, zr_post_samples], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				hsmm[a].init_hmm_kbins(z_encoded)
				hsmm[a].em(z_encoded)
			print('testing model')
			test_recon, test_kl, test_loss, xr_gen, zr_post_samples, x, iters = run_iteration(test_iterator, model_r, optimizer, hsmm, model_h)
			print('writing testing summaries')
			# write_summaries_robot(writer, test_recon, test_kl, test_loss, xr_gen, zr_post_samples, x, steps_done, 'test', model_r)
			# write_summaries_hr(writer, test_recon, test_kl, test_loss, xr_gen, zr_post_samples, x, steps_done, 'test', model_r)
			writer.add_scalar('test/loss', sum(test_loss), steps_done)
			writer.add_scalar('test/kl_div', sum(test_kl), steps_done)
			writer.add_scalar('test/recon_loss', sum(test_recon), steps_done)
		
		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, checkpoint_file)
