import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, hsmm, model, optimizer):
	iters = 0
	total_recon = []
	total_reg = []
	total_loss = []
	z_dim = model.latent_dim
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		# idx = idx[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		# if model.window_size>1:
		# 	bp_idx = np.zeros((seq_len-model.window_size, model.window_size))
		# 	for i in range(model.window_size):
		# 		bp_idx[:,i] = idx[i:seq_len-model.window_size+i]
		# 	x = x[:, bp_idx].flatten(2)

		# if isinstance(model, networks.VAE):
		if hsmm!=[]:
			# mu_prior = torch.Tensor([hsmm[label].mu[:,:z_dim], hsmm[label].mu[:,z_dim:]]).to(device)
			# Sigma_prior = torch.Tensor([hsmm[label].sigma[:, :z_dim, :z_dim], hsmm[label].sigma[:, z_dim:, z_dim:]]).to(device)
			if model.window_size >1:
				mu_prior = torch.Tensor([hsmm[label].mu[:,:z_dim], hsmm[label].mu[:,z_dim:]]).to(device)
				Sigma_prior = torch.Tensor([hsmm[label].sigma[:, :z_dim, :z_dim], hsmm[label].sigma[:, z_dim:, z_dim:]]).to(device)
			else:
				mu_prior = torch.Tensor([hsmm[label].mu[:,:z_dim], hsmm[label].mu[:,2*z_dim:3*z_dim]]).to(device)
				Sigma_prior = torch.Tensor([hsmm[label].sigma[:, :z_dim, :z_dim], hsmm[label].sigma[:, 2*z_dim:3*z_dim, 2*z_dim:3*z_dim]]).to(device)
			alpha_hsmm, _, _, _, _ = hsmm[label].compute_messages(marginal=[], sample_size=seq_len)
			if np.any(np.isnan(alpha_hsmm)):
				print('Alpha Nan')
				alpha_hsmm = forward_variable(hsmm[label], n_step=seq_len)
			# alpha_hsmm = torch.Tensor(alpha_hsmm).to(device)
			seq_alpha = alpha_hsmm.argmax(0)

		if model.training:
			optimizer.zero_grad()

		x_gen, zpost_samples, zpost_dist = model(x)

			
		if model.training and isinstance(model, networks.VAE):
			recon_loss = F.mse_loss(x[None].repeat(11,1,1,1), x_gen, reduction='sum')
		else:
			recon_loss = F.mse_loss(x, x_gen, reduction='sum')

		reg_loss = 0.
		if isinstance(model, networks.VAE):
			z_prior = torch.distributions.MultivariateNormal(mu_prior[:, seq_alpha], Sigma_prior[:, seq_alpha])
			kld = torch.distributions.kl_divergence(zpost_dist, z_prior).mean(0)
			reg_loss += kld.mean()
		else:
			delta = mu_prior[:, seq_alpha] - zpost_samples
			kld0 = torch.bmm(delta[0,:,None], torch.matmul(torch.inverse(Sigma_prior[0, seq_alpha]), delta[0,:,:,None]))
			kld1 = torch.bmm(delta[1,:,None], torch.matmul(torch.inverse(Sigma_prior[1, seq_alpha]), delta[1,:,:,None]))
			reg_loss += (kld0+kld1).mean()
		
		if not model.training:
			if model.window_size >1:
				z2, _ = hsmm[label].condition(zpost_samples[0].cpu().numpy(), dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			else:
				z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
				z2, _ = hsmm[label].condition(torch.concat([zpost_samples[0], z1_vel], dim=-1).cpu().numpy(), dim_in=slice(0, 2*z_dim), dim_out=slice(2*z_dim, 3*z_dim))
			x2_gen = model._output(model._decoder(torch.Tensor(z2).to(device)))
			x1_gen = x_gen[0]
			x_gen = torch.concat([x1_gen[None], x2_gen[None]])
		
		loss = recon_loss + model.beta*reg_loss

		total_recon.append(recon_loss)
		total_reg.append(reg_loss)
		total_loss.append(loss)

		if model.training:
			loss.backward()
			optimizer.step()
		iters += 1
	
	return total_recon, total_reg, total_loss, x_gen, zpost_samples, x, iters

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/results/'+datetime.datetime.now().strftime("%m%d%H%M"), metavar='RES',
						help='Path for saving results (default: ./logs/results/MMDDHHmm).')
	parser.add_argument('--src', type=str, default='/home/vignesh/playground/hsmmvae/data/buetepage/traj_data.npz', metavar='RES',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/data/buetepage/traj_data.npz).')
	parser.add_argument('--pretrained', type=str, default=None, metavar='CKPT',
						help='Path to read pre-trained network.')
	parser.add_argument('--hsmm-components', type=int, default=8, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 10).')
	parser.add_argument('--latent-dim', type=int, default=5, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 10).')
	parser.add_argument('--model', type=str, default='AE', metavar='ARCH', choices=['AE', 'VAE', 'WAE', 'FullCovVAE'],
						help='Model to use: AE, VAE or WAE (default: VAE).')
	parser.add_argument('--dataset', type=str, default='buetepage', metavar='DATASET', choices=['buetepage', 'nuitrack', 'shakefive'],
						help='Dataset to use: buetepage, hhoi or shakefive (default: buetepage).')
	args = parser.parse_args()
	# torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	global_config = getattr(config, args.dataset).global_config()
	ae_config = getattr(config, args.dataset).ae_config()
	ae_config.latent_dim = args.latent_dim

	DEFAULT_RESULTS_FOLDER = args.results
	MODELS_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "models")
	SUMMARIES_FOLDER = os.path.join(DEFAULT_RESULTS_FOLDER, "summary")
	global_step = 0
	global_epochs = 0

	# elif os.path.exists(os.path.join(MODELS_FOLDER,'hyperparams.npz')):
	# 	hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	# 	args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	# 	global_config = hyperparams['global_config'].item()
	# 	ae_config = hyperparams['ae_config'].item()
	# 	print(ae_config.__dict__)

	print("Creating Model and Optimizer")

	if os.path.exists(os.path.join(MODELS_FOLDER, 'final.pth')):
		print("Loading Checkpoints")
		ckpt = torch.load(os.path.join(MODELS_FOLDER, 'final.pth'),device)
		model = getattr(networks, args.model)(**(ae_config.__dict__)).to(device)
		optimizer = getattr(torch.optim, global_config.optimizer)(model.parameters(), lr=global_config.lr)
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		global_epochs = ckpt['epoch']
	elif args.pretrained is not None:
		PRETRAINED_RESULTS_FOLDER = os.path.dirname(args.pretrained)
		pretrained_hyperparams = np.load(os.path.join(PRETRAINED_RESULTS_FOLDER,'hyperparams.npz'),allow_pickle=True)
		pretrained_args = pretrained_hyperparams['args']
		pretrained_global_config = pretrained_hyperparams['global_config']
		pretrained_ae_config = pretrained_hyperparams['ae_config']
		ckpt = torch.load(args.pretrained,device)

		if 'post_mean.weight' in ckpt['model']:
			ae_config.latent_dim = ckpt['model']['post_mean.bias'].shape[0]
		else:
			ae_config.latent_dim = ckpt['model']['latent.bias'].shape[0]

		if ckpt['model']['_output.bias'].shape[0] == 12:
			ae_config.window_size = 1
		elif ckpt['model']['_output.bias'].shape[0] == 480:
			ae_config.window_size = 5

		model = getattr(networks, args.model)(**(ae_config.__dict__)).to(device)
		optimizer = getattr(torch.optim, global_config.optimizer)(model.parameters(), lr=global_config.lr)
		with torch.no_grad():
			model._encoder[0].bias.copy_(ckpt['model']['_encoder.0.bias'])
			model._encoder[2].weight.copy_(ckpt['model']['_encoder.2.weight'])
			model._encoder[2].bias.copy_(ckpt['model']['_encoder.2.bias'])
			model._decoder[0].weight.copy_(ckpt['model']['_decoder.0.weight'])
			model._decoder[0].bias.copy_(ckpt['model']['_decoder.0.bias'])
			model._decoder[2].weight.copy_(ckpt['model']['_decoder.2.weight'])
			model._decoder[2].bias.copy_(ckpt['model']['_decoder.2.bias'])
			if 'post_mean.weight' in ckpt['model']:
				if isinstance(model, networks.AE):
					model.latent.weight.copy_(ckpt['model']['post_mean.weight'])
					model.latent.bias.copy_(ckpt['model']['post_mean.bias'])
				else:
					model.post_mean.weight.copy_(ckpt['model']['post_mean.weight'])
					model.post_mean.bias.copy_(ckpt['model']['post_mean.bias'])
					model.post_logstd.weight.copy_(ckpt['model']['post_logstd.weight'])
					model.post_logstd.bias.copy_(ckpt['model']['post_logstd.bias'])
					if 'post_cholesky.weight' in ckpt['model'] and isinstance(model, networks.FullCovVAE):
						model.post_cholesky.weight.copy_(ckpt['model']['post_cholesky.weight'])
						model.post_cholesky.bias.copy_(ckpt['model']['post_cholesky.bias'])
			else:
				model.latent.weight.copy_(ckpt['model']['latent.weight'])
				model.latent.bias.copy_(ckpt['model']['latent.bias'])
			if ae_config.window_size==1:
				model._encoder[0].weight.copy_(ckpt['model']['_encoder.0.weight'])
				model._encoder[0].bias.copy_(ckpt['model']['_encoder.0.bias'])
				model._output.weight.copy_(ckpt['model']['_output.weight'])
				model._output.bias.copy_(ckpt['model']['_output.bias'])
			else:
				model._encoder[0].weight.copy_(ckpt['model']['_encoder.0.weight'][:, :model.input_dim])
				model._output.weight.copy_(ckpt['model']['_output.weight'][:model.input_dim, :])
				model._output.bias.copy_(ckpt['model']['_output.bias'][:model.input_dim])

	print("Reading Data")
	dataset = getattr(dataloaders, args.dataset)
	if model.window_size ==1:
		train_iterator = DataLoader(dataset.SequenceDataset(args.src, train=True), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceDataset(args.src, train=False), batch_size=1, shuffle=True)
	else:
		train_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=True, window_length=model.window_size), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=False, window_length=model.window_size), batch_size=1, shuffle=True)
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

	if not os.path.exists(DEFAULT_RESULTS_FOLDER):
		print("Creating Result Directories")
		os.makedirs(DEFAULT_RESULTS_FOLDER)
		os.makedirs(MODELS_FOLDER)
		os.makedirs(SUMMARIES_FOLDER)
		np.savez_compressed(os.path.join(MODELS_FOLDER,'hyperparams.npz'), args=args, global_config=global_config, ae_config=ae_config)

	print("Starting Epochs")
	hsmm = []
	if model.window_size == 1:
		nb_dim = 4*model.latent_dim
	else:
		nb_dim = 2*model.latent_dim
	nb_states = args.hsmm_components
	for i in range(NUM_ACTIONS):
		hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
		hsmm[-1].mu = np.zeros((nb_states, nb_dim))
		hsmm[-1].sigma = np.eye(nb_dim)[None].repeat(nb_states,0)
		hsmm[-1].Mu_Pd = np.zeros(nb_states)
		hsmm[-1].Sigma_Pd = np.ones(nb_states)
		hsmm[-1].Trans_Pd = np.ones((nb_states, nb_states))/nb_states

	for epoch in range(global_epochs,global_epochs+global_config.EPOCHS):
		model.train()
		train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iteration(train_iterator, hsmm, model, optimizer)
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
			test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iteration(test_iterator, hsmm, model, optimizer)
			write_summaries_vae(writer, test_recon, test_kl, test_loss, x_gen, zx_samples, x, steps_done, 'test', model)
			print(np.any(np.isnan(x_gen.detach().cpu().numpy())))
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
					if model.window_size == 1:
						z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
						z2_vel = torch.diff(zpost_samples[1], prepend=zpost_samples[1][0:1], dim=0)
						z_encoded.append(torch.concat([zpost_samples[0], z1_vel, zpost_samples[1], z2_vel], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
					else:
						z_encoded.append(torch.concat([zpost_samples[0], zpost_samples[1]], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				hsmm[a].init_hmm_kbins(z_encoded)
				hsmm[a].em(z_encoded)

		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step, 'hsmm':hsmm}, checkpoint_file)
