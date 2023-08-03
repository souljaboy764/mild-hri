import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from vae import *
import config
from utils import *
import dataloaders

import pbdlib as pbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, hsmm, model, optimizer, args):
	iters = 0
	total_recon = []
	total_reg = []
	total_loss = []
	z_dim = args.latent_dim
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		# idx = idx[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		# if args.window_size>1:
		# 	bp_idx = np.zeros((seq_len-args.window_size, args.window_size))
		# 	for i in range(args.window_size):
		# 		bp_idx[:,i] = idx[i:seq_len-args.window_size+i]
		# 	x = x[:, bp_idx].flatten(2)

		# if isinstance(model, networks.VAE):
		if hsmm!=[]:
			# mu_prior = torch.Tensor([hsmm[label].mu[:,:z_dim], hsmm[label].mu[:,z_dim:]]).to(device)
			# Sigma_prior = torch.Tensor([hsmm[label].sigma[:, :z_dim, :z_dim], hsmm[label].sigma[:, z_dim:, z_dim:]]).to(device)
			if args.window_size >1:
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

			
		if model.training and isinstance(model, VAE):
			recon_loss = F.mse_loss(x[None].repeat(args.mce_samples+1,1,1,1), x_gen, reduction='sum')
		else:
			recon_loss = F.mse_loss(x, x_gen, reduction='sum')

		reg_loss = 0.
		if isinstance(model, VAE):
			z_prior = torch.distributions.MultivariateNormal(mu_prior[:, seq_alpha], Sigma_prior[:, seq_alpha])
			kld = torch.distributions.kl_divergence(zpost_dist, z_prior).mean(0)
			reg_loss += kld.mean()
		else:
			delta = mu_prior[:, seq_alpha] - zpost_samples
			kld0 = torch.bmm(delta[0,:,None], torch.matmul(torch.inverse(Sigma_prior[0, seq_alpha]), delta[0,:,:,None]))
			kld1 = torch.bmm(delta[1,:,None], torch.matmul(torch.inverse(Sigma_prior[1, seq_alpha]), delta[1,:,:,None]))
			reg_loss += (kld0+kld1).mean()
		
		if not model.training:
			if args.window_size >1:
				z2, _ = hsmm[label].condition(zpost_samples[0].cpu().numpy(), Sigma_in=None, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			else:
				z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
				z2, _ = hsmm[label].condition(torch.concat([zpost_samples[0], z1_vel], dim=-1).cpu().numpy(), Sigma_in=None, dim_in=slice(0, 2*z_dim), dim_out=slice(2*z_dim, 3*z_dim))
			x2_gen = model._output(model._decoder(torch.Tensor(z2).to(device)))
			x1_gen = x_gen[0]
			x_gen = torch.concat([x1_gen[None], x2_gen[None]])
		
		loss = recon_loss + args.beta*reg_loss

		total_recon.append(recon_loss)
		total_reg.append(reg_loss)
		total_loss.append(loss)

		if model.training:
			loss.backward()
			optimizer.step()
		iters += 1
	
	return total_recon, total_reg, total_loss, x_gen, zpost_samples, x, iters

if __name__=='__main__':
	args = training_argparse()
	print('Random Seed',args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.autograd.set_detect_anomaly(True)

	ae_config = config.buetepage.ae_config()
	ae_config.latent_dim = args.latent_dim
	ae_config.window_size = args.window_size
	ae_config.mce_samples = args.mce_samples
	if args.dataset == 'buetepage_pepper':
		robot_vae_config = config.buetepage.robot_vae_config()
		robot_vae_config.num_joints = 4
		dataset = dataloaders.buetepage.PepperWindowDataset
	elif args.dataset == 'buetepage':
		robot_vae_config = config.buetepage.ae_config()
		dataset = dataloaders.buetepage.HHWindowDataset
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
	model = FullCovVAE(**(ae_config.__dict__)).to(device)
	params = model.parameters()
	named_params = model.named_parameters()
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

	print("Starting Epochs")
	hsmm = []
	nb_dim = 2*args.latent_dim
	nb_states = args.hsmm_components
	for i in range(NUM_ACTIONS):
		hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
		hsmm[-1].mu = np.zeros((nb_states, nb_dim))
		hsmm[-1].sigma = np.eye(nb_dim)[None].repeat(nb_states,0)
		hsmm[-1].Mu_Pd = np.zeros(nb_states)
		hsmm[-1].Sigma_Pd = np.ones(nb_states)
		hsmm[-1].Trans_Pd = np.ones((nb_states, nb_states))/nb_states

	for epoch in range(global_epochs,args.epochs):
		model.train()
		train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iteration(train_iterator, hsmm, model, optimizer, args)
		steps_done = (epoch+1)*iters
		write_summaries_vae(writer, train_recon, train_kl, steps_done, 'train')
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
					if args.window_size == 1:
						z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
						z2_vel = torch.diff(zpost_samples[1], prepend=zpost_samples[1][0:1], dim=0)
						z_encoded.append(torch.concat([zpost_samples[0], z1_vel, zpost_samples[1], z2_vel], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
					else:
						z_encoded.append(torch.concat([zpost_samples[0], zpost_samples[1]], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				hsmm[a].init_hmm_kbins(z_encoded)
				hsmm[a].em(z_encoded)

				z_encoded = np.concatenate(z_encoded)
				for zdim in range(args.latent_dim):
					writer.add_histogram(f'z_h/{a}_{zdim}', z_encoded[:,zdim], steps_done)
					writer.add_histogram(f'z_r/{a}_{zdim}', z_encoded[:,args.latent_dim+zdim], steps_done)
				writer.add_image(f'hmm_{a}_trans', hsmm[a].Trans*255, steps_done, dataformats='HW')
				alpha = np.zeros((nb_states*10, 100))
				alpha_hsmm = hsmm[a].forward_variable(marginal=[], sample_size=100)
				for n in range(nb_states):
					alpha[n*10:(n+1)*10, :] = alpha_hsmm[n]
				writer.add_image(f'hmm_{a}_alpha', alpha*255, steps_done, dataformats='HW')
				
				writer.add_histogram(f'alpha/{a}', alpha_hsmm.argmax(0), steps_done)
			test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iteration(test_iterator, hsmm, model, optimizer, args)
			write_summaries_vae(writer, test_recon, test_kl, steps_done, 'test')

		if epoch % 10 == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step, 'hsmm':hsmm}, checkpoint_file)
