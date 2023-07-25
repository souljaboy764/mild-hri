import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Sophia import SophiaG 

import numpy as np
import os, datetime, argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
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
					model_h:networks.VAE, 
					model_r:networks.VAE, 
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

		if args.cov_cond:
			data_Sigma_in = zh_post.covariance_matrix
		else:
			data_Sigma_in = None

		if args.variant==2 and epoch >= args.pretrain:
			# Sample Conditioning: Sampling from Posterior and then Conditioning the HMM
			xr_cond = []
			zr_cond_mean = []
			for zh in zh_samples:
				zr_cond = hsmm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
												return_cov=False, data_Sigma_in=data_Sigma_in)
				zr_cond_mean.append(zr_cond[None])
			zr_cond_mean = torch.concat(zr_cond_mean)
			xr_cond = model_r._output(model_r._decoder(zr_cond_mean))

		elif args.variant==3 and epoch >= args.pretrain:
			# Conditioned Sampling: Conditioning on the Posterior and then Sampling from the conditional distribution
			zr_cond_mean, zr_cond_sigma = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
											return_cov=True, data_Sigma_in=data_Sigma_in)
			if model_h.training and model_r.mce_samples>0:
				try:
					zr_cond = torch.distributions.MultivariateNormal(zr_cond_mean, zr_cond_sigma)
				except Exception as e:
					zr_cond = torch.distributions.MultivariateNormal(zr_cond_mean, batchNearestPD(zr_cond_sigma, eps=hsmm[label].reg[0][0]))
				xr_cond = model_r._output(model_r._decoder(torch.concat([zr_cond.rsample((model_r.mce_samples,)), zr_cond_mean[None]], dim=0)))
			else:
				xr_cond = model_r._output(model_r._decoder(zr_cond_mean))

		if model_h.training:
			# Pretraining
			if args.variant==1 or (args.variant!=1 and epoch < args.pretrain):
				recon_loss = ((xh_gen - x_h[None])**2).mean() + ((xr_gen - x_r[None])**2).mean()
			# conditional training
			else:
				if args.gamma!=1:
					factor = args.gamma**(epoch - args.pretrain)
					recon_loss = (((xh_gen - x_h[None])**2).mean() + ((xr_gen - x_r[None])**2).mean())*factor + ((xr_cond - x_r[None])**2).mean()*(1-factor)
				else:
					recon_loss = (((xh_gen - x_h[None])**2).mean() + ((xr_gen - x_r[None])**2).mean()) + ((xr_cond - x_r[None])**2).mean()
		else:
			if args.variant==1 or (args.variant!=1 and epoch < args.pretrain):
				# Simple conditioninal reconstruction
				zr_cond_mean = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h, 
												return_cov=False, data_Sigma_in=data_Sigma_in)
				xr_cond = model_r._output(model_r._decoder(zr_cond_mean))
			recon_loss = ((xr_cond - x_r)**2).sum()
	
		loss = recon_loss

		if model_h.beta!=0 and epoch < args.epochs:
			with torch.no_grad():
				zh_prior = torch.distributions.MultivariateNormal(mu_prior[label][0, :, None], Sigma_prior[label][0, :, None])
				zr_prior = torch.distributions.MultivariateNormal(mu_prior[label][1, :, None], Sigma_prior[label][1, :, None])
			reg_loss = args.beta * (fwd_h*(torch.distributions.kl_divergence(zh_post, zh_prior) + \
					torch.distributions.kl_divergence(zr_post, zr_prior))).mean()
		else:
			reg_loss = torch.zeros_like(recon_loss)

		loss += reg_loss
		total_recon.append(recon_loss.mean())
		total_reg.append(reg_loss.mean())
		total_loss.append(loss.mean())
		if model_h.training:
			if args.grad_clip!=0:
				torch.nn.utils.clip_grad_norm_(model_h.parameters(), args.grad_clip)
				torch.nn.utils.clip_grad_norm_(model_r.parameters(), args.grad_clip)
			loss.backward()
			optimizer.step()
	return total_recon, total_reg, total_loss, i

# def finetune_dataset(iterator:DataLoader, hsmm:List[pbd_torch.HMM], model_h:networks.VAE, training:bool, args):
# 	io_pairs = []
# 	z_dim = model_h.latent_dim
# 	with torch.no_grad():
# 		for i, x in enumerate(iterator):
# 			x, label = x
# 			x = x[0]
# 			label = label[0]
# 			x = torch.Tensor(x).to(device)
# 			x_h = x[:, :model_h.input_dim]
# 			x_r = x[:, model_h.input_dim:]

# 			zh_post = model_h(x_h, dist_only=True)
# 			assert not torch.any(torch.isnan(zh_post.mean))
# 			assert not torch.any(torch.isinf(zh_post.mean))
# 			if args.cov_cond:
# 				data_Sigma_in=zh_post.covariance_matrix
# 			else:
# 				data_Sigma_in = None

# 			if args.variant==1 or args.variant==2:
# 				if training and model_h.mce_samples>0:
# 					zh_samples = torch.concat([zh_post.rsample((model_h.mce_samples,)), zh_post.mean[None]], dim=0)
# 				else:
# 					zh_samples = [zh_post.mean]
# 				# Sample Conditioning: Sampling from Posterior and then Conditioning the HMM
# 				zr_cond = []
# 				for zh in zh_samples:
# 					fwd_h = hsmm[label].forward_variable(demo=zh, marginal=slice(0, z_dim))
# 					assert not torch.any(torch.isnan(fwd_h))
# 					assert not torch.any(torch.isinf(fwd_h))
# 					zr_cond_mean = hsmm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h,
# 													return_cov=False, data_Sigma_in=data_Sigma_in)
# 					zr_cond.append(zr_cond_mean[None])
# 				if training:
# 					zr_cond = torch.concat(zr_cond)
# 				else:
# 					zr_cond = torch.concat(zr_cond)[0]
			
				
# 			elif args.variant==3:
# 				# Conditioned Sampling: Conditioning on the Posterior and then Sampling from the conditional distribution
# 				fwd_h = hsmm[label].forward_variable(demo=zh, marginal=slice(0, z_dim))
# 				assert not torch.any(torch.isnan(fwd_h))
# 				assert not torch.any(torch.isinf(fwd_h))
# 				zr_cond_mean, zr_cond_sigma = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h,
# 												return_cov=True, data_Sigma_in=data_Sigma_in)
# 				if training and model_h.mce_samples>0:
# 					try:
# 						zr_cond_dist = torch.distributions.MultivariateNormal(zr_cond_mean, zr_cond_sigma)
# 					except Exception as e:
# 						zr_cond_dist = torch.distributions.MultivariateNormal(zr_cond_mean, batchNearestPD(zr_cond_sigma, eps=hsmm[label].reg[0][0]))
# 					zr_cond = torch.concat([zr_cond_dist.rsample((model_h.mce_samples,)), zr_cond_mean[None]], dim=0)
# 				else:
# 					zr_cond = zr_cond_mean
			
# 			assert not torch.any(torch.isnan(zr_cond))
# 			assert not torch.any(torch.isinf(zr_cond))
# 			zr = zr_cond.detach().cpu().numpy().reshape(-1, z_dim)
# 			if training and model_h.mce_samples>0:
# 				x_r = x_r[None].repeat(model_h.mce_samples+1,1,1)
# 			xr = x_r.detach().cpu().numpy().reshape(-1, x_r.shape[-1])
# 			io = np.concatenate([zr, xr], axis=-1)
# 			io_pairs += io.tolist()

# 	io_pairs = np.array(io_pairs).astype(np.float32)
# 	return io_pairs


# def run_finetune_iteration(iterator:DataLoader, model_r:networks.VAE, optimizer:torch.optim.Optimizer, training:bool):
# 	total_loss = []
# 	z_dim = model_r.latent_dim
# 	for i, val in enumerate(iterator):
# 		if training:
# 			optimizer.zero_grad()
# 		val = val.to(device)
# 		xr_gt = val[:, z_dim:]
# 		z_in = val[:, :z_dim]
# 		xr_pred = model_r._output(model_r._decoder(z_in))
# 		if training:
# 			loss = ((xr_pred - xr_gt)**2).mean()
# 		else:
# 			loss = ((xr_pred - xr_gt)**2).sum()
# 		total_loss.append(loss)
# 		if training:
# 			loss.backward()
# 			optimizer.step()
	
# 	return total_loss, i


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
	parser.add_argument('--latent-dim', type=int, default=5, metavar='Z',
						help='Latent space dimension (default: 5)')
	parser.add_argument('--cov-reg', type=float, default=1e-3, metavar='EPS',
						help='Positive value to add to covariance diagonal (default: 1e-3)')
	parser.add_argument('--beta', type=float, default=0.005, metavar='BETA',
						help='Scaling factor for KL divergence (default: 0.005)')
	parser.add_argument('--window-size', type=int, default=5, metavar='WINDOW',
						help='Window Size for inputs (default: 5)')
	parser.add_argument('--downsample', type=float, default=0.2, metavar='DOWNSAMPLE',
						help='Factor for downsampling the data (default: 0.2)')
	parser.add_argument('--mce-samples', type=int, default=10, metavar='MCE',
						help='Number of Monte Carlo samples to draw (default: 10)')
	parser.add_argument('--grad-clip', type=float, default=0.5, metavar='CLIP',
						help='Value to clip gradients at (default: 0.5)')
	parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
						help='Number of epochs to train for (default: 100)')
	parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
						help='Starting Learning Rate (default: 5e-4)')
	parser.add_argument('--pretrain', type=int, default=0, metavar='PRETRAIN',
						help='Number of epochs to pretrain for (default: 0)')
	parser.add_argument('--finetune', type=int, default=0, metavar='FINETUNE',
						help='Number of epochs to finetune for after training (default: 0)')
	parser.add_argument('--gamma', type=float, default=0.95, metavar='GAMMA',
						help='Starting Relative weight for VAE vs conditional reconstruction (default: 0.95)')
	parser.add_argument('--optimizer', type=str, default='AdamW', metavar='OPTIM', choices=['AdamW', 'Adam', 'Adagrad', 'RMSprop'],
						help='Optimizer to use: AdamW, Adam, Adagrad or RMSprop (default: AdamW).')
	parser.add_argument('--variant', type=int, default=2, metavar='VARIANT', choices=[1, 2, 3],
						help='Which variant to use 1 - vanilla, 2 - sample conditioning, 3 - conditional sampling (default: 1).')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	parser.add_argument('--cov-cond', action='store_true', 
						help='Whether to use covariance for conditioning or not')
	args = parser.parse_args()
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
	robot_vae_config.beta = ae_config.beta = args.beta
	robot_vae_config.window_size = ae_config.window_size = args.window_size
	robot_vae_config.mce_samples = ae_config.mce_samples = args.mce_samples

	print("Reading Data")
	train_iterator = DataLoader(dataset(args.src, train=True, window_length=args.window_size, downsample=args.downsample), batch_size=1, shuffle=True)
	test_iterator = DataLoader(dataset(args.src, train=False, window_length=args.window_size, downsample=args.downsample), batch_size=1, shuffle=False)
	MODELS_FOLDER = os.path.join(args.results, "models")
	SUMMARIES_FOLDER = os.path.join(args.results, "summary")
	global_step = 0
	global_epochs = 0

	print("Creating Model and Optimizer")

	model_h = getattr(networks, args.model)(**(ae_config.__dict__)).to(device)
	model_r = getattr(networks, args.model)(**(robot_vae_config.__dict__)).to(device)
	params = list(model_h.parameters()) + list(model_r.parameters())
	named_params = list(model_h.named_parameters()) + list(model_r.named_parameters())
	optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr)
	# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.965)
	# if args.pretrain!=0:
	# 	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.pretrain)
	
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
		hyperparams = np.load(os.path.join(os.path.dirname(args.ckpt),'hyperparams.npz'), allow_pickle=True)
		seed = hyperparams['args'].item().seed
		torch.manual_seed(seed)
		np.random.seed(seed)
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
		params = []
		grads = []
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
				hsmm_np.em(z_encoded, reg=args.cov_reg, reg_finish=args.cov_reg)
				# hsmm_np.em(np.concatenate(z_encoded))

				hsmm[a].mu = torch.Tensor(hsmm_np.mu).to(device).requires_grad_(False)
				hsmm[a].sigma = torch.Tensor(hsmm_np.sigma).to(device).requires_grad_(False)
				hsmm[a].priors = torch.Tensor(hsmm_np.priors).to(device).requires_grad_(False)
				hsmm[a].reg = torch.Tensor(hsmm_np.reg).to(device).requires_grad_(False)
				hsmm[a].trans = torch.Tensor(hsmm_np.trans).to(device).requires_grad_(False)
				hsmm[a].Trans = torch.Tensor(hsmm_np.Trans).to(device).requires_grad_(False)
				hsmm[a].init_priors = torch.Tensor(hsmm_np.init_priors).to(device).requires_grad_(False)
				# hsmm[a].mu_d = torch.Tensor(hsmm_np.mu_d).to(device).requires_grad_(False)
				# hsmm[a].sigma_d = torch.Tensor(hsmm_np.sigma_d).to(device).requires_grad_(False)
				# hsmm[a].trans_d = torch.Tensor(hsmm_np.trans_d).to(device).requires_grad_(False)
				# hsmm[a].Mu_Pd = torch.Tensor(hsmm_np.Mu_Pd).to(device).requires_grad_(False)
				# hsmm[a].Sigma_Pd = torch.Tensor(hsmm_np.Sigma_Pd).to(device).requires_grad_(False)
				# hsmm[a].Trans_Pd = torch.Tensor(hsmm_np.Trans_Pd).to(device).requires_grad_(False)
		for name, param in named_params:
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
			test_recon, test_kl, test_loss, iters = run_iteration(test_iterator, hsmm, model_h, model_r, optimizer, args, epoch)
			write_summaries_vae(writer, test_recon, test_kl, steps_done, 'test')

		if epoch % 10 == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, f'{epoch:04d}.pth')
			torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)
		# scheduler.step()
		print(epoch,'epochs done')
	# if epoch != global_epochs:
	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch+1}.pth')
	torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

	# ### Finetuning
	# if args.finetune >0:
	# 	train_finetune_iterator = DataLoader(finetune_dataset(train_iterator, hsmm, model_h, True, args), batch_size=50, shuffle=True)
	# 	test_finetune_iterator = DataLoader(finetune_dataset(test_iterator, hsmm, model_h, False, args), batch_size=1000, shuffle=False)
	# 	params = []
	# 	for n,p in model_r.named_parameters():
	# 		if n.split('.')[0] == '_decoder' or n.split('.')[0] == '_output':
	# 			p.requires_grad = True
	# 			params.append(p)
	# 		else:
	# 			p.requires_grad = False
	# 	for p in model_h.parameters():
	# 		p.requires_grad = False

	# 	# params = list(model_r._output.parameters()) + list(model_r._decoder.parameters())
	# 	named_params = list(model_r._output.named_parameters()) + list(model_r._decoder.named_parameters())
	# 	finetune_optimizer = getattr(torch.optim, args.optimizer)(params, lr=args.lr)

	# 	for epoch in range(epoch+1, args.finetune):# + global_epochs):
	# 		model_r._output.train()
	# 		model_r._decoder.train()

	# 		train_recon, iters = run_finetune_iteration(train_finetune_iterator, model_r, finetune_optimizer, True)
	# 		steps_done = (epoch+1)*iters
	# 		write_summaries_vae(writer, train_recon, [], steps_done, 'train')
	# 		params = []
	# 		grads = []
	# 		for name, param in named_params:
	# 			if param.grad is None:
	# 				continue
	# 			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
	# 			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
	# 			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
	# 				print('zero grad for',name)
			
	# 		model_r._output.eval()
	# 		model_r._decoder.eval()
	# 		with torch.no_grad():				
	# 			test_recon, iters = run_finetune_iteration(test_finetune_iterator, model_r, finetune_optimizer, False)
	# 			write_summaries_vae(writer, test_recon, [], steps_done, 'test')

	# 		if epoch % 10 == 0:
	# 			checkpoint_file = os.path.join(MODELS_FOLDER, f'finetune_{epoch:04d}.pth')
	# 			torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

	# 		print(f'{epoch:02d}\t{sum(train_recon):.4e}\t{sum(test_recon):.4e}')
	# 	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_finetune_{epoch+1}.pth')
	# 	torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)
	
	writer.flush()

