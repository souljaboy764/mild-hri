import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import mild_hri.vae
from mild_hri.utils import *
from mild_hri.dataloaders import buetepage

import pbdlib as pbd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iteration(iterator, ssm, model, optimizer, args, epoch):
	iters = 0
	total_recon, total_reg, total_loss, mu_prior, Sigma_prior = [], [], [], [], []
	z_dim = args.latent_dim
	if ssm!=[]:
		with torch.no_grad():
			for i in range(len(ssm)):
				mu_prior.append(torch.concat([ssm[i].mu[None,:,:z_dim], ssm[i].mu[None, :,z_dim:]]))
				Sigma_prior.append(torch.concat([ssm[i].sigma[None, :, :z_dim, :z_dim], ssm[i].sigma[None, :, z_dim:, z_dim:]]))
	for i, x in enumerate(iterator):
		x, label = x
		x = x[0]
		# idx = idx[0]
		label = label[0]
		x = torch.Tensor(x).to(device)
		seq_len, dims = x.shape
		x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
		
		if model.training:
			optimizer.zero_grad()

		with torch.cuda.amp.autocast(dtype=torch.bfloat16):
		# if True:
			x_gen, zpost_samples, zpost_dist = model(x)

			if model.training:
				recon_loss = F.mse_loss(x[None].repeat(args.mce_samples+1,1,1,1), x_gen, reduction='sum')
			else:
				recon_loss = F.mse_loss(x, x_gen, reduction='sum')
				# z2_cond, _ = ssm[label].condition(zpost_samples[0], data_Sigma_in=None, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
				# z1_cond, _ = ssm[label].condition(zpost_samples[1], data_Sigma_in=None, dim_in=slice(z_dim, 2*z_dim), dim_out=slice(0, z_dim))
				# x_gen = model._output(model._decoder(torch.concat([z1_cond[None], z2_cond[None]])))
				# recon_loss = F.mse_loss(x, x_gen, reduction='sum')
			
			if model.training and epoch!=0:	
				with torch.no_grad():
					alpha_ssm = ssm[label].forward_variable(marginal=[], sample_size=seq_len)
					seq_alpha = alpha_ssm.argmax(0)
					z_prior = torch.distributions.MultivariateNormal(mu_prior[label][:, seq_alpha], Sigma_prior[label][:, seq_alpha])
				reg_loss = torch.distributions.kl_divergence(zpost_dist, z_prior).mean()
				total_reg.append(reg_loss)
				loss = recon_loss + args.beta*reg_loss
			else:
				loss = recon_loss
				total_reg.append(0)

			total_recon.append(recon_loss)
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

	assert args.dataset != 'buetepage_pepper'
	
	if args.dataset == 'buetepage':
		dataset = buetepage.HHWindowDataset
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
	model = getattr(mild_hri.vae, args.model)(**(args.__dict__)).to(device)
	params = model.parameters()
	named_params = model.named_parameters()
	optimizer = torch.optim.AdamW(params, lr=args.lr, fused=True)

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

	if args.ckpt is not None:
		ckpt = torch.load(args.ckpt)
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		ssm = ckpt['ssm']
		global_epochs = ckpt['epoch']

	print("Starting Epochs")
	ssm = init_ssm_torch(2*args.latent_dim, args.ssm_components, args.ssm, NUM_ACTIONS, device)
	if args.variant == 3:
		cov_reg = torch.diag_embed(torch.tile(torch.linspace(0.91, 1.0, args.latent_dim),(2,)))*args.cov_reg
	else:
		cov_reg = args.cov_reg
	
	torch.compile(model)
	for epoch in range(global_epochs,args.epochs):
		model.train()
		train_recon, train_kl, train_loss, x_gen, zx_samples, x, iters = run_iteration(train_iterator, ssm, model, optimizer, args, epoch)
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
				lens = []
				for j in range(s[0], s[1]):
				# for j in np.random.randint(s[0], s[1], 12):
					x, label = train_iterator.dataset[j]
					assert np.all(label == a)
					x = torch.Tensor(x).to(device)
					seq_len, dims = x.shape
					lens.append(seq_len)
					x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
					
					zpost_samples = model(x, encode_only=True)
					z_encoded.append(torch.concat([zpost_samples[0], zpost_samples[1]], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				ssm_np = getattr(pbd, args.ssm)(nb_dim=2*args.latent_dim, nb_states=args.ssm_components)
				ssm_np.init_hmm_kbins(z_encoded)
				ssm_np.em(z_encoded)
				for k in vars(ssm_np).keys():
					if isinstance(ssm_np.__getattribute__(k), np.ndarray):
						ssm[a].__setattr__(k, torch.Tensor(ssm_np.__getattribute__(k)).to(device).requires_grad_(False))
					else:
						ssm[a].__setattr__(k, ssm_np.__getattribute__(k))
				if args.variant == 3:
					ssm[a].reg = cov_reg.to(device).requires_grad_(False)
				else:
					ssm[a].reg = torch.Tensor(ssm_np.reg).to(device).requires_grad_(False)

				z_encoded = np.concatenate(z_encoded)
				for zdim in range(args.latent_dim):
					writer.add_histogram(f'z_h/{a}_{zdim}', z_encoded[:,zdim], steps_done)
					writer.add_histogram(f'z_r/{a}_{zdim}', z_encoded[:,args.latent_dim+zdim], steps_done)
				writer.add_image(f'hmm_{a}_trans', ssm[a].Trans, steps_done, dataformats='HW')
				alpha_ssm = ssm[a].forward_variable(marginal=[], sample_size=np.mean(lens).astype(int))
				writer.add_histogram(f'alpha/{a}', alpha_ssm.argmax(0), steps_done)
			test_recon, test_kl, test_loss, x_gen, zx_samples, x, iters = run_iteration(test_iterator, ssm, model, optimizer, args, epoch)
			write_summaries_vae(writer, test_recon, test_kl, steps_done, 'test')

		if epoch % 10 == 0:
			checkpoint_file = os.path.join(MODELS_FOLDER, '%0.4d.pth'%(epoch))
			torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ssm':ssm, 'args':args}, checkpoint_file)

		print(epoch,'epochs done')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, 'final.pth')
	torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': global_step, 'ssm':ssm, 'args':args}, checkpoint_file)
