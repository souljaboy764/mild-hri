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

def finetune_dataset(iterator:DataLoader, hsmm:List[pbd_torch.HMM], model_h:networks.VAE, training:bool, args):
	io_pairs = []
	z_dim = model_h.latent_dim
	with torch.no_grad():
		for i, x in enumerate(iterator):
			x, label = x
			x = x[0]
			label = label[0]
			x = torch.Tensor(x).to(device)
			x_h = x[:, :model_h.input_dim]
			x_r = x[:, model_h.input_dim:]

			zh_post = model_h(x_h, dist_only=True)
			assert not torch.any(torch.isnan(zh_post.mean))
			assert not torch.any(torch.isinf(zh_post.mean))
			if args.cov_cond:
				data_Sigma_in=zh_post.covariance_matrix
			else:
				data_Sigma_in = None

			if args.variant==1 or args.variant==2:
				if training and model_h.mce_samples>0:
					zh_samples = torch.concat([zh_post.rsample((model_h.mce_samples,)), zh_post.mean[None]], dim=0)
				else:
					zh_samples = [zh_post.mean]
				# Sample Conditioning: Sampling from Posterior and then Conditioning the HMM
				zr_cond = []
				for zh in zh_samples:
					fwd_h = hsmm[label].forward_variable(demo=zh, marginal=slice(0, z_dim))
					assert not torch.any(torch.isnan(fwd_h))
					assert not torch.any(torch.isinf(fwd_h))
					zr_cond_mean = hsmm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h,
													return_cov=False, data_Sigma_in=data_Sigma_in)
					zr_cond.append(zr_cond_mean[None])
				if training:
					zr_cond = torch.concat(zr_cond)
				else:
					zr_cond = torch.concat(zr_cond)[0]
			
				
			elif args.variant==3:
				# Conditioned Sampling: Conditioning on the Posterior and then Sampling from the conditional distribution
				fwd_h = hsmm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim))
				assert not torch.any(torch.isnan(fwd_h))
				assert not torch.any(torch.isinf(fwd_h))
				zr_cond_mean, zr_cond_sigma = hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=fwd_h,
												return_cov=True, data_Sigma_in=data_Sigma_in)
				if training and model_h.mce_samples>0:
					try:
						zr_cond_dist = torch.distributions.MultivariateNormal(zr_cond_mean, zr_cond_sigma)
					except Exception as e:
						zr_cond_dist = torch.distributions.MultivariateNormal(zr_cond_mean, batchNearestPD(zr_cond_sigma, eps=hsmm[label].reg[0][0]))
					zr_cond = torch.concat([zr_cond_dist.rsample((model_h.mce_samples,)), zr_cond_mean[None]], dim=0)
				else:
					zr_cond = zr_cond_mean
			
			assert not torch.any(torch.isnan(zr_cond))
			assert not torch.any(torch.isinf(zr_cond))
			zr = zr_cond.detach().cpu().numpy().reshape(-1, z_dim)
			if training and model_h.mce_samples>0:
				x_r = x_r[None].repeat(model_h.mce_samples+1,1,1)
			xr = x_r.detach().cpu().numpy().reshape(-1, x_r.shape[-1])
			io = np.concatenate([zr, xr], axis=-1)
			io_pairs += io.tolist()

	io_pairs = np.array(io_pairs).astype(np.float32)
	return io_pairs


def run_iteration(iterator:DataLoader, model_r:networks.VAE, optimizer:torch.optim.Optimizer, training:bool, args):
	total_loss = []
	z_dim = model_r.latent_dim
	for i, val in enumerate(iterator):
		if training:
			optimizer.zero_grad()
		val = val.to(device)
		xr_gt = val[:, z_dim:]
		z_in = val[:, :z_dim]
		xr_pred = model_r._output(model_r._decoder(z_in))
		if training:
			loss = ((xr_pred - xr_gt)**2).mean()
		else:
			loss = ((xr_pred - xr_gt)**2).sum()
		total_loss.append(loss)
		if training:
			if args.grad_clip!=0:
				torch.nn.utils.clip_grad_norm_(model_r._output.parameters(), args.grad_clip)
				torch.nn.utils.clip_grad_norm_(model_r._decoder.parameters(), args.grad_clip)
			loss.backward()
			optimizer.step()
	
	return total_loss, i
	
if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	args = parser.parse_args()
	ckpt = torch.load(args.ckpt)
	MODELS_FOLDER = os.path.dirname(args.ckpt)
	hyperparams = np.load(os.path.join(MODELS_FOLDER, 'hyperparams.npz'), allow_pickle=True)
	args_ckpt = hyperparams['args'].item()
	ae_config = hyperparams['ae_config'].item()
	robot_vae_config = hyperparams['robot_vae_config'].item()
	if args_ckpt.dataset == 'buetepage_pepper':
		dataset = dataloaders.buetepage.PepperWindowDataset
	elif args_ckpt.dataset == 'buetepage':
		dataset = dataloaders.buetepage.HHWindowDataset
	# TODO: Nuitrack

	train_iterator = DataLoader(dataset(args_ckpt.src, train=True, window_length=args_ckpt.window_size, downsample=args_ckpt.downsample), batch_size=1, shuffle=False)
	test_iterator = DataLoader(dataset(args_ckpt.src, train=False, window_length=args_ckpt.window_size, downsample=args_ckpt.downsample), batch_size=1, shuffle=False)

	model_h = getattr(networks, args_ckpt.model)(**(ae_config.__dict__)).to(device)
	model_h.load_state_dict(ckpt['model_h'])
	model_r = getattr(networks, args_ckpt.model)(**(robot_vae_config.__dict__)).to(device)
	model_r.load_state_dict(ckpt['model_r'])
	hsmm = ckpt['hsmm']
	epochs_done = ckpt['epoch']

	params = []
	for n,p in model_r.named_parameters():
		if n.split('.')[0] == '_decoder' or n.split('.')[0] == '_output':
			p.requires_grad = True
			params.append(p)
		else:
			p.requires_grad = False
	for p in model_h.parameters():
		p.requires_grad = False

	# params = list(model_r._output.parameters()) + list(model_r._decoder.parameters())
	named_params = list(model_r._output.named_parameters()) + list(model_r._decoder.named_parameters())
	optimizer = torch.optim.AdamW(params, lr=1e-3)
	
	train_finetune_iterator = DataLoader(finetune_dataset(train_iterator, hsmm, model_h, True, args_ckpt), batch_size=100, shuffle=True)
	test_finetune_iterator = DataLoader(finetune_dataset(test_iterator, hsmm, model_h, False, args_ckpt), batch_size=1000, shuffle=False)
	
	print("Building Writer")
	SUMMARIES_FOLDER = os.path.join(os.path.dirname(MODELS_FOLDER), "summary")
	if not os.path.exists(SUMMARIES_FOLDER):
		print("Creating Model Directory")
		os.makedirs(SUMMARIES_FOLDER)
	writer = SummaryWriter(SUMMARIES_FOLDER)
	print("Starting Epochs")

	for epoch in range(epochs_done, epochs_done+int(args_ckpt.epochs*0.5)):
		model_r._output.train()
		model_r._decoder.train()

		train_recon, iters = run_iteration(train_finetune_iterator, model_r, optimizer, True, args_ckpt)
		steps_done = (epoch+1)*iters
		write_summaries_vae(writer, train_recon, [], steps_done, 'train')
		params = []
		grads = []
		for name, param in named_params:
			if param.grad is None:
				continue
			writer.add_histogram('grads/'+name, param.grad.reshape(-1), steps_done)
			writer.add_histogram('param/'+name, param.reshape(-1), steps_done)
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print('zero grad for',name)
		
		model_r._output.eval()
		model_r._decoder.eval()
		with torch.no_grad():				
			test_recon, iters = run_iteration(test_finetune_iterator, model_r, optimizer, False, args_ckpt)
			write_summaries_vae(writer, test_recon, [], steps_done, 'test')

		print(f'{epoch:02d}\t{sum(train_recon):.4e}\t{sum(test_recon):.4e}')

	writer.flush()

	# checkpoint_file = args.ckpt[:-4]+'_finetuning.pth'
	# torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)
