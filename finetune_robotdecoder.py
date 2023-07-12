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

def finetune_dataset(iterator:DataLoader, hsmm:List[pbd_torch.HMM], model_h:networks.VAE, cov:bool):
	training_inputs = []
	training_targets = []
	z_dim = model_h.latent_dim
	with torch.no_grad():
		for i, x in enumerate(iterator):
			x, label = x
			x = x[0]
			label = label[0]
			x = torch.Tensor(x).to(device)
			x_h = x[:, :model_h.input_dim]
			x_r = x[:, model_h.input_dim:]
			training_targets += x_r.detach().cpu().numpy().tolist()
			if cov:
				zh_post = model_h(x_h, dist_only=True)
				training_inputs += hsmm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), 
														return_cov=False,
														data_Sigma_in=zh_post.covariance_matrix
														).detach().cpu().numpy().tolist()
			else:
				zh = model_h(x_h, encode_only=True)
				training_inputs +=  hsmm[label].condition(zh, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), 
						   								return_cov=False,
														data_Sigma_in = None,
														).detach().cpu().numpy().tolist()

	return np.hstack([training_inputs, training_targets]).astype(np.float32)


def run_iteration(iterator:DataLoader, model_r:networks.VAE, optimizer:torch.optim.Optimizer, training:bool):
	total_loss = []
	z_dim = model_r.latent_dim
	for i, val in enumerate(iterator):
		if training:
			optimizer.zero_grad()
		val = val.to(device)
		xr_gt = val[:, z_dim:]
		z_in = val[:, :z_dim]
		xr_pred = model_r._output(model_r._decoder(z_in))
		loss = ((xr_pred - xr_gt)**2).reshape(-1, model_r.window_size, model_r.num_joints).sum(-1).mean(-1).sum()

		total_loss.append(loss)
		if training:
			loss.backward()
			optimizer.step()
	
	return total_loss, i
	
if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	parser.add_argument('--cov', action='store_true', 
						help='Whether to use covariance for conditioning or not')
	args = parser.parse_args()
	ckpt = torch.load(args.ckpt)
	MODELS_FOLDER = os.path.dirname(args.ckpt)
	hyperparams = np.load(os.path.join(MODELS_FOLDER, 'hyperparams.npz'), allow_pickle=True)
	args_ckpt = hyperparams['args'].item()
	global_config = hyperparams['global_config'].item()
	ae_config = hyperparams['ae_config'].item()
	robot_vae_config = hyperparams['robot_vae_config'].item()
	if args_ckpt.dataset == 'buetepage_pepper':
		dataset = dataloaders.buetepage.PepperWindowDataset
	elif args_ckpt.dataset == 'buetepage':
		dataset = dataloaders.buetepage.HHWindowDataset
	# TODO: Nuitrack

	train_iterator = DataLoader(dataset(args_ckpt.src, train=True, window_length=global_config.window_size, downsample=global_config.downsample), batch_size=1, shuffle=False)
	test_iterator = DataLoader(dataset(args_ckpt.src, train=False, window_length=global_config.window_size, downsample=global_config.downsample), batch_size=1, shuffle=False)

	model_h = getattr(networks, args_ckpt.model)(**(ae_config.__dict__)).to(device)
	model_h.load_state_dict(ckpt['model_h'])
	model_r = getattr(networks, args_ckpt.model)(**(robot_vae_config.__dict__)).to(device)
	model_r.load_state_dict(ckpt['model_r'])

	params = []
	for n,p in model_r.named_parameters():
		if n.split('.')[0] == '_decoder' or n.split('.')[0] == '_output':
			p.requires_grad = True
			params.append(p)
		else:
			p.requires_grad = False
	for p in model_h.parameters():
		p.requires_grad = False
	hsmm = ckpt['hsmm']


	# params = list(model_r._output.parameters()) + list(model_r._decoder.parameters())
	named_params = list(model_r._output.named_parameters()) + list(model_r._decoder.named_parameters())
	optimizer = torch.optim.AdamW(params, lr=0.001)
	
	train_finetune_iterator = DataLoader(finetune_dataset(train_iterator, hsmm, model_h, args.cov), batch_size=50, shuffle=True)
	test_finetune_iterator = DataLoader(finetune_dataset(test_iterator, hsmm, model_h, args.cov), batch_size=1000, shuffle=False)
	
	print("Building Writer")
	SUMMARIES_FOLDER = os.path.join(os.path.dirname(MODELS_FOLDER), "summary_finetuning")
	if not os.path.exists(SUMMARIES_FOLDER):
		print("Creating Model Directory")
		os.makedirs(SUMMARIES_FOLDER)
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
	
	print("Starting Epochs")

	for epoch in range(100):
		model_r._output.train()
		model_r._decoder.train()

		train_recon, iters = run_iteration(train_finetune_iterator, model_r, optimizer, True)
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
			test_recon, iters = run_iteration(test_finetune_iterator, model_r, optimizer, False)
			write_summaries_vae(writer, test_recon, [], steps_done, 'test')

		if epoch % global_config.EPOCHS_TO_SAVE == 0:
			if os.path.exists(os.path.join(MODELS_FOLDER, 'last_ckpt_finetuning.pth')):
				os.rename(os.path.join(MODELS_FOLDER, 'last_ckpt_finetuning.pth'), os.path.join(MODELS_FOLDER, '2ndlast_ckpt_finetuning.pth'))
			checkpoint_file = os.path.join(MODELS_FOLDER, 'last_ckpt_finetuning.pth')
			torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)

		print(f'{epoch:02d}\t{sum(train_recon):.4e}\t{sum(test_recon):.4e}')

	writer.flush()

	checkpoint_file = os.path.join(MODELS_FOLDER, f'final_{epoch+1}_finetuning.pth')
	torch.save({'model_h': model_h.state_dict(), 'model_r': model_r.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'hsmm':hsmm}, checkpoint_file)
