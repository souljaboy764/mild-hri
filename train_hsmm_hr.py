import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, datetime, argparse
from config.buetepage import robot_vae_config

import networks
import config
from utils import *
import dataloaders

import pbdlib as pbd

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/vae_hr_07080033/', metavar='RES',
						help='Path for saving results (default: ./logs/results/MMDDHHmm).')
	parser.add_argument('--src', type=str, default='/home/vignesh/playground/hsmmvae/data/buetepage_hr/labelled_sequences.npz', metavar='RES',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/data/buetepage_hr/labelled_sequences.npz).')
	parser.add_argument('--human-ckpt', type=str, default='/home/vignesh/playground/hsmmvae/logs/ablation/ae_rarm_window_07101153_z3/models/final.pth', metavar='CKPT',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/logs/ablation/ae_rarm_window_07101153_z3/models/final.pth).')
	parser.add_argument('--robot-ckpt', type=str, default='/home/vignesh/playground/hsmmvae/logs/ae_robot_window_07140932_humanckpt/models/final_old.pth', metavar='CKPT',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/logs/ae_robot_window_07140932_humanckpt/models/final_old.pth).')
	parser.add_argument('--hsmm-components', type=int, default=10, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 10).')
	args = parser.parse_args()
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	HUMAN_MODELS_FOLDER = os.path.dirname(args.human_ckpt)
	hyperparams_h = np.load(os.path.join(HUMAN_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	saved_args_h = hyperparams_h['args'].item() # overwrite args if loading from checkpoint
	vae_config_h = hyperparams_h['ae_config'].item()
	vae_config_h.latent_dim = 5
	
	ROBOT_MODELS_FOLDER = os.path.dirname(args.robot_ckpt)
	hyperparams_r = np.load(os.path.join(ROBOT_MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	saved_args_r = hyperparams_r['args'].item() # overwrite args if loading from checkpoint
	vae_config_r = hyperparams_r['ae_config'].item()

	model_h = getattr(networks, saved_args_h.model)(**(vae_config_h.__dict__)).to(device)
	ckpt = torch.load(args.human_ckpt,device)
	model_h.load_state_dict(ckpt['model'])
	model_h.eval()

	# model_r = getattr(networks, saved_args_r.model)(**(vae_config_r.__dict__)).to(device)
	ckpt = torch.load(saved_args_r.human_ckpt,device)
	if 'post_mean.weight' in ckpt['model']:
		vae_config_r.latent_dim = ckpt['model']['post_mean.bias'].shape[0]
	else:
		vae_config_r.latent_dim = ckpt['model']['latent.bias'].shape[0]

	if ckpt['model']['_output.bias'].shape[0] == 12:
		vae_config_r.window_size = 1
	elif ckpt['model']['_output.bias'].shape[0] == 480:
		vae_config_r.window_size = 40
	vae_config_r.num_joints = 7
	vae_config_r.joint_dims = 1
	
	model_r = getattr(networks, saved_args_h.model)(**(vae_config_r.__dict__)).to(device)
	ckpt = torch.load(args.robot_ckpt,device)
	model_r.load_state_dict(ckpt['model_r'])
	model_r.eval()
	
	print("Reading Data")
	dataset = dataloaders.buetepage_hr
	if model_h.window_size ==1:
		nb_dim = 2*model_h.latent_dim + 2*model_r.latent_dim
		train_iterator = DataLoader(dataset.SequenceDataset(args.src, train=True), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceDataset(args.src, train=False), batch_size=1, shuffle=True)
	else:
		nb_dim = model_h.latent_dim + model_r.latent_dim
		train_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=True, window_length=model_h.window_size), batch_size=1, shuffle=True)
		test_iterator = DataLoader(dataset.SequenceWindowDataset(args.src, train=False, window_length=model_h.window_size), batch_size=1, shuffle=True)
	NUM_ACTIONS = len(test_iterator.dataset.actidx)
	actions = ['Waving', 'Handshaking', 'Rocket Fistbump', 'Parachute Fistbump']
	
	# nb_states = args.hsmm_components
	# hsmm = []
	# for i in range(NUM_ACTIONS):
	# 	hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
	# 	print('Training HSMM',i)
	# 	z_encoded = []
	# 	s = train_iterator.dataset.actidx[i]
	# 	for j in range(s[0], s[1]):
	# 		x, label = train_iterator.dataset[j]
	# 		assert np.all(label == i)
	# 		x = torch.Tensor(x).to(device)
	# 		seq_len, dims = x.shape
	# 		# xh = x[:, :12]
	# 		# xr = x[:, 12:] # x[0] = Agent 1, x[1] = Agent 2
	# 		xh = x[:, :model_h.input_dim]
	# 		xr = x[:, -model_r.input_dim:]
			
	# 		with torch.no_grad():
	# 			zh = model_h(xh, encode_only=True)
	# 			zr = model_r(xr, encode_only=True)

	# 		if model_h.window_size == 1:
	# 			zh_vel = torch.diff(zh, prepend=zh[0:1], dim=0)
	# 			zr_vel = torch.diff(zr, prepend=zr[0:1], dim=0)
	# 			z_encoded.append(torch.concat([zh, zh_vel, zr, zr_vel], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
	# 		else:
	# 			z_encoded.append(torch.concat([zh, zr], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
	# 	hsmm[i].init_hmm_kbins(z_encoded)
	# 	hsmm[i].em(z_encoded)
	# 	print('')

	# np.savez_compressed('hr_hsmm_vae/hsmm.npz',hsmm=hsmm)
	# np.savez_compressed('hr_hsmm_vae/args.npz',args=args)
	hsmm = np.load('hr_hsmm/hsmm.npz',allow_pickle=True)['hsmm']

	reconstruction_error, gt_data, gen_data, vae_gen_data, lens = [], [], [], [], []
	zh_dim = model_h.latent_dim
	zr_dim = model_r.latent_dim
	model_h.eval()
	model_r.eval()
	for i in range(NUM_ACTIONS):
		s = test_iterator.dataset.actidx[i]
		for j in range(s[0], s[1]):
			x, label = test_iterator.dataset[j]
			gt_data.append(x)
			print(label,i)
			assert np.all(label == i)
			x = torch.Tensor(x).to(device)
			seq_len, dims = x.shape
			xh = x[:, :model_h.input_dim]
			xr = x[:, -model_r.input_dim:]

			with torch.no_grad():
				xh_vae, zh, _ = model_h(xh)
				xr_vae, zr, _  = model_r(xr)
				vae_gen_data.append(torch.concat([xh_vae,xr_vae],dim=-1).detach().cpu().numpy())

			if model_h.window_size == 1:
				zh_vel = torch.diff(zh, prepend=zh[0:1], dim=0)
				zr_hsmm, _ = hsmm[i].condition(torch.concat([zh, zh_vel],dim=-1).detach().cpu().numpy(), dim_in=slice(0, 2*zh_dim), dim_out=slice(2*zh_dim, 2*zh_dim+zr_dim))
			else:
				zr_hsmm, _ = hsmm[i].condition(zh.detach().cpu().numpy(), dim_in=slice(0, zh_dim), dim_out=slice(zh_dim, zh_dim+zr_dim))
			if np.any(np.isnan(zr_hsmm)):
				print('zr_hsmm nan',actions[i],i,j)
			with torch.no_grad():
				xr_hsmm = model_r._output(model_r._decoder(torch.Tensor(zr_hsmm).to(device)))
			if torch.any(torch.isnan(xr_hsmm)):
				print('xr_hsmm nan',actions[i],i,j)
			
			reconstruction_error.append(F.mse_loss(xr, xr_hsmm,reduction='none').detach().cpu().numpy())
			gen_data.append(xr_hsmm.detach().cpu().numpy())
			lens.append(seq_len)
		
		# np.savez_compressed(os.path.join('hr_hsmm/predictions_action_'+str(i)), 
		# 									x1_gt=xh.detach().cpu().numpy(), 
		# 									x2_gt=xr.detach().cpu().numpy(), 
		# 									x2_gen=xr_hsmm.detach().cpu().numpy(), 
		# 									x2_vae=xr_vae.detach().cpu().numpy())
		np.set_printoptions(precision=5)
	reconstruction_error = np.concatenate(reconstruction_error,axis=0)
	# reconstruction_error = reconstruction_error.reshape((-1,model_r.window_size,model_r.num_joints,3)).sum(-1).mean(-1)#.mean(-1)
	# np.savez_compressed(os.path.join('hr_hsmm_vae/recon_error.npz'), error=reconstruction_error)
	gt_data = np.concatenate(gt_data,axis=0)
	gen_data = np.concatenate(gen_data,axis=0)
	vae_gen_data = np.concatenate(vae_gen_data,axis=0)
	print(reconstruction_error.shape, gt_data.shape, test_iterator.dataset.actidx,lens,np.cumsum(lens))
	# np.savez_compressed(os.path.join('hr_hsmm_vae/predictions.npz'), gt=gt_data, vae_gen=vae_gen_data, xr_hsmm=gen_data,lens=lens)
			