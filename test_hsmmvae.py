import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import matplotlib.pyplot as plt


import networks
import dataloaders

import pbdlib as pbd
import time

import config, networks, dataloaders

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--ckpt', type=str, metavar='CKPT', required=True, # logs/fullvae_rarm_window_07081252_klconditionedclass/models/final.pth
						help='Checkpoint to test')
	args = parser.parse_args()
	device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	MODELS_FOLDER = os.path.dirname(args.ckpt)
	# hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	# saved_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	# vae_config = getattr(config, saved_args.dataset).ae_config()
	# vae_config = hyperparams['ae_config'].item()
	# vae_config.window_size=40
	vae_config = config.nuitrack.ae_config()
	vae_config.latent_dim = 5
	# vae_config.window_size = 5


	# vae_config.latent_dim = saved_args.latent_dim
	# vae_config.latent_dim = 3
	print("Creating Model")
	# model = getattr(networks, saved_args.model)(**(vae_config.__dict__)).to(device)
	model = networks.FullCovVAE(**(vae_config.__dict__)).to(device)
	
	z_dim = model.latent_dim
	# print('Latent Dim:', z_dim, saved_args.latent_dim)
	
	print("Loading Checkpoints")
	ckpt = torch.load(args.ckpt,device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print("Reading Data")
	# if model.window_size ==1:
	# 	train_dataset = dataloaders, saved_args.dataset).SequenceDataset(saved_args.src, train=True)
	# 	dataset = getattr(dataloaders, saved_args.dataset).SequenceDataset(saved_args.src, train=False)
	# else:
	# 	train_dataset = getattr(dataloaders, saved_args.dataset).SequenceWindowDataset(saved_args.src, train=True, window_length=model.window_size)
	# 	dataset = getattr(dataloaders, saved_args.dataset).SequenceWindowDataset(saved_args.src, train=False, window_length=model.window_size)
	src = 'data/nuitrack/labelled_sequences.npz'
	if model.window_size ==1:
		train_dataset = dataloaders.nuitrack.SequenceDataset(src, train=True)
		dataset = dataloaders.nuitrack.SequenceDataset(src, train=False)
	else:
		train_dataset = dataloaders.nuitrack.SequenceWindowDataset(src, train=True, window_length=model.window_size)
		dataset = dataloaders.nuitrack.SequenceWindowDataset(src, train=False, window_length=model.window_size)
	NUM_ACTIONS = len(dataset.actidx)
	traj_idx = [[4, 3, 22, 2, 5, 23, 20, 18, 14, 17, 7, 1, 10, 0, 11],
				[45, 43, 38, 26, 49, 41, 25, 39, 42, 52, 36, 46, 33, 31, 34],
				[94, 61, 96, 85, 77, 109, 73, 105, 68, 107, 63, 102, 101, 108, 59],
				[137, 130, 144, 120, 128, 147, 141, 118, 112, 145, 124, 135, 133, 140, 123]]

	if isinstance(model, networks.VAE):
		hsmm = ckpt['hsmm']
		RESULTS_FOLDER = os.path.join(MODELS_FOLDER,'results')
		os.makedirs(RESULTS_FOLDER, exist_ok=True)
	else:
	# for nb_states in [10]:#[4,6,8,10]:
		nb_states = 8
		hsmm = []
		RESULTS_FOLDER = os.path.join(MODELS_FOLDER,'results_hsmm_'+str(nb_states))
		os.makedirs(RESULTS_FOLDER, exist_ok=True)
		if model.window_size == 1:
			nb_dim = 4*model.latent_dim
		else:
			nb_dim = 2*model.latent_dim
		
		for a in range(len(train_dataset.actidx)):
			hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
			s = train_dataset.actidx[a]
			z_encoded = []
			# for j in range(s[0], s[1]):
			# for j in np.random.choice(np.arange(s[0], s[1]), 15, replace=False):
			for j in traj_idx[a]:
				# print('IDX:',j)
				x, label = train_dataset[j]
				assert np.all(label == a)
				x = torch.Tensor(x).to(device)
				seq_len, dims = x.shape
				x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
				# if model.window_size>1:
				# 	bp_idx = np.zeros((seq_len-model.window_size, model.window_size))
				# 	for i in range(model.window_size):
				# 		bp_idx[:,i] = idx[i:seq_len-model.window_size+i]
				# 	x = x[:, bp_idx].flatten(2)
				with torch.no_grad():
					zpost_samples = model(x, encode_only=True)
				if model.window_size == 1:
					z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
					z2_vel = torch.diff(zpost_samples[1], prepend=zpost_samples[1][0:1], dim=0)
					z_encoded.append(torch.concat([zpost_samples[0], z1_vel, zpost_samples[1], z2_vel], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
				else:
					z_encoded.append(torch.concat([zpost_samples[0], zpost_samples[1]], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
			hsmm[a].init_hmm_kbins(z_encoded)
			hsmm[a].em(z_encoded)
	print('Results Folder:',RESULTS_FOLDER)
	print("Starting")
	actions = ['Waving', 'Handshaking', 'Rocket Fistbump', 'Parachute Fistbump']
	reconstruction_error, gt_data, gen_data, lens,times = [], [], [], [], []
	for i in range(NUM_ACTIONS):
		s = dataset.actidx[i]
		for j in range(s[0], s[1]):
			x, label = dataset[j]
			gt_data.append(x)
			assert np.all(label == i)
			x = torch.Tensor(x).to(device)
			seq_len, dims = x.shape
			x2_gt = x[:, dims//2:]
			seq_len, dims = x.shape
			# if model.window_size>1:
			# 	x = x[:, :dims//2]
			# 	bp_idx = np.zeros((seq_len-model.window_size, model.window_size))
			# 	for k in range(model.window_size):
			# 		bp_idx[:,k] = idx[k:seq_len-model.window_size+k]
			# 	z1 = model(x[bp_idx].flatten(1), encode_only=True).detach().cpu().numpy()
			# 	x2_gt = x2_gt[bp_idx].flatten(1)
			# else:
			t0 = time.time()
			z1 = model(x[:, :dims//2], encode_only=True)
			if model.window_size == 1:
				z1_vel = torch.diff(z1, prepend=z1[0:1], dim=0)
				z2, _ = hsmm[i].condition(torch.concat([z1, z1_vel],dim=-1).detach().cpu().numpy(), dim_in=slice(0, 2*z_dim), dim_out=slice(2*z_dim, 3*z_dim))
			else:
				z2, _ = hsmm[i].condition(z1.detach().cpu().numpy(), dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			if np.any(np.isnan(z2)):
				print('z2 nan',actions[i],nb_states,os.path.basename(args.ckpt))
			x2_gen = model._output(model._decoder(torch.Tensor(z2).to(device)))
			if torch.any(torch.isnan(x2_gen)):
				print('x2_gen nan',actions[i],nb_states,os.path.basename(args.ckpt))
			times.append((time.time()-t0)/seq_len)
			reconstruction_error.append(F.mse_loss(x2_gt, x2_gen,reduction='none').detach().cpu().numpy())
			gen_data.append(x2_gen.detach().cpu().numpy())
			lens.append(seq_len)
			
		if model.window_size>1:
			x1_gt = gt_data[-1][:, :dims//2].reshape(-1, model.num_joints*model.window_size, 3)
			x2_gt = gt_data[-1][:, dims//2:].reshape(-1, model.num_joints*model.window_size, 3)
			x2_gen = gen_data[-1].reshape(-1, model.window_size*model.num_joints, 3)
		else:
			x1_gt = gt_data[-1][:, :dims//2].reshape(-1, model.num_joints, 3)
			x2_gt = gt_data[-1][:, dims//2:].reshape(-1, model.num_joints, 3)
			x2_gen = gen_data[-1].reshape(-1, model.num_joints, 3)
		np.savez_compressed(os.path.join(RESULTS_FOLDER, 'predictions_action_'+str(i)), x1_gt=x1_gt, x2_gt=x2_gt, x2_gen=x2_gen)
		np.set_printoptions(precision=5)
	reconstruction_error = np.concatenate(reconstruction_error,axis=0)
	gen_data = np.concatenate(gen_data,axis=0)
	gt_data = np.concatenate(gt_data,axis=0)
	print(gt_data.shape, gen_data.shape)
	reconstruction_error = reconstruction_error.reshape((-1,model.window_size,model.num_joints,3)).sum(-1).mean(-1)#.mean(-1)
	np.savez_compressed(os.path.join(RESULTS_FOLDER, 'recon_error_hsmm.npz'), error=reconstruction_error)
	np.savez_compressed(os.path.join(RESULTS_FOLDER, 'hsmm.npz'), hsmm=hsmm)
	np.savez_compressed(os.path.join(RESULTS_FOLDER, 'predictions.npz'), x_gt=gt_data, x2_gen=gen_data,lens=lens)
	
	print(np.mean(times))
	# fig = plt.figure()
	# ax = fig.add_subplot(1, 1, 1, projection='3d')
	# ax.view_init(20, -45)
	# plt.ion()
	# for t in range(seq_len):
	# 	plt.cla()
	# 	ax.set_xlim3d([-0.2, 0.8])
	# 	ax.set_ylim3d([-0.3, 0.7])
	# 	ax.set_zlim3d([-0.5, 0.5])
	# 	ax.set_title(actions[i])
	# 	ax.plot(x1_gt[t, :, 2], -x1_gt[t, :, 0], x1_gt[t, :, 1], color='k', marker='o', markerfacecolor='r', label='Agent 1 GT')
	# 	ax.plot(0.65-x2_gt[t, :, 2], 0.3+x2_gt[t, :, 0], x2_gt[t, :, 1], color='k', marker='o', markerfacecolor='g', label='Agent 2 GT')
	# 	ax.plot(0.65-x2_gen[t, :, 2], 0.3+x2_gen[t, :, 0], x2_gen[t, :, 1], color='k', marker='o', markerfacecolor='b', linestyle='--', label='Agent 2 Pred')
	# 	plt.legend()
	# 	plt.pause(0.03)
	# 	if not plt.fignum_exists(fig.number):
	# 		break
	# plt.ioff()
	# # plt.close("all")
	# plt.show()

