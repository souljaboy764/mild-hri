import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import matplotlib.pyplot as plt
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import dataloaders
from config.buetepage import robot_vae_config

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--ckpt', type=str, metavar='CKPT', required=True,
						help='Checkpoint to test')
	args = parser.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(128542)
	torch.autograd.set_detect_anomaly(True)

	MODELS_FOLDER = os.path.dirname(args.ckpt)
	hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
	saved_args = hyperparams['args'].item() # overwrite args if loading from checkpoint
	# vae_config = getattr(config, saved_args.dataset).ae_config()
	ae_config = hyperparams['ae_config'].item()
	ae_config.window_size=40

	print("Creating Model")
	model_h = getattr(networks, saved_args.model)(**(ae_config.__dict__)).to(device)
	model_r = getattr(networks, saved_args.model)(**(robot_vae_config().__dict__)).to(device)
	zh_dim = model_h.latent_dim
	zr_dim = model_r.latent_dim
	
	print("Loading Checkpoints")
	ckpt = torch.load(args.ckpt)
	model_h.load_state_dict(ckpt['model_h'])
	model_r.load_state_dict(ckpt['model_r'])
	hsmm = ckpt['hsmm']
	model_r.eval()
	model_h.eval()
	
	print("Reading Data")
	if model_h.window_size ==1:
		dataset = dataloaders.buetepage_hr.SequenceDataset(saved_args.src, train=False)
	else:
		dataset = dataloaders.buetepage_hr.SequenceWindowDataset(saved_args.src, train=False, window_length=model_r.window_size)
	NUM_ACTIONS = len(dataset.actidx)

	print("Starting")
	actions = ['Waving', 'Handshaking', 'Rocket Fistbump', 'Parachute Fistbump']
	reconstruction_error, gt_data, gen_data, lens = [], [], [], []
	for i in range(NUM_ACTIONS):
		s = dataset.actidx[i]
		for j in range(s[0], s[1]):
			x, label = dataset[j]
			gt_data.append(x)
			assert np.all(label == i)
			x = torch.Tensor(x).to(device)
			seq_len, dims = x.shape
			lens.append(seq_len)
			print('x.shape',seq_len, dims)
			x2_gt = x[:, -model_r.input_dim:]
			# if model.window_size>1:
			# 	x = x[:, :dims//2]
			# 	bp_idx = np.zeros((seq_len-model.window_size, model.window_size))
			# 	for k in range(model.window_size):
			# 		bp_idx[:,k] = idx[k:seq_len-model.window_size+k]
			# 	z1 = model(x[bp_idx].flatten(1), encode_only=True).detach().cpu().numpy()
			# 	x2_gt = x2_gt[bp_idx].flatten(1)
			# else:
			z1 = model_h(x[:, :model_h.input_dim], encode_only=True)#.detach().cpu().numpy()
			if model_h.window_size == 1:
				z1_vel = torch.diff(z1, prepend=z1[0:1], dim=0)
				z2, _ = hsmm[i].condition(torch.concat([z1, z1_vel],dim=-1).detach().cpu().numpy(), dim_in=slice(0, 2*zh_dim), dim_out=slice(2*zh_dim, 2*zh_dim+zr_dim))
			else:
				z2, _ = hsmm[i].condition(z1.detach().cpu().numpy(), dim_in=slice(0, zh_dim), dim_out=slice(zh_dim, zh_dim+zr_dim))
			# z2, _ = hsmm[i].condition(z1, dim_in=slice(0, zh_dim), dim_out=slice(zh_dim, 2*zh_dim))
			x2_gen = model_r._output(model_r._decoder(torch.Tensor(z2).to(device)))
			reconstruction_error.append(F.mse_loss(x2_gt, x2_gen,reduction='none').detach().cpu().numpy())
			gen_data.append(x2_gen.detach().cpu().numpy())
			
		if model_h.window_size>1:
			x1_gt = gt_data[-1][:, :model_h.input_dim].reshape(-1, model_h.num_joints*model_h.window_size, model_h.joint_dims)
			x2_gt = gt_data[-1][:, -model_r.input_dim:].reshape(-1, model_r.num_joints*model_r.window_size, model_r.joint_dims)
			x2_gen = gen_data[-1].reshape(-1, model_r.window_size*model_r.num_joints, model_r.joint_dims)
		else:
			x1_gt = gt_data[-1][:, :model_h.input_dim].reshape(-1, model_h.num_joints, model_h.joint_dims)
			x2_gt = gt_data[-1][:, -model_r.input_dim:].reshape(-1, model_r.num_joints, model_r.joint_dims)
			x2_gen = gen_data[-1].reshape(-1, model_r.num_joints, model_r.joint_dims)
		np.savez_compressed('hsmmvae_hri_test.npz', x_gen=np.array(gen_data), test_data=np.array(gt_data), lens=lens)
		# np.savez_compressed('predictions_action_'+str(i), x1_gt=x1_gt, x2_gt=x2_gt, x2_gen=x2_gen)
		# np.set_printoptions(precision=5)
	# if model_h.window_size>1:
	# 	reconstruction_error = np.concatenate(reconstruction_error,axis=0).reshape((-1,model_r.window_size,model_r.num_joints, model_r.joint_dims)).sum(-1).mean(-1).mean(-1)
	# else:
	# 	reconstruction_error = np.concatenate(reconstruction_error,axis=0).reshape((-1,model_r.num_joints, model_r.joint_dims)).sum(-1).mean(-1).mean(-1)
	# np.savez_compressed('recon_error.npz', error=reconstruction_error)
	# print(reconstruction_error.shape)
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

	