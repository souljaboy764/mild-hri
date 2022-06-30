from sklearn.cluster import k_means
import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import matplotlib.pyplot as plt
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import dataloaders

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
	vae_config = hyperparams['ae_config'].item()

	print("Creating Model")
	model = getattr(networks, saved_args.model)(**(vae_config.__dict__)).to(device)
	z_dim = model.latent_dim
	
	print("Loading Checkpoints")
	ckpt = torch.load(args.ckpt)
	model.load_state_dict(ckpt['model'])
	hsmm = ckpt['hsmm']
	model.eval()
	
	print("Reading Data")
	dataset = getattr(dataloaders, saved_args.dataset).SequenceDataset(saved_args.src, train=False)
	NUM_ACTIONS = len(dataset.actidx)

	print("Starting")
	actions = ['Waving', 'Handshaking', 'Rocket Fistbump', 'Parachute Fistbump']
	reconstruction_error, gt_data, gen_data = [], [], []
	for i in range(NUM_ACTIONS):
		s = dataset.actidx[i]
		for j in range(s[0], s[1]):
			x, idx, label = dataset[j]
			gt_data.append(x)
			assert np.all(label == i)
			x = torch.Tensor(x).to(device)
			seq_len, dims = x.shape
			x2_gt = x[:, dims//2:]
			seq_len, dims = x.shape
			if model.window_size>1:
				x = x[:, :dims//2]
				bp_idx = np.zeros((seq_len-model.window_size, model.window_size))
				for k in range(model.window_size):
					bp_idx[:,k] = idx[k:seq_len-model.window_size+k]
				z1 = model(x[bp_idx].flatten(1), encode_only=True).detach().cpu().numpy()
				x2_gt = x2_gt[bp_idx].flatten(1)
			else:
				z1 = model(x[:, :dims//2], encode_only=True).detach().cpu().numpy()
			z2, _ = hsmm[i].condition(z1, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			x2_gen = model._output(model._decoder(torch.Tensor(z2).to(device)))
			reconstruction_error.append(F.mse_loss(x2_gt, x2_gen).detach().cpu().numpy())
			gen_data.append(x2_gen.detach().cpu().numpy())

		x1_gt = gt_data[-1][:, :dims//2].reshape(-1, model.num_joints, 3)
		x2_gt = gt_data[-1][:, dims//2:].reshape(-1, model.num_joints, 3)
		if model.window_size>1:
			x2_gen = np.zeros_like(x2_gt)
			x2_gendata = gen_data[-1].reshape(-1, model.window_size, model.num_joints, 3)
			x2_gen[:seq_len-model.window_size] = x2_gendata[:, 0]
			x2_gen[seq_len-model.window_size:] = x2_gendata[-1, -1]
		else:
			x2_gen = gen_data[-1].reshape(-1, model.num_joints, 3)
		np.savez_compressed('predictions_action_'+str(i), x1_gt=x1_gt, x2_gt=x2_gt, x2_gen=x2_gen)
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

	