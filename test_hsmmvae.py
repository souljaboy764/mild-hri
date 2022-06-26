import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import matplotlib.pyplot as plt
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import dataloaders
import config

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
	vae_config = getattr(config, saved_args.dataset).ae_config()
	# vae_config = hyperparams['vae_config'].item()

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
			z1 = model(x[:, :dims//2], encode_only=True).detach().cpu().numpy()
			z2, _ = hsmm[i].condition(z1, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim))
			x2_gen = model._output(model._decoder(torch.Tensor(z2).to(device)))
			reconstruction_error.append(F.mse_loss(x2_gt, x2_gen).detach().cpu().numpy())
			gen_data.append(x2_gen.detach().cpu().numpy())

		x_gen = gen_data[-1].reshape(-1, model.num_joints, 3)
		x1_gt = gt_data[-1][:, dims//2:].reshape(-1, model.num_joints, 3)
		x2_gt = gt_data[-1][:, :dims//2].reshape(-1, model.num_joints, 3)
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		ax.view_init(20, -45)
		plt.ion()
		for t in range(seq_len):
			ax.scatter(x_gen[i, :, 0], x_gen[i, :, 1], x_gen[i, :, 2], color='b', marker='o')
			ax.scatter(x1_gt[i, :, 0], x1_gt[i, :, 1], x1_gt[i, :, 2], color='g', marker='o')
			ax.scatter(x2_gt[i, :, 0], x2_gt[i, :, 1], x2_gt[i, :, 2], color='r', marker='o')
			plt.pause(0.03)	
			if plt.fignum_exists(fig.number):
				break
		plt.ioff()
		# plt.close("all")
		plt.show()

	