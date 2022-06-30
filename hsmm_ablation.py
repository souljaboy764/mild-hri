import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import dataloaders
import pbdlib as pbd

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
	model.eval()
	
	print("Reading Data")
	train_dataset = getattr(dataloaders, saved_args.dataset).SequenceDataset(saved_args.src, train=True)
	test_dataset = getattr(dataloaders, saved_args.dataset).SequenceDataset(saved_args.src, train=True)
	NUM_ACTIONS = len(train_dataset.actidx)

	print("Starting")
	reconstruction_error, gt_data, gen_data = [], [], []
	hsmm_list = []
	nb_dim = 2*model.latent_dim
	nb_states = np.arange(8,9)
	train_mse = np.zeros((len(nb_states), NUM_ACTIONS))
	test_mse = np.zeros((len(nb_states), NUM_ACTIONS))
	actions = ['Waving', 'Handshaking', 'Rocket Fistbump', 'Parachute Fistbump']
	with torch.no_grad():
		def hsmm_mse(hsmm, z_in, x2_gt):
			z2_pred, _ = hsmm.condition(z_in, dim_in=slice(0, 2*z_dim), dim_out=slice(2*z_dim, 3*z_dim))
			x1_gen = model._output(model._decoder(torch.Tensor(z_in[:, :z_dim]).to(device)))
			x2_gen = model._output(model._decoder(torch.Tensor(z2_pred).to(device)))
			return F.mse_loss(x2_gt, x2_gen, reduction='mean').detach().cpu().numpy(), x1_gen, x2_gen
		for i in range(NUM_ACTIONS):
			print('Action',i)
			s = train_dataset.actidx[i]
			x1_gt_train, x1_gt_test, x2_gt_train, x2_gt_test, z_encoded_train, z_encoded_test = [], [], [], [], [], []
			for j in range(s[0], s[1]): 
				x, idx, label = train_dataset[j]
				assert np.all(label == i)
				x = torch.Tensor(x).to(device)
				seq_len, dims = x.shape
				x1_gt_train.append(x[:, :dims//2])
				x2_gt_train.append(x[:, dims//2:])
				x = torch.concat([x[None, :, :dims//2], x[None, :, dims//2:]]) # x[0] = Agent 1, x[1] = Agent 2
				zpost_samples = model(x, encode_only=True)

				z1_vel = torch.diff(zpost_samples[0], prepend=zpost_samples[0][0:1], dim=0)
				z2_vel = torch.diff(zpost_samples[1], prepend=zpost_samples[1][0:1], dim=0)
				z_encoded_train.append(torch.concat([zpost_samples[0], z1_vel, zpost_samples[1], z2_vel], dim=-1).cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
			
			s = test_dataset.actidx[i]
			for j in range(s[0], s[1]):
				x, idx, label = test_dataset[j]
				x = torch.Tensor(x).to(device)
				seq_len, dims = x.shape
				x1_gt_test.append(x[:, :dims//2])
				x2_gt_test.append(x[:, dims//2:])
				zpost_samples = model(x[:, :dims//2], encode_only=True)
				z_vel = torch.diff(zpost_samples, prepend=zpost_samples[0:1], dim=0)
				z = torch.concat([zpost_samples,z_vel], dim=-1)
				z_encoded_test.append(z.cpu().numpy()) # (num_trajs, seq_len, 2*z_dim)
			
			for n in range(len(nb_states)):
				nb_state = nb_states[n]
				print('Running',nb_state,'states')
				hsmm = pbd.HSMM(nb_dim=nb_dim, nb_states=nb_state)
				hsmm.init_hmm_kbins(z_encoded_train)
				hsmm.em(z_encoded_train)

				mse = []
				for j in range(len(x2_gt_train)):
					mse_j, x1_gen, x2_gen = hsmm_mse(hsmm, z_encoded_train[j][:, :2*z_dim], x2_gt_train[j])
					mse.append(mse_j)
				train_mse[n, i] = np.sum(mse)

				mse = []
				for j in range(len(x2_gt_test)):
					mse_j, x1_gen,  x2_gen = hsmm_mse(hsmm, z_encoded_test[j], x2_gt_test[j])
					mse.append(mse_j)
				test_mse[n, i] = np.sum(mse)
				print('Finished',nb_state,'states',test_mse[n, i],train_mse[n, i])

			x1_gen = x1_gen.reshape(-1, model.num_joints, 3).cpu().numpy()
			x2_gen = x2_gen.reshape(-1, model.num_joints, 3).cpu().numpy()
			x1_gt = x1_gt_test[j].reshape(-1, model.num_joints, 3).cpu().numpy()
			x2_gt = x2_gt_test[j].reshape(-1, model.num_joints, 3).cpu().numpy()
			np.savez_compressed('predictions_action_'+str(i), x1_gt=x1_gt, x2_gt=x2_gt, x1_gen=x1_gen, x2_gen=x2_gen)
	# 		fig = plt.figure()
	# 		ax = fig.add_subplot(1, 1, 1, projection='3d')
	# 		ax.view_init(20, -45)
	# 		plt.ion()
	# 		for t in range(seq_len):
	# 			plt.cla()
	# 			ax.set_xlim3d([-0.2, 0.8])
	# 			ax.set_ylim3d([-0.3, 0.7])
	# 			ax.set_zlim3d([-0.5, 0.5])
	# 			ax.set_title(actions[i])
	# 			ax.plot(x1_gt[t, :, 2], -x1_gt[t, :, 0], x1_gt[t, :, 1], color='k', marker='o', markerfacecolor='r', label='Agent 1 GT')
	# 			ax.plot(0.65-x2_gt[t, :, 2], 0.3+x2_gt[t, :, 0], x2_gt[t, :, 1], color='k', marker='o', markerfacecolor='g', label='Agent 2 GT')
	# 			ax.plot(0.65-x2_gen[t, :, 2], 0.3+x2_gen[t, :, 0], x2_gen[t, :, 1], color='k', marker='o', markerfacecolor='b', linestyle='--', label='Agent 2 Pred')
	# 			plt.legend()
	# 			plt.draw()
	# 			plt.pause(0.03)
	# 			if not plt.fignum_exists(fig.number):
	# 				break
	# 		plt.ioff()
	# 		# plt.close("all")
	# 		plt.show()

	# fig = plt.figure()
	# ax = fig.add_subplot(1, 2, 1)
	# ax.imshow(train_mse)
	# ax.set_title('Training MSE')
	# ax = fig.add_subplot(1, 2, 2)
	# ax.imshow(test_mse)
	# ax.set_title('Testing MSE')
	# plt.show()
	# plt.savefig('hsmm_ablation.png')
	# print(train_mse)
	# print(test_mse)
	