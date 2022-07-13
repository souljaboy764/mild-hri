import numpy as np
import os, argparse

import dataloaders

import pbdlib as pbd

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Raw Data HSMM Training')
	parser.add_argument('--src', type=str, default='/home/vignesh/playground/hsmmvae/data/buetepage/traj_data.npz', metavar='RES',
						help='Path to read training and testing data (default: /home/vignesh/playground/hsmmvae/data/buetepage/traj_data.npz).')
	args = parser.parse_args()
	window_size = 1
	nb_states = 8
	hsmm = []

	if window_size == 1:
		train_dataset = dataloaders.buetepage.SequenceDataset(args.src, train=True)
		dataset = dataloaders.buetepage.SequenceDataset(args.src, train=False)
		RESULTS_FOLDER = os.path.join('logs/rawdata_hsmm_/results_hsmm_'+str(nb_states))
		nb_dim = 4*4*3
	else:
		train_dataset = dataloaders.buetepage.SequenceWindowDataset(args.src, train=True, window_length=window_size)
		dataset = dataloaders.buetepage.SequenceWindowDataset(args.src, train=False, window_length=window_size)
		RESULTS_FOLDER = os.path.join('logs/rawdata_window_hsmm_/results_hsmm_'+str(nb_states))
		nb_dim = 2*40*4*3
	NUM_ACTIONS = len(dataset.actidx)
	print(RESULTS_FOLDER)
	os.makedirs(RESULTS_FOLDER, exist_ok=True)
	
	for a in range(len(train_dataset.actidx)):
		hsmm.append(pbd.HSMM(nb_dim=nb_dim, nb_states=nb_states))
		s = train_dataset.actidx[a]
		# for j in range(s[0], s[1]):
		samples = []
		for j in np.random.choice(np.arange(s[0], s[1]), 15):
			x, label = train_dataset[j]
			assert np.all(label == a)
			seq_len, dims = x.shape
			if window_size == 1:
				x_vel = np.diff(x, prepend=x[0:1], axis=0)
				samples.append(np.concatenate([x[:, :dims//2], x_vel[:, :dims//2], x[:, dims//2:], x_vel[:, dims//2:]], axis=-1))
			else:
				samples.append(x)
		hsmm[-1].init_hmm_kbins(samples)
		print('Training',a)
		hsmm[-1].em(samples)
		# print(hsmm[-1].Trans==0)
	print("Starting")
	actions = ['Waving', 'Handshaking', 'Rocket Fistbump', 'Parachute Fistbump']
	reconstruction_error, gt_data, gen_data, lens = [], [], [], []
	for i in range(NUM_ACTIONS):
		s = dataset.actidx[i]
		for j in range(s[0], s[1]):
			x, label = dataset[j]
			gt_data.append(x)
			assert np.all(label == i)
			seq_len, dims = x.shape
			x_in = x[:, :dims//2]
			x2_gt = x[:, dims//2:]
			seq_len, dims = x.shape
			if window_size == 1:
				x_vel = np.diff(x_in, prepend=x_in[0:1], axis=0)
				x2_gen, _ = hsmm[i].condition(np.concatenate([x_in, x_vel], axis=-1), dim_in=slice(0, 24), dim_out=slice(24, 36))
			else:
				x2_gen, _ = hsmm[i].condition(x_in, dim_in=slice(0, dims//2), dim_out=slice(dims//2, dims))
			reconstruction_error.append((x2_gt - x2_gen)**2)
			gen_data.append(x2_gen)
			lens.append(seq_len)
			
		if window_size>1:
			x1_gt = gt_data[-1][:, :dims//2].reshape(-1, 4*window_size, 3)
			x2_gt = gt_data[-1][:, dims//2:].reshape(-1, 4*window_size, 3)
			x2_gen = gen_data[-1].reshape(-1, 4*window_size, 3)
		else:
			x1_gt = gt_data[-1][:, :dims//2].reshape(-1, 4, 3)
			x2_gt = gt_data[-1][:, dims//2:].reshape(-1, 4, 3)
			x2_gen = gen_data[-1].reshape(-1, 4, 3)
		np.savez_compressed(os.path.join(RESULTS_FOLDER, 'predictions_action_'+str(i)), x1_gt=x1_gt, x2_gt=x2_gt, x2_gen=x2_gen)
		np.set_printoptions(precision=5)
	reconstruction_error = np.concatenate(reconstruction_error,axis=0)
	reconstruction_error = reconstruction_error.reshape((-1, window_size, 4, 3)).sum(-1).mean(-1)#.mean(-1)
	print(np.any(np.isnan(reconstruction_error)))
	print(reconstruction_error.shape)
	np.savez_compressed(os.path.join(RESULTS_FOLDER, 'recon_error_hsmm.npz'), error=reconstruction_error)
	np.savez_compressed(os.path.join(RESULTS_FOLDER, 'hsmm.npz'), hsmm=hsmm)
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

	