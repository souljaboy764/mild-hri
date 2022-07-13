import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import matplotlib.pyplot as plt
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import dataloaders

import pbdlib as pbd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class Gaussian():
	'''
	class for multivariate gaussian distribution
	'''
	def __init__(self, mu=0, sigma=0):
		self.mu = np.atleast_1d(mu)              #turns a scalar into 1D array otherwise preserves the arrray
		if np.array(sigma).ndim == 0:             #when sigma is scalar
			self.Sigma = np.atleast_2d(sigma**2)  #turns a scalar into 2D array otherwise preserves the arrray
		else:
			self.Sigma = sigma

	def density(self, x):
		n,d = x.shape
		xm = (x-self.mu[None,:])                                                    
		normalization = ((2*np.pi)**(-d/2.)) * np.linalg.det(self.Sigma)**(-1/2.)
		quadratic = np.sum((xm @ np.linalg.inv(self.Sigma)) * xm, axis=1)          #Note the @ sign here denotes matrix multiplication
		return normalization * np.exp(-.5 *  quadratic)


def eigsorted(cov):
	vals, vecs = np.linalg.eigh(cov)
	order = vals.argsort()[::-1]
	return vals[order], vecs[:,order]

def confidence_ellipse(mu, cov, ax, n_std=2.0, facecolor='none', **kwargs):
	"""
	Create a plot of the covariance confidence ellipse of *x* and *y*.

	Parameters
	----------
	x, y : array-like, shape (n, )
		Input data.

	ax : matplotlib.axes.Axes
		The axes object to draw the ellipse into.

	n_std : float
		The number of standard deviations to determine the ellipse's radiuses.

	**kwargs
		Forwarded to `~matplotlib.patches.Ellipse`

	Returns
	-------
	matplotlib.patches.Ellipse
	"""
	pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
	# Using a special case to obtain the eigenvalues of this
	# two-dimensionl dataset.
	ell_radius_x = np.sqrt(1 + pearson)
	ell_radius_y = np.sqrt(1 - pearson)
	lambda_, v = np.linalg.eig(cov)
	lambda_ = np.sqrt(lambda_)
	theta = np.rad2deg(np.arccos(v[0, 0]))

	# ellipse = Ellipse((0, 0), width=lambda_[0] * 6, height=lambda_[1] * 6, angle=theta,
	# 				  facecolor=facecolor, **kwargs)
	ellipse = Ellipse(xy=(mu[0], mu[1]),
				  width=lambda_[0]*n_std, height=lambda_[1]*n_std,
				  angle=np.rad2deg(np.arccos(v[0, 0])), facecolor=facecolor, **kwargs)
	return ax.add_artist(ellipse)

	# Calculating the stdandard deviation of x from
	# the squareroot of the variance and multiplying
	# with the given number of standard deviations.
	scale_x = np.sqrt(cov[0, 0]) * n_std
	mean_x = mu[0]

	# calculating the stdandard deviation of y ...
	scale_y = np.sqrt(cov[1, 1]) * n_std
	mean_y = mu[1]

	# transf = transforms.Affine2D() \
	# 	.rotate_deg(-45) \
	# 	.scale(scale_x, scale_y) \
	# 	.translate(mean_x, mean_y)

	# ellipse.set_transform(transf + ax.transData)
	return ax.add_patch(ellipse)

def plot_gaussian(mu, Sigma, ax, color = 'r', alpha=1.):
	# r1 = mu[0]-2*np.sqrt(Sigma[0,0]), mu[0]+2*np.sqrt(Sigma[0,0])     #get the range of x axis in the grid
	# r2 = mu[1]-2*np.sqrt(Sigma[1,1]), mu[1]+2*np.sqrt(Sigma[1,1])     #get the range of y axis in the grid
	# x1, x2 = np.mgrid[r1[0]:r1[1]:.01, r2[0]:r2[1]:.01]               #get the meshgrid       
	# x = np.vstack((x1.ravel(), x2.ravel())).T         #flatten it
	# p = Gaussian(mu,Sigma).density(x)                 #get the probability density values over the grid 
	#ax.set_aspect(1)
	# plt.contourf(x1, x2, p.reshape(x1.shape),colors=color, alpha=alpha)           #plot the contours
	ax.scatter(mu[0], mu[1], color=color)
	confidence_ellipse(mu, Sigma, ax, facecolor=color,alpha=alpha)


	
if __name__=='__main__':
	parser = argparse.ArgumentParser(description='SKID Training')
	parser.add_argument('--ckpt', type=str, metavar='CKPT', required=True, # logs/fullvae_rarm_window_07081252_klconditionedclass/models/final.pth
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
	# vae_config.window_size=40

	vae_config.latent_dim = 3
	print("Creating Model")
	model = getattr(networks, saved_args.model)(**(vae_config.__dict__)).to(device)
	z_dim = model.latent_dim
	# print('Latent Dim:', z_dim, saved_args.latent_dim)
	
	print("Loading Checkpoints")
	ckpt = torch.load(args.ckpt)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print("Reading Data")
	if model.window_size ==1:
		dataset = getattr(dataloaders, saved_args.dataset).SequenceDataset(saved_args.src, train=True)
	else:
		dataset = getattr(dataloaders, saved_args.dataset).SequenceWindowDataset(saved_args.src, train=True, window_length=model.window_size)
	NUM_ACTIONS = len(dataset.actidx)

	if isinstance(model, networks.VAE):
		hsmm = ckpt['hsmm']
	else:
		RESULTS_FOLDER = os.path.join(MODELS_FOLDER,'results_hsmm_10')
		print('Results Folder:',RESULTS_FOLDER)
		hsmm = np.load(os.path.join(RESULTS_FOLDER, 'hsmm.npz'), allow_pickle=True)['hsmm']
	# os.makedirs(RESULTS_FOLDER, exist_ok=True)
	# else:
	print("Starting")
	actions = ['Hand Waving', 'Handshaking', 'Rocket', 'Parachute']
	actions_filename = ['handwave', 'handshake', 'rocket', 'parachute']
	reconstruction_error, gt_data, gen_data, lens,times = [], [], [], [], []
	idx = [[0,0], [0,1], [1,0], [1,1]]
	traj_idx = [[4, 3, 22, 2, 5, 23, 20, 18, 14, 17, 7, 1, 10, 0, 11],
				[45, 43, 38, 26, 49, 41, 25, 39, 42, 52, 36, 46, 33, 31, 34],
				[94, 61, 96, 85, 77, 109, 73, 105, 68, 107, 63, 102, 101, 108, 59],
				[137, 130, 144, 120, 128, 147, 141, 118, 112, 145, 124, 135, 133, 140, 123]]
	for i in range(NUM_ACTIONS):
		fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='3d'))
		s = dataset.actidx[i]
		z1_encoded = []
		z2_encoded = []
		d1 = 0
		d2 = 1
		# plt.clf()
		# plt.cla()
		# ax = axs[idx[i][0],idx[i][1]]
		dim = np.array([d1,d2])
		sl = np.ix_(dim, dim)
		# for c in range(hsmm[i].mu.shape[0]):
		# 	plot_gaussian(hsmm[i].mu[c][dim], hsmm[i].sigma[c][sl], color='r', alpha=0.3, ax=ax)
		
		# dim=[2*z_dim, 2*z_dim+1]
		dim=[z_dim, z_dim+1]
		sl = np.ix_(dim, dim)
		# for c in range(hsmm[i].mu.shape[0]):
		# 	plot_gaussian(hsmm[i].mu[c][dim], hsmm[i].sigma[c][sl], color='b', alpha=0.3, ax=ax)

		# for j in np.random.choice(np.arange(s[0], s[1]),10):
		for j in traj_idx[i]:
			x, label = dataset[j]
			gt_data.append(x)
			assert np.all(label == i)
			x = torch.Tensor(x).to(device)
			seq_len, dims = x.shape
			x2_gt = x[:, dims//2:]
			seq_len, dims = x.shape
			z1 = model(x[:, :dims//2], encode_only=True).detach().cpu().numpy()
			z2 = model(x[:, dims//2:], encode_only=True).detach().cpu().numpy()
			# dims = np.random.randint(0,z_dim,2)
			# d1 = dims[0]
			# d2 = dims[1] 
			# ax.plot(z1[::2, d1], z1[::2, d2], 'r--', alpha=0.3)
			# ax.plot(z2[::2, d1], z2[::2, d2], 'b--', alpha=0.3)
			ax.scatter(z1[::20,0], z1[::20,1], z1[::20,2], color='r', marker='o', s=50, alpha=0.15)
			ax.scatter(z2[::20,0], z2[::20,1], z2[::20,2], color='b', marker='*', s=50, alpha=0.15)
		pbd.plot_gmm3d(ax, hsmm[i].mu[:, :3], hsmm[i].sigma[:, :3, :3], color='red', alpha=0.3)
		# if model.window_size ==1:
		pbd.plot_gmm3d(ax, hsmm[i].mu[:, z_dim:z_dim+3], hsmm[i].sigma[:, z_dim:z_dim+3, z_dim:z_dim+3], color='blue', alpha=0.3)
		# else:
		# 	pbd.plot_gmm(hsmm[i].mu, hsmm[i].sigma, dim=[z_dim, z_dim+1], color=[0, 0, 1], alpha=0.3)
		# ax.set_title(actions[i],fontweight='bold',fontsize=35)
		# ax.set_xlabel('Latent Dim 0', fontweight='bold',fontsize=20)
		# ax.set_ylabel('Latent Dim 1', fontweight='bold',fontsize=20)
		# ax.set_ylabel('Latent Dim 2', fontweight='bold',fontsize=20)
		plt.tight_layout()
		plt.axis('off')
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	# np.savez_compressed('latent_actions.npz',axs = axs, fig=fig)
		plt.show()
	# plt.savefig('latent_actions.pdf')