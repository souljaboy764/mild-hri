import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import grid_sample, affine_grid
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from mild_hri.transformations import _AXES2TUPLE, _TUPLE2AXES, _NEXT_AXIS, euler_matrix
from mild_hri import dataloaders
from mild_hri.vae import VAE

import pbdlib as pbd
import pbdlib_torch as pbd_torch

def write_summaries_vae(writer, recon, kl, steps_done, prefix):
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)

def batchNearestPDCholesky(A:torch.Tensor, eps = torch.finfo(torch.float32).eps):
	"""Find the nearest positive-definite matrix to input taken from [1] 
	which is a A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [2], 
	which credits [3].
	[1] https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
	[2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
	[3] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

	Modified as the input will always be symmetic (covariance matrix), therefore can go ahead with eigh from the beginning and no need to ensure symmetry
	Additionally, to get potentially faster covergence, we use a similar approach as in https://github.com/LLNL/spdlayers of ensuring positive eigenvalues.
	"""

	try:
		return torch.linalg.cholesky(A)
	except:
		pass
	# A_ = A.clone().detach()
	# The above is different from [1]. It appears that MATLAB's `chol` Cholesky
	# decomposition will accept matrixes with exactly 0-eigenvalue, whereas
	# torch will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
	# for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
	# will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
	# the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
	# `spacing` will, for Gaussian random matrixes of small dimension, be on
	# othe order of 1e-16. In practice, both ways converge
	with torch.no_grad():
		I = torch.eye(A.shape[-1]).repeat(A.shape[0],1,1).to(A.device)
	dtype = A.dtype
	A_ = A.detach().clone().to(torch.float32)
	for k in range(1,31):
		# eigvals, eigvecs = torch.linalg.eigh(A_)
		# eigvals_matrix = torch.diag_embed(torch.nn.ReLU()(eigvals) + eps)
		# A_ = eigvecs @ eigvals_matrix @ eigvecs.transpose(-1,-2)
		eigvals, eigvecs = torch.linalg.eigh(A_)
		A_ = A_ + I * (torch.abs(eigvals[:,0]) * k**2 + eps)[:, None, None]
		try:
			return torch.linalg.cholesky(A_).to(dtype)# + A - A.detach()) # keeping the same gradients as A but value of A_
		except:
			continue
	for a in A.to(torch.float32):
		try:
			torch.linalg.cholesky(a)
		except:
			print(a, torch.linalg.cholesky_ex(a).L)
	raise ValueError(f"Unable to convert matrix to Positive Definite after {k} iterations")

# joints = ["none", "head", "neck", "torso", "waist", "left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand", "left_fingertip", "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand", "right_fingertip", "left_hip", "left_knee", "left_ankle", "left_foot", "right_hip", "right_knee", "right_ankle", "right_foot"]
# # joints = ['head', 'neck', 'torso', 'waist', 'left_shoulder', 'left_elbow', 'left_hand', 'right_shoulder', 'right_elbow', 'right_hand']
# # joints = ['neck', 'right_shoulder', 'right_elbow', 'right_hand']
# joints_dic = {joints[i]:i for i in range(len(joints))}


# joints = ["none", "head", "neck", "torso", "waist", "left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand", "left_fingertip", "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand", "right_fingertip", "left_hip", "left_knee", "left_ankle", "left_foot", "right_hip", "right_knee", "right_ankle", "right_foot"]
joints = ["waist", "torso", "neck", "head", "left_shoulder", "right_shoulder", "right_elbow", "right_wrist"]
joints_in = ['neck', 'right_shoulder', 'right_elbow', 'right_wrist']
joints_dic = {joints[i]:i for i in range(len(joints))}



def angle(a,b):
	dot = np.dot(a,b)
	cos = dot/(np.linalg.norm(a)*np.linalg.norm(b))
	if np.allclose(cos, 1):
		cos = 1
	elif np.allclose(cos, -1):
		cos = -1
	return np.arccos(cos)

def projectToPlane(plane, vec):
	return (vec - plane)*np.dot(plane,vec)

def rotation_normalization(skeleton):
	leftShoulder = skeleton[joints_dic["left_shoulder"]]
	rightShoulder = skeleton[joints_dic["right_shoulder"]]
	waist = skeleton[joints_dic["waist"]]
	
	xAxisHelper = waist - rightShoulder
	yAxis = leftShoulder - rightShoulder # right to left
	xAxis = np.cross(xAxisHelper, yAxis) # out of the human(like an arrow in the back)
	zAxis = np.cross(xAxis, yAxis) # like spine, but straight
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	return np.array([[xAxis[0], xAxis[1], xAxis[2]],
					 [yAxis[0], yAxis[1], yAxis[2]],
					 [zAxis[0], zAxis[1], zAxis[2]]])

def joint_angle_extraction(skeleton): # Based on the Pepper Robot URDF, with the limits
	# Recreating arm with upper and under arm
	rightUpperArm = skeleton[1] - skeleton[0]
	rightUnderArm = skeleton[2] - skeleton[1]


	rightElbowAngle = np.clip(angle(rightUpperArm, rightUnderArm), 0.0087, 1.562)
	
	rightYaw = np.clip(np.arcsin(min(rightUpperArm[1],-0.0087)/np.linalg.norm(rightUpperArm)), -1.562, -0.0087)
	
	rightPitch = np.arctan2(max(rightUpperArm[0],0), rightUpperArm[2])
	rightPitch -= np.pi/2 # Needed for pepper frame
	
	# Recreating under Arm Position with known Angles(without roll)
	rightRotationAroundY = euler_matrix(0, rightPitch, 0,)[:3,:3]
	rightRotationAroundX = euler_matrix(0, 0, rightYaw)[:3,:3]
	rightElbowRotation = euler_matrix(0, 0, rightElbowAngle)[:3,:3]

	rightUnderArmInZeroPos = np.array([np.linalg.norm(rightUnderArm), 0, 0.])
	rightUnderArmWithoutRoll = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightElbowRotation,rightUnderArmInZeroPos)))

	# Calculating the angle betwenn actual under arm position and the one calculated without roll
	rightRoll = angle(rightUnderArmWithoutRoll, rightUnderArm)

	return np.array([rightPitch, rightYaw, rightRoll, rightElbowAngle]).astype(np.float32)

def prepare_axis():
	fig = plt.figure()
	ax = fig.add_subplot(1,2,1, projection='3d')
	# plt.ion()
	ax.view_init(25, -155)
	ax.set_xlim3d([-0.05, 0.75])
	ax.set_ylim3d([-0.3, 0.5])
	ax.set_zlim3d([-0.8, 0.2])
	return fig, ax

def reset_axis(ax, variant = None, action = None, frame_idx = None):
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	zlim = ax.get_zlim()
	ax.cla()
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_facecolor('none')
	ax.set_xlim3d(xlim)
	ax.set_ylim3d(ylim)
	ax.set_zlim3d(zlim)
	title = ""
	if variant is not None and action is not None and frame_idx is not None:
		ax.set_title(variant + " " + action + "\nFrame: {}".format(frame_idx))
	return ax

def visualize_skeleton(ax, trajectory, **kwargs):
	# trajectory shape: W, J, D (window size x num joints x joint dims)
	# Assuming that num joints = 4 and dims = 3
	# assert len(trajectory.shape) ==  3 #and trajectory.shape[1] == 4 and trajectory.shape[2] == 3
	for w in range(trajectory.shape[0]):
		ax.plot(trajectory[w, :, 0], trajectory[w, :, 1], trajectory[w, :, 2], color='k', marker='o', **kwargs)
	
	return ax

def downsample_trajs(train_data, downsample_len, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
	# train_data shape: seq_len, J, D : J - num joints, D - dimensions
	num_trajs = len(train_data)
	for i in range(num_trajs):
		seq_len, J, D = train_data[i].shape
		theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(J,1,1)
		train_data[i] = train_data[i].transpose(1,2,0) # J, D, seq_len
		train_data[i] = torch.Tensor(train_data[i]).to(device).unsqueeze(2) # J, D, 1, seq_len
		train_data[i] = torch.concat([train_data[i], torch.zeros_like(train_data[i])], dim=2) # J, D, 2 seq_len
		
		grid = affine_grid(theta, torch.Size([J, D, 2, downsample_len]), align_corners=True)
		train_data[i] = grid_sample(train_data[i].type(torch.float32), grid, align_corners=True) # J, D, 2 downsample_len
		train_data[i] = train_data[i][:, :, 0].cpu().detach().numpy() # J, D, downsample_len
		train_data[i] = train_data[i].transpose(2,0,1) # downsample_len, J, D
	return train_data

def init_ssm_np(nb_dim, nb_states, ssm_type, NUM_ACTIONS):
	ssm = []
	for i in range(NUM_ACTIONS):
		ssm_i = getattr(pbd, ssm_type)(nb_dim=nb_dim, nb_states=nb_states)
		ssm_i.mu = np.zeros((nb_states, nb_dim))
		ssm_i.sigma = np.tile(np.eye(nb_dim)[None],(nb_states,1,1))
		if ssm_type!='GMM':
			ssm_i.init_priors = np.ones((nb_states,))/nb_states
			ssm_i.Trans = np.ones((nb_states, nb_states))/nb_states
		else:
			ssm_i.priors = np.ones((nb_states,))/nb_states
		if ssm_type=='HSMM':
			ssm_i.Mu_Pd = np.zeros(nb_states)
			ssm_i.Sigma_Pd = np.ones(nb_states)
			ssm_i.Trans_Pd = np.ones((nb_states, nb_states))/nb_states
		ssm.append(ssm_i)
	return ssm

def init_ssm_torch(nb_dim, nb_states, ssm_type, NUM_ACTIONS, device):
	ssm = []
	for i in range(NUM_ACTIONS):
		ssm_i = getattr(pbd_torch, ssm_type)(nb_dim=nb_dim, nb_states=nb_states)
		ssm_i.mu = torch.zeros((nb_states, nb_dim), device=device)
		ssm_i.sigma = torch.eye(nb_dim, device=device)[None].repeat(nb_states,1,1)
		if ssm_type!='GMM':
			ssm_i.init_priors = torch.ones((nb_states,), device=device)/nb_states
			ssm_i.Trans = torch.ones((nb_states, nb_states), device=device)/nb_states
		else:
			ssm_i.priors = torch.ones((nb_states,), device=device)/nb_states
		if ssm_type=='HSMM':
			ssm_i.Mu_Pd = torch.zeros(nb_states, device=device)
			ssm_i.Sigma_Pd = torch.ones(nb_states, device=device)
			ssm_i.Trans_Pd = torch.ones((nb_states, nb_states), device=device)/nb_states
		ssm.append(ssm_i)
	return ssm

# Thanks to https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7/45734500#45734500
def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return
	
def window_concat(traj_data, window_length, robot=None):
	window_trajs = []
	for i in range(len(traj_data)):
		trajs_concat = []
		traj_shape = traj_data[i].shape
		dim = traj_shape[-1]
		if robot is None:
			input_dim = dim//2
			# input_dim = int(2*dim/3)
		elif robot=='pepper':
			input_dim = dim-4
		elif robot=='yumi':
			input_dim = dim-7
		idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - 2*window_length)])
		trajs_concat.append(traj_data[i][:,:input_dim][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*input_dim)))
		idx = np.array([np.arange(i,i+window_length) for i in range(window_length, traj_shape[0] + 1 - window_length)])
		trajs_concat.append(traj_data[i][:,input_dim:][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*(dim-input_dim))))
		# elif robot=='pepper':
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - 2*window_length)])
		# 	trajs_concat.append(traj_data[i][:,:dim-4][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*(dim-4))))
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(window_length, traj_shape[0] + 1 - window_length)])
		# 	trajs_concat.append(traj_data[i][:,-4:][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*4)))
		# elif robot=='yumi':
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - 2*window_length)])
		# 	trajs_concat.append(traj_data[i][:,:dim-7][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*(dim-7))))
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(window_length, traj_shape[0] + 1 - window_length)])
		# 	trajs_concat.append(traj_data[i][:,-7:][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*7)))

		trajs_concat = np.concatenate(trajs_concat,axis=-1)
		window_trajs.append(trajs_concat)
	return window_trajs

def evaluate_ckpt_hh(ckpt_path):
	ckpt = torch.load(ckpt_path)
	args_ckpt = ckpt['args']
	if args_ckpt.dataset == 'buetepage':
		dataset = dataloaders.buetepage.HHWindowDataset
	if args_ckpt.dataset == 'nuisi':
		dataset = dataloaders.nuisi.HHWindowDataset
	
	test_iterator = DataLoader(dataset(args_ckpt.src, train=False, window_length=args_ckpt.window_size, downsample=args_ckpt.downsample), batch_size=1, shuffle=False)

	model = VAE(**(args_ckpt.__dict__)).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	ssm = ckpt['ssm']

	return evaluate_ckpt(model, model, ssm, args_ckpt.cov_cond, test_iterator, None)

def evaluate_ckpt_hr(ckpt_path):
	ckpt = torch.load(ckpt_path)
	args_h = ckpt['args_h']
	args_r = ckpt['args_r']
	if args_r.dataset == 'buetepage_pepper':
		dataset = dataloaders.buetepage.PepperWindowDataset
	if args_r.dataset == 'buetepage_yumi':
		dataset = dataloaders.buetepage_hr.YumiWindowDataset
	# TODO: BP_Yumi, Nuisi_Pepper
	
	test_iterator = DataLoader(dataset('../'+args_r.src, train=False, window_length=args_r.window_size, downsample=args_r.downsample), batch_size=1, shuffle=False)

	model_h = VAE(**(args_h.__dict__)).to(device)
	model_h.load_state_dict(ckpt['model_h'])
	model_r = VAE(**{**(args_h.__dict__), **(args_r.__dict__)})
	# model_r._output = nn.Sequential(model_r._output, nn.Sigmoid())
	model_r.to(device)

	model_r.load_state_dict(ckpt['model_r'])
	
	model_h.eval()
	model_r.eval()
	ssm = ckpt['ssm']

	return evaluate_ckpt(model_h, model_r, ssm, args_r.cov_cond, test_iterator, args_r)

def evaluate_ckpt(model_h, model_r, ssm, use_cov, test_iterator, args_r):
	pred_mse_total = []
	pred_mse_action = []
	pred_mse_nowave = []
	# pred_mse_wave = []
	# pred_mse_shake = []
	# pred_mse_rocket = []
	# pred_mse_parachute = []

	x_in = []
	x_vae = []
	x_cond = []
	with torch.no_grad():
		# for i, x in enumerate(test_iterator):
		for idx in test_iterator.dataset.actidx:
			pred_mse_action.append([])
			for i in range(idx[0],idx[1]):
				x, label = test_iterator.dataset[i]
				# x = x[0]
				# label = label[0]
				# x_in.append(x.cpu().numpy())
				x = torch.Tensor(x).to(device)
				x_h = x[:, :model_h.input_dim]
				x_r = x[:, model_h.input_dim:]
				z_dim = model_h.latent_dim
				
				zh_post = model_h(x_h, dist_only=True)
				# xr_gen, _, _ = model_r(x_r)
				# x_vae.append(xr_gen.cpu().numpy())
				if use_cov:
					data_Sigma_in = zh_post.covariance_matrix
				else: 
					data_Sigma_in = None
				zr_cond = ssm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), 
												data_Sigma_in=data_Sigma_in,
												return_cov=False) 
				xr_cond = model_r._output(model_r._decoder(zr_cond)) # * args_r.joints_range + args_r.joints_min
				x_cond.append(xr_cond.cpu().numpy())
				
				mse_i = ((xr_cond - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints, model_r.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()
				pred_mse_total += mse_i
				pred_mse_action[-1] += mse_i

				# Buetepage HH & Pepper
				if i>7:
					pred_mse_nowave += mse_i

				# # Buetepage Yumi
				# if i>2:
				# 	pred_mse_nowave += mse_i
				
				# # NuiSI v2
				# if i>3:
				# 	pred_mse_nowave += mse_i
				
				
				# vae_mse += ((xr_gen - x_r)**2).reshape((x_r.shape[0], model_r.window_size, model_r.num_joints, model_r.joint_dims)).sum(-1).mean(-1).mean(-1).detach().cpu().numpy().tolist()
	# x_in = np.array(x_in,dtype=object)
	# x_cond = np.array(x_cond,dtype=object)
	# x_vae = np.array(x_vae,dtype=object)

	# np.savez_compressed('x_test_cond_norm.npz', x_in=x_in, x_cond=x_cond, x_vae=x_vae)
	
	return pred_mse_total, pred_mse_action, pred_mse_nowave#, pred_mse_wave, pred_mse_shake, pred_mse_rocket, pred_mse_parachute


def training_hh_argparse(args=None):
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	# Results and Paths
	parser.add_argument('--results', type=str, default='./logs/debug',#+datetime.datetime.now().strftime("%m%d%H%M"),
						help='Path for saving results (default: ./logs/results/MMDDHHmm).', metavar='RES')
	parser.add_argument('--src', type=str, default='./data/buetepage/traj_data.npz', metavar='SRC',
						help='Path to read training and testing data (default: ./data/buetepage/traj_data.npz).')
	parser.add_argument('--dataset', type=str, default='buetepage', metavar='DATASET', choices=['buetepage', "nuisi"],
						help='Dataset to use: buetepage or nuisi (default: buetepage).')
	parser.add_argument('--seed', type=int, default=np.random.randint(0,np.iinfo(np.int32).max), metavar='SEED',
						help='Random seed for training (randomized by default).')
	
	# Input data shapers
	parser.add_argument('--downsample', type=float, default=0.2, metavar='DOWNSAMPLE',
						help='Factor for downsampling the data (default: 0.2)')
	parser.add_argument('--window-size', type=int, default=5, metavar='WINDOW',
						help='Window Size for inputs (default: 5)')
	parser.add_argument('--num-joints', default=3, type=int,
		     			help='Number of joints in the input data')
	parser.add_argument('--joint-dims', default=3, type=int,
		     			help='Number of Dimensions of each joint in the input data')
	
	# SSM args
	parser.add_argument('--ssm', type=str, default='HMM', metavar='SSM', choices=['HMM', 'HSMM'],
						help='Which State Space Model to use: HMM or HSMM (default: HMM).')
	parser.add_argument('--ssm-components', type=int, default=5, metavar='N_COMPONENTS',
						help='Number of components to use in SSM Prior (default: 5).')
	parser.add_argument('--cov-reg', type=float, default=1e-2, metavar='EPS',
						help='Positive value to add to covariance diagonal (default: 1e-3)')
	
	# VAE args
	parser.add_argument('--model', type=str, default='VAE', metavar='MODEL', choices=['VAE', 'FullCovVAE'],
						help='Which VAE to use: VAE or FullCovVAE (default: VAE).')
	parser.add_argument('--latent-dim', type=int, default=5, metavar='Z',
						help='Latent space dimension (default: 5)')
	parser.add_argument('--variant', type=int, default=2, metavar='VARIANT', choices=[1, 2, 3, 4],
						help='Which variant to use 1 - vanilla, 2 - sample conditioning, 3 - conditional sampling (default: 1).')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	parser.add_argument('--cov-cond', action='store_true',
						help='Whether to use covariance for conditioning or not')
	parser.add_argument('--hidden-sizes', default=[250,150], nargs='+', type=int,
		     			help='List of weights for the VAE layers (default: [250,150] )')
	parser.add_argument('--activation', default='LeakyReLU', type=str,
		     			help='Activation Function for the VAE layers')
	
	# Hyperparameters
	parser.add_argument('--mce-samples', type=int, default=10, metavar='MCE',
						help='Number of Monte Carlo samples to draw (default: 10)')
	parser.add_argument('--grad-clip', type=float, default=0.5, metavar='CLIP',
						help='Value to clip gradients at (default: 0.5)')
	parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
						help='Number of epochs to train for (default: 100)')
	parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
						help='Starting Learning Rate (default: 5e-4)')
	parser.add_argument('--beta', type=float, default=0.005, metavar='BETA',
						help='Scaling factor for KL divergence (default: 0.005)')

	return parser.parse_args(args)

def training_hr_argparse(args=None):
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	# Results and Paths
	parser.add_argument('--results', type=str, default='./logs/debug',#+datetime.datetime.now().strftime("%m%d%H%M"),
						help='Path for saving results (default: ./logs/results/MMDDHHmm).', metavar='RES')
	parser.add_argument('--src', type=str, default='./data/buetepage/traj_data.npz', metavar='SRC',
						help='Path to read training and testing data (default: ./data/buetepage/traj_data.npz).')
	parser.add_argument('--dataset', type=str, default='buetepage_pepper', metavar='DATASET', choices=['buetepage_yumi', 'buetepage_pepper', "nuisi_pepper"],
						help='Dataset to use: buetepage_yumi, buetepage_pepper or nuisi_pepper (default: buetepage_pepper).')
	parser.add_argument('--seed', type=int, default=np.random.randint(0,np.iinfo(np.int32).max), metavar='SEED',
						help='Random seed for training (randomized by default).')
	
	# Input data shapers
	parser.add_argument('--downsample', type=float, default=0.2, metavar='DOWNSAMPLE',
						help='Factor for downsampling the data (default: 0.2)')
	parser.add_argument('--window-size', type=int, default=5, metavar='WINDOW',
						help='Window Size for inputs (default: 5)')
	parser.add_argument('--num-joints', default=4, type=int,
		     			help='Number of joints in the input data')

	# VAE args
	parser.add_argument('--variant', type=int, default=2, metavar='VARIANT', choices=[1, 2, 3, 4],
						help='Which variant to use 1 - vanilla, 2 - sample conditioning, 3 - conditional sampling (default: 1).')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	parser.add_argument('--ckpt-h', type=str, default=None, metavar='CKPT', required=True,
						help='Checkpoint to the human VAE (default: None)')
	parser.add_argument('--cov-cond', action='store_true',
						help='Whether to use covariance for conditioning or not')

	# Hyperparameters
	parser.add_argument('--mce-samples', type=int, default=10, metavar='MCE',
						help='Number of Monte Carlo samples to draw (default: 10)')
	parser.add_argument('--grad-clip', type=float, default=0.5, metavar='CLIP',
						help='Value to clip gradients at (default: 0.5)')
	parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
						help='Number of epochs to train for (default: 100)')
	parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
						help='Starting Learning Rate (default: 5e-4)')
	parser.add_argument('--beta', type=float, default=0.005, metavar='BETA',
						help='Scaling factor for KL divergence (default: 0.005)')

	return parser.parse_args(args)

