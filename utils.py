import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import argparse

import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample, affine_grid

from transformations import *

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
	A_ = A.detach().clone()
	for k in range(1,31):
		with torch.no_grad():
			# eigvals, eigvecs = torch.linalg.eigh(A_)
			# eigvals_matrix = torch.diag_embed(torch.nn.ReLU()(eigvals) + eps)
			# A_ = eigvecs @ eigvals_matrix @ eigvecs.transpose(-1,-2)
			eigvals, eigvecs = torch.linalg.eigh(A_)
			A_ = A_ + I * (torch.abs(eigvals[:,0]) * k**2 + eps)[:, None, None]
		try:
			return torch.linalg.cholesky(A_ + A - A.detach()) # keeping the same gradients as A but value of A_
		except:
			continue
	for a in A:
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

def joint_angle_extraction(skeleton): # Based on the Pepper Robot URDF
	
	rightShoulder = skeleton[0]
	rightElbow = skeleton[1]
	rightHand = skeleton[2]
	
	rightYaw = 0
	rightPitch = 0
	rightRoll = 0
	rightElbowAngle = 0
	
	# Recreating arm with upper and under arm
	rightUpperArm = rightElbow - rightShoulder
	rightUnderArm = rightHand - rightElbow

	rightElbowAngle = angle(rightUpperArm, rightUnderArm)

	rightYaw = np.arctan2(rightUpperArm[1],-rightUpperArm[2]) # Comes from robot structure
	# rightYaw -= 0.009
	rightPitch = np.arctan2(max(rightUpperArm[0],0), rightUpperArm[2]) # Comes from robot structure
	rightPitch -= np.pi/2 # for pepper frame
	
	# Recreating under Arm Position with known Angles(without roll)
	rightRotationAroundY = euler_matrix(0, rightPitch, 0,)[:3,:3]
	rightRotationAroundX = euler_matrix(0, 0, rightYaw)[:3,:3]
	rightElbowRotation = euler_matrix(0, 0, rightElbowAngle)[:3,:3]

	rightUnderArmInZeroPos = np.array([np.linalg.norm(rightUnderArm), 0, 0.])
	rightUnderArmWithoutRoll = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightElbowRotation,rightUnderArmInZeroPos)))

	# Calculating the angle betwenn actual under arm position and the one calculated without roll
	rightRoll = angle(rightUnderArmWithoutRoll, rightUnderArm)
	
	# # This is a check which sign the angle has as the calculation only produces positive angles
	# rightRotationAroundArm = euler_matrix(0, 0, -rightRoll)[:3, :3]
	# rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	# r1saver = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	# rightRotationAroundArm = euler_matrix(0, 0, rightRoll)[:3, :3]
	# rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	# r1 = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	# if (r1 > r1saver):
	# 	rightRoll = -rightRoll

	return np.array([rightPitch, rightYaw, rightRoll, rightElbowAngle])

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
	seq_len, J, D  = train_data[0].shape
	theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(J,1,1)
	for i in range(num_trajs):
		seq_len, J, D = train_data[i].shape
		train_data[i] = train_data[i].transpose(1,2,0) # J, D, seq_len
		train_data[i] = torch.Tensor(train_data[i]).to(device).unsqueeze(2) # J, D, 1, seq_len
		train_data[i] = torch.concat([train_data[i], torch.zeros_like(train_data[i])], dim=2) # J, D, 2 seq_len
		
		grid = affine_grid(theta, torch.Size([J, D, 2, downsample_len]), align_corners=True)
		train_data[i] = grid_sample(train_data[i].type(torch.float32), grid, align_corners=True) # J, D, 2 downsample_len
		train_data[i] = train_data[i][:, :, 0].cpu().detach().numpy() # J, D, downsample_len
		train_data[i] = train_data[i].transpose(2,0,1) # downsample_len, J, D
	return np.array(train_data)


def training_argparse():
	parser = argparse.ArgumentParser(description='HSMM VAE Training')
	parser.add_argument('--results', type=str, default='./logs/debug',#+datetime.datetime.now().strftime("%m%d%H%M"),
						help='Path for saving results (default: ./logs/results/MMDDHHmm).', metavar='RES')
	parser.add_argument('--src', type=str, default='./data/buetepage/traj_data.npz', metavar='SRC',
						help='Path to read training and testing data (default: ./data/buetepage/traj_data.npz).')
	parser.add_argument('--hsmm-components', type=int, default=5, metavar='N_COMPONENTS', 
						help='Number of components to use in HSMM Prior (default: 5).')
	parser.add_argument('--dataset', type=str, default='buetepage_pepper', metavar='DATASET', choices=['buetepage', 'buetepage_pepper'],
						help='Dataset to use: buetepage, buetepage_pepper or nuitrack (default: buetepage_pepper).')
	parser.add_argument('--seed', type=int, default=np.random.randint(0,np.iinfo(np.int32).max), metavar='SEED',
						help='Random seed for training (randomized by default).')
	parser.add_argument('--latent-dim', type=int, default=5, metavar='Z',
						help='Latent space dimension (default: 5)')
	parser.add_argument('--cov-reg', type=float, default=1e-3, metavar='EPS',
						help='Positive value to add to covariance diagonal (default: 1e-3)')
	parser.add_argument('--beta', type=float, default=0.005, metavar='BETA',
						help='Scaling factor for KL divergence (default: 0.005)')
	parser.add_argument('--window-size', type=int, default=5, metavar='WINDOW',
						help='Window Size for inputs (default: 5)')
	parser.add_argument('--downsample', type=float, default=0.2, metavar='DOWNSAMPLE',
						help='Factor for downsampling the data (default: 0.2)')
	parser.add_argument('--mce-samples', type=int, default=4, metavar='MCE',
						help='Number of Monte Carlo samples to draw (default: 4)')
	parser.add_argument('--grad-clip', type=float, default=0.5, metavar='CLIP',
						help='Value to clip gradients at (default: 0.5)')
	parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
						help='Number of epochs to train for (default: 100)')
	parser.add_argument('--gamma', type=float, default=1.0, metavar='GAMMA',
						help='Decay Factor to for the relative weight of the conditional loss  (default: 1.0)')
	parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
						help='Starting Learning Rate (default: 5e-4)')
	parser.add_argument('--variant', type=int, default=2, metavar='VARIANT', choices=[1, 2, 3, 4],
						help='Which variant to use 1 - vanilla, 2 - sample conditioning, 3 - conditional sampling (default: 1).')
	parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
						help='Checkpoint to resume training from (default: None)')
	parser.add_argument('--cov-cond', action='store_true', 
						help='Whether to use covariance for conditioning or not')
	return parser.parse_args()