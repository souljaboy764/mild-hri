import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# from tf.transformations import *

import networks
colors_10 = get_cmap('tab10')

def MMD(x, y, reduction='mean'):
	"""Emprical maximum mean discrepancy with rbf kernel. The lower the result
	   the more evidence that distributions are the same.
	   https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html

	Args:
		x: first sample, distribution P
		y: second sample, distribution Q
		kernel: kernel type such as "multiscale" or "rbf"
	"""
	xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
	rx = (xx.diag().unsqueeze(0).expand_as(xx))
	ry = (yy.diag().unsqueeze(0).expand_as(yy))

	dxx = rx.t() + rx - 2. * xx # Used for A in (1)
	dyy = ry.t() + ry - 2. * yy # Used for B in (1)
	dxy = rx.t() + ry - 2. * zz # Used for C in (1)

	XX, YY, XY = (torch.zeros_like(xx),
				  torch.zeros_like(xx),
				  torch.zeros_like(xx))

	bandwidth_range = [10, 15, 20, 50]
	for a in bandwidth_range:
		XX += torch.exp(-0.5*dxx/a)
		YY += torch.exp(-0.5*dyy/a)
		XY += torch.exp(-0.5*dxy/a)

	if reduction=='none':
		return XX + YY - 2. * XY
	
	return getattr(torch, reduction)(XX + YY - 2. * XY)


def multi_variate_normal(x, mu, sigma=None, log=True, gmm=False, lmbda=None):
	"""
	Multivariatve normal distribution PDF

	:param x:		np.array([nb_samples, nb_dim])
	:param mu: 		np.array([nb_dim])
	:param sigma: 	np.array([nb_dim, nb_dim])
	:param log: 	bool
	:return:
	"""
	if not gmm:
		if type(sigma) is float:
			sigma = np.array(sigma, ndmin=2)
		if type(mu) is float:
			mu = np.array(mu, ndmin=1)
		if sigma is not None:
			sigma = sigma[None, None] if sigma.shape == () else sigma

		mu = mu[None] if mu.shape == () else mu
		x = x[:, None] if x.ndim == 1 else x

		dx = mu - x
		lmbda_ = np.linalg.inv(sigma) if lmbda is None else lmbda

		log_lik = -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', lmbda_, dx))

		if sigma is not None:
			log_lik -= 0.5 * (x.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1])
		else:
			log_lik -= 0.5 * (x.shape[1] * np.log(2 * np.pi) - np.linalg.slogdet(lmbda)[1])


		return log_lik if log else np.exp(log_lik)
	else:
		raise NotImplementedError


def forward_variable(hsmm, n_step):
	nbD = np.round(4 * n_step // hsmm.nb_states)
	hsmm.Pd = np.zeros((hsmm.nb_states, nbD))
	for i in range(hsmm.nb_states):
		hsmm.Pd[i, :] = multi_variate_normal(np.arange(nbD), hsmm.Mu_Pd[i], hsmm.Sigma_Pd[i]+1e-8, log=False)
		hsmm.Pd[i, :] = hsmm.Pd[i, :] / (np.sum(hsmm.Pd[i, :])+1e-8)

	h = np.zeros((hsmm.nb_states, n_step))

	ALPHA, S, h[:, 0] = hsmm._fwd_init_ts(nbD)

	for i in range(1, n_step):
		ALPHA, S, h[:, i] = hsmm._fwd_step_ts(ALPHA, S, nbD)

	h /= (np.sum(h, axis=0)+1e-8)
	return h

# def forward_variable(hsmm, n_step=None, demo=None, marginal=None, dep=None, p_obs=None):
# 		"""
# 		Compute the forward variable with some observations

# 		:param demo: 	[np.array([nb_timestep, nb_dim])]
# 		:param dep: 	[A x [B x [int]]] A list of list of dimensions
# 			Each list of dimensions indicates a dependence of variables in the covariance matrix
# 			E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
# 			E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
# 			covariance with dim [2]
# 		:param table: 	np.array([nb_states, nb_demos]) - composed of 0 and 1
# 			A mask that avoid some demos to be assigned to some states
# 		:param marginal: [slice(dim_start, dim_end)] or []
# 			If not None, compute messages with marginals probabilities
# 			If [] compute messages without observations, use size
# 			(can be used for time-series regression)
# 		:param p_obs: 		np.array([nb_states, nb_timesteps])
# 				custom observation probabilities
# 		:return:
# 		"""
# 		if isinstance(demo, np.ndarray):
# 			n_step = demo.shape[0]
# 		elif isinstance(demo, dict):
# 			n_step = demo['x'].shape[0]

# 		nbD = np.round(4 * n_step // hsmm.nb_states)
# 		if nbD == 0:
# 			nbD = 10
# 		hsmm.Pd = np.zeros((hsmm.nb_states, nbD))
# 		# Precomputation of duration probabilities
# 		for i in range(hsmm.nb_states):
# 			hsmm.Pd[i, :] = multi_variate_normal(np.arange(nbD), hsmm.Mu_Pd[i], hsmm.Sigma_Pd[i], log=False)
# 			hsmm.Pd[i, :] = hsmm.Pd[i, :] / (np.sum(hsmm.Pd[i, :])+np.finfo(np.float64).tiny)
# 		# compute observation marginal probabilities
# 		p_obs, _ = hsmm.obs_likelihood(demo, dep, marginal, n_step)

# 		hsmm._B = p_obs

# 		h = np.zeros((hsmm.nb_states, n_step))
# 		bmx, ALPHA, S, h[:, 0] = hsmm._fwd_init(nbD, p_obs[:, 0])

# 		for i in range(1, n_step):
# 			bmx, ALPHA, S, h[:, i] = hsmm._fwd_step(bmx, ALPHA, S, nbD, p_obs[:, i])

# 		h /= np.sum(h, axis=0)

# 		return h


def forward_variable_single(hsmm, t_step, n_step):
	nbD = np.round(4 * n_step // hsmm.nb_states)
	hsmm.Pd = np.zeros((hsmm.nb_states, nbD))
	for i in range(hsmm.nb_states):
		hsmm.Pd[i, :] = multi_variate_normal(np.arange(nbD), hsmm.Mu_Pd[i], hsmm.Sigma_Pd[i]+1e-8, log=False)
		hsmm.Pd[i, :] = hsmm.Pd[i, :] / (np.sum(hsmm.Pd[i, :])+1e-8)

	h = np.zeros(hsmm.nb_states)

	ALPHA, S, h[:, 0] = hsmm._fwd_init_ts(nbD)

	for i in range(1, n_step):
		ALPHA, S, h[:, i] = hsmm._fwd_step_ts(ALPHA, S, nbD)

	h /= (np.sum(h, axis=0)+1e-8)
	return h

def write_summaries_vae(writer, recon, kl, loss, x_gen, zx_samples, x, steps_done, prefix, model):
	writer.add_histogram(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)
	
	# writer.add_embedding(zx_samples[:100],global_step=steps_done, tag=prefix+'/q(z|x)')
	if model.training and isinstance(model, networks.VAE):
		x_gen = x_gen[-1]
	_, seq_len, dims = x_gen.shape
	x_gen = x_gen.detach().cpu().numpy()
	x = x.detach().cpu().numpy()
	
	if model.window_size>1:
		x = x.reshape(2,-1,model.window_size,model.num_joints,3)
		x_gen = x_gen.reshape(2,-1,model.window_size,model.num_joints,3)
		fig, ax = plt.subplots(nrows=5, ncols=model.num_joints, figsize=(28, 16), sharex=True)
		fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		plt.subplots_adjust(
			left=0.05,  # the left side of the subplots of the figure
			right=0.95,  # the right side of the subplots of the figure
			bottom=0.05,  # the bottom of the subplots of the figure
			top=0.95,  # the top of the subplots of the figure
			wspace=0.05,  # the amount of width reserved for blank space between subplots
			hspace=0.05,  # the amount of height reserved for white space between subplots
		)
		for i in range(5):
			idx = np.random.randint(0, seq_len)
			for j in range(model.num_joints):
				ax[i][j].set(xlim=(0, model.window_size - 1))
				color_counter = 0
				for dim in range(model.joint_dims):
					ax[i][j].plot(x[0, idx, :, j, dim], color=colors_10(color_counter%10))
					ax[i][j].plot(x_gen[0, idx, :, j, dim], linestyle='--', color=colors_10(color_counter % 10))
					ax[i][j].plot(x[1, idx, :, j, dim], color=colors_10(color_counter%10))
					ax[i][j].plot(x_gen[1, idx, :, j, dim], linestyle='-.', color=colors_10(color_counter % 10))
					color_counter += 1
	else:
		fig, ax = plt.subplots(nrows=model.num_joints, ncols=2, figsize=(28, 16), sharex=True)
		fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		plt.subplots_adjust(
			left=0.05,  # the left side of the subplots of the figure
			right=0.95,  # the right side of the subplots of the figure
			bottom=0.05,  # the bottom of the subplots of the figure
			top=0.95,  # the top of the subplots of the figure
			wspace=0.05,  # the amount of width reserved for blank space between subplots
			hspace=0.05,  # the amount of height reserved for white space between subplots
		)
		for i in range(model.num_joints):
			for j in range(2):
				ax[i][j].set(xlim=(0, seq_len - 1))
				color_counter = 0
				for dim in range(3):
					ax[i][j].plot(x[j, :, i*3+dim], color=colors_10(color_counter%10))
					ax[i][j].plot(x_gen[j, :, i*3+dim], linestyle='--', color=colors_10(color_counter % 10))
					color_counter += 1

	fig.canvas.draw()
	writer.add_figure(prefix+'/sample reconstruction', fig, steps_done)
	plt.close(fig)

def write_summaries_hr(writer, recon, kl, loss, x_gen, zr_samples, x, steps_done, prefix, model):
	writer.add_scalar(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)
	
	# # # writer.add_embedding(zx_samples[:100],global_step=steps_done, tag=prefix+'/q(z|x)')
	# x_gen = x_gen.detach().cpu().numpy()
	# x = x.detach().cpu().numpy()
	# if prefix=='train' and len(x_gen.shape)==3:
	# 	x_gen = x_gen[-1]
	
	# seq_len, dims = x.shape
	
	# fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(28, 16), sharex=True)
	# fig.tight_layout(pad=0, h_pad=0, w_pad=0)

	# plt.subplots_adjust(
	# 	left=0.05,  # the left side of the subplots of the figure
	# 	right=0.95,  # the right side of the subplots of the figure
	# 	bottom=0.05,  # the bottom of the subplots of the figure
	# 	top=0.95,  # the top of the subplots of the figure
	# 	wspace=0.05,  # the amount of width reserved for blank space between subplots
	# 	hspace=0.05,  # the amount of height reserved for white space between subplots
	# )

	# if model.window_size>1:
	# 	# xh_gt = x[:, :480].reshape(seq_len,model.window_size,4,3)
	# 	# xr_gt = x[:, 480:].reshape(seq_len,model.window_size,7)
	# 	# xh_gen = x_gen[:, :480].reshape(seq_len,model.window_size,4,3)
	# 	# xr_gen = x_gen[:, 480:].reshape(seq_len,model.window_size,7)
	# 	xh_gt = x[:, :12].reshape(seq_len,4,3)
	# 	xr_gt = x[:, 480:480+7]
	# 	xh_gen = x_gen[:, :12].reshape(seq_len,4,3)
	# 	xr_gen = x_gen[:, 480:480+7]
		
		# # fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(28, 16), sharex=True)
		# # fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		# # plt.subplots_adjust(
		# # 	left=0.05,  # the left side of the subplots of the figure
		# # 	right=0.95,  # the right side of the subplots of the figure
		# # 	bottom=0.05,  # the bottom of the subplots of the figure
		# # 	top=0.95,  # the top of the subplots of the figure
		# # 	wspace=0.05,  # the amount of width reserved for blank space between subplots
		# # 	hspace=0.05,  # the amount of height reserved for white space between subplots
		# # )
		# # for i in range(4):
		# # 	idx = np.random.randint(0, seq_len)
		# # 	# Plot human dofs
		# # 	for j in range(4):
		# # 		ax[i][j].set(xlim=(0, model.window_size - 1))
		# # 		color_counter = 0
		# # 		for dim in range(3):
		# # 			ax[i][j].plot(xh_gt[idx, :, j, dim], color=colors_10(color_counter%10))
		# # 			ax[i][j].plot(xh_gen[idx, :, j, dim], linestyle='--', color=colors_10(color_counter % 10))
		# # 			color_counter += 1
		# # 	# Plot robot dofs
		# # 	ax[i][4].set(xlim=(0, model.window_size - 1))
		# # 	color_counter = 0
		# # 	for dim in range(7):
		# # 		ax[i][4].plot(xr_gt[idx, :, dim], color=colors_10(color_counter%10))
		# # 		ax[i][4].plot(xr_gen[idx, :, dim], linestyle='--', color=colors_10(color_counter % 10))
		# # 		color_counter += 1
		# fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(28, 16), sharex=True)
		# fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		# plt.subplots_adjust(
		# 	left=0.05,  # the left side of the subplots of the figure
		# 	right=0.95,  # the right side of the subplots of the figure
		# 	bottom=0.05,  # the bottom of the subplots of the figure
		# 	top=0.95,  # the top of the subplots of the figure
		# 	wspace=0.05,  # the amount of width reserved for blank space between subplots
		# 	hspace=0.05,  # the amount of height reserved for white space between subplots
		# )
		# for i in range(4):
		# 	ax[i].set(xlim=(0, seq_len - 1))
		# 	color_counter = 0
		# 	for dim in range(3):
		# 		ax[i].plot(xh_gt[:,0, i, dim], color=colors_10(color_counter%10))
		# 		ax[i].plot(xh_gen[:,0, i, dim], linestyle='--', color=colors_10(color_counter % 10))
		# 		for t in range(seq_len):
		# 			ax[i].plot(np.arange(t+1,t+model.window_size), xh_gt[t,1:, i, dim], color=colors_10(color_counter%10), alpha=0.4)
		# 			ax[i].plot(np.arange(t+1,t+model.window_size), xh_gen[t,1:, i, dim], linestyle='--', color=colors_10(color_counter % 10), alpha=0.4)
		# 		color_counter += 1
		# ax[4].set(xlim=(0, seq_len - 1))
		# color_counter = 0
		# for dim in range(7):
		# 	ax[4].plot(xr_gt[:, dim], color=colors_10(color_counter%10))
		# 	ax[4].plot(xr_gen[:, dim], linestyle='--', color=colors_10(color_counter % 10))
		# 	for t in range(seq_len):
		# 		ax[4].plot(np.arange(t+1,t+model.window_size), xr_gt[t,1:, dim], color=colors_10(color_counter%10), alpha=0.4)
		# 		ax[4].plot(np.arange(t+1,t+model.window_size), xr_gen[t,1:, dim], linestyle='--', color=colors_10(color_counter % 10), alpha=0.4)
		# 	color_counter += 1
	# else:

	# 	xh_gt = x[:, :12].reshape(seq_len,4,3)
	# 	xr_gt = x[:, 12:]
	# 	xh_gen = x_gen[:, :12].reshape(seq_len,4,3)
	# 	xr_gen = x_gen[:, 12:]

	# fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(28, 16), sharex=True)
	# fig.tight_layout(pad=0, h_pad=0, w_pad=0)
	# plt.subplots_adjust(
	# 	left=0.05,  # the left side of the subplots of the figure
	# 	right=0.95,  # the right side of the subplots of the figure
	# 	bottom=0.05,  # the bottom of the subplots of the figure
	# 	top=0.95,  # the top of the subplots of the figure
	# 	wspace=0.05,  # the amount of width reserved for blank space between subplots
	# 	hspace=0.05,  # the amount of height reserved for white space between subplots
	# )
	# for i in range(4):
	# 	ax[i].set(xlim=(0, seq_len - 1))
	# 	color_counter = 0
	# 	for dim in range(3):
	# 		ax[i].plot(xh_gt[:, i, dim], color=colors_10(color_counter%10))
	# 		ax[i].plot(xh_gen[:, i, dim], linestyle='--', color=colors_10(color_counter % 10))
	# 		color_counter += 1
	# ax[4].set(xlim=(0, seq_len - 1))
	# color_counter = 0
	# for dim in range(7):
	# 	ax[4].plot(xr_gt[:, dim], color=colors_10(color_counter%10))
	# 	ax[4].plot(xr_gen[:, dim], linestyle='--', color=colors_10(color_counter % 10))
	# 	color_counter += 1

	# fig.canvas.draw()
	# writer.add_figure(prefix+'/sample reconstruction', fig, steps_done)
	# plt.close(fig)


def write_summaries_robot(writer, recon, kl, loss, x_gen, zr_samples, x, steps_done, prefix, model):
	writer.add_scalar(prefix+'/loss', sum(loss), steps_done)
	writer.add_scalar(prefix+'/kl_div', sum(kl), steps_done)
	writer.add_scalar(prefix+'/recon_loss', sum(recon), steps_done)
	
	# # writer.add_embedding(zx_samples[:100],global_step=steps_done, tag=prefix+'/q(z|x)')
	x_gen = x_gen.detach().cpu().numpy()
	x = x.detach().cpu().numpy()
	if prefix=='train' and len(x_gen.shape)==3:
		x_gen = x_gen[-1]
	
	seq_len, dims = x.shape
	
	if model.window_size>1:
		xr_gt = x[:, 480:].reshape(seq_len,model.window_size,7)
		xr_gen = x_gen.reshape(seq_len,model.window_size,7)
		
		fig, ax = plt.subplots(nrows=7, ncols=4, figsize=(28, 16), sharex=True, sharey='row')
		fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		plt.subplots_adjust(
			left=0.05,  # the left side of the subplots of the figure
			right=0.95,  # the right side of the subplots of the figure
			bottom=0.05,  # the bottom of the subplots of the figure
			top=0.95,  # the top of the subplots of the figure
			wspace=0.05,  # the amount of width reserved for blank space between subplots
			hspace=0.05,  # the amount of height reserved for white space between subplots
		)
		for i in range(4):
			idx = np.random.randint(0, seq_len)
			# Plot robot dofs
			color_counter = 0
			for dim in range(7):
				ax[dim][i].set(xlim=(0, model.window_size - 1))
				ax[dim][i].plot(xr_gt[idx, :, dim], color=colors_10(color_counter%10))
				ax[dim][i].plot(xr_gen[idx, :, dim], linestyle='--', color=colors_10(color_counter % 10))
				color_counter += 1
	else:
		fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(28, 16), sharex=True)
		fig.tight_layout(pad=0, h_pad=0, w_pad=0)
		
		plt.subplots_adjust(
			left=0.05,  # the left side of the subplots of the figure
			right=0.95,  # the right side of the subplots of the figure
			bottom=0.05,  # the bottom of the subplots of the figure
			top=0.95,  # the top of the subplots of the figure
			wspace=0.05,  # the amount of width reserved for blank space between subplots
			hspace=0.05,  # the amount of height reserved for white space between subplots
		)
		xr_gt = x[:, 12:]
		xr_gen = x_gen
		color_counter = 0
		for dim in range(7):
			ax[dim].set(xlim=(0, seq_len - 1))
			ax[dim].plot(xr_gt[:, dim], color=colors_10(color_counter%10))
			ax[dim].plot(xr_gen[:, dim], linestyle='--', color=colors_10(color_counter % 10))
			color_counter += 1

	fig.canvas.draw()
	writer.add_figure(prefix+'/sample reconstruction', fig, steps_done)
	plt.close(fig)

def kl_div(mu0, Sigma0, mu1, Sigma1, reduction='sum'):
	diff = (mu0 - mu1)
	Sigma1_inv = torch.linalg.inv(Sigma1)
	return 0.5 * torch.diagonal(Sigma1_inv.matmul(Sigma0),dim1=-1, dim2=-2).sum(-1) + \
		diff.unsqueeze(-2).matmul(Sigma1_inv).matmul(diff.unsqueeze(-1))[...,0,0] + \
		torch.logdet(Sigma1) - torch.logdet(Sigma0) - diff.shape[-1]

def kl_div_diag(mu0, sigma0, mu1, sigma1, reduction='sum'):
	var0 = sigma0**2
	var1 = sigma1**2
	return getattr(torch, reduction)(torch.log(sigma1/sigma0) + (var0 + (mu0-mu1)**2)/(2*var1) - 0.5)


def batchNearestPD(A):
	"""Find the nearest positive-definite matrix to input
	A Python/Numpy port [1] of John D'Errico's `nearestSPD` MATLAB code [2], which
	credits [3].
	[1] https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
	[2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
	[3] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
	"""

	B = (A + torch.transpose(A,-2,-1))/2.
	_, s, V = torch.svd(B)
	sV = torch.bmm(torch.diag_embed(s),V)
	H = torch.bmm(torch.transpose(V,-2,-1),sV)

	A2 = (B + H) / 2

	A3 = (A2 + torch.transpose(A2,-2,-1)) / 2.

	if batchIsPD(A3):
		return A3

	spacing = torch.finfo(torch.float32).eps
	# The above is different from [1]. It appears that MATLAB's `chol` Cholesky
	# decomposition will accept matrixes with exactly 0-eigenvalue, whereas
	# Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
	# for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
	# will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
	# the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
	# `spacing` will, for Gaussian random matrixes of small dimension, be on
	# othe order of 1e-16. In practice, both ways converge, as the unit test
	# below suggests.
	with torch.no_grad():
		I = torch.eye(A.shape[-1]).repeat(A.shape[0],1,1).to(A.device)
	k = 1
	while not batchIsPD(A3):
		mineig = torch.real(torch.symeig(A3)[0].type(torch.cfloat))[:,0]
		v = (-mineig[:,None,None] * k**2 + spacing).repeat(1,A.shape[-1],A.shape[-1])
		A3 += I * v
		k += 1

	return A3

def batchIsPD(B):
	"""Returns true when input is positive-definite, via Cholesky"""
	try:
		torch.cholesky(B)
		return True
	except:
		return False

joints = ["none", "head", "neck", "torso", "waist", "left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand", "left_fingertip", "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand", "right_fingertip", "left_hip", "left_knee", "left_ankle", "left_foot", "right_hip", "right_knee", "right_ankle", "right_foot"]
# joints = ['head', 'neck', 'torso', 'waist', 'left_shoulder', 'left_elbow', 'left_hand', 'right_shoulder', 'right_elbow', 'right_hand']
# joints = ['neck', 'right_shoulder', 'right_elbow', 'right_hand']
joints_dic = {joints[i]:i for i in range(len(joints))}


def angle(a,b):
	dot = np.dot(a,b)
	return np.arccos(dot/(np.linalg.norm(a)*np.linalg.norm(b)))

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
	

	# T = np.eye(4)
	# T[:3,3] = skeleton[0]
	# T[:3,:3] = rotation_normalization(skeleton)
	# rightShoulder = skeleton[joints_dic["right_shoulder"]] - T[:3,3]
	# rightShoulder = T[:3,:3].dot(rightShoulder)
	# rightElbow = skeleton[joints_dic["right_elbow"]] - T[:3,3]
	# rightElbow = T[:3,:3].dot(rightElbow)
	# rightHand = skeleton[joints_dic["right_hand"]] - T[:3,3]
	# rightHand = T[:3,:3].dot(rightHand)

	# rightShoulder = skeleton[joints_dic["right_shoulder"]]
	# rightElbow = skeleton[joints_dic["right_elbow"]]
	# rightHand = skeleton[joints_dic["right_hand"]]

	rightShoulder = skeleton[1]
	rightElbow = skeleton[2]
	rightHand = skeleton[3]
	
	rightYaw = 0
	rightPitch = 0
	rightRoll = 0
	rightElbowAngle = 0
	
	# Recreating arm with upper and under arm
	rightUpperArm = rightElbow - rightShoulder
	rightUnderArm = rightHand - rightElbow

	rightElbowAngle = angle(rightUpperArm, rightUnderArm)

	armlengthRight = np.linalg.norm(rightUpperArm)
	rightYaw = np.arctan2(rightUpperArm[1],-rightUpperArm[2]) # Comes from robot structure
	# rightYaw -= 0.009
	rightPitch = np.arctan2(rightUpperArm[0], rightUpperArm[2]) # Comes from robot structure
	rightPitch -= np.pi/2
	
	# Recreating under Arm Position with known Angles(without roll)
	rightRotationAroundY = euler_matrix(0, rightPitch, 0,)[:3,:3]
	rightRotationAroundX = euler_matrix(0, 0, rightYaw)[:3,:3]
	rightElbowRotation = euler_matrix(0, 0, rightElbowAngle)[:3,:3]

	rightUnderArmInZeroPos = np.array([np.linalg.norm(rightUnderArm), 0, 0.])
	rightUnderArmWithoutRoll = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightElbowRotation,rightUnderArmInZeroPos)))

	# Calculating the angle betwenn actual under arm position and the one calculated without roll
	rightRoll = angle(rightUnderArmWithoutRoll, rightUnderArm)
	
	#This is a check which sign the angle has as the calculation only produces positive angles
	rightRotationAroundArm = euler_matrix(0, 0, -rightRoll)[:3, :3]
	rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	r1saver = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	rightRotationAroundArm = euler_matrix(0, 0, rightRoll)[:3, :3]
	rightShouldBeWristPos = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightRotationAroundArm,np.dot(rightElbowRotation,rightUnderArmInZeroPos))))
	r1 = np.linalg.norm(rightUnderArm - rightShouldBeWristPos)
	
	# if (r1 > r1saver):
	# 	rightRoll = -rightRoll

	return np.array([rightPitch, rightYaw, rightRoll, rightElbowAngle])

