import torch
import torch.nn.functional as F
import numpy as np

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
