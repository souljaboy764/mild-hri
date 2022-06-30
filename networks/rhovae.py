import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl_divergence

import numpy as np

from networks import VAE

class RhoVAE(VAE):
	"""
	Implementation of Rho-VAE: https://arxiv.org/pdf/1909.06236.pdf
	Parts taken from https://github.com/sssohrab/rho_VAE/
	"""
	def __init__(self, **kwargs):
		super(RhoVAE, self).__init__(**kwargs)
		
		self.rho = nn.Linear(self.enc_sizes[-1], 1)
		self.log_s = nn.Linear(self.enc_sizes[-1], 1)
		self.z_prior = Normal(self.z_prior_mean, self.z_prior_std)
		self.pow_idx = np.arange(self.latent_dim)
		self.cov_idx = np.zeros(self.latent_dim, self.latent_dim)
		for i in reversed(range(self.latent_dim)):
			self.cov_idx[self.pow_idx,np.clip(self.pow_idx+i,0,self.latent_dim-1)] = i
			self.cov_idx[np.clip(self.pow_idx+i,0,self.latent_dim-1),self.pow_idx] = i

	def forward(self, x, encode_only = False):
		enc = self._encoder(x)
		z_mean = self.post_mean(enc)
		if encode_only:
			return z_mean
		rho = torch.tanh(self.rho(enc))
		log_s = self.log_s(enc)
		z_cov = rho[...]
		rho_d = rho
		for i in range(self.latent_dim):
			for j in self.latent_dim:
			z_cov[..., ]

		zpost_dist = MultivariateNormal(z_mean, z_cov)
			
		if self.training:
			zpost_samples = torch.concat([zpost_dist.rsample((10,)), z_mean[None]], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, zpost_dist

	def latent_loss(self, zpost_samples, zpost_dist):
		if isinstance(self.z_prior, Normal):
			return kl_divergence(zpost_dist, self.z_prior).mean()
		if isinstance(self.z_prior, list):
			kld = 0
			for p in self.z_prior:
				kld += kl_divergence(zpost_dist, p).mean()
			return kld
