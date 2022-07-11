import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl_divergence

from networks import AE
from utils import *

class VAE(AE):
	def __init__(self, **kwargs):
		super(VAE, self).__init__(**kwargs)
		
		self.post_mean = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		self.post_logstd = nn.Linear(self.enc_sizes[-1], self.latent_dim)
		
	def forward(self, x, encode_only = False):
		enc = self._encoder(x)
		z_mean = self.post_mean(enc)
		if encode_only:
			return z_mean
		# z_std = self.post_logstd(enc).exp()
		# zpost_dist = Normal(z_mean, z_std)
		z_std = self.post_logstd(enc).exp() + 1e-3
		# zpost_dist = MultivariateNormal(z_mean, z_std)
			
		if self.training:
			eps = torch.randn((10,)+z_mean.shape).to(z_mean.device)
			zpost_samples = z_mean + eps*z_std
			zpost_samples = torch.concat([zpost_samples, z_mean[None]], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		# return x_gen, zpost_samples, z_mean, torch.diag_embed(z_std**2)
		return x_gen, zpost_samples, MultivariateNormal(z_mean, torch.diag_embed(z_std**2))

	def latent_loss(self, zpost_dist, zpost_samples):
		return kl_divergence(zpost_dist, Normal(0, 1)).mean()

class FullCovVAE(VAE):
	def __init__(self, **kwargs):
		super(FullCovVAE, self).__init__(**kwargs)
		
		# # Not mentioned in the paper what is used to ensure stddev>0, using softplus for now
		# # self.post_std = nn.Sequential(nn.Linear(self.enc_sizes[-1], self.latent_dim), nn.Softplus())
		self.post_cholesky = nn.Linear(self.enc_sizes[-1], (self.latent_dim*(self.latent_dim+1))//2)
		self.z_prior = Normal(self.z_prior_mean, self.z_prior_std)
		self.diag_idx = torch.arange(self.latent_dim)
		self.tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)

	def forward(self, x, encode_only = False):
		enc = self._encoder(x)
		
		z_mean = self.post_mean(enc)
		if encode_only:
			return z_mean
		# Colesky Matrix Prediction 
		# Dorta et al. "Structured Uncertainty Prediction Networks" CVPR'18
		# Dorta et al. "Training VAEs Under Structured Residuals" 2018
		z_std = self.post_cholesky(enc)
		z_chol = torch.zeros(z_std.shape[:-1]+(self.latent_dim, self.latent_dim)).to(z_std.device)
		z_chol[..., self.tril_indices[0], self.tril_indices[1]] = z_std
		z_chol[..., self.diag_idx,self.diag_idx] = 2*torch.abs(z_chol[..., self.diag_idx,self.diag_idx]) + 1e-2
		zpost_dist = MultivariateNormal(z_mean, scale_tril=z_chol)
			
		if self.training:
			zpost_samples = torch.concat([zpost_dist.rsample((10,)), z_mean[None]], dim=0)
		else:
			zpost_samples = z_mean
		
		x_gen = self._output(self._decoder(zpost_samples))
		return x_gen, zpost_samples, zpost_dist

	def latent_loss(self, zpost_dist, zpost_samples):
		if isinstance(self.z_prior, Normal):
			return kl_divergence(zpost_dist, self.z_prior).mean()
		if isinstance(self.z_prior, list):
			kld = []
			for p in self.z_prior:
				kld.append(kl_divergence(zpost_dist, p))
			return kld
