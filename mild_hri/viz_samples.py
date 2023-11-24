import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import sklearn

import os

from mild_hri.utils import *
import mild_hri.dataloaders
import mild_hri.vae

parser = argparse.ArgumentParser(description='RMDVAE Testing')
parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
						help='Checkpoint to evaluate')
args = parser.parse_args()
ckpt = torch.load(args.ckpt)
args_h = ckpt['args_h']
args_r = ckpt['args_r']
if args_r.dataset == 'buetepage_pepper':
    dataset = dataloaders.buetepage.PepperWindowDataset
if args_r.dataset == 'nuisi_pepper':
    dataset = dataloaders.nuisi.PepperWindowDataset
if args_r.dataset == 'buetepage_yumi':
    dataset = dataloaders.buetepage_hr.YumiWindowDataset
# TODO: BP_Yumi, Nuisi_Pepper

test_iterator = DataLoader(dataset(args_r.src, train=False, window_length=args_r.window_size, downsample=args_r.downsample), batch_size=1, shuffle=False)

model_h = VAE(**(args_h.__dict__)).to(device)
model_h.load_state_dict(ckpt['model_h'])
model_r = VAE(**{**(args_h.__dict__), **(args_r.__dict__)})
model_r.to(device)

model_r.load_state_dict(ckpt['model_r'])

model_h.eval()
model_r.eval()
ssm = ckpt['ssm']

with torch.no_grad():
    x, label = test_iterator.dataset[16]
    x = torch.Tensor(x).to(device)
    x_h = x[:, :model_h.input_dim]
    x_r = x[:, model_h.input_dim:]
    z_dim = model_h.latent_dim

    zh_post = model_h(x_h, dist_only=True)
    if args_r.cov_cond:
        data_Sigma_in = zh_post.covariance_matrix
    else: 
        data_Sigma_in = None
    # print(label, label.dtype)
    h = ssm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim))#, cpp=False)
    zr_cond = ssm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), h=h,
                                    data_Sigma_in=data_Sigma_in,
                                    return_cov=False) 
    print(h.argmax(0))
    xr_cond = model_r._output(model_r._decoder(zr_cond))
    xr_cond = xr_cond.cpu().numpy()

np.savez_compressed('logs/samples/pepper_fistbump.npz', xh = x_h.cpu().numpy(), xr = xr_cond, zh=zr_cond.cpu().numpy())