import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from utils import *
from phd_utils.dataloaders import *
from phd_utils.visualization import *

parser = argparse.ArgumentParser(description='Buetepage et al. (2020) Human-Human Interaction Testing')
parser.add_argument('--ckpt', type=str, metavar='CKPT', required=True,
					help='Path to the checkpoint to test.')
args = parser.parse_args()

ckpt_path = args.ckpt
ckpt = torch.load(ckpt_path)
ckpt = torch.load(ckpt_path)
args_ckpt = ckpt['args']
if args_ckpt.dataset == 'buetepage':
	dataset = buetepage.HHWindowDataset
if args_ckpt.dataset == 'nuisi':
	dataset = nuisi.HHWindowDataset
if args_ckpt.dataset == 'alap':
	dataset = alap.HHWindowDataset
if args_ckpt.dataset == 'kobo':
	dataset = alap.KoboWindowDataset

test_iterator = DataLoader(dataset(train=False, window_length=args_ckpt.window_size, downsample=args_ckpt.downsample), batch_size=1, shuffle=False)

model = VAE(**(args_ckpt.__dict__)).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
ssm = ckpt['ssm']
z_dim = model.latent_dim


with torch.no_grad():
    for idx in [test_iterator.dataset.actidx[0]]:
        # ax_latent = reset_axis(ax_latent)
        # for i in range(ssm[0].mu.shape[0]):
        #     pbd_torch.plot_gauss3d(ax_latent, ssm[0].mu[i, :3].detach().cpu().numpy(), ssm[0].sigma[i, :3, :3].detach().cpu().numpy(), color='red', alpha=0.2)
        #     pbd_torch.plot_gauss3d(ax_latent, ssm[0].mu[i, z_dim:z_dim+3].detach().cpu().numpy(), ssm[0].sigma[i, z_dim:z_dim+3, z_dim:z_dim+3].detach().cpu().numpy(), color='blue', alpha=0.2)
        #     ax_latent.text(ssm[0].mu[i, 0].detach().cpu().numpy(), ssm[0].mu[i, 1].detach().cpu().numpy(), ssm[0].mu[i, 2].detach().cpu().numpy(), str(i), None)
        #     ax_latent.text(ssm[0].mu[i, z_dim].detach().cpu().numpy(), ssm[0].mu[i, z_dim+1].detach().cpu().numpy(), ssm[0].mu[i, z_dim+2].detach().cpu().numpy(), str(i), None)

        for i in range(idx[0],idx[1]):
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1, projection='3d')
            ax_alpha = fig.add_subplot(1,2,2)
            ax_alpha.set_xlim([0, 1])
            ax_alpha.set_ylim([0, 1])

            # plt.ion()
            ax.view_init(6, -88)
            ax.set_xlim3d([0.4, 1.4])
            ax.set_ylim3d([-0.5, 0.5])
            ax.set_zlim3d([0, 1])
            plt.ion()
            plt.show(block=False)
            x, label = test_iterator.dataset[i]
            # x = x[0]
            # label = label[0]
            # x_in.append(x.cpu().numpy())
            x = torch.Tensor(x).to(device)
            x_h = x[:, :model.input_dim]
            x_r = x[:, model.input_dim:]
            
            
            zh_post = model(x_h, dist_only=True)
            # xr_gen, _, _ = model_r(x_r)
            # x_vae.append(xr_gen.cpu().numpy())
            if args_ckpt.cov_cond:
                data_Sigma_in = zh_post.covariance_matrix
            else: 
                data_Sigma_in = None
            # print(label, label.dtype)
            zr_cond = ssm[label].condition(zh_post.mean, dim_in=slice(0, z_dim), dim_out=slice(z_dim, 2*z_dim), 
                                            data_Sigma_in=data_Sigma_in,
                                            return_cov=False)
            alpha = ssm[label].forward_variable(demo=zh_post.mean, marginal=slice(0, z_dim)).cpu().numpy().T
            xr_cond = model._output(model._decoder(zr_cond)).cpu().numpy().reshape((x_r.shape[0], model.window_size, model.num_joints, model.joint_dims))
            x_h = x_h.cpu().numpy().reshape((x_r.shape[0], model.window_size, model.num_joints, model.joint_dims))
            zh = zh_post.mean.cpu().numpy().reshape((x_r.shape[0], z_dim))
            zr_cond = zr_cond.cpu().numpy().reshape((x_r.shape[0], z_dim))

            times = np.linspace(0,1,x_r.shape[0])
            for t in range(x_r.shape[0]):
                ax = reset_axis(ax)
                ax = visualize_skeleton(ax, x_h[t, :, :2], facecolor='red')
                ax = visualize_skeleton(ax, xr_cond[t, :, :2], facecolor='blue')

                ax_alpha.cla()
                ax_alpha.set_xlim([0, 1])
                ax_alpha.set_ylim([0, 1])
                ax_alpha.set_xlabel('t')
                ax_alpha.set_ylabel('alpha')
                ax_alpha.set_facecolor('none')
                for j in range(alpha.shape[1]):
                    ax_alpha.plot(times[:t], alpha[:t, j], label=str(j+1))
                ax_alpha.legend()
                mypause(0.05)
                # if not plt.fignum_exists(fig.number):
                #       break
            plt.ioff()
            plt.show()
        #     if not plt.fignum_exists(fig.number):
        #         break
        # if not plt.fignum_exists(fig.number):
        #     break
