# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

import numpy as np
import os, datetime, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import networks
import config
from utils import *
from dataloaders.buetepage import PepperWindowDataset

import pbdlib as pbd
import pbdlib_torch as pbd_torch
from typing import List

window_size = 5
num_joints = 4
joint_dims = 1
train_dataset = PepperWindowDataset('./data/buetepage/traj_data.npz', train=False, window_length=window_size, downsample=0.2)
test_dataset = PepperWindowDataset('./data/buetepage/traj_data.npz', train=False, window_length=window_size, downsample=0.2)
for nb_states in [5, 10, 12]:
    pred_mse = []
    for a in range(4):
        train_traj_idx = train_dataset.actidx[a]
        start = datetime.datetime.now()
        hmm = pbd.HMM(nb_dim=train_dataset.traj_data[0].shape[-1], nb_states=nb_states)
        hmm.init_hmm_kbins(train_dataset.traj_data[train_traj_idx[0]:train_traj_idx[1]])
        hmm.em(train_dataset.traj_data[train_traj_idx[0]:train_traj_idx[1]])
        print(nb_states,a, (datetime.datetime.now()-start).total_seconds())
        test_traj_idx = test_dataset.actidx[a]
        start = datetime.datetime.now()
        for i in range(test_traj_idx[0], test_traj_idx[1]):
            traj_data = test_dataset.traj_data[i]
            dims = traj_data.shape[-1]
            x_h = traj_data[:dims-4]
            x_r = traj_data[-4:]
            xr_cond = hmm.condition(x_h, Sigma_in=None, dim_in=slice(0, dims-4), dim_out=slice(dims-4, dims), return_cov=False)
            pred_mse += ((xr_cond - x_r)**2).reshape((x_r.shape[0], window_size, num_joints, joint_dims)).sum(-1).mean(-1).mean(-1).tolist()
        print(nb_states,a, (datetime.datetime.now()-start).total_seconds())
    print(f'{nb_states}\t{np.mean(pred_mse):.3e} pm {np.std(pred_mse):.3e}')