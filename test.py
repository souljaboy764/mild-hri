import numpy as np
import torch
import argparse

from utils import *
from phd_utils.dataloaders import *

parser = argparse.ArgumentParser(description='Buetepage et al. (2020) Human-Human Interaction Testing')
parser.add_argument('--ckpt', type=str, metavar='CKPT', required=True,
					help='Path to the checkpoint to test.')
args = parser.parse_args()

ckpt_path = args.ckpt
ckpt = torch.load(ckpt_path)
if 'args_r' in ckpt.keys():
	pred_mse_action_ckpt = evaluate_ckpt_hr(ckpt_path)
else:
	pred_mse_action_ckpt = evaluate_ckpt_hh(ckpt_path)

s = ''
for i in range(len(pred_mse_action_ckpt)):
	s += f'{np.mean(pred_mse_action_ckpt[i]):.4e} $pm$ {np.std(pred_mse_action_ckpt[i]):.4e}\t'
print(s)
