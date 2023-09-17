import numpy as np
import torch

from mild_hri.utils import *
from mild_hri.dataloaders import *

pred_mse = []
pred_mse_nowave = []
pred_mse_wave = []
pred_mse_shake = []
pred_mse_rocket = []
pred_mse_parachute = []

epochs = np.arange(0,401,10)
# epochs = [100]
model_types = [
					# # HRI
					# 'v1_1/diaghmm_z3h5',
					# 'v1_1/diaghmm_z3h6',
					# 'v1_1/diaghmm_z3h7',
					# 'v1_1/diaghmm_z3h8',
					# 'v1_1/diaghmm_z5h5',
					# 'v1_1/diaghmm_z5h6',
					# 'v1_1/diaghmm_z5h7',
					# 'v1_1/diaghmm_z5h8',
					# 'v1_1/diaghmm_z8h5',
					# 'v1_1/diaghmm_z8h6',
					# 'v1_1/diaghmm_z8h7',
					# 'v1_1/diaghmm_z8h8',
					# 'v2_1/diaghmm_z3h5',
					# 'v2_1/diaghmm_z3h6',
					# 'v2_1/diaghmm_z3h7',
					# 'v2_1/diaghmm_z3h8',
					# 'v2_1/diaghmm_z5h5',
					# 'v2_1/diaghmm_z5h6',
					# 'v2_1/diaghmm_z5h7',
					# 'v2_1/diaghmm_z5h8',
					# 'v2_1/diaghmm_z8h5',
					# 'v2_1/diaghmm_z8h6',
					# 'v2_1/diaghmm_z8h7',
					# 'v2_1/diaghmm_z8h8',
					# 'v2_2/diaghmm_z3h5',
					# 'v2_2/diaghmm_z3h6',
					# 'v2_2/diaghmm_z3h7',
					# 'v2_2/diaghmm_z3h8',
					# 'v2_2/diaghmm_z5h5',
					# 'v2_2/diaghmm_z5h6',
					# 'v2_2/diaghmm_z5h7',
					# 'v2_2/diaghmm_z5h8',
					# 'v2_2/diaghmm_z8h5',
					# 'v2_2/diaghmm_z8h6',
					# 'v2_2/diaghmm_z8h7',
					# 'v2_2/diaghmm_z8h8',
					# 'v3_1/diaghmm_z3h5',
					# 'v3_1/diaghmm_z3h6',
					# 'v3_1/diaghmm_z3h7',
					# 'v3_1/diaghmm_z3h8',
					# 'v3_1/diaghmm_z5h5',
					# 'v3_1/diaghmm_z5h6',
					# 'v3_1/diaghmm_z5h7',
					# 'v3_1/diaghmm_z5h8',
					# 'v3_1/diaghmm_z8h5',
					# 'v3_1/diaghmm_z8h6',
					# 'v3_1/diaghmm_z8h7',
					# 'v3_1/diaghmm_z8h8',
					# 'v3_2/diaghmm_z3h5',
					# 'v3_2/diaghmm_z3h6',
					# 'v3_2/diaghmm_z3h7',
					# 'v3_2/diaghmm_z3h8',
					# 'v3_2/diaghmm_z5h5',
					# 'v3_2/diaghmm_z5h6',
					# 'v3_2/diaghmm_z5h7',
					# 'v3_2/diaghmm_z5h8',
					# 'v3_2/diaghmm_z8h5',
					# 'v3_2/diaghmm_z8h6',
					# 'v3_2/diaghmm_z8h7',
					# 'v3_2/diaghmm_z8h8',
					# 'v4_1/diaghmm_z3h5',
					# 'v4_1/diaghmm_z3h6',
					# 'v4_1/diaghmm_z3h7',
					# 'v4_1/diaghmm_z3h8',
					# 'v4_1/diaghmm_z5h5',
					# 'v4_1/diaghmm_z5h6',
					# 'v4_1/diaghmm_z5h7',
					# 'v4_1/diaghmm_z5h8',
					# 'v4_1/diaghmm_z8h5',
					# 'v4_1/diaghmm_z8h6',
					# 'v4_1/diaghmm_z8h7',
					# 'v4_1/diaghmm_z8h8',
					# 'v4_2/diaghmm_z3h5',
					# 'v4_2/diaghmm_z3h6',
					# 'v4_2/diaghmm_z3h7',
					# 'v4_2/diaghmm_z3h8',
					# 'v4_2/diaghmm_z5h5',
					# 'v4_2/diaghmm_z5h6',
					# 'v4_2/diaghmm_z5h7',
					# 'v4_2/diaghmm_z5h8',
					# 'v4_2/diaghmm_z8h5',
					# 'v4_2/diaghmm_z8h6',
					# 'v4_2/diaghmm_z8h7',
					# 'v4_2/diaghmm_z8h8',


					# HHI
					# 'v1_1/z3h5',
					# 'v1_1/z3h6',
					# 'v1_1/z3h7',
					# 'v1_1/z3h8',
					# 'v1_1/z5h5',
					# 'z5h6',
					# 'z5h5',
					'z5h6',
					# 'v1_1/z5h7',
					# 'v1_1/z5h8',

					# HRI
					# 'v1_1/z5h6',
					# 'v2_1/z5h6',
					# 'v2_2/z5h6',
					# 'v3_1/z5h6',
					# 'v3_2/z5h6',
					# 'v1_1/z5h7',
					# 'v2_1/z5h7',
					# 'v2_2/z5h7',
					# 'v3_1/z5h7',
					# 'v3_2/z5h7',
				]
# Buetepage & NuiSI-v2 ['waving', 'handshake2', 'rocket', 'parachute']
# NuiSI-v1 ['clapfist', 'fistbump', 'handshake', 'highfive', 'rocket', 'wave1']

# print('Epochs\tTrial\tPred. MSE (all)\t\tPred. MSE w/o waving\t\tPred. MSE waving\t\tPred. MSE handshake\t\tPred. MSE rocket\t\tPred. MSE parachute')
# print('\t\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma')
print('Model\tEpochs\tTrial\tPred. MSE (all)\t\tPred. MSE w/o waving\t\tPred. MSE Clapfist\t\tPred. MSE Fistbump\t\tPred. MSE handshake\t\tPred. MSE highfive\t\tPred. MSE rocket\t\tPred. MSE waving')
print('\t\t\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma\tmean\tsigma')
for model_type in model_types:
	for epoch in epochs:
		pred_mse_k = []
		pred_mse_action_k = []
		for i in range(4):
			pred_mse_action_k.append([])
		pred_mse_nowave_k = []
		# pred_mse_wave_k = []
		# pred_mse_shake_k = []
		# pred_mse_rocket_k = []
		# pred_mse_parachute_k = []
		for trial in range(1):
			if epoch == 400:
				ckpt_path = f'../logs/2023/bp_hh_20hz_3joints_xvel/{model_type}/trial{trial}/models/final_399.pth'
				# ckpt_path = f'../logs/debug/models/final_399.pth'
			else:
				ckpt_path = f'../logs/2023/bp_hh_20hz_3joints_xvel/{model_type}/trial{trial}/models/' + '%0.3d'%epoch + '.pth'
				# ckpt_path = f'../logs/debug/models/' + '%0.3d'%epoch + '.pth'
	
			ckpt = torch.load(ckpt_path)
			# pred_mse_ckpt, pred_mse_nowave_ckpt, pred_mse_wave_ckpt, pred_mse_shake_ckpt, pred_mse_rocket_ckpt, pred_mse_parachute_ckpt = evaluate_ckpt_hh(ckpt_path)
			pred_mse_ckpt, pred_mse_action_ckpt, pred_mse_nowave_ckpt = evaluate_ckpt_hh(ckpt_path)

			if np.any(np.isnan(pred_mse_ckpt)):
				print(model_type, trial)
				continue
			pred_mse_k += pred_mse_ckpt
			pred_mse_nowave_k += pred_mse_nowave_ckpt
			s = f'{model_type}\t{epoch}\t{trial}\t{np.mean(pred_mse_ckpt):.4e}\t{np.std(pred_mse_ckpt):.4e}\t{np.mean(pred_mse_nowave_ckpt):.4e}\t{np.std(pred_mse_nowave_ckpt):.4e}'
			for i in range(len(pred_mse_action_ckpt)):
				pred_mse_action_k[i] += pred_mse_action_ckpt[i]
				s += f'\t{np.mean(pred_mse_action_ckpt[i]):.4e}\t{np.std(pred_mse_action_ckpt[i]):.4e}'
			print(s)
			# pred_mse_wave_k += pred_mse_wave_ckpt
			# pred_mse_shake_k += pred_mse_shake_ckpt
			# pred_mse_rocket_k += pred_mse_rocket_ckpt
			# pred_mse_parachute_k += pred_mse_parachute_ckpt
			# print(f'{model_type}\t{epoch}\t{trial}\t{np.mean(pred_mse_ckpt):.4e} ± {np.std(pred_mse_ckpt):.4e}')#\t{np.mean(pred_mse_nowave_ckpt):.4e} ± {np.std(pred_mse_nowave_ckpt):.4e}\t{np.mean(pred_mse_wave_ckpt):.4e} ± {np.std(pred_mse_wave_ckpt):.4e}\t{np.mean(pred_mse_shake_ckpt):.4e} ± {np.std(pred_mse_shake_ckpt):.4e}\t{np.mean(pred_mse_rocket_ckpt):.4e} ± {np.std(pred_mse_rocket_ckpt):.4e}\t{np.mean(pred_mse_parachute_ckpt):.4e} ± {np.std(pred_mse_parachute_ckpt):.4e}')
		# print(f'{model_type}\t{epoch}\tall\t{np.mean(pred_mse_k):.4e} ± {np.std(pred_mse_k):.4e}')#\t{np.mean(pred_mse_nowave_k):.4e} ± {np.std(pred_mse_nowave_k):.4e}\t{np.mean(pred_mse_wave_k):.4e} ± {np.std(pred_mse_wave_k):.4e}\t{np.mean(pred_mse_shake_k):.4e} ± {np.std(pred_mse_shake_k):.4e}\t{np.mean(pred_mse_rocket_k):.4e} ± {np.std(pred_mse_rocket_k):.4e}\t{np.mean(pred_mse_parachute_k):.4e} ± {np.std(pred_mse_parachute_k):.4e}')
		s = f'{model_type}\t{epoch}\tall\t{np.mean(pred_mse_k):.4e}\t{np.std(pred_mse_k):.4e}\t{np.mean(pred_mse_nowave_k):.4e}\t{np.std(pred_mse_nowave_k):.4e}'
		for i in range(len(pred_mse_action_k)):
			if len(pred_mse_action_k[i]) == 0:
				continue
			s += f'\t{np.mean(pred_mse_action_k[i]):.4e}\t{np.std(pred_mse_action_k[i]):.4e}'
		print(s)
	# print('\n')