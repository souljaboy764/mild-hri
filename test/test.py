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

epochs = np.arange(50,401,10)
model_types = [
					'v1_1/diaghmm_z3h5',
					'v1_1/diaghmm_z3h6',
					'v1_1/diaghmm_z3h7',
					'v1_1/diaghmm_z3h8',
					'v1_1/diaghmm_z5h5',
					'v1_1/diaghmm_z5h6',
					'v1_1/diaghmm_z5h7',
					'v1_1/diaghmm_z5h8',
					'v1_1/diaghmm_z8h5',
					'v1_1/diaghmm_z8h6',
					'v1_1/diaghmm_z8h7',
					'v1_1/diaghmm_z8h8',
					'v2_1/diaghmm_z3h5',
					'v2_1/diaghmm_z3h6',
					'v2_1/diaghmm_z3h7',
					'v2_1/diaghmm_z3h8',
					'v2_1/diaghmm_z5h5',
					'v2_1/diaghmm_z5h6',
					'v2_1/diaghmm_z5h7',
					'v2_1/diaghmm_z5h8',
					'v2_1/diaghmm_z8h5',
					'v2_1/diaghmm_z8h6',
					'v2_1/diaghmm_z8h7',
					'v2_1/diaghmm_z8h8',
					'v2_2/diaghmm_z3h5',
					'v2_2/diaghmm_z3h6',
					'v2_2/diaghmm_z3h7',
					'v2_2/diaghmm_z3h8',
					'v2_2/diaghmm_z5h5',
					'v2_2/diaghmm_z5h6',
					'v2_2/diaghmm_z5h7',
					'v2_2/diaghmm_z5h8',
					'v2_2/diaghmm_z8h5',
					'v2_2/diaghmm_z8h6',
					'v2_2/diaghmm_z8h7',
					'v2_2/diaghmm_z8h8',
					'v3_1/diaghmm_z3h5',
					'v3_1/diaghmm_z3h6',
					'v3_1/diaghmm_z3h7',
					'v3_1/diaghmm_z3h8',
					'v3_1/diaghmm_z5h5',
					'v3_1/diaghmm_z5h6',
					'v3_1/diaghmm_z5h7',
					'v3_1/diaghmm_z5h8',
					'v3_1/diaghmm_z8h5',
					'v3_1/diaghmm_z8h6',
					'v3_1/diaghmm_z8h7',
					'v3_1/diaghmm_z8h8',
					'v3_2/diaghmm_z3h5',
					'v3_2/diaghmm_z3h6',
					'v3_2/diaghmm_z3h7',
					'v3_2/diaghmm_z3h8',
					'v3_2/diaghmm_z5h5',
					'v3_2/diaghmm_z5h6',
					'v3_2/diaghmm_z5h7',
					'v3_2/diaghmm_z5h8',
					'v3_2/diaghmm_z8h5',
					'v3_2/diaghmm_z8h6',
					'v3_2/diaghmm_z8h7',
					'v3_2/diaghmm_z8h8',
					'v4_1/diaghmm_z3h5',
					'v4_1/diaghmm_z3h6',
					'v4_1/diaghmm_z3h7',
					'v4_1/diaghmm_z3h8',
					'v4_1/diaghmm_z5h5',
					'v4_1/diaghmm_z5h6',
					'v4_1/diaghmm_z5h7',
					'v4_1/diaghmm_z5h8',
					'v4_1/diaghmm_z8h5',
					'v4_1/diaghmm_z8h6',
					'v4_1/diaghmm_z8h7',
					'v4_1/diaghmm_z8h8',
					'v4_2/diaghmm_z3h5',
					'v4_2/diaghmm_z3h6',
					'v4_2/diaghmm_z3h7',
					'v4_2/diaghmm_z3h8',
					'v4_2/diaghmm_z5h5',
					'v4_2/diaghmm_z5h6',
					'v4_2/diaghmm_z5h7',
					'v4_2/diaghmm_z5h8',
					'v4_2/diaghmm_z8h5',
					'v4_2/diaghmm_z8h6',
					'v4_2/diaghmm_z8h7',
					'v4_2/diaghmm_z8h8',
				]

print('Model Type\tEpochs\tTrial\tPred. MSE (all)\tPred. MSE w/o waving\tPred. MSE waving\tPred. MSE handshake\tPred. MSE rocket\tPred. MSE parachute\n')
for model_type in model_types:
	for epoch in epochs:
		pred_mse_k = []
		pred_mse_nowave_k = []
		pred_mse_wave_k = []
		pred_mse_shake_k = []
		pred_mse_rocket_k = []
		pred_mse_parachute_k = []
		for trial in range(4):
			if epoch == 400:
				ckpt_path = f'logs/2023/bp_pepper_20hz/{model_type}/trial{trial}/models/final_399.pth'
			else:
				ckpt_path = f'logs/2023/bp_pepper_20hz/{model_type}/trial{trial}/models/' + '%0.3d'%epoch + '.pth'
	
			ckpt = torch.load(ckpt_path)
			pred_mse_ckpt, pred_mse_nowave_ckpt, pred_mse_wave_ckpt, pred_mse_shake_ckpt, pred_mse_rocket_ckpt, pred_mse_parachute_ckpt = evaluate_ckpt_hr(ckpt_path)
			if np.any(np.isnan(pred_mse_ckpt)):
				print(model_type, trial)
				continue
			pred_mse_k += pred_mse_ckpt
			pred_mse_nowave_k += pred_mse_nowave_ckpt
			pred_mse_wave_k += pred_mse_wave_ckpt
			pred_mse_shake_k += pred_mse_shake_ckpt
			pred_mse_rocket_k += pred_mse_rocket_ckpt
			pred_mse_parachute_k += pred_mse_parachute_ckpt
			print(f'{model_type}\t{epoch}\t{trial}\t{np.mean(pred_mse_ckpt):.4e} ± {np.std(pred_mse_ckpt):.4e}\t{np.mean(pred_mse_nowave_ckpt):.4e} ± {np.std(pred_mse_nowave_ckpt):.4e}\t{np.mean(pred_mse_wave_ckpt):.4e} ± {np.std(pred_mse_wave_ckpt):.4e}\t{np.mean(pred_mse_shake_ckpt):.4e} ± {np.std(pred_mse_shake_ckpt):.4e}\t{np.mean(pred_mse_rocket_ckpt):.4e} ± {np.std(pred_mse_rocket_ckpt):.4e}\t{np.mean(pred_mse_parachute_ckpt):.4e} ± {np.std(pred_mse_parachute_ckpt):.4e}')
		print(f'{model_type}\t{epoch}\tall\t{np.mean(pred_mse_k):.4e} ± {np.std(pred_mse_k):.4e}\t{np.mean(pred_mse_nowave_k):.4e} ± {np.std(pred_mse_nowave_k):.4e}\t{np.mean(pred_mse_wave_k):.4e} ± {np.std(pred_mse_wave_k):.4e}\t{np.mean(pred_mse_shake_k):.4e} ± {np.std(pred_mse_shake_k):.4e}\t{np.mean(pred_mse_rocket_k):.4e} ± {np.std(pred_mse_rocket_k):.4e}\t{np.mean(pred_mse_parachute_k):.4e} ± {np.std(pred_mse_parachute_k):.4e}')
	print('\n')