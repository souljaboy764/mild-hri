# HSMMVAE

Exploring the use of HSMMs for learning latent space dynamics in VAEs

## Datsets

TODO:

- Update preproc instructions form original repos
- Update readme

Current Good Model: logs/fullvae_rarm_window_alphaargmaxnocond_07132220/models/0570.pth


Best HH 20hz model:
v1_1/diaghmm_z5h5 trial2 390.pth
v1_1/diaghmm_z8h5 trial3 320.pth


```python
from mild_hri.utils import *
model_type = 'v3_1/diaghmm_z3h5'
trial = 1
epoch = 300
ckpt_path = f'logs/2023/bp_pepper_20hz_norm/{model_type}/trial{trial}/models/final_399.pth'
_ = evaluate_ckpt_hr(ckpt_path)
```

HH Models used for HR:
logs/2023/bp_hh_20hz/v1_1/diaghmm_z3h5/trial0/models/100.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z3h6/trial3/models/080.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z3h7/trial2/models/080.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z3h8/trial1/models/120.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z5h5/trial2/models/390.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z5h6/trial2/models/190.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z5h7/trial0/models/290.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z5h8/trial2/models/340.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z8h5/trial3/models/320.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z8h6/trial2/models/340.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z8h7/trial1/models/240.pth
logs/2023/bp_hh_20hz/v1_1/diaghmm_z8h8/trial3/models/120.pth
