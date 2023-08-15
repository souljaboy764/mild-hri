# HSMMVAE

Exploring the use of HSMMs for learning latent space dynamics in VAEs

## Datsets

TODO:

- Update preproc instructions form original repos
- Update readme

Current Good Model: logs/fullvae_rarm_window_alphaargmaxnocond_07132220/models/0570.pth



```python
from mild_hri.utils import *
model_type = 'v3_1/diaghmm_z3h5'
trial = 1
epoch = 300
ckpt_path = f'logs/2023/bp_pepper_20hz_norm/{model_type}/trial{trial}/models/final_399.pth'
_ = evaluate_ckpt_hr(ckpt_path)
```
