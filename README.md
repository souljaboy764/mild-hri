# HSMMVAE

Exploring the use of HSMMs for learning latent space dynamics in VAEs

## Datsets

TODO:

- Update preproc instructions form original repos
- Update readme

Current Good Model: logs/fullvae_rarm_window_alphaargmaxnocond_07132220/models/0570.pth



```python
from mild_hri.utils import *
model_type = 'v1_1/diaghmm_z3h5'
trial = 1
epoch = 300
ckpt_path = f'logs/2023/bp_pepper_20hz/{model_type}/trial{trial}/models/' + ('%0.3d'%epoch) + '.pth'
_ = evaluate_ckpt_hr(ckpt_path)
```