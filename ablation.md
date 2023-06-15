# VAE Ablation Results

Averaged over 5 trials

MSE - Squared Euclidean distance between predicted and ground truth arm locations summed over 4 joints and over the time window $\mathcal{W}$ \& averaged over all trajectories

(average over time window is the current MSE divided by window length)

# $x \in \mathcal{R}^{\mathcal{W} \times4\times3}$
# MSE = $\frac{\sum_{i=1}^{|\mathcal{D}|} \sum_{t=1}^{T_i} \sum_{w=1}^{\mathcal{W}} \sum_{j=1}^{4} ||\hat{\mathbf{x}}[t,w,j] - \mathbf{x}[t,w,j]||^2}{\sum_{i=1}^{|\mathcal{D}|} T_i}$

<!-- ## $\Sigma_{VAE} + \Sigma_{HSMM}$ -->

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| vae | 3 | 8.323e-01 ± 1.258e+00 |
| mild | 3 | 8.609e-01 ± 1.429e+00 |
| mild (cond. KL) | 3 | 8.853e-01 ± 1.448e+00 |
| vae | 5 | 7.087e-01 ± 9.950e-01 |
| mild | 5 | 7.662e-01 ± 1.066e+00 |
| mild (cond. KL) | 5 | 7.234e-01 ± 1.058e+00 |
| vae | 8 | 6.948e-01 ± 9.684e-01 |
| mild | 8 | 6.710e-01 ± 9.613e-01 |
| mild (cond. KL) | 8 | __6.527e-01 ± 9.233e-01__ |
| vae | 10 | 6.839e-01 ± 9.418e-01 |
| mild | 10 | 6.569e-01 ± 9.600e-01 |
| mild (cond. KL) | 10 | 6.717e-01 ± 9.450e-01 |

<!-- ## $\Sigma_{HSMM}$

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| vae | 3 | 8.324e-01 ± 1.258e+00 |
| mild | 3 | 8.614e-01 ± 1.429e+00 |
| mild (cond. KL) | 3 | 8.860e-01 ± 1.449e+00 |
| vae | 5 | 7.088e-01 ± 9.951e-01 |
| mild | 5 | 7.672e-01 ± 1.067e+00 |
| mild (cond. KL) | 5 | 7.247e-01 ± 1.059e+00 |
| vae | 8 | 6.952e-01 ± 9.696e-01 |
| mild | 8 | 6.742e-01 ± 9.664e-01 |
| mild (cond. KL) | 8 | __6.577e-01 ± 9.262e-01__ |
| vae | 10 | 6.844e-01 ± 9.436e-01 |
| mild | 10 | 6.625e-01 ± 9.653e-01 |
| mild (cond. KL) | 10 | 6.812e-01 ± 9.551e-01 | -->

VAE Ablation

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| ablation_vae | 3 | 8.3238e-01 ± 1.2583e+00 |
| vae_crosskl | 3 | 8.3685e-01 ± 1.2672e+00 |
| vae_onlycrosskl | 3 | 8.4034e-01 ± 1.3129e+00 |
| vae_sophia_onlycrosskl | 3 | 8.2272e-01 ± 1.3904e+00 |
| vae_sophia_addcrosskl | 3 | 8.0949e-01 ± 1.3140e+00 |
|:-----:|:-----:|:----:|
| ablation_vae | 5 | 7.0876e-01 ± 9.9508e-01 |
| vae_crosskl | 5 | 8.8036e-01 ± 1.4217e+00 |
| vae_onlycrosskl | 5 | 6.8970e-01 ± 9.4087e-01 |
| vae_sophia_onlycrosskl | 5 | 7.6130e-01 ± 1.1069e+00 |
| vae_sophia_addcrosskl | 5 | 7.7347e-01 ± 1.1211e+00 |
|:-----:|:-----:|:----:|
| ablation_vae | 8 | 6.9484e-01 ± 9.6848e-01 |
| vae_crosskl | 8 | 8.8375e-01 ± 1.4125e+00 |
| vae_onlycrosskl | 8 | 6.7312e-01 ± 9.1082e-01 |
| vae_sophia_onlycrosskl | 8 | 7.1415e-01 ± 1.1189e+00 |
| vae_sophia_addcrosskl | 8 | 6.9836e-01 ± 1.1246e+00 |
|:-----:|:-----:|:----:|
| ablation_vae | 10 | 6.8391e-01 ± 9.4185e-01 |
| vae_crosskl | 10 | 7.6653e-01 ± 1.0788e+00 |
| vae_onlycrosskl | 10 | 6.5030e-01 ± 9.3792e-01 |
| vae_sophia_onlycrosskl | 10 | 7.2432e-01 ± 1.1596e+00 |
| vae_sophia_addcrosskl | 10 | 7.2149e-01 ± 1.2061e+00 |
|:-----:|:-----:|:----:|

MILD Ablation

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| ablation_mild | 3 | 8.6093e-01 ± 1.4293e+00 |
| mild_crosskl | 3 | 8.3761e-01 ± 1.2148e+00 |
| mild_sophia_onlycrosskl | 3 | 8.2231e-01 ± 1.3833e+00 |
| mild_sophia_addcrosskl | 3 | 8.1908e-01 ± 1.3026e+00 |
|:-----:|:-----:|:----:|
| ablation_mild | 5 | 7.6629e-01 ± 1.0660e+00 |
| mild_crosskl | 5 | 8.2818e-01 ± 1.2739e+00 |
| mild_sophia_onlycrosskl | 5 | 7.2780e-01 ± 1.0366e+00 |
| mild_sophia_addcrosskl | 5 | 7.3063e-01 ± 1.0317e+00 |
|:-----:|:-----:|:----:|
| ablation_mild | 8 | 6.7108e-01 ± 9.6133e-01 |
| mild_crosskl | 8 | 8.7088e-01 ± 1.2077e+00 |
| mild_sophia_onlycrosskl | 8 | 6.8773e-01 ± 1.0645e+00 |
| mild_sophia_addcrosskl | 8 | 6.7076e-01 ± 1.0725e+00 |
|:-----:|:-----:|:----:|
| ablation_mild | 10 | 6.5695e-01 ± 9.6006e-01 |
| mild_crosskl | 10 | 8.5855e-01 ± 1.1252e+00 |
| mild_sophia_onlycrosskl | 10 | 6.6621e-01 ± 1.0662e+00 |
| mild_sophia_addcrosskl | 10 | 7.0181e-01 ± 1.1109e+00 |