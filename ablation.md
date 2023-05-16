# VAE Ablation Results

Averaged over 5 trials
MSE - Euclidean distance between predicted and ground truth arm locations summed over joints, mean over window \& trajectory

<!-- $\frac{\sum_{i=1}^{|\mathcal{D}|} \sum_{t=1}^{T_i} \sum_{j=1}^{4} ||\hat{x}[t,j] - x[t,j]}{\sum_{i=1}^{|\mathcal{D}|} T_i}$ -->

## $\Sigma_{VAE} + \Sigma_{HSMM}$

### Window Mean

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| vae | 3 | 2.0810e-02 ± 3.1458e-02 |
| mild | 3 | 2.1523e-02 ± 3.5733e-02 |
| vae | 5 | 1.7719e-02 ± 2.4877e-02 |
| mild | 5 | 1.9157e-02 ± 2.6650e-02 |
| vae | 8 | 1.7371e-02 ± 2.4212e-02 |
| mild | 8 | 1.6777e-02 ± 2.4033e-02 |
| vae | 10 | 1.7098e-02 ± 2.3546e-02 |
| mild | 10 | 1.6424e-02 ± 2.4002e-02 |

### Window Sum

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| vae | 3 | 8.3238e-01 ± 1.2583e+00 |
| mild | 3 | 8.6093e-01 ± 1.4293e+00 |
| vae | 5 | 7.0876e-01 ± 9.9508e-01 |
| mild | 5 | 7.6629e-01 ± 1.0660e+00 |
| vae | 8 | 6.9484e-01 ± 9.6848e-01 |
| mild | 8 | 6.7108e-01 ± 9.6133e-01 |
| vae | 10 | 6.8391e-01 ± 9.4185e-01 |
| mild | 10 | 6.5695e-01 ± 9.6006e-01 |

## $\Sigma_{HSMM}$

### Window Mean

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| vae | 3 | 2.0810e-02 ± 3.1459e-02 |
| mild | 3 | 2.1536e-02 ± 3.5740e-02 |
| vae | 5 | 1.7720e-02 ± 2.4878e-02 |
| mild | 5 | 1.9181e-02 ± 2.6674e-02 |
| vae | 8 | 1.7380e-02 ± 2.4240e-02 |
| mild | 8 | 1.6856e-02 ± 2.4161e-02 |
| vae | 10 | 1.7111e-02 ± 2.3591e-02 |
| mild | 10 | 1.6562e-02 ± 2.4134e-02 |

### Window Sum

| Model | Z Dim | MSE |
|:-----:|:-----:|:----:|
| vae | 3 | 8.3241e-01 ± 1.2584e+00 |
| mild | 3 | 8.6145e-01 ± 1.4296e+00 |
| vae | 5 | 7.0881e-01 ± 9.9514e-01 |
| mild | 5 | 7.6722e-01 ± 1.0670e+00 |
| vae | 8 | 6.9520e-01 ± 9.6960e-01 |
| mild | 8 | 6.7423e-01 ± 9.6643e-01 |
| vae | 10 | 6.8444e-01 ± 9.4364e-01 |
| mild | 10 | 6.6250e-01 ± 9.6536e-01 |
