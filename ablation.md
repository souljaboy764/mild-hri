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
