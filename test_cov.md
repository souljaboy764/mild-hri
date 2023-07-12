# Pepper Buetepage

## without hand waving

Model                       MSE                         Reconstruction                                              Input cov during Conditioning
mild_vanilla                3.4058e-02 ± 4.8561e-02     posterior mean                                              only HSMM
mild_vanilla with covcond   3.4058e-02 ± 4.8561e-02     posterior mean                                              HSMM + Full
mild_vanilla_onlycovcond    3.3608e-02 ± 4.7621e-02     posterior mean                                              only Full
mild_crossrecon_nocovcond   2.2037e-02 ± 3.5099e-02     posterior mean and HSMM cov conditioned mean                only HSMM
mild_crossrecon_covcond     2.4946e-02 ± 3.8571e-02     posterior mean and HSMM cov + Full cov conditioned mean     HSMM + Full

vae_vanilla                 3.4147e-02 ± 4.8365e-02     posterior mean                                              HSMM
vae_vanilla with covcond    3.5461e-02 ± 5.0352e-02     posterior mean                                              HSMM + Diag
vae_vanilla_onlycovcond     3.5032e-02 ± 4.9129e-02     posterior mean                                              only Diag
vae_crossrecon_nocovcond    2.0811e-02 ± 3.1877e-02     posterior mean and HSMM cov conditioned mean                only HSMM
vae_crossrecon_covcond      2.1926e-02 ± 3.1960e-02     posterior mean and HSMM cov + Diag cov conditioned mean     HSMM + Diag



mild_vanilla False      1.7696e-01 ± 2.4886e-01         3.8383e-02 ± 5.9870e-02
mild_crossrecon_nocovcond False         1.2509e-01 ± 1.6022e-01         7.4698e-02 ± 1.0581e-01
mild_crossrecon_covcond True    1.3258e-01 ± 1.6776e-01         7.9680e-02 ± 1.2056e-01
mild_crossrecon_samplecond False        1.0734e-01 ± 1.6924e-01         5.6105e-02 ± 8.5978e-02
vae_vanilla False       1.8111e-01 ± 2.4859e-01         4.5384e-02 ± 6.7237e-02
vae_crossrecon_nocovcond False  1.0470e-01 ± 1.3238e-01         5.0419e-02 ± 7.1895e-02
vae_crossrecon_covcond True     1.1582e-01 ± 1.3811e-01         5.5158e-02 ± 7.5922e-02
vae_crossrecon_samplecond False         1.0625e-01 ± 1.6096e-01         4.9654e-02 ± 6.8719e-02
