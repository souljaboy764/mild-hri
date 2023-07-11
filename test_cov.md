# Pepper Buetepage

## without hand waving

Model                       MSE                         Reconstruction                              Input cov during Conditioning
mild_vanilla                3.3662e-02 ± 4.8302e-02     AE                                          only HSMM
mild_vanilla with covcond   3.4058e-02 ± 4.8561e-02     AE                                          HSMM + Full
mild_vanilla_onlycovcond    3.3608e-02 ± 4.7621e-02     AE                                          only Full
mild_crossrecon_nocovcond   2.0180e-02 ± 3.4425e-02     AE and HSMM cov conditioned                 only HSMM
mild_crossrecon_covcond     2.3818e-02 ± 4.0188e-02     AE and HSMM cov + Full cov conditioned      HSMM + Full

vae_vanilla                 3.4147e-02 ± 4.8365e-02     AE                                          HSMM
vae_vanilla with covcond    3.5461e-02 ± 5.0352e-02     AE                                          HSMM + Diag
vae_vanilla_onlycovcond     3.5032e-02 ± 4.9129e-02     AE                                          only Diag
vae_crossrecon_nocovcond    2.0811e-02 ± 3.1877e-02     AE and HSMM cov conditioned                 only HSMM
vae_crossrecon_covcond      2.1926e-02 ± 3.1960e-02     AE and HSMM cov + Diag cov conditioned      HSMM + Diag
