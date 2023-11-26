# MILD-HRI

Repository for the work "MILD: Multimodal Interactive Latent Dynamics for Learning Human-Robot Interaction"

## Requirements

Install the requirements in [`requirements.txt`](requirements.txt) by running

```bash
pip install -r requirements.txt
```

Clone the repo `https://github.com/souljaboy764/phd_utils` and follow the installation instructions in its README. This repository has the datasets to be used already preprocessed.

Clone the pbdlib repositories:

Pbdlib Python:

```bash
git clone https://gitlab.idiap.ch/rli/pbdlib-python
git clone https://gist.github.com/5d551c432d4a4ebf1433615595cfd87d.git
cd pbdlib-python
git checkout d211d86fa81b50cadec63c5135c312ece861e508
cd ..
patch --strip=1 --directory=pbdlib-python/ < 5d551c432d4a4ebf1433615595cfd87d/pbdlib_python3.patch
cd pbdlib-python
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -e .
```

Pbdlib PyTorch:

```bash
git clone https://git.ias.informatik.tu-darmstadt.de/prasad/pbdlib-torch
cd pbdlib-torch
pip install -e .
```

## Training

First the VAE and HMMs are trained on the Human-Human dataset, after which the human VAE and the latent HMM are frozen and then the robot VAE is trained.

### Human-Human Interaction Configurations

Buetepage Dataset:

```bash
python train_hh.py --results path/to/buetepage_hh_results --latent-dim 5 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-3 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 --num-joints 3
```

NuiSI Dataset:

```bash
python train_hh.py --results path/to/nuisi_hh_results --ckpt path/to/buetepage_ckpt.pth --latent-dim 5 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-3 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset nuisi  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 --num-joints 3
```

### Human-Robot Interaction Configurations

In the HRI scenarios, we additionally train the conditional generation of the robot motions based on the human's observations. There are 3 modes in training the HRI models with [`train_hr.py`](train_hr.py):
1 - Vanilla approach of just training the Robot VAE as is with the HMM prior
2 - Sampling from the prior and then conditioning the samples
3 - Conditioning the prior distribution and then sampling from the conditioned distribution.
This can be set using the `--variant` option.
Additionally, to choose whether to use the VAE posterior covariance in the conditioning, this can be set using the `--cov-cond` flag.

In the paper, we show totally 5 variants:
v1 - Vanilla approach of just training the Robot VAE as is with the HMM prior: `--variant 1`
v2.1 - Sampling from the prior and then conditioning the samples without the posterior covariance: `--variant 2`
v2.2 - Sampling from the prior and then conditioning the samples with the posterior covariance: `--variant 2 --cov-cond`
v3.1 - Conditioning the prior distribution and then sampling from the conditioned distribution without the posterior covariance: `--variant 3`
v3.2 - Conditioning the prior distribution and then sampling from the conditioned distribution with the posterior covariance: `--variant 3 --cov-cond`

Buetepage Pepper Dataset:

```bash
python train_hr.py --results path/to/buetepage_pepperr_results --ckpt-h path/to/buetepage_hh_results/XXXX.pth --num-joints 4 --beta 0.005 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --epochs 400 --variant V [--cov-cond]
```

Buetepage Yumi Dataset:

```bash
python train_hr.py --results path/to/buetepage_yumi_results --ckpt-h path/to/buetepage_hh_results/XXXX.pth --num-joints 4 --beta 0.005 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_yumi --epochs 400 --variant V [--cov-cond]
```

NuiSI Pepper Dataset:

```bash
python train_hr.py --results path/to/nuisi_pepper_results --ckpt-h path/to/nuisi_hh_results/XXXX.ckpt --num-joints 4 --beta 0.005 --grad-clip 0.0 --lr 5e-4 --downsample 1 --window-size 5 --mce-samples 10 --dataset nuisi_pepper --epochs 400  --variant V [--cov-cond]
```
