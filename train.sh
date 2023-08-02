CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial0 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 1 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 --dataset buetepage >> logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial0 &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial1 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 1 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 --dataset buetepage >> logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial1 &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial2 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 1 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 --dataset buetepage >> logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial2 &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial3 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 1 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 --dataset buetepage >> logs/2023aug/bp_hh_downsampled/z3h5/v1_0/trial3 &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v3_1 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 3 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 >> logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v3_1.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v4_1 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 4 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 >> logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v4_1.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v2_2 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 2 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 --cov-cond >> logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v2_2.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v3_2 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 3 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 --cov-cond >> logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v3_2.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v4_2 --latent-dim 3 --hsmm-components 5 --beta 5e-3 --variant 4 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-4  --mce-samples 10 --cov-cond >> logs/2023aug/bp_pepper_downsampled_robotfuture/z03h05/gmm_v4_2.txt &


CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial1 --latent-dim 5 --hsmm-components 10 --beta 5e-3 --variant 2 --grad-clip 0.5 --lr 5e-4 --cov-reg 1e-3  --mce-samples 10 >> logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial1.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial2 --latent-dim 5 --hsmm-components 10 --beta 5e-3 --variant 2 --grad-clip 0.5 --lr 5e-4 --cov-reg 1e-3  --mce-samples 10 >> logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial2.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial3 --latent-dim 5 --hsmm-components 10 --beta 5e-3 --variant 2 --grad-clip 0.5 --lr 5e-4 --cov-reg 1e-3  --mce-samples 10 >> logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial3.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial4 --latent-dim 5 --hsmm-components 10 --beta 5e-3 --variant 2 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-3  --mce-samples 10 >> logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial4.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial5 --latent-dim 5 --hsmm-components 10 --beta 5e-3 --variant 2 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-3  --mce-samples 10 >> logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial5.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial6 --latent-dim 5 --hsmm-components 10 --beta 5e-3 --variant 2 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-3  --mce-samples 10 >> logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial6.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial7 --latent-dim 5 --hsmm-components 10 --beta 5e-3 --variant 2 --grad-clip 1.0 --lr 5e-4 --cov-reg 1e-3  --mce-samples 10 >> logs/2023/bp_pepper_downsampled_robotfuture/z05h10/v2_1/trial7.txt &
