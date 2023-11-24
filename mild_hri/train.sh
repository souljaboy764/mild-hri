# Human-Human Interaction settings

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h5/trial0 --latent-dim 3 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h5/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h5/trial1 --latent-dim 3 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h5/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h5/trial2 --latent-dim 3 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h5/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h5/trial3 --latent-dim 3 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h5/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h6/trial0 --latent-dim 3 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h6/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h6/trial1 --latent-dim 3 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h6/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h6/trial2 --latent-dim 3 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h6/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h6/trial3 --latent-dim 3 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h6/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h7/trial0 --latent-dim 3 --ssm-components 7 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h7/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h7/trial1 --latent-dim 3 --ssm-components 7 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h7/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h7/trial2 --latent-dim 3 --ssm-components 7 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h7/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z3h7/trial3 --latent-dim 3 --ssm-components 7 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z3h7/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h7/trial0 --latent-dim 3 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h7/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h7/trial1 --latent-dim 3 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h7/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h7/trial2 --latent-dim 3 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h7/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h7/trial3 --latent-dim 3 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h7/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h5/trial0 --latent-dim 5 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h5/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h5/trial1 --latent-dim 5 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h5/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h5/trial2 --latent-dim 5 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h5/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h5/trial3 --latent-dim 5 --ssm-components 5 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h5/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/2023/unsupervised/bp_hh/trial0 --latent-dim 5 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-3 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/2023/unsupervised/bp_hh/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/2023/unsupervised/bp_hh/trial1 --latent-dim 5 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-3 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/2023/unsupervised/bp_hh/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/2023/unsupervised/bp_hh/trial2 --latent-dim 5 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-3 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/2023/unsupervised/bp_hh/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/2023/unsupervised/bp_hh/trial3 --latent-dim 5 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-3 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/2023/unsupervised/bp_hh/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h10/trial0 --latent-dim 5 --ssm-components 10 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 100 >> logs/092023/bp_hh_20hz/z5h10/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h10/trial1 --latent-dim 5 --ssm-components 10 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 100 >> logs/092023/bp_hh_20hz/z5h10/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h10/trial2 --latent-dim 5 --ssm-components 10 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 100 >> logs/092023/bp_hh_20hz/z5h10/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h10/trial3 --latent-dim 5 --ssm-components 10 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 100 >> logs/092023/bp_hh_20hz/z5h10/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h8/trial0 --latent-dim 5 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h8/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h8/trial1 --latent-dim 5 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h8/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h8/trial2 --latent-dim 5 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h8/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --results logs/092023/bp_hh_20hz/z5h8/trial3 --latent-dim 5 --ssm-components 8 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage --src ./data/buetepage/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/bp_hh_20hz/z5h8/trial3.txt &


CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --results logs/092023/nuisiv2_hh/z5h6/trial0 --ckpt logs/2023/unsupervised/bp_hh/trial1/models/050.pth --latent-dim 5 --ssm-components 6 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-3 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset nuisi --src ./data/nuisi/traj_data.npz  --model VAE --ssm HMM --hidden-sizes 40 20 --epochs 400 >> logs/092023/nuisiv2_hh/z5h6/trial0.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/092023/nuisiv2_pepper/z5h6/v1_1/trial0 --ckpt-h logs/092023/nuisiv2_hh/z5h6/trial0/models/050.pth --num-joints 4 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --downsample 1 --window-size 5 --mce-samples 10 --dataset nuisi_pepper --src ./data/nuisi/traj_data.npz  --epochs 400 >> logs/092023/nuisiv2_pepper/z5h6/v1_1/trial0.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/092023/nuisiv2_pepper/z5h6/v2_1/trial0 --ckpt-h logs/092023/nuisiv2_hh/z5h6/trial0/models/050.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 1 --window-size 5 --mce-samples 10 --dataset nuisi_pepper --src ./data/nuisi/traj_data.npz  --epochs 400 >> logs/092023/nuisiv2_pepper/z5h6/v2_1/trial0.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/092023/nuisiv2_pepper/z5h6/v2_2/trial0 --ckpt-h logs/092023/nuisiv2_hh/z5h6/trial0/models/050.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 1 --window-size 5 --mce-samples 10 --dataset nuisi_pepper --src ./data/nuisi/traj_data.npz  --epochs 400 --cov-cond >> logs/092023/nuisiv2_pepper/z5h6/v2_2/trial0.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/092023/nuisiv2_pepper/z5h6/v3_1/trial0 --ckpt-h logs/092023/nuisiv2_hh/z5h6/trial0/models/050.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 1 --window-size 5 --mce-samples 10 --dataset nuisi_pepper --src ./data/nuisi/traj_data.npz  --epochs 400 >> logs/092023/nuisiv2_pepper/z5h6/v3_1/trial0.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/092023/nuisiv2_pepper/z5h6/v3_2/trial0 --ckpt-h logs/092023/nuisiv2_hh/z5h6/trial0/models/050.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 1 --window-size 5 --mce-samples 10 --dataset nuisi_pepper --src ./data/nuisi/traj_data.npz  --epochs 400 --cov-cond >> logs/092023/nuisiv2_pepper/z5h6/v3_2/trial0.txt &

# Human-Robot Interaction settings

mkdir -p logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7
mkdir -p logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7
mkdir -p logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7
mkdir -p logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7
mkdir -p logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial0 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial1 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial2 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial3 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v1_1/z5h7/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial0 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial1 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial2 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial3 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_1/z5h7/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial0 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial1 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial2 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial3 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 2 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v2_2/z5h7/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial0 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial1 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial2 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial3 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_1/z5h7/trial3.txt &

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial0 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial1 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial2 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hr.py --results logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial3 --ckpt-h logs/092023/bp_hh_20hz/z5h7/trial2/models/399.pth --num-joints 4 --beta 0.005 --variant 3 --grad-clip 0.0 --lr 5e-4 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset buetepage_pepper --src ./data/buetepage/traj_data.npz  --cov-cond --epochs 400 >> logs/2023/nuisiv2_3joints_xvel/v3_2/z5h7/trial3.txt &

# Human-Human Interaction settings for Handover dataset

CUDA_VISIBLE_DEVICES=0 nohup python3 mild_hri/train_hh.py --num-joints 6 --results logs/alap_hh/trial0 --latent-dim 10 --ssm-components 3 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-2 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset alap --src /home/vignesh/playground/rmdn_hri/data/alap_dataset_combined.npz  --model VAE --ssm HMM --hidden-sizes 80 40 --epochs 400 >> logs/alap_hh/trial0.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 mild_hri/train_hh.py --num-joints 6 --results logs/alap_hh/trial1 --latent-dim 10 --ssm-components 3 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-2 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset alap --src /home/vignesh/playground/rmdn_hri/data/alap_dataset_combined.npz  --model VAE --ssm HMM --hidden-sizes 80 40 --epochs 400 >> logs/alap_hh/trial1.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 mild_hri/train_hh.py --num-joints 6 --results logs/alap_hh/trial2 --latent-dim 10 --ssm-components 3 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-2 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset alap --src /home/vignesh/playground/rmdn_hri/data/alap_dataset_combined.npz  --model VAE --ssm HMM --hidden-sizes 80 40 --epochs 400 >> logs/alap_hh/trial2.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 mild_hri/train_hh.py --num-joints 6 --results logs/alap_hh/trial3 --latent-dim 10 --ssm-components 3 --beta 0.005 --variant 1 --grad-clip 0.0 --lr 5e-4 --cov-reg 5e-2 --downsample 0.2 --window-size 5 --mce-samples 10 --dataset alap --src /home/vignesh/playground/rmdn_hri/data/alap_dataset_combined.npz  --model VAE --ssm HMM --hidden-sizes 80 40 --epochs 400 >> logs/alap_hh/trial3.txt &