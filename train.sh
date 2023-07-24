CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_1/beta10_ --beta 10. --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_1/beta10_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_1/beta05_ --beta 5.0 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_1/beta05_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_1/beta01_ --beta 1.0 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_1/beta01_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_1/beta0_5 --beta 0.5 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_1/beta0_5.txt &

CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_2/beta10_ --cov-cond --beta 10. --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_2/beta10_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_2/beta05_ --cov-cond --beta 5.0 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_2/beta05_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_2/beta01_ --cov-cond --beta 1.0 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_2/beta01_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v2_2/beta0_5 --cov-cond --beta 0.5 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v2_2/beta0_5.txt &


CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_1/beta10_ --beta 10. --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_1/beta10_.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_1/beta05_ --beta 5.0 --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_1/beta05_.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_1/beta01_ --beta 1.0 --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_1/beta01_.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_1/beta0_5 --beta 0.5 --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_1/beta0_5.txt &

CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_2/beta10_ --cov-cond --beta 10. --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_2/beta10_.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_2/beta05_ --cov-cond --beta 5.0 --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_2/beta05_.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_2/beta01_ --cov-cond --beta 1.0 --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_2/beta01_.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v3_2/beta0_5 --cov-cond --beta 0.5 --variant 2 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v3_2/beta0_5.txt &


CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v1/beta10_ --beta 10. --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v1/beta10_.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v1/beta05_ --beta 5.0 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v1/beta05_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v1/beta01_ --beta 1.0 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v1/beta01_.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation_beta/v1/beta0_5 --beta 0.5 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation_beta/v1/beta0_5.txt &
