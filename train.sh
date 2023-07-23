CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW00 --variant 1 --grad-clip 5.0 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW00.txt &
CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW01 --variant 1 --grad-clip 1.0 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW01.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW02 --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW02.txt &
CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW03 --variant 1 --grad-clip 0.0 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW03.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW04 --variant 1 --grad-clip 5.0 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW04.txt &
CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW05 --variant 1 --grad-clip 1.0 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW05.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW06 --variant 1 --grad-clip 0.5 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW06.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v1/AdamW07 --variant 1 --grad-clip 0.0 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v1/AdamW07.txt &

CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW00 --cov-cond --variant 1 --grad-clip 5.0 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW00.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW01 --cov-cond --variant 1 --grad-clip 1.0 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW01.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW02 --cov-cond --variant 1 --grad-clip 0.5 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW02.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW03 --cov-cond --variant 1 --grad-clip 0.0 --lr 5e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW03.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW04 --cov-cond --variant 1 --grad-clip 5.0 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW04.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW05 --cov-cond --variant 1 --grad-clip 1.0 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW05.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW06 --cov-cond --variant 1 --grad-clip 0.5 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW06.txt &
CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --results logs/2023/bp_pepper_ablation/v2_2/AdamW07 --cov-cond --variant 1 --grad-clip 0.0 --lr 1e-4 --optimizer AdamW >> logs/2023/bp_pepper_ablation/v2_2/AdamW07.txt &
