#!/bin/bash
#SBATCH --job-name gms_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_custext.out


# for seed in 60 61
#     do


# 50
sed -i 's/"epsilon": 1/"epsilon": 50/g' ./configs/2-slice/custext/gms_20.json
python main_pipeline_llm_MIA_test.py --seed 60 --configs 2-slice/custext/gms_20

# 30
sed -i 's/"epsilon": 50/"epsilon": 30/g' ./configs/2-slice/custext/gms_20.json
python main_pipeline_llm_MIA_test.py --seed 60 --configs 2-slice/custext/gms_20

# 1
sed -i 's/"epsilon": 30/"epsilon": 1/g' ./configs/2-slice/custext/gms_20.json
python main_pipeline_llm_MIA_test.py --seed 60 --configs 2-slice/custext/gms_20

# 0.01
sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/custext/gms_20.json
python main_pipeline_llm_MIA_test.py --seed 60 --configs 2-slice/custext/gms_20

sed -i 's/"epsilon": 0.01/"epsilon": 1/g' ./configs/2-slice/custext/gms_20.json


