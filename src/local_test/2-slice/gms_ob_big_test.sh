#!/bin/bash
#SBATCH --job-name gms_250           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_250.out
#SBATCH --mem 160000MB


# 250 

# 1000
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/ob/gms_big_test

# 1e4
sed -i 's/"epsilon": 1000/"epsilon": 1e4/g' ./configs/2-slice/ob/gms_big_test.json
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/ob/gms_big_test

# 1e5
sed -i 's/"epsilon": 1e4/"epsilon": 1e5/g' ./configs/2-slice/ob/gms_big_test.json
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/ob/gms_big_test


sed -i 's/"epsilon": 1e5/"epsilon": 1000/g' ./configs/2-slice/ob/gms_big_test.json

