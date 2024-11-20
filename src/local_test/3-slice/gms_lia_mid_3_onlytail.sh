#!/bin/bash
#SBATCH --job-name gms_lia_mid_3_onlytail      # 任务名叫 example
#SBATCH --gres gpu:a100:4                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_lia_mid_3_onlytail.out



# only tail
python main_pipeline_llm_LIA.py --seed 60 --configs 3-slice/gms_lia_mid_3_onlytail 

sed -i 's/"lr": 5e-6/"lr": 1e-5/g' ./configs/3-slice/gms_lia_mid_3_onlytail.json
python main_pipeline_llm_LIA.py --seed 60 --configs 3-slice/gms_lia_mid_3_onlytail

sed -i 's/"lr": 1e-5/"lr": 5e-5/g' ./configs/3-slice/gms_lia_mid_3_onlytail.json
python main_pipeline_llm_LIA.py --seed 60 --configs 3-slice/gms_lia_mid_3_onlytail

sed -i 's/"lr": 5e-5/"lr": 5e-6/g' ./configs/3-slice/gms_lia_mid_3_onlytail.json

