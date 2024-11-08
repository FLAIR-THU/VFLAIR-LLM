#!/bin/bash
#SBATCH --job-name gms_lia_mid_1           # 任务名叫 example
#SBATCH --gres gpu:a100:4                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_lia_mid_1.out


for seed in {60,61}
    do
    # lr=1e-5
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_mid_1

    # lr=5e-6
    sed -i 's/"lr": 1e-5/"lr": 5e-6/g' ./configs/3-slice/gms_lia_mid_1.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_mid_1

    sed -i 's/"lr": 5e-6/"lr": 1e-5/g' ./configs/3-slice/gms_lia_mid_1.json


done
