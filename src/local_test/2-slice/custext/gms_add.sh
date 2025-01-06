#!/bin/bash
#SBATCH --job-name gms_add_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_add_custext.out


for seed in 60 61 62
    do


    # 5
    sed -i 's/"epsilon": 1/"epsilon": 5/g' ./configs/2-slice/custext/gms_add.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/gms_add

    # 15
    sed -i 's/"epsilon": 5/"epsilon": 15/g' ./configs/2-slice/custext/gms_add.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/gms_add

    sed -i 's/"epsilon": 15/"epsilon": 1/g' ./configs/2-slice/custext/gms_add.json


done
