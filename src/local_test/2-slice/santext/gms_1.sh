#!/bin/bash
#SBATCH --job-name gms1_santext           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms1_santext.out

for seed in 60 61 62
    do

    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms_1


done
