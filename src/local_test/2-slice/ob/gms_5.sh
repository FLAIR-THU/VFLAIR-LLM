#!/bin/bash
#SBATCH --job-name gms_1_ob          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_1_ob.out
#SBATCH --mem 160000MB


for seed in 60 61 62 63 64 65
    do 

    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/gms_5

done
