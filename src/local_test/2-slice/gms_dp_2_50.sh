#!/bin/bash
#SBATCH --job-name gms_mia_dp_2_50          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_mia_dp_2_50.out

for seed in {60,61,62,63,64,65}
    do
   
    python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_dp_2_50


done
