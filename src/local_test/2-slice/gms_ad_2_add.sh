#!/bin/bash
#SBATCH --job-name gms_mia_ad_2_add          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_mia_ad_2_add.out

for seed in {7,8,9,10,11}
    do
    # 1.0
    python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_ad_2_add

done
