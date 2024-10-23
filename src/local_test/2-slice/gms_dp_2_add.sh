#!/bin/bash
#SBATCH --job-name gms_dp_2_add           # 任务名叫 example
#SBATCH --gres gpu:a100:3                  # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完


for seed in {10,11,12}
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_1000
  
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_700

    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_5000

done