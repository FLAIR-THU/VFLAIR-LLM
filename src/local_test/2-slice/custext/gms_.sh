#!/bin/bash
#SBATCH --job-name gms_custext_           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_custext_.out


for seed in 1 2 3 4
    do
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/custext/gms_wo
    
    # 30
    sed -i 's/"epsilon": 1/"epsilon": 30/g' ./configs/2-slice/custext/gms_.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/custext/gms_

    # 50
    sed -i 's/"epsilon": 30/"epsilon": 50/g' ./configs/2-slice/custext/gms_.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/custext/gms_

    sed -i 's/"epsilon": 50/"epsilon": 1/g' ./configs/2-slice/custext/gms_.json


done
