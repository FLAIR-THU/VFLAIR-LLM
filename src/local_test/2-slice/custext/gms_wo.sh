#!/bin/bash
#SBATCH --job-name gms_wo           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_wo.out


for seed in 63 64 65 66 67
    do
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/custext/gms_wo
done