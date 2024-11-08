#!/bin/bash
#SBATCH --job-name gms_lia_wo_1_fromraw           # 任务名叫 example
#SBATCH --gres gpu:a100:4                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_lia_wo_1_fromraw.out

for seed in {60,61}
    do
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_wo_1_fromraw
done
