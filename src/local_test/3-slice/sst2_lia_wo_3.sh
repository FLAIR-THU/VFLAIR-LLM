#!/bin/bash
#SBATCH --job-name sst2_wo_lia           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_wo_lia_3_1.out


for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_wo_3_1
done

for seed in {10,11,12,13}
    do
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_wo_3_1
done