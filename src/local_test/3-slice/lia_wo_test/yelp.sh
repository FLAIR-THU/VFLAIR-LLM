#!/bin/bash
#SBATCH --job-name both_wo           # 任务名叫 example
#SBATCH --gres gpu:a100:2               # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/both_wo.out
#SBATCH --mem 80000MB

# lr 1e-5
for seed in 60 61 62 63
    do
    python main_pipeline_llm.py --prefix "lia_wo_test" --seed $seed --configs 3-slice/lia/yelp_wo_new
   
done
