#!/bin/bash
#SBATCH --job-name yelp_both_wo_3           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_both_wo_3.out

for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/yelp_both_wo_3

 
done
