#!/bin/bash
#SBATCH --job-name lia/yelp_ad_3           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_ad_3.out


for seed in 60 61 10
    do

    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_wo_new

  
done
