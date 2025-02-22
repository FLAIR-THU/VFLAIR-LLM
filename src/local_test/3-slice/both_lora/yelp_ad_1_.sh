#!/bin/bash
#SBATCH --job-name yelp_ad_1_           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_ad_1_.out
#SBATCH --mem 100000MB                

for seed in 65 64
    do

    
    
    # 5
    sed -i 's/"lambda": 0.001/"lambda": 5.0/g' ./configs/3-slice/both_lora/yelp_ad_1_.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_1_

    # 1
    sed -i 's/"lambda": 5.0/"lambda": 1.0/g' ./configs/3-slice/both_lora/yelp_ad_1_.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_1_


    sed -i 's/"lambda": 1.0/"lambda": 0.001/g' ./configs/3-slice/both_lora/yelp_ad_1_.json
    
done
