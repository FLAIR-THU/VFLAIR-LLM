#!/bin/bash
#SBATCH --job-name yelp_ad_1           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_ad_1.out


for seed in 65 64
    do

    # 0.1
    sed -i 's/"lambda": 0.001/"lambda": 0.1/g' ./configs/3-slice/both_lora/yelp_ad_1.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_1


    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/3-slice/both_lora/yelp_ad_1.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_1

    # 0.001
    sed -i 's/"lambda": 0.1/"lambda": 0.001/g' ./configs/3-slice/both_lora/yelp_ad_1.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_1

    
done
