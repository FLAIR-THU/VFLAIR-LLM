#!/bin/bash
#SBATCH --job-name yelp_lora_ad_           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_lora_ad_.out
#SBATCH --qos high


for seed in 60 61 62 #SBATCH --mem 120000MB                    # 子任务 1 天 1 小时就能跑完
    do

    # 0.001
    # python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/3-slice/both_lora/yelp_ad_.json
    # python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/3-slice/both_lora/yelp_ad_.json
    # python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/3-slice/both_lora/yelp_ad_.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/3-slice/both_lora/yelp_ad_.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/yelp_ad_


    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/3-slice/both_lora/yelp_ad_.json
    
done
