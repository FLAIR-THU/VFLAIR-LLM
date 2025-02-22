#!/bin/bash
#SBATCH --job-name yelp_lora_mid1_           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_lora_mid1_.out

# 0.0001
sed -i 's/"lambda": 0.5/"lambda": 0.0001/g' ./configs/3-slice/both_lora/yelp_mid_1_.json
python main_pipeline_llm.py --prefix "both_lora_add" --seed 65 --configs 3-slice/both_lora/yelp_mid_1_

# 1e-5
sed -i 's/"lambda": 0.0001/"lambda": 1e-5/g' ./configs/3-slice/both_lora/yelp_mid_1_.json
python main_pipeline_llm.py --prefix "both_lora_add" --seed 65 --configs 3-slice/both_lora/yelp_mid_1_

sed -i 's/"lambda": 1e-5/"lambda": 0.5/g' ./configs/3-slice/both_lora/yelp_mid_1_.json


for seed in 64 63
    do
    # 0.5
    # python main_pipeline_llm.py --prefix "both_lora_add" --seed $seed --configs 3-slice/both_lora/yelp_mid_1_

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/both_lora/yelp_mid_1_.json
    # python main_pipeline_llm.py --prefix "both_lora_add" --seed $seed --configs 3-slice/both_lora/yelp_mid_1_

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/3-slice/both_lora/yelp_mid_1_.json
    # python main_pipeline_llm.py --prefix "both_lora_add" --seed $seed --configs 3-slice/both_lora/yelp_mid_1_

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/3-slice/both_lora/yelp_mid_1_.json
    python main_pipeline_llm.py --prefix "both_lora_add" --seed $seed --configs 3-slice/both_lora/yelp_mid_1_

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/3-slice/both_lora/yelp_mid_1_.json
    python main_pipeline_llm.py --prefix "both_lora_add" --seed $seed --configs 3-slice/both_lora/yelp_mid_1_

    # 1e-5
    sed -i 's/"lambda": 0.0001/"lambda": 1e-5/g' ./configs/3-slice/both_lora/yelp_mid_1_.json
    python main_pipeline_llm.py --prefix "both_lora_add" --seed $seed --configs 3-slice/both_lora/yelp_mid_1_

    sed -i 's/"lambda": 1e-5/"lambda": 0.5/g' ./configs/3-slice/both_lora/yelp_mid_1_.json

done
