#!/bin/bash
#SBATCH --job-name lia/yelp_mid_1           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_mid_1.out
#SBATCH --mem 100000MB

for seed in 65 64
    do
    
    # 0.5
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_mid_1

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/lia/yelp_mid_1.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_mid_1

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/3-slice/lia/yelp_mid_1.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_mid_1

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/3-slice/lia/yelp_mid_1.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_mid_1

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/3-slice/lia/yelp_mid_1.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_mid_1

    # 0.00001
    sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/3-slice/lia/yelp_mid_1.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_mid_1

    sed -i 's/"lambda": 0.00001/"lambda": 0.5/g' ./configs/3-slice/lia/yelp_mid_1.json

done
