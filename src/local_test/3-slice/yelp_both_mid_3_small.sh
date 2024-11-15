#!/bin/bash
#SBATCH --job-name yelp_both_mid_3_small           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_both_mid_3_small.out

for seed in {61,62,63,64,65,1,2,3}
    do
  
    # 0.01
    sed -i 's/"lambda": 0.5/"lambda": 0.01/g' ./configs/3-slice/yelp_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/yelp_both_mid_3_small

    # 1e-3
    sed -i 's/"lambda": 0.01/"lambda": 1e-3/g' ./configs/3-slice/yelp_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/yelp_both_mid_3_small

    # 1e-4
    sed -i 's/"lambda": 1e-3/"lambda": 1e-4/g' ./configs/3-slice/yelp_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/yelp_both_mid_3_small

    # 1e-5
    sed -i 's/"lambda": 1e-4/"lambda": 1e-5/g' ./configs/3-slice/yelp_both_mid_3_small.json
    # python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/yelp_both_mid_3_small

    # 1e-6
    sed -i 's/"lambda": 1e-5/"lambda": 1e-6/g' ./configs/3-slice/yelp_both_mid_3_small.json
    # python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/yelp_both_mid_3_small

    # 1e-9
    sed -i 's/"lambda": 1e-6/"lambda": 1e-9/g' ./configs/3-slice/yelp_both_mid_3_small.json
    # python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/yelp_both_mid_3_small


    sed -i 's/"lambda": 1e-9/"lambda": 0.5/g' ./configs/3-slice/yelp_both_mid_3_small.json

done
