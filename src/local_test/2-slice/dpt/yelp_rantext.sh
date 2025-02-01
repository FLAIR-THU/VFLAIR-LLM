#!/bin/bash
#SBATCH --job-name yelp_ran           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_ran.out
#SBATCH --mem 160000MB


for seed in 60 61 62 63
    do

    # 10
    sed -i 's/"epsilon": 1/"epsilon": 10/g' ./configs/2-slice/dpt/yelp.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/yelp

    # 15
    sed -i 's/"epsilon": 10/"epsilon": 15/g' ./configs/2-slice/dpt/yelp.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/yelp

    # 20
    sed -i 's/"epsilon": 15/"epsilon": 20/g' ./configs/2-slice/dpt/yelp.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/yelp

    # 25
    sed -i 's/"epsilon": 20/"epsilon": 25/g' ./configs/2-slice/dpt/yelp.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/yelp

    # 30
    sed -i 's/"epsilon": 25/"epsilon": 30/g' ./configs/2-slice/dpt/yelp.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/yelp

    sed -i 's/"epsilon": 30/"epsilon": 1/g' ./configs/2-slice/dpt/yelp.json

done
