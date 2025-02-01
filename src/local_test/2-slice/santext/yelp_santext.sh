#!/bin/bash
#SBATCH --job-name yelp_santext           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_santext.out

for seed in 60 61 62 63
    do

    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/santext/yelp.json
    python main_pipeline_llm.py --prefix "santext" --seed $seed --configs 2-slice/santext/yelp


    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/santext/yelp.json
    python main_pipeline_llm.py --prefix "santext" --seed $seed --configs 2-slice/santext/yelp

    # 1
    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/santext/yelp.json
    python main_pipeline_llm.py --prefix "santext" --seed $seed --configs 2-slice/santext/yelp

    # 5
    sed -i 's/"epsilon": 1/"epsilon": 5/g' ./configs/2-slice/santext/yelp.json
    python main_pipeline_llm.py --prefix "santext" --seed $seed --configs 2-slice/santext/yelp

    sed -i 's/"epsilon": 5/"epsilon": 1/g' ./configs/2-slice/santext/yelp.json


done
