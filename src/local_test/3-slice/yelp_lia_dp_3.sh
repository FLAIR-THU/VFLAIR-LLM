#!/bin/bash
#SBATCH --job-name yelp_dp_lia_3           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_dp_lia_3.out

# 1e9
sed -i 's/"epsilon": 50/"epsilon": 1e9/g' ./configs/3-slice/yelp_lia_dp_3.json
python main_pipeline_llm_LIA.py --seed 62 --configs 3-slice/yelp_lia_dp_3

sed -i 's/"epsilon": 1e9/"epsilon": 50/g' ./configs/3-slice/yelp_lia_dp_3.json


for seed in {63,64,65}
    do
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_wo_3

    # 1e5
    sed -i 's/"epsilon": 50/"epsilon": 1e5/g' ./configs/3-slice/yelp_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_dp_3

    # 1e6
    sed -i 's/"epsilon": 1e5/"epsilon": 1e6/g' ./configs/3-slice/yelp_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_dp_3

    # 1e7
    sed -i 's/"epsilon": 1e6/"epsilon": 1e7/g' ./configs/3-slice/yelp_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_dp_3

    # 1e8
    sed -i 's/"epsilon": 1e7/"epsilon": 1e8/g' ./configs/3-slice/yelp_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_dp_3

    # 1e9
    sed -i 's/"epsilon": 1e8/"epsilon": 1e9/g' ./configs/3-slice/yelp_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_dp_3

    sed -i 's/"epsilon": 1e9/"epsilon": 50/g' ./configs/3-slice/yelp_lia_dp_3.json

done

for seed in {10,11,12,13}
    do
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_wo_3
done