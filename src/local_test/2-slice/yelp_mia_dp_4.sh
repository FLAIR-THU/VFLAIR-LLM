#!/bin/bash
#SBATCH --job-name yelp_dp_4         # 任务名叫 example
#SBATCH --gres gpu:a100:1                  # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00    

for seed in {60,61,62,63,64,65}
    do  
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_wo_4

    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_dp_4

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/yelp_mia_dp_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_dp_4

    # 90
    sed -i 's/"epsilon": 70/"epsilon": 90/g' ./configs/2-slice/yelp_mia_dp_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_dp_4

    # 100
    sed -i 's/"epsilon": 90/"epsilon": 100/g' ./configs/2-slice/yelp_mia_dp_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_dp_4

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/yelp_mia_dp_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_dp_4

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/yelp_mia_dp_4.json
done