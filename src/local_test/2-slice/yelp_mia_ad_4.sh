#!/bin/bash
#SBATCH --job-name yelp_ad_4          # 任务名叫 example
#SBATCH --gres gpu:a100:1                  # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00   

for seed in {60,61,62,63,64,65}
    do
    # 0.001
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_4

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/2-slice/yelp_mia_ad_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_4

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/2-slice/yelp_mia_ad_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_4

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/2-slice/yelp_mia_ad_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_4

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/2-slice/yelp_mia_ad_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_4

    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/2-slice/yelp_mia_ad_4.json

done
