#!/bin/bash
#SBATCH --job-name lia_yelp_gs_           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_gs_.out


for seed in 60 61 62 63 64 65
    do

    
    # 98
    sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 98/g' ./configs/3-slice/lia/yelp_gs_.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_gs_

    # 97
    sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/3-slice/lia/yelp_gs_.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_gs_

    # 96
    sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/3-slice/lia/yelp_gs_.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_gs_

    # 95
    sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/3-slice/lia/yelp_gs_.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/lia/yelp_gs_


    sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/3-slice/lia/yelp_gs_.json

done
