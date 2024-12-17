#!/bin/bash
#SBATCH --job-name sst2_ob_2           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_ob_2.out

for seed in 60 61 62 63 64 65
    do

    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/sst2_2

    # 100
    sed -i 's/"cluster_num": 50/"cluster_num": 100/g' ./configs/2-slice/ob/sst2_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/sst2_2

    # 150
    sed -i 's/"cluster_num": 100/"cluster_num": 150/g' ./configs/2-slice/ob/sst2_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/sst2_2

    # 200
    sed -i 's/"cluster_num": 150/"cluster_num": 200/g' ./configs/2-slice/ob/sst2_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/sst2_2

    # 250
    sed -i 's/"cluster_num": 200/"cluster_num": 250/g' ./configs/2-slice/ob/sst2_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/sst2_2

    sed -i 's/"cluster_num": 250/"cluster_num": 50/g' ./configs/2-slice/ob/sst2_2.json

done
