#!/bin/bash
#SBATCH --job-name sst2_1           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_1.out
#SBATCH --mem 160000MB


for seed in 60 61 62 63 64
    do


    # 100
    sed -i 's/"epsilon": 1/"epsilon": 100/g' ./configs/2-slice/dpt/sst2_1.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/sst2_1

    # 1000
    sed -i 's/"epsilon": 100/"epsilon": 1000/g' ./configs/2-slice/dpt/sst2_1.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/sst2_1

    # 1e4
    sed -i 's/"epsilon": 1000/"epsilon": 1e4/g' ./configs/2-slice/dpt/sst2_1.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/sst2_1

    sed -i 's/"epsilon": 1e4/"epsilon": 1/g' ./configs/2-slice/dpt/sst2_1.json

done
