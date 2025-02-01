#!/bin/bash
#SBATCH --job-name gms4           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms4.out
#SBATCH --mem 160000MB


for seed in 61 62 63 64
    do

    # 10
    sed -i 's/"epsilon": 1/"epsilon": 10/g' ./configs/2-slice/dpt/gms4.json
    # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms4

    # 15
    sed -i 's/"epsilon": 10/"epsilon": 15/g' ./configs/2-slice/dpt/gms4.json
    # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms4

    # 20
    sed -i 's/"epsilon": 15/"epsilon": 20/g' ./configs/2-slice/dpt/gms4.json
    # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms4

    # 30
    sed -i 's/"epsilon": 20/"epsilon": 30/g' ./configs/2-slice/dpt/gms4.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms4

    # 100
    sed -i 's/"epsilon": 30/"epsilon": 100/g' ./configs/2-slice/dpt/gms4.json
    # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms4

    sed -i 's/"epsilon": 100/"epsilon": 1/g' ./configs/2-slice/dpt/gms4.json

done
