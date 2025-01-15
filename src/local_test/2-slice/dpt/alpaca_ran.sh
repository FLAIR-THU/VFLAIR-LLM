#!/bin/bash
#SBATCH --job-name alpaca_ran           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_ran.out
#SBATCH --mem 160000MB


for seed in 61 62 63 64
    do

    # 10
    sed -i 's/"epsilon": 1/"epsilon": 10/g' ./configs/2-slice/dpt/alpaca_ran.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/alpaca_ran

    # 15
    sed -i 's/"epsilon": 10/"epsilon": 15/g' ./configs/2-slice/dpt/alpaca_ran.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/alpaca_ran

    # 100
    sed -i 's/"epsilon": 15/"epsilon": 100/g' ./configs/2-slice/dpt/alpaca_ran.json
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/alpaca_ran

    sed -i 's/"epsilon": 100/"epsilon": 1/g' ./configs/2-slice/dpt/alpaca_ran.json

done
