#!/bin/bash
#SBATCH --job-name alpaca_ran_           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_ran_.out
#SBATCH --mem 160000MB


for seed in 60 61 62 63 64
    do

    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/dpt/alpaca_ran_.json
    # python main_pipeline_llm.py --prefix "rantext_new" --seed $seed --configs 2-slice/dpt/alpaca_ran_


    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/dpt/alpaca_ran_.json
    # python main_pipeline_llm.py --prefix "rantext_new" --seed $seed --configs 2-slice/dpt/alpaca_ran_

    # 1
    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/dpt/alpaca_ran_.json
    # python main_pipeline_llm.py --prefix "rantext_new" --seed $seed --configs 2-slice/dpt/alpaca_ran_

    # 3
    sed -i 's/"epsilon": 1/"epsilon": 3/g' ./configs/2-slice/dpt/alpaca_ran_.json
    python main_pipeline_llm.py --prefix "rantext_new" --seed $seed --configs 2-slice/dpt/alpaca_ran_

    # 5
    sed -i 's/"epsilon": 3/"epsilon": 5/g' ./configs/2-slice/dpt/alpaca_ran_.json
    python main_pipeline_llm.py --prefix "rantext_new" --seed $seed --configs 2-slice/dpt/alpaca_ran_

    # 10
    sed -i 's/"epsilon": 5/"epsilon": 10/g' ./configs/2-slice/dpt/alpaca_ran_.json
    python main_pipeline_llm.py --prefix "rantext_new" --seed $seed --configs 2-slice/dpt/alpaca_ran_


    sed -i 's/"epsilon": 10/"epsilon": 1/g' ./configs/2-slice/dpt/alpaca_ran_.json

done
