#!/bin/bash
#SBATCH --job-name alpaca_original_dpt           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_original_dpt.out
#SBATCH --mem 160000MB


for seed in 1 2 3 4 5
    do

    # 0.00001
    sed -i 's/"epsilon": 1/"epsilon": 0.00001/g' ./configs/2-slice/dpt/alpaca_original.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/alpaca_original

    # 0.0001
    sed -i 's/"epsilon": 0.00001/"epsilon": 0.0001/g' ./configs/2-slice/dpt/alpaca_original.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/alpaca_original

    # 0.001
    sed -i 's/"epsilon": 0.0001/"epsilon": 0.001/g' ./configs/2-slice/dpt/alpaca_original.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/alpaca_original

    # 0.01
    sed -i 's/"epsilon": 0.001/"epsilon": 0.01/g' ./configs/2-slice/dpt/alpaca_original.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/alpaca_original

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/dpt/alpaca_original.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/alpaca_original


    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/dpt/alpaca_original.json

done
