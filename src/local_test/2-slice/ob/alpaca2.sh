#!/bin/bash
#SBATCH --job-name alpaca2_ob          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca2_ob.out
#SBATCH --mem 160000MB


for seed in 61 62 63 64 65
    do 

    # 0.5
    sed -i 's/"epsilon": 1/"epsilon": 0.5/g' ./configs/2-slice/ob/alpaca2.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca2

    # 1
    sed -i 's/"epsilon": 0.5/"epsilon": 1/g' ./configs/2-slice/ob/alpaca2.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca2

    # 2
    sed -i 's/"epsilon": 1/"epsilon": 2/g' ./configs/2-slice/ob/alpaca2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca2

    # 5
    sed -i 's/"epsilon": 2/"epsilon": 5/g' ./configs/2-slice/ob/alpaca2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca2

    # 10
    sed -i 's/"epsilon": 5/"epsilon": 10/g' ./configs/2-slice/ob/alpaca2.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca2

    sed -i 's/"epsilon": 10/"epsilon": 1/g' ./configs/2-slice/ob/alpaca2.json

done
