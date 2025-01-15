#!/bin/bash
#SBATCH --job-name alpaca3_ob          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca3_ob.out
#SBATCH --mem 160000MB


for seed in 61 62 63 64 65
    do 

    # 0.5
    sed -i 's/"epsilon": 1/"epsilon": 0.5/g' ./configs/2-slice/ob/alpaca3.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca3

    # 1
    sed -i 's/"epsilon": 0.5/"epsilon": 1/g' ./configs/2-slice/ob/alpaca3.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca3

    # 2
    sed -i 's/"epsilon": 1/"epsilon": 2/g' ./configs/2-slice/ob/alpaca3.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca3

    # 5
    sed -i 's/"epsilon": 2/"epsilon": 5/g' ./configs/2-slice/ob/alpaca3.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca3

    # 10
    sed -i 's/"epsilon": 5/"epsilon": 10/g' ./configs/2-slice/ob/alpaca3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/alpaca3

    sed -i 's/"epsilon": 10/"epsilon": 1/g' ./configs/2-slice/ob/alpaca3.json

done
