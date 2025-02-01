#!/bin/bash
#SBATCH --job-name alpaca1_ob          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca1_ob.out
#SBATCH --mem 160000MB


for seed in 10 11 12
    do 

   
    # 1
    python main_pipeline_llm.py --prefix "obeps" --seed $seed --configs 2-slice/ob/cluster/alpaca1

    # 100
    sed -i 's/"epsilon": 50/"epsilon": 100/g' ./configs/2-slice/ob/cluster/alpaca1.json
    python main_pipeline_llm.py --prefix "obeps" --seed $seed --configs 2-slice/ob/cluster/alpaca1

    # 150
    sed -i 's/"epsilon": 100/"epsilon": 150/g' ./configs/2-slice/ob/cluster/alpaca1.json
    python main_pipeline_llm.py --prefix "obeps" --seed $seed --configs 2-slice/ob/cluster/alpaca1

    # 200
    sed -i 's/"epsilon": 150/"epsilon": 200/g' ./configs/2-slice/ob/cluster/alpaca1.json
    python main_pipeline_llm.py --prefix "obeps" --seed $seed --configs 2-slice/ob/cluster/alpaca1

    # 250
    sed -i 's/"epsilon": 200/"epsilon": 250/g' ./configs/2-slice/ob/cluster/alpaca1.json
    python main_pipeline_llm.py --prefix "obeps" --seed $seed --configs 2-slice/ob/cluster/alpaca1
    
    sed -i 's/"epsilon": 250/"epsilon": 50/g' ./configs/2-slice/ob/cluster/alpaca1.json


done
