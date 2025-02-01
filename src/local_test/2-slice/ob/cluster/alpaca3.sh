#!/bin/bash
#SBATCH --job-name alpaca3_ob          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca3_ob.out
#SBATCH --mem 160000MB


for seed in 13 14
    do 

   
    # 50
    python main_pipeline_llm.py --prefix "obcluster" --seed $seed --configs 2-slice/ob/cluster/alpaca3

    # 100
    sed -i 's/"cluster_num": 50/"cluster_num": 100/g' ./configs/2-slice/ob/cluster/alpaca3.json
    python main_pipeline_llm.py --prefix "obcluster" --seed $seed --configs 2-slice/ob/cluster/alpaca3

    # 150
    sed -i 's/"cluster_num": 100/"cluster_num": 150/g' ./configs/2-slice/ob/cluster/alpaca3.json
    python main_pipeline_llm.py --prefix "obcluster" --seed $seed --configs 2-slice/ob/cluster/alpaca3

    # 200
    sed -i 's/"cluster_num": 150/"cluster_num": 200/g' ./configs/2-slice/ob/cluster/alpaca3.json
    # python main_pipeline_llm.py --prefix "obcluster" --seed $seed --configs 2-slice/ob/cluster/alpaca3

    # 250
    sed -i 's/"cluster_num": 200/"cluster_num": 250/g' ./configs/2-slice/ob/cluster/alpaca3.json
    # python main_pipeline_llm.py --prefix "obcluster" --seed $seed --configs 2-slice/ob/cluster/alpaca3
    
    sed -i 's/"cluster_num": 250/"cluster_num": 50/g' ./configs/2-slice/ob/cluster/alpaca3.json


done
