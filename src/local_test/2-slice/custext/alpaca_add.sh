#!/bin/bash
#SBATCH --job-name gms_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_custext.out


for seed in 1 2 3
    do

    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/custext/alpaca_add.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca_add

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/custext/alpaca_add.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca_add

    # 1
    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/custext/alpaca_add.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca_add

    # 5
    sed -i 's/"epsilon": 1/"epsilon": 5/g' ./configs/2-slice/custext/alpaca_add.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca_add

    sed -i 's/"epsilon": 5/"epsilon": 1/g' ./configs/2-slice/custext/alpaca_add.json
   
done
