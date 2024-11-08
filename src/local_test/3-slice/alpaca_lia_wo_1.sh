#!/bin/bash
#SBATCH --job-name alpaca_lia_wo_1      # 任务名叫 example
#SBATCH --gres gpu:a100:5             # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_lia_wo_1.out
#SBATCH --qos high

for seed in {60,61}
    do
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/alpaca_lia_wo_1

    sed -i 's/"lr": 5e-6/"lr": 1e-5/g' ./configs/3-slice/alpaca_lia_wo_1.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/alpaca_lia_wo_1

    sed -i 's/"lr": 1e-5/"lr": 5e-5/g' ./configs/3-slice/alpaca_lia_wo_1.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/alpaca_lia_wo_1

    sed -i 's/"lr": 5e-5/"lr": 5e-6/g' ./configs/3-slice/alpaca_lia_wo_1.json


done
