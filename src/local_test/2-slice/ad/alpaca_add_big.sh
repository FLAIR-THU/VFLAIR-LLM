#!/bin/bash
#SBATCH --job-name alpaca_big          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_big.out

for seed in 21 22 23 24 25
    do

    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ad/alpaca_5
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ad/alpaca_1


done
