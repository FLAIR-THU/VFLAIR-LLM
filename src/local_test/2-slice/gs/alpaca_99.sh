#!/bin/bash
#SBATCH --job-name alpaca_gs           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 8:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_gs.out


for seed in 71 72 73 74
    do

    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca_99

    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca_98


done
