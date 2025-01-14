#!/bin/bash
#SBATCH --job-name alpaca_dpt           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_dpt.out
#SBATCH --mem 160000MB


python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/dpt/alpaca_wo

# for seed in 60 61
#     do
   
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/alpaca_wo

# done
