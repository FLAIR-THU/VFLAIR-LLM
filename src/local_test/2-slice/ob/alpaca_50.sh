#!/bin/bash
#SBATCH --job-name alpaca_ob          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_ob.out
#SBATCH --mem 160000MB


for seed in 15 16 17 18 19
    do 

   
    python main_pipeline_llm.py --prefix "ob_add" --seed $seed --configs 2-slice/ob/alpaca_50

done
