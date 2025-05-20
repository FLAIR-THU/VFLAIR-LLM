#!/bin/bash
#SBATCH --job-name dolly_lo         # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/dolly_lo.out

# epsilon=50 l10
python main_pipeline_llm.py  --save_model 1 --prefix "ft_lo" --seed 97 --configs finetune/dolly_lo

