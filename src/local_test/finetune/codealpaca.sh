#!/bin/bash
#SBATCH --job-name codealpaca_inf         # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/codealpaca_inf.out

# epsilon=50 l10
python main_pipeline_llm.py  --save_model 1 --prefix "ft" --seed 60 --configs finetune/codealpaca

