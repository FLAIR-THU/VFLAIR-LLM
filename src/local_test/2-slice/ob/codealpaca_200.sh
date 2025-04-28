#!/bin/bash
#SBATCH --job-name ob_200          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/ob_200.out
#SBATCH --mem 160000MB
#SBATCH --qos high

# epsilon=50 l10
python main_pipeline_llm.py --save_model 1 --prefix "ob" --seed 65 --configs 2-slice/ob/codealpaca_200

