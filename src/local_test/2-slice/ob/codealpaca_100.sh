#!/bin/bash
#SBATCH --job-name ob_100          # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/ob_100.out
# epsilon=50 l10
python main_pipeline_llm.py --save_model 1 --prefix "ob" --seed 65 --configs 2-slice/ob/codealpaca_100
