#!/bin/bash
#SBATCH --job-name codealpaca_dp_500           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 8:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/codealpaca_dp_500.out


python main_pipeline_llm.py --attack_only true --prefix "wo" --seed 60 --configs 2-slice/dp/codealpaca_wo_attackonly
python main_pipeline_llm.py --attack_only true --prefix "wo" --seed 61 --configs 2-slice/dp/codealpaca_wo_attackonly
python main_pipeline_llm.py --attack_only true --prefix "wo" --seed 62 --configs 2-slice/dp/codealpaca_wo_attackonly

