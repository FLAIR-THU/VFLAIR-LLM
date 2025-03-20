#!/bin/bash
#SBATCH --job-name codealpaca_dp_50           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 8:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/codealpaca_dp_50.out

python main_pipeline_llm.py --prefix "dp" --seed 61 --configs 2-slice/dp/codealpaca_wo
python main_pipeline_llm.py --prefix "dp" --seed 61 --configs 2-slice/dp/codealpaca_50



