#!/bin/bash
#SBATCH --job-name dolly_mid_0.0001          # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/dolly_mid_0.0001.out

python main_pipeline_llm.py --prefix "pmc_dolly_mid" --attack_only 0 --save_model 1 --seed 60 --configs 3-slice/dolly/llama_mid_0.0001

