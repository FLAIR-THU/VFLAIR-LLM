#!/bin/bash
#SBATCH --job-name cola_both_lora_mid           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/cola_both_lora_mid.out

python main_pipeline_llm.py --prefix "both_lora_test" --seed 60 --configs 3-slice/both_lora/cola_mid_test
