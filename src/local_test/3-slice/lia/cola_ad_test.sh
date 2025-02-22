#!/bin/bash
#SBATCH --job-name cola_lia_ad           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/cola_lia_ad.out
#SBATCH --mem 100000MB

python main_pipeline_llm.py --prefix "lia_test" --seed 60 --configs 3-slice/lia/cola_ad_test

# python main_pipeline_llm.py --prefix "lia_test" --seed 60 --configs 3-slice/lia/cola_mid_test

# python main_pipeline_llm.py --prefix "lia_test" --seed 60 --configs 3-slice/lia/cola_mid_test1

    