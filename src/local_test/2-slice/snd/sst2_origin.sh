#!/bin/bash
#SBATCH --job-name sst2_snd           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_snd.out

python main_pipeline_llm.py --prefix "snd_test" --seed 60 --configs 2-slice/snd/sst2_origin
