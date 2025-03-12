#!/bin/bash
#SBATCH --job-name codealpaca_0.001       # 任务名叫 example
#SBATCH --gres gpu:a100:4                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/codealpaca_0.001.out
#SBATCH --mem 160000MB


sed -i 's/"pipeline": "finetune"/"pipeline": "pretrained"/g' ./configs/2-slice/mid/codealpaca_0.001.json
python main_pipeline_llm.py --prefix "mid" --seed 60 --configs 2-slice/mid/codealpaca_0.001


sed -i 's/"pipeline": "pretrained"/"pipeline": "finetune"/g' ./configs/2-slice/mid/codealpaca_0.001.json
python main_pipeline_llm.py --prefix "mid" --seed 61 --configs 2-slice/mid/codealpaca_0.001
