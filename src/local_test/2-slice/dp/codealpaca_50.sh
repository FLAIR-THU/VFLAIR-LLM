#!/bin/bash
#SBATCH --job-name dp_new           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 8:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/dp_new.out

python main_pipeline_llm.py --attack_only 1 --prefix "dp_new" --seed 60 --configs 2-slice/dp/codealpaca_50
python main_pipeline_llm.py --attack_only 1 --prefix "dp_new" --seed 61 --configs 2-slice/dp/codealpaca_50


python main_pipeline_llm.py --attack_only 1 --prefix "dp_new" --seed 60 --configs 2-slice/dp/codealpaca_70

python main_pipeline_llm.py --attack_only 1 --prefix "dp_new" --seed 60 --configs 2-slice/dp/codealpaca_100

python main_pipeline_llm.py --attack_only 1 --prefix "dp_new" --seed 60 --configs 2-slice/dp/codealpaca_500

