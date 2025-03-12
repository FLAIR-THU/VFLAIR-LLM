#!/bin/bash
#SBATCH --job-name sst2_va_ad0.1           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_va_ad0.1.out



# 0.1
python main_pipeline_llm.py --prefix "both_vanilla" --seed 65 --configs 3-slice/both_vanilla/sst2_ad_0.1

# 1.0
python main_pipeline_llm.py --prefix "both_vanilla" --seed 65 --configs 3-slice/both_vanilla/sst2_ad_1.0
