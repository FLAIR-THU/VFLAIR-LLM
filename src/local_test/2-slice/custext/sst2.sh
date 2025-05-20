#!/bin/bash
#SBATCH --job-name sst2_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_custext.out

for seed in 65 64 63
    do

    # 1
    python main_pipeline_llm.py --seed $seed --configs 2-slice/custext/sst2

    # 0.1
    sed -i 's/"epsilon": 1/"epsilon": 0.1/g' ./configs/2-slice/custext/sst2.json
    python main_pipeline_llm.py --seed $seed --configs 2-slice/custext/sst2

    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/custext/sst2.json

done
