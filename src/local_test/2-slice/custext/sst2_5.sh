#!/bin/bash
#SBATCH --job-name sst2_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_custext.out

sed -i 's/"epsilon": 1/"epsilon": 5/g' ./configs/2-slice/custext/sst2.json
for seed in 60 61 62 63 64 65
    do

    # 5
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/sst2

done
sed -i 's/"epsilon": 5/"epsilon": 1/g' ./configs/2-slice/custext/sst2.json
