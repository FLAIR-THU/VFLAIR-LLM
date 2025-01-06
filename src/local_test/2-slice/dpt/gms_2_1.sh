#!/bin/bash
#SBATCH --job-name gms_dpt_1           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_dpt_1.out
#SBATCH --mem 160000MB


for seed in {64,65} # with decode  0.01 0.1
    do

    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/dpt/gms_2_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/gms_2_1

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/dpt/gms_2_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/gms_2_1


    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/dpt/gms_2_1.json

done
