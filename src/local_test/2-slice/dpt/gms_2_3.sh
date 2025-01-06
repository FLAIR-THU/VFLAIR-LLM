#!/bin/bash
#SBATCH --job-name gms_dpt_3           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_dpt_3.out
#SBATCH --mem 160000MB

# with decode  0.00001 0.0001 0.001
for seed in 1 2
    do

    # 0.00001
    sed -i 's/"epsilon": 1/"epsilon": 0.00001/g' ./configs/2-slice/dpt/gms_2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/gms_2_3

    # 0.0001
    sed -i 's/"epsilon": 0.00001/"epsilon": 0.0001/g' ./configs/2-slice/dpt/gms_2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/gms_2_3

    sed -i 's/"epsilon": 0.0001/"epsilon": 1/g' ./configs/2-slice/dpt/gms_2_3.json

done
