#!/bin/bash
#SBATCH --job-name gms_2_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_2_custext.out


for seed in 1 2 3 4 5
    do


    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/custext/gms_2.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/gms_2

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/custext/gms_2.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/gms_2

    # 1
    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/custext/gms_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/gms_2

    # 5
    sed -i 's/"epsilon": 1/"epsilon": 5/g' ./configs/2-slice/custext/gms_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/gms_2


done
