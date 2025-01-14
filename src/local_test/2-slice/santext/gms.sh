#!/bin/bash
#SBATCH --job-name gms_santext           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_santext.out

for seed in 60 61 62 63 64 65
    do

    # 1
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/santext/gms.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/santext/gms.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    # 0.5
    sed -i 's/"epsilon": 0.1/"epsilon": 0.5/g' ./configs/2-slice/santext/gms.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    # 5
    sed -i 's/"epsilon": 0.5/"epsilon": 5/g' ./configs/2-slice/santext/gms.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    # 10
    sed -i 's/"epsilon": 5/"epsilon": 10/g' ./configs/2-slice/santext/gms.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    # 30
    sed -i 's/"epsilon": 10/"epsilon": 30/g' ./configs/2-slice/santext/gms.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    # 50
    sed -i 's/"epsilon": 30/"epsilon": 50/g' ./configs/2-slice/santext/gms.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/santext/gms

    sed -i 's/"epsilon": 50/"epsilon": 1/g' ./configs/2-slice/santext/gms.json


done
