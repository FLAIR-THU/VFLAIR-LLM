#!/bin/bash
#SBATCH --job-name gms_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_custext.out


for seed in 60 61 62 63 64
    do

    # 50
    sed -i 's/"epsilon": 1/"epsilon": 50/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 30
    sed -i 's/"epsilon": 50/"epsilon": 30/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 15
    sed -i 's/"epsilon": 30/"epsilon": 15/g' ./configs/2-slice/custext/alpaca.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 5
    sed -i 's/"epsilon": 15/"epsilon": 5/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 1
    sed -i 's/"epsilon": 5/"epsilon": 1/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 0.5
    sed -i 's/"epsilon": 1/"epsilon": 0.5/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca


    # 0.1
    sed -i 's/"epsilon": 0.5/"epsilon": 0.1/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca


    # 0.01
    sed -i 's/"epsilon": 0.1/"epsilon": 0.01/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    sed -i 's/"epsilon": 0.01/"epsilon": 1/g' ./configs/2-slice/custext/alpaca.json

done
