#!/bin/bash
#SBATCH --job-name sst2_snd           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_snd.out


for seed in 60 61 62 63 64
    do

    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 100
    sed -i 's/"train_eta": 50/"train_eta": 100/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 150
    sed -i 's/"train_eta": 100/"train_eta": 150/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 200
    sed -i 's/"train_eta": 150/"train_eta": 200/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 10
    sed -i 's/"train_eta": 200/"train_eta": 10/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    sed -i 's/"train_eta": 10/"train_eta": 50/g' ./configs/2-slice/snd/sst2.json


done
