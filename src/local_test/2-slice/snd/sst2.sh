#!/bin/bash
#SBATCH --job-name sst2_snd           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_snd.out

for seed in 60 61 62 63 64 65
    do

    # 100
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 10
    sed -i 's/"test_eta": 100/"test_eta": 10/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 1e3
    sed -i 's/"test_eta": 10/"test_eta": 1e3/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 1e4
    sed -i 's/"test_eta": 1e3/"test_eta": 1e4/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    # 1e5
    sed -i 's/"test_eta": 1e4/"test_eta": 1e5/g' ./configs/2-slice/snd/sst2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/snd/sst2

    sed -i 's/"test_eta": 1e5/"test_eta": 100/g' ./configs/2-slice/snd/sst2.json


done
