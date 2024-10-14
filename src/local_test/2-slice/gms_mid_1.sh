#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

for seed in {60,61,62,63,64,65}
    do
    # 0.5
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid_1

    # 1e-5
    sed -i 's/"lambda": 0.5/"lambda": 1e-5/g' ./configs/2-slice/gms_mia_mid_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid_1

    # 1e-6
    sed -i 's/"lambda": 1e-5/"lambda": 1e-6/g' ./configs/2-slice/gms_mia_mid_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid_1

    # 1e-7
    sed -i 's/"lambda": 1e-6/"lambda": 1e-7/g' ./configs/2-slice/gms_mia_mid_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid_1

    # 1e-8
    sed -i 's/"lambda": 1e-7/"lambda": 1e-8/g' ./configs/2-slice/gms_mia_mid_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid_1

    # 1e-9
    sed -i 's/"lambda": 1e-8/"lambda": 1e-9/g' ./configs/2-slice/gms_mia_mid_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid_1


    sed -i 's/"lambda": 1e-9/"lambda": 0.5/g' ./configs/2-slice/gms_mia_mid_1.json

done
