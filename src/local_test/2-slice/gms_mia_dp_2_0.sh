#!/bin/bash
#SBATCH --job-name gms_dp_2           # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

# 100
sed -i 's/"epsilon": 50/"epsilon": 1000/g' ./configs/2-slice/gms_mia_dp_2_0.json
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/gms_mia_dp_2_0

sed -i 's/"epsilon": 1000/"epsilon": 50/g' ./configs/2-slice/gms_mia_dp_2_0.json

for seed in {61,62,63,64,65}
    do
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_wo_2

    # 50
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_0

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 700/g' ./configs/2-slice/gms_mia_dp_2_0.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_0

    # 80
    sed -i 's/"epsilon": 700/"epsilon": 800/g' ./configs/2-slice/gms_mia_dp_2_0.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_0

    # 100
    sed -i 's/"epsilon": 800/"epsilon": 1000/g' ./configs/2-slice/gms_mia_dp_2_0.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_0

    # 500
    sed -i 's/"epsilon": 1000/"epsilon": 5000/g' ./configs/2-slice/gms_mia_dp_2_0.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_0

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/gms_mia_dp_2_0.json

done