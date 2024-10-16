#!/bin/bash
#SBATCH --job-name gms_mia_dp_2_1           # 任务名叫 example
#SBATCH --gres gpu:a100:2                 # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_mia_dp_2_1.out

for seed in {61,62,63,64,65}
    do
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_wo_2

    # 50
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_1

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 700/g' ./configs/2-slice/gms_mia_dp_2_1.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_1

    # 80
    sed -i 's/"epsilon": 700/"epsilon": 800/g' ./configs/2-slice/gms_mia_dp_2_1.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_1

    # 100
    sed -i 's/"epsilon": 800/"epsilon": 1000/g' ./configs/2-slice/gms_mia_dp_2_1.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_1

    # 500
    sed -i 's/"epsilon": 1000/"epsilon": 5000/g' ./configs/2-slice/gms_mia_dp_2_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_2_1

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/gms_mia_dp_2_1.json

done

for seed in {1,2,3,4,5}
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_wo_2
done