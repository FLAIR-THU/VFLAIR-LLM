#!/bin/bash
#SBATCH --job-name gms_dp           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_dp.out

python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/acc_only/gms_wo
for seed in 60 61 62
    do

    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_dp

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/acc_only/gms_dp.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_dp

    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/2-slice/acc_only/gms_dp.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_dp

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/acc_only/gms_dp.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_dp

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/acc_only/gms_dp.json

done