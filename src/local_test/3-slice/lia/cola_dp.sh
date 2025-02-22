#!/bin/bash
#SBATCH --job-name cola_lia_dp           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/cola_lia_dp.out
#SBATCH --mem 60000MB
#SBATCH --qos high

for seed in 60 61 62 63 64 65
    do
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_wo_new

    # 50
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_dp

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/3-slice/lia/cola_dp.json
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_dp

   
    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/3-slice/lia/cola_dp.json
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_dp

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/3-slice/lia/cola_dp.json
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_dp

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/3-slice/lia/cola_dp.json

done
