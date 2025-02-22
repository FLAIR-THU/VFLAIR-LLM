#!/bin/bash
#SBATCH --job-name cola_lia_gs          # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/cola_lia_gs.out
#SBATCH --mem 100000MB


for seed in 1 2 3 4 5
    do
    # 98
    sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 98/g' ./configs/3-slice/lia/cola_gs.json
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_gs

    # 97
    sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/3-slice/lia/cola_gs.json
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_gs

    # 96
    sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/3-slice/lia/cola_gs.json
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_gs

    # 95
    sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/3-slice/lia/cola_gs.json
    python main_pipeline_llm.py --prefix "lia" --seed $seed --configs 3-slice/lia/cola_gs


    sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/3-slice/lia/cola_gs.json

done
