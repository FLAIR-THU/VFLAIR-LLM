#!/bin/bash
#SBATCH --job-name alpaca_gs           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 8:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_gs.out


# # 99
# sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99/g' ./configs/2-slice/gs/alpaca.json
# for seed in 10 11 12 13 14 15
#     do
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca
# done
# sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/gs/alpaca.json


for seed in 20 21 22 23
    do

    # 99
    sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99/g' ./configs/2-slice/gs/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca

    # 98
    sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/2-slice/gs/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca

    # 97
    sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/2-slice/gs/alpaca.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca

    # 96
    sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/2-slice/gs/alpaca.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca

    # 95
    sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/2-slice/gs/alpaca.json
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gs/alpaca


    sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/gs/alpaca.json

done
