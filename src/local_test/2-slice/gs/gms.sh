#!/bin/bash
#SBATCH --job-name gms_gs           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 8:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_gs.out

for seed in {60,61,62}
    do

    # 99
    sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99/g' ./configs/2-slice/gs/gms.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/gs/gms

    # 98
    sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/2-slice/gs/gms.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/gs/gms

    # 97
    sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/2-slice/gs/gms.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/gs/gms

    # 96
    sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/2-slice/gs/gms.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/gs/gms

    # 95
    sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/2-slice/gs/gms.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/gs/gms


    sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/gs/gms.json

done

