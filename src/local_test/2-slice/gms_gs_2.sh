#!/bin/bash
#SBATCH --job-name gms_gs_96           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_gs_96.out


# 99
sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 96/g' ./configs/2-slice/gms_mia_gs_2.json
for seed in 1 2 3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_gs_2
done
sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/gms_mia_gs_2.json

# # 98
# sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/2-slice/gms_mia_gs_2.json
# python main_pipeline_llm_MIA.py --seed 61 --configs 2-slice/gms_mia_gs_2

# # 97
# sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/2-slice/gms_mia_gs_2.json
# python main_pipeline_llm_MIA.py --seed 61 --configs 2-slice/gms_mia_gs_2

# # 96
# sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/2-slice/gms_mia_gs_2.json
# python main_pipeline_llm_MIA.py --seed 61 --configs 2-slice/gms_mia_gs_2

# # 95
# sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/2-slice/gms_mia_gs_2.json
# python main_pipeline_llm_MIA.py --seed 61 --configs 2-slice/gms_mia_gs_2


# sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/gms_mia_gs_2.json



