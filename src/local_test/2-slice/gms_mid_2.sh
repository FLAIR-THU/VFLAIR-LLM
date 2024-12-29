#!/bin/bash
#SBATCH --job-name gms_mia_mid       # 任务名叫 example
#SBATCH --gres gpu:a100:3                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_mia_mid.out

# 1e-5
sed -i 's/"lambda": 0.5/"lambda": 0.00001/g' ./configs/2-slice/gms_mia_mid.json
for seed in 1 5 6
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid
done
sed -i 's/"lambda": 0.00001/"lambda": 0.5/g' ./configs/2-slice/gms_mia_mid.json


# 1e-4
sed -i 's/"lambda": 0.5/"lambda": 0.0001/g' ./configs/2-slice/gms_mia_mid.json
for seed in 1 5 6
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid
done
sed -i 's/"lambda": 0.0001/"lambda": 0.5/g' ./configs/2-slice/gms_mia_mid.json


# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/2-slice/gms_mia_mid.json
for seed in 1 5 6
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid
done
sed -i 's/"lambda": 0.1/"lambda": 0.5/g' ./configs/2-slice/gms_mia_mid.json


# 0.5
for seed in 1 5 6
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid
done


# 0.001
sed -i 's/"lambda": 0.5/"lambda": 0.001/g' ./configs/2-slice/gms_mia_mid.json
for seed in 1 5 6
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid
done
sed -i 's/"lambda": 0.001/"lambda": 0.5/g' ./configs/2-slice/gms_mia_mid.json


# 0.01
sed -i 's/"lambda": 0.5/"lambda": 0.01/g' ./configs/2-slice/gms_mia_mid.json
for seed in 1 5 6
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_mid
done
sed -i 's/"lambda": 0.01/"lambda": 0.5/g' ./configs/2-slice/gms_mia_mid.json
