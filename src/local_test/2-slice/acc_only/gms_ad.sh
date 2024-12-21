#!/bin/bash
#SBATCH --job-name gms_ad          # 任务名叫 example
#SBATCH --gres gpu:a100:3                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_ad.out
#SBATCH --mem 100000MB

for seed in 60 61 62
    do

    # 0.001
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_ad

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/2-slice/acc_only/gms_ad.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_ad

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/2-slice/acc_only/gms_ad.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_ad

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/2-slice/acc_only/gms_ad.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_ad

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/2-slice/acc_only/gms_ad.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/acc_only/gms_ad


    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/2-slice/acc_only/gms_ad.json
    
done
