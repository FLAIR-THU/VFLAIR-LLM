#!/bin/bash
#SBATCH --job-name gms_bi_mid_3          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_bi_mid_3.out


# 0.5
for seed in 60 63 1
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/gms_mia_mid
done

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/2-slice/bi/gms_mia_mid.json
for seed in 60 61 62 63 1 3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/gms_mia_mid
done

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/2-slice/bi/gms_mia_mid.json
for seed in 61 63 65 3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/gms_mia_mid
done


sed -i 's/"lambda": 0.01/"lambda": 0.5/g' ./configs/2-slice/bi/gms_mia_mid.json


