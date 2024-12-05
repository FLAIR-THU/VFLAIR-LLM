#!/bin/bash
#SBATCH --job-name gms_bi_mid_3_add          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_bi_mid_3_add.out



# 0.001
sed -i 's/"lambda": 0.5/"lambda": 0.001/g' ./configs/2-slice/bi/gms_mia_mid_add.json
for seed in 61 62 63 64 1 2 3 4
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/gms_mia_mid_add
done

# 0.0001
sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/2-slice/bi/gms_mia_mid_add.json
for seed in 62 64 1 2 3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/gms_mia_mid_add
done

# 1e-5
sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/2-slice/bi/gms_mia_mid_add.json
for seed in 61 64 1 3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/gms_mia_mid_add
done

sed -i 's/"lambda": 0.00001/"lambda": 0.5/g' ./configs/2-slice/bi/gms_mia_mid_add.json


