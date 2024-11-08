#!/bin/bash
#SBATCH --job-name sst2_midbig_lia           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完

for seed in {60,61,62,63,64,65}
    do
    # 0.5
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_mid_3_1_big

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/sst2_lia_mid_3_1_big.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_mid_3_1_big

    # 5
    sed -i 's/"lambda": 0.1/"lambda": 5/g' ./configs/3-slice/sst2_lia_mid_3_1_big.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_mid_3_1_big

    # 10
    sed -i 's/"lambda": 5/"lambda": 10/g' ./configs/3-slice/sst2_lia_mid_3_1_big.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_mid_3_1_big

    # 50
    sed -i 's/"lambda": 10/"lambda": 50/g' ./configs/3-slice/sst2_lia_mid_3_1_big.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_mid_3_1_big

    # 100
    sed -i 's/"lambda": 50/"lambda": 100/g' ./configs/3-slice/sst2_lia_mid_3_1_big.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_mid_3_1_big

    sed -i 's/"lambda": 100/"lambda": 0.5/g' ./configs/3-slice/sst2_lia_mid_3_1_big.json

done
