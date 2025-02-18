#!/bin/bash
#SBATCH --job-name sst2_both_vanilla_mid           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_both_vanilla_mid.out

for seed in 60 61 62 63 64 65
    do
    # 0.5
    python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_mid

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/both_vanilla/sst2_mid.json
    python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_mid

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/3-slice/both_vanilla/sst2_mid.json
    python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_mid

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/3-slice/both_vanilla/sst2_mid.json
    # python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_mid

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/3-slice/both_vanilla/sst2_mid.json
    # python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_mid

    # 1e-5
    sed -i 's/"lambda": 0.0001/"lambda": 1e-5/g' ./configs/3-slice/both_vanilla/sst2_mid.json
    # python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_mid

    sed -i 's/"lambda": 1e-5/"lambda": 0.5/g' ./configs/3-slice/both_vanilla/sst2_mid.json

done
