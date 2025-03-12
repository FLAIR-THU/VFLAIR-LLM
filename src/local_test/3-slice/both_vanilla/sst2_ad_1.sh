#!/bin/bash
#SBATCH --job-name sst2_va_ad1           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-5:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_va_ad1.out


for seed in 64 65 1 2
    do

    # 0.001
    python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_ad_1

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/3-slice/both_vanilla/sst2_ad_1.json
    python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_ad_1

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/3-slice/both_vanilla/sst2_ad_1.json
    python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_ad_1

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/3-slice/both_vanilla/sst2_ad_1.json
    # python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_ad_1

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/3-slice/both_vanilla/sst2_ad_1.json
    # python main_pipeline_llm.py --prefix "both_vanilla" --seed $seed --configs 3-slice/both_vanilla/sst2_ad_1


    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/3-slice/both_vanilla/sst2_ad_1.json
    
done
