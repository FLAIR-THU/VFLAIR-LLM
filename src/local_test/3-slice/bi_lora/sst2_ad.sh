#!/bin/bash
#SBATCH --job-name sst2_both_both_ad           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_both_both_ad.out

sed -i 's/"lambda": 0.001/"lambda": 5.0/g' ./configs/3-slice/both/sst2_ad.json
python main_pipeline_llm_Both_LoRA.py --seed 60 --configs 3-slice/both/sst2_ad
sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/3-slice/both/sst2_ad.json
    
for seed in 61 62 63 64 65
    do

    # 0.001
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/both/sst2_ad

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/3-slice/both/sst2_ad.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/both/sst2_ad

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/3-slice/both/sst2_ad.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/both/sst2_ad

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/3-slice/both/sst2_ad.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/both/sst2_ad

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/3-slice/both/sst2_ad.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/both/sst2_ad


    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/3-slice/both/sst2_ad.json
    
done
