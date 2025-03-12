#!/bin/bash
#SBATCH --job-name sst2_lora_ad3           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_lora_ad3.out
#SBATCH --qos high
# python main_pipeline_llm.py --prefix "both_lora_test" --seed 60 --configs 3-slice/both_lora/sst2_ad_3

for seed in 62 63 64 65 1 2 3
    do

    # 0.001
    # python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/sst2_ad_3

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/3-slice/both_lora/sst2_ad_3.json
    # python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/sst2_ad_3

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/3-slice/both_lora/sst2_ad_3.json
    # python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/sst2_ad_3

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/3-slice/both_lora/sst2_ad_3.json
    # python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/sst2_ad_3

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/3-slice/both_lora/sst2_ad_3.json
    python main_pipeline_llm.py --prefix "both_lora" --seed $seed --configs 3-slice/both_lora/sst2_ad_3


    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/3-slice/both_lora/sst2_ad_3.json
    
done
