#!/bin/bash
#SBATCH --job-name yelp_both_bi_mid           # 任务名叫 example
#SBATCH --gres gpu:a100:4                # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_both_bi_mid.out
#SBATCH --mem 120000MB 

for seed in 61 62 63 
    do
    # 0.5
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_mid_3

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/bi_lora/yelp_both_mid_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_mid_3

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/3-slice/bi_lora/yelp_both_mid_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_mid_3

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/3-slice/bi_lora/yelp_both_mid_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_mid_3

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/3-slice/bi_lora/yelp_both_mid_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_mid_3

    # 1e-5
    sed -i 's/"lambda": 0.0001/"lambda": 1e-5/g' ./configs/3-slice/bi_lora/yelp_both_mid_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_mid_3

    sed -i 's/"lambda": 1e-5/"lambda": 0.5/g' ./configs/3-slice/bi_lora/yelp_both_mid_3.json

done
