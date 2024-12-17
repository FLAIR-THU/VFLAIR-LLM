#!/bin/bash
#SBATCH --job-name yelp_both_bi_lora_dp           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_both_bi_lora_dp.out
#SBATCH --mem 60000MB

# 50
python main_pipeline_llm_Both_LoRA.py --seed 60 --configs 3-slice/bi_lora/yelp_both_dp_3

# 70
sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json
python main_pipeline_llm_Both_LoRA.py --seed 60 --configs 3-slice/bi_lora/yelp_both_dp_3

# 100
sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json
python main_pipeline_llm_Both_LoRA.py --seed 60 --configs 3-slice/bi_lora/yelp_both_dp_3

# 500
sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json
python main_pipeline_llm_Both_LoRA.py --seed 60 --configs 3-slice/bi_lora/yelp_both_dp_3

sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json

for seed in 61 62 62
    do
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_wo_3

    # 50
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_dp_3

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_dp_3

   

    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_dp_3

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json
    python main_pipeline_llm_Both_LoRA.py --seed $seed --configs 3-slice/bi_lora/yelp_both_dp_3

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/3-slice/bi_lora/yelp_both_dp_3.json

done
