#!/bin/bash
#SBATCH --job-name sst2_both_both_dp           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_both_both_dp.out

for seed in 60 61 62 63 64 65
    do
    python main_pipeline_llm_Both_Vanilla.py --seed $seed --configs 3-slice/both_va/sst2_wo

    # 50
    python main_pipeline_llm_Both_Vanilla.py --seed $seed --configs 3-slice/both_va/sst2_dp

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/3-slice/both_va/sst2_dp.json
    python main_pipeline_llm_Both_Vanilla.py --seed $seed --configs 3-slice/both_va/sst2_dp

   
    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/3-slice/both_va/sst2_dp.json
    python main_pipeline_llm_Both_Vanilla.py --seed $seed --configs 3-slice/both_va/sst2_dp

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/3-slice/both_va/sst2_dp.json
    python main_pipeline_llm_Both_Vanilla.py --seed $seed --configs 3-slice/both_va/sst2_dp

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/3-slice/both_va/sst2_dp.json

done
