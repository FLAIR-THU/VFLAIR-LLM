#!/bin/bash
#SBATCH --job-name alpaca_dp           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 8:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_dp.out


for seed in 2 3 4 5
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/alpaca_wo

    # # 50
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/alpaca

    # # 70
    # sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/dp/alpaca.json
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/alpaca

    # # 100
    # sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/2-slice/dp/alpaca.json
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/alpaca

    # # 500
    # sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/dp/alpaca.json
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/alpaca

    # sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/dp/alpaca.json

done