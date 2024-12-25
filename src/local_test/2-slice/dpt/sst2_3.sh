#!/bin/bash
#SBATCH --job-name sst2_dpt           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_dpt.out

for seed in 60 61 62 63
    do

    # 1
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/sst2_3

    # 0.1
    sed -i 's/"epsilon": 1/"epsilon": 0.1/g' ./configs/2-slice/dpt/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/sst2_3

    # 0.01
    sed -i 's/"epsilon": 0.1/"epsilon": 0.01/g' ./configs/2-slice/dpt/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/sst2_3

    # 0.001
    sed -i 's/"epsilon": 0.01/"epsilon": 0.001/g' ./configs/2-slice/dpt/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/sst2_3

    # 0.0001
    sed -i 's/"epsilon": 0.001/"epsilon": 0.0001/g' ./configs/2-slice/dpt/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/sst2_3

    # 0.00001
    sed -i 's/"epsilon": 0.0001/"epsilon": 0.00001/g' ./configs/2-slice/dpt/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dpt/sst2_3


    sed -i 's/"epsilon": 0.00001/"epsilon": 1/g' ./configs/2-slice/dpt/sst2_3.json


done
