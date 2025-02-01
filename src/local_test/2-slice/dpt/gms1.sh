#!/bin/bash
#SBATCH --job-name gms1           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-18:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms1.out
#SBATCH --mem 160000MB

sed -i 's/"epsilon": 1/"epsilon": 25/g' ./configs/2-slice/dpt/gms1.json

for seed in 60 61 62
    do

    # 25
    python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms1

    # # 15
    # sed -i 's/"epsilon": 10/"epsilon": 15/g' ./configs/2-slice/dpt/gms1.json
    # # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms1

    # # 20
    # sed -i 's/"epsilon": 15/"epsilon": 20/g' ./configs/2-slice/dpt/gms1.json
    # # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms1

    # # 30
    # sed -i 's/"epsilon": 20/"epsilon": 30/g' ./configs/2-slice/dpt/gms1.json
    # # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms1

    # # 100
    # sed -i 's/"epsilon": 30/"epsilon": 100/g' ./configs/2-slice/dpt/gms1.json
    # # python main_pipeline_llm.py --prefix "rantext" --seed $seed --configs 2-slice/dpt/gms1


    # sed -i 's/"epsilon": 100/"epsilon": 1/g' ./configs/2-slice/dpt/gms1.json

done
sed -i 's/"epsilon": 25/"epsilon": 1/g' ./configs/2-slice/dpt/gms1.json
