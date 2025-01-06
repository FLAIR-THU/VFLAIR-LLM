#!/bin/bash
#SBATCH --job-name sst2_custext           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_custext.out


# 1
# python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/custext/sst2_3

# 30
sed -i 's/"epsilon": 1/"epsilon": 30/g' ./configs/2-slice/custext/sst2_3.json
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/custext/sst2_3

# 50
sed -i 's/"epsilon": 30/"epsilon": 50/g' ./configs/2-slice/custext/sst2_3.json
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/custext/sst2_3

sed -i 's/"epsilon": 50/"epsilon": 1/g' ./configs/2-slice/custext/sst2_3.json

for seed in 62 63 64 65
    do

    # 1
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/sst2_3

    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/custext/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/sst2_3


    # 10
    sed -i 's/"epsilon": 0.01/"epsilon": 10/g' ./configs/2-slice/custext/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/sst2_3

    # 30
    sed -i 's/"epsilon": 10/"epsilon": 30/g' ./configs/2-slice/custext/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/sst2_3

    # 50
    sed -i 's/"epsilon": 30/"epsilon": 50/g' ./configs/2-slice/custext/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/sst2_3

    sed -i 's/"epsilon": 50/"epsilon": 1/g' ./configs/2-slice/custext/sst2_3.json


done
