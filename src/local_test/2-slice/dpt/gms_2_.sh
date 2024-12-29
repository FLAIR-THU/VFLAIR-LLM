#!/bin/bash
#SBATCH --job-name gms_dpt_           # 任务名叫 example
#SBATCH --gres gpu:a100:4                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_dpt_.out

# test dpt with decode
for seed in {60,61,62}
    do
    # 0.00001
    sed -i 's/"epsilon": 1/"epsilon": 0.00001/g' ./configs/2-slice/dpt/gms_2_.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_2_

    # 0.0001
    sed -i 's/"epsilon": 0.00001/"epsilon": 0.0001/g' ./configs/2-slice/dpt/gms_2_.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_2_

    # 0.001
    sed -i 's/"epsilon": 0.0001/"epsilon": 0.001/g' ./configs/2-slice/dpt/gms_2_.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_2_

    # 0.01
    sed -i 's/"epsilon": 0.001/"epsilon": 0.01/g' ./configs/2-slice/dpt/gms_2_.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_2_

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/dpt/gms_2_.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_2_


    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/dpt/gms_2_.json


done
