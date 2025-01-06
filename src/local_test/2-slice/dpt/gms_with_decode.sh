#!/bin/bash
#SBATCH --job-name gms_with_decode           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_with_decode.out

# test dpt with decode
for seed in {60,61}
    do
    # 0.00001
    sed -i 's/"epsilon": 1/"epsilon": 0.00001/g' ./configs/2-slice/dpt/gms_with_decode.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_with_decode

    # 0.0001
    sed -i 's/"epsilon": 0.00001/"epsilon": 0.0001/g' ./configs/2-slice/dpt/gms_with_decode.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_with_decode

    # 0.001
    sed -i 's/"epsilon": 0.0001/"epsilon": 0.001/g' ./configs/2-slice/dpt/gms_with_decode.json
    python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_with_decode

    # 0.01
    sed -i 's/"epsilon": 0.001/"epsilon": 0.01/g' ./configs/2-slice/dpt/gms_with_decode.json
    # python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_with_decode

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/dpt/gms_with_decode.json
    # python main_pipeline_llm_MIA_test.py --seed $seed --configs 2-slice/dpt/gms_with_decode


    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/dpt/gms_with_decode.json


done
