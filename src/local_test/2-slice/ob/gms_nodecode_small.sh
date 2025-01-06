#!/bin/bash
#SBATCH --job-name gms_ob_pretrain_small          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-16:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_ob_pretrain_small.out
#SBATCH --mem 160000MB


for seed in 6 7
    do 

    # 250
    sed -i 's/"cluster_num": 50/"cluster_num": 250/g' ./configs/2-slice/ob/gms_nodecode_small.json
    python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_small

    # 200
    sed -i 's/"cluster_num": 250/"cluster_num": 200/g' ./configs/2-slice/ob/gms_nodecode_small.json
    python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_small

    # 150
    sed -i 's/"cluster_num": 200/"cluster_num": 150/g' ./configs/2-slice/ob/gms_nodecode_small.json
    python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_small
    

    # # 100
    # sed -i 's/"cluster_num": 150/"cluster_num": 100/g' ./configs/2-slice/ob/gms_nodecode_small.json
    # python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_small

    # # 50
    # sed -i 's/"cluster_num": 100/"cluster_num": 50/g' ./configs/2-slice/ob/gms_nodecode_small.json
    # python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_small


done
