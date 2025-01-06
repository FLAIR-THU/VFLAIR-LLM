#!/bin/bash
#SBATCH --job-name gms_ob_pretrain_big          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-16:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_ob_pretrain_big.out
#SBATCH --mem 160000MB


for seed in 2 3
    do 

    # 250
    sed -i 's/"cluster_num": 50/"cluster_num": 250/g' ./configs/2-slice/ob/gms_nodecode_big.json
    python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_big

    # 200
    sed -i 's/"cluster_num": 250/"cluster_num": 200/g' ./configs/2-slice/ob/gms_nodecode_big.json
    python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_big

    # 150
    sed -i 's/"cluster_num": 200/"cluster_num": 150/g' ./configs/2-slice/ob/gms_nodecode_big.json
    python main_pipeline_llm_MIA_nodecode.py --seed $seed --configs 2-slice/ob/gms_nodecode_big
    
    sed -i 's/"cluster_num": 150/"cluster_num": 50/g' ./configs/2-slice/ob/gms_nodecode_big.json

done
