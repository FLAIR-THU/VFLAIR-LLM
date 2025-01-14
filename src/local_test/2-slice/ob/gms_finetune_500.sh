#!/bin/bash
#SBATCH --job-name gms_ob_pretrain          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_ob_pretrain.out
#SBATCH --mem 160000MB


# gms finetune epsilon-500
for seed in 4 5
    do 

    # 250
    sed -i 's/"cluster_num": 50/"cluster_num": 250/g' ./configs/2-slice/ob/gms_finetune_500.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/gms_finetune_500

    # 200
    sed -i 's/"cluster_num": 250/"cluster_num": 200/g' ./configs/2-slice/ob/gms_finetune_500.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/gms_finetune_500

    # 150
    sed -i 's/"cluster_num": 200/"cluster_num": 150/g' ./configs/2-slice/ob/gms_finetune_500.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/gms_finetune_500
    

    # 100
    sed -i 's/"cluster_num": 150/"cluster_num": 100/g' ./configs/2-slice/ob/gms_finetune_500.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/gms_finetune_500

    # 50
    sed -i 's/"cluster_num": 100/"cluster_num": 50/g' ./configs/2-slice/ob/gms_finetune_500.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/ob/gms_finetune_500


done
