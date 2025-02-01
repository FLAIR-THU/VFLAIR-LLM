#!/bin/bash
#SBATCH --job-name squad_finetune_2           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/squad_finetune_2.out
#SBATCH --mem 160000MB

for seed in 63 64 65
do
    # 2-slice va
    python main_pipeline_llm_full.py --seed $seed --configs 2-slice/vanilla/squad_full
    python main_pipeline_llm_passive.py --seed $seed --configs 2-slice/vanilla/squad_passive

    # # 3-slice va
    # python main_pipeline_llm_full.py --seed $seed --configs 3-slice/vanilla/squad_full
    # python main_pipeline_llm_passive.py --seed $seed --configs 3-slice/vanilla/squad_passive
    
    # 2-slice lora
    # python main_pipeline_llm_full.py --seed $seed --configs 2-slice/lora/squad_full
    # python main_pipeline_llm_passive.py --seed $seed --configs 2-slice/lora/squad_passive

    # # 3-slice lora
    # python main_pipeline_llm_full.py --seed $seed --configs 3-slice/lora/squad_full
    # python main_pipeline_llm_passive.py --seed $seed --configs 3-slice/lora/squad_passive

    
done
