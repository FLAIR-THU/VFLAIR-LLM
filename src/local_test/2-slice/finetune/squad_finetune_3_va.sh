#!/bin/bash
#SBATCH --job-name squad_finetune_3           # 任务名叫 example
#SBATCH --gres gpu:a100:2                # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/squad_finetune_3.out
#SBATCH --mem 160000MB


for seed in 60 61 62
do
    # 2-slice va
    # python main_pipeline_llm_full.py --seed $seed --configs 2-slice/vanilla/squad_full
    # python main_pipeline_llm_passive.py --seed $seed --configs 2-slice/vanilla/squad_passive

    # 3-slice va
    python main_pipeline_llm_full.py --seed $seed --configs 3-slice/vanilla/squad_full
    python main_pipeline_llm_passive.py --seed $seed --configs 3-slice/vanilla/squad_passive
    
    # 2-slice lora_new
    # python main_pipeline_llm_full.py --seed $seed --configs 2-slice/lora_new/squad_full
    # python main_pipeline_llm_passive.py --seed $seed --configs 2-slice/lora_new/squad_passive

    # 3-slice lora_new
    # python main_pipeline_llm_full.py --seed $seed --configs 3-slice/lora_new/squad_full
    # python main_pipeline_llm_passive.py --seed $seed --configs 3-slice/lora_new/squad_passive

    
done
