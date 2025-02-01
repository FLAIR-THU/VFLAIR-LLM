#!/bin/bash
#SBATCH --job-name 3_lambada_finetune           # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-10:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/all_lambada_finetune.out


for seed in 63 64 65
do
    # python main_pipeline_llm_full.py --seed $seed --configs 3-slice/vanilla/lambada_full
    # python main_pipeline_llm_full.py --seed $seed --configs 3-slice/lora/lambada_full

    # python main_pipeline_llm_passive.py --seed $seed --configs 3-slice/lora/lambada_passive
    python main_pipeline_llm_passive.py --seed $seed --configs 3-slice/vanilla/lambada_passive

    
    # python main_pipeline_llm_full.py --seed $seed --configs 2-slice/lora/lambada_full
    # python main_pipeline_llm_full.py --seed $seed --configs 2-slice/vanilla/lambada_full

    # python main_pipeline_llm_passive.py --seed $seed --configs 2-slice/lora/lambada_passive
    # python main_pipeline_llm_passive.py --seed $seed --configs 2-slice/vanilla/lambada_passive
    
done
