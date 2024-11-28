#!/bin/bash
#SBATCH --job-name gms_lia_mid_3      # 任务名叫 example
#SBATCH --gres gpu:a100:4                # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/gms_lia_mid_3.out



# head+tail lr = 0.0001
python main_pipeline_llm_LIA.py --seed 60 --configs 3-slice/gms_lia_mid_3 

sed -i 's/"lambda": 1e-5/"lambda": 0.5/g' ./configs/3-slice/gms_lia_mid_3.json
python main_pipeline_llm_LIA.py --seed 60 --configs 3-slice/gms_lia_mid_3 

sed -i 's/"lambda": 0.5/"lambda": 1e-3/g' ./configs/3-slice/gms_lia_mid_3.json
python main_pipeline_llm_LIA.py --seed 60 --configs 3-slice/gms_lia_mid_3 

sed -i 's/"lambda": 1e-3/"lambda": 1e-5/g' ./configs/3-slice/gms_lia_mid_3.json
