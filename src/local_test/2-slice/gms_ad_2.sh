#!/bin/bash
#SBATCH --job-name gms_mia_ad_2            # 任务名叫 example
#SBATCH --gres gpu:a100:3                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 4-1:00:00                    # 子任务 1 天 1 小时就能跑完

# 0.1
sed -i 's/"lambda": 0.001/"lambda": 0.1/g' ./configs/2-slice/gms_mia_ad_2.json
python main_pipeline_llm_MIA.py --seed 64  --configs 2-slice/gms_mia_ad_2
sed -i 's/"lambda": 0.1/"lambda": 0.001/g' ./configs/2-slice/gms_mia_ad_2.json



# 0.001
python main_pipeline_llm_MIA.py --seed 65 --configs 2-slice/gms_mia_ad_2

# 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/2-slice/gms_mia_ad_2.json
python main_pipeline_llm_MIA.py --seed 65 --configs 2-slice/gms_mia_ad_2

# 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/2-slice/gms_mia_ad_2.json
python main_pipeline_llm_MIA.py --seed 65 --configs 2-slice/gms_mia_ad_2

# 1
sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/2-slice/gms_mia_ad_2.json
# python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_ad_2

# 5
sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/2-slice/gms_mia_ad_2.json
# python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_ad_2


sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/2-slice/gms_mia_ad_2.json

