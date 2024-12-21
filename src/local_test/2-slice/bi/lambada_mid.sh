#!/bin/bash
#SBATCH --job-name CoLA Finetuned             # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完

# # 0.5
# for seed in 1 3 #60 64
#     do
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_mid
# done

# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/2-slice/bi/lambada_mid.json
for seed in 64 #1 6 7  #61 
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_mid
done

# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/2-slice/bi/lambada_mid.json
for seed in 60 1 #2 3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_mid
done

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/2-slice/bi/lambada_mid.json
for seed in 60 61 #62 64 1 2 3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_mid
done


# 0.0001
sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/2-slice/bi/lambada_mid.json
for seed in 62 64 #1 6
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_mid
done


# 1e-5
sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/2-slice/bi/lambada_mid.json
for seed in 63 64 #3
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_mid
done

sed -i 's/"lambda": 0.00001/"lambda": 0.5/g' ./configs/2-slice/bi/lambada_mid.json

