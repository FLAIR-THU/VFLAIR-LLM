#!/bin/bash
#SBATCH --job-name lambada_bi_ad          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/lambada_bi_ad.out
#SBATCH --mem 100000MB

# 0.001
for seed in 61 #62
    # 62
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_ad
done


# 0.01
sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/2-slice/bi/lambada_ad.json
for seed in 61 #62
    #61 62
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_ad
done

# 0.1
sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/2-slice/bi/lambada_ad.json
for seed in 62 #63
    # 62 63
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_ad
done

# 1
sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/2-slice/bi/lambada_ad.json
for seed in 61 #62
    #61 62
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_ad
done


# 5
sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/2-slice/bi/lambada_ad.json
for seed in 61
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/lambada_ad
done
sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/2-slice/bi/lambada_ad.json

