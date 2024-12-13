#!/bin/bash
#SBATCH --job-name yelp_mia_mid_3          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 3-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_mia_mid_3.out


    # 0.5
for seed in 60 61 65
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_3
done
    
# 0.1
sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/2-slice/yelp_mia_mid_3.json
for seed in 60 61 62 63 64 65
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_3
done


# 0.01
sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/2-slice/yelp_mia_mid_3.json
for seed in 60 61 62 63 64 65
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_3
done

# 0.001
sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/2-slice/yelp_mia_mid_3.json
for seed in 60 61 62 63 64 65
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_3
done

# 0.0001
sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/2-slice/yelp_mia_mid_3.json
for seed in 60 61 62 63 64 65
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_3
done

# 0.00001
sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/2-slice/yelp_mia_mid_3.json
for seed in 60 61 62 65
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_3
done

sed -i 's/"lambda": 0.00001/"lambda": 0.5/g' ./configs/2-slice/yelp_mia_mid_3.json
