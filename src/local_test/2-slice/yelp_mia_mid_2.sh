#!/bin/bash
#SBATCH --job-name yelp_mia_mid_2          # 任务名叫 example
#SBATCH --gres gpu:a100:2                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_mia_mid_2.out

# # Get the GPU IDs assigned to this job
# GPU_IDS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n")
# echo "Applied GPU: $GPU_IDS"

# # Split the GPU IDs into an array
# IFS=$'\n' read -d '' -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
# echo "GPU_ID_ARRAY: ${GPU_ID_ARRAY[@]}"

# cuda0=${GPU_ID_ARRAY[0]}
# echo "CUDA 0: $cuda0"

# cuda1=${GPU_ID_ARRAY[1]}
# echo "CUDA 1: $cuda1"

# cuda2=${GPU_ID_ARRAY[2]}
# echo "CUDA 2: $cuda2"

# CUDA_VISIBLE_DEVICES=$cuda0,$cuda1 python main_pipeline_llm_Both.py --seed 61 --configs 3-slice/yelp_both_dp_100
# CUDA_VISIBLE_DEVICES=$cuda2,$cuda3 python main_pipeline_llm_Both.py --seed 62 --configs 3-slice/yelp_both_dp_100

for seed in 60 61 62 63 64 65
    do
    ##########
    # 0.5
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_2

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/2-slice/yelp_mia_mid_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_2


    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/2-slice/yelp_mia_mid_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_2

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/2-slice/yelp_mia_mid_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_2


    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/2-slice/yelp_mia_mid_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_2


    # 0.00001
    sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/2-slice/yelp_mia_mid_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_mid_2



    sed -i 's/"lambda": 0.00001/"lambda": 0.5/g' ./configs/2-slice/yelp_mia_mid_2.json

done