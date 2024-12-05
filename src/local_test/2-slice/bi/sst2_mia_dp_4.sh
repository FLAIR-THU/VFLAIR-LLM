#!/bin/bash
#SBATCH --job-name sst2_bi_dp_4          # 任务名叫 example
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 1-1:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/sst2_bi_dp_4.out

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

# cuda3=${GPU_ID_ARRAY[3]}
# echo "CUDA 3: $cuda3"

# for seed in 1 2 3 4 5 6
# do
#     CUDA_VISIBLE_DEVICES=$cuda0,$cuda1 python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_mid_0.5 &
#     CUDA_VISIBLE_DEVICES=$cuda2,$cuda3 python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_mid_0.1

#     CUDA_VISIBLE_DEVICES=$cuda0,$cuda1 python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_mid_1e-2 &
#     CUDA_VISIBLE_DEVICES=$cuda2,$cuda3 python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_mid_1e-3

#     CUDA_VISIBLE_DEVICES=$cuda0,$cuda1 python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_mid_1e-4 &
#     CUDA_VISIBLE_DEVICES=$cuda2,$cuda3 python main_pipeline_llm_MIA_new.py --seed $seed --configs 2-slice/gms_mia_mid_1e-5
# done
for seed in 60 61 62 63 64 65
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/sst2_mia_wo_4

    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_dp_4

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/bi/sst2_mia_dp_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_dp_4

  
    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/2-slice/bi/sst2_mia_dp_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_dp_4

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/bi/sst2_mia_dp_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_dp_4

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/bi/sst2_mia_dp_4.json

done
