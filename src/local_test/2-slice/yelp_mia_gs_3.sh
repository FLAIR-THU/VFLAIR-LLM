#!/bin/bash
#SBATCH --job-name yelp_mia_gs_3_2           # 任务名叫 example
#SBATCH --gres gpu:a100:1                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/yelp_mia_gs_3_2.out


for seed in 60 61 62 63 64 65
    do

    # 100
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_3

    # 99.5
    sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/2-slice/yelp_mia_gs_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_3

    # 99
    sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99/g' ./configs/2-slice/yelp_mia_gs_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_3

    # 98
    sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/2-slice/yelp_mia_gs_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_3

    # 97
    sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/2-slice/yelp_mia_gs_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_3

    # 96
    sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/2-slice/yelp_mia_gs_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_3

    # 95
    sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/2-slice/yelp_mia_gs_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_3


    sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/yelp_mia_gs_3.json



    # 100
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

    # 99.5
    sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/2-slice/yelp_mia_gs_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

    # 99
    sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99/g' ./configs/2-slice/yelp_mia_gs_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

    # 98
    sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/2-slice/yelp_mia_gs_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

    # 97
    sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/2-slice/yelp_mia_gs_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

    # 96
    sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/2-slice/yelp_mia_gs_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

    # 95
    sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/2-slice/yelp_mia_gs_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2


    sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/yelp_mia_gs_2.json


done


# for seed in {60,61,62,63,64,65}
#     do

#     # 100
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

#     # 99.5
#     sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/2-slice/yelp_mia_gs_2.json
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

#     # 99
#     sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99/g' ./configs/2-slice/yelp_mia_gs_2.json
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

#     # 98
#     sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/2-slice/yelp_mia_gs_2.json
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

#     # 97
#     sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/2-slice/yelp_mia_gs_2.json
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

#     # 96
#     sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/2-slice/yelp_mia_gs_2.json
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2

#     # 95
#     sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/2-slice/yelp_mia_gs_2.json
#     python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_gs_2


#     sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/2-slice/yelp_mia_gs_2.json

# done
