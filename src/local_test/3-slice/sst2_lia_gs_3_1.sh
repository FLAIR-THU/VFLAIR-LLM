for seed in {60,61,62,63,64,65}
    do

    # 100
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_gs_3_1

    # 99.5
    sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/3-slice/sst2_lia_gs_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_gs_3_1

    # 99
    sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99/g' ./configs/3-slice/sst2_lia_gs_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_gs_3_1

    # 98
    sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/3-slice/sst2_lia_gs_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_gs_3_1

    # 97
    sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/3-slice/sst2_lia_gs_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_gs_3_1

    # 96
    sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/3-slice/sst2_lia_gs_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_gs_3_1

    # 95
    sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/3-slice/sst2_lia_gs_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_gs_3_1


    sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/3-slice/sst2_lia_gs_3_1.json

done
