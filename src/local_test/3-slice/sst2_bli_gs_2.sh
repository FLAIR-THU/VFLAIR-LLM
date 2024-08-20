for seed in {61,62,63,64,65}
    do

    # 100
    python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_gs_2

    # # 99.5
    # sed -i 's/"gradient_sparse_rate": 100.0/"gradient_sparse_rate": 99.5/g' ./configs/sst2_bli_gs_2.json
    # python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_gs_2

    # # 99
    # sed -i 's/"gradient_sparse_rate": 99.5/"gradient_sparse_rate": 99/g' ./configs/sst2_bli_gs_2.json
    # python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_gs_2

    # # 98
    # sed -i 's/"gradient_sparse_rate": 99/"gradient_sparse_rate": 98/g' ./configs/sst2_bli_gs_2.json
    # python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_gs_2

    # # 97
    # sed -i 's/"gradient_sparse_rate": 98/"gradient_sparse_rate": 97/g' ./configs/sst2_bli_gs_2.json
    # python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_gs_2

    # # 96
    # sed -i 's/"gradient_sparse_rate": 97/"gradient_sparse_rate": 96/g' ./configs/sst2_bli_gs_2.json
    # python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_gs_2

    # # 95
    # sed -i 's/"gradient_sparse_rate": 96/"gradient_sparse_rate": 95/g' ./configs/sst2_bli_gs_2.json
    # python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_gs_2


    # sed -i 's/"gradient_sparse_rate": 95/"gradient_sparse_rate": 100.0/g' ./configs/sst2_bli_gs_2.json

done
