for seed in {60,61,62,63,64,65}
    do
    # # 0.5
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    # 1e-3
    sed -i 's/"lambda": 0.5/"lambda": 1e-3/g' ./configs/3-slice/sst2_bli_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    # 1e-4
    sed -i 's/"lambda": 1e-3/"lambda": 1e-4/g' ./configs/3-slice/sst2_bli_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    # 1e-5
    sed -i 's/"lambda": 1e-4/"lambda": 1e-5/g' ./configs/3-slice/sst2_bli_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    # 1e-6
    sed -i 's/"lambda": 1e-5/"lambda": 1e-6/g' ./configs/3-slice/sst2_bli_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    # 1e-7
    sed -i 's/"lambda": 1e-6/"lambda": 1e-7/g' ./configs/3-slice/sst2_bli_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    # 1e-8
    sed -i 's/"lambda": 1e-7/"lambda": 1e-8/g' ./configs/3-slice/sst2_bli_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    # 1e-9
    sed -i 's/"lambda": 1e-8/"lambda": 1e-9/g' ./configs/3-slice/sst2_bli_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_bli_mid_3

    sed -i 's/"lambda": 0.05/"lambda": 0.5/g' ./configs/3-slice/sst2_bli_mid_3.json

done
