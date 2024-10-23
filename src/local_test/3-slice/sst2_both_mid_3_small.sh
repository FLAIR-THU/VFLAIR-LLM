for seed in {60,61,62,63,64,65}
    do
  
    # 0.01
    sed -i 's/"lambda": 0.5/"lambda": 0.01/g' ./configs/3-slice/sst2_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_small

    # 1e-3
    sed -i 's/"lambda": 0.01/"lambda": 1e-3/g' ./configs/3-slice/sst2_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_small

    # 1e-4
    sed -i 's/"lambda": 1e-3/"lambda": 1e-4/g' ./configs/3-slice/sst2_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_small

    # 1e-5
    sed -i 's/"lambda": 1e-4/"lambda": 1e-5/g' ./configs/3-slice/sst2_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_small

    # 1e-6
    sed -i 's/"lambda": 1e-5/"lambda": 1e-6/g' ./configs/3-slice/sst2_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_small

    # 1e-9
    sed -i 's/"lambda": 1e-6/"lambda": 1e-9/g' ./configs/3-slice/sst2_both_mid_3_small.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_small


    sed -i 's/"lambda": 1e-9/"lambda": 0.5/g' ./configs/3-slice/sst2_both_mid_3_small.json

done
