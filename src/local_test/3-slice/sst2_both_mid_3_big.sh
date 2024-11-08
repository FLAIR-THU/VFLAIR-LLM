for seed in {1,2,3,4,5}
    do
    # 0.5
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_big

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/sst2_both_mid_3_big.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_big

    # 5
    sed -i 's/"lambda": 0.1/"lambda": 5/g' ./configs/3-slice/sst2_both_mid_3_big.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_big

    # 10
    sed -i 's/"lambda": 5/"lambda": 10/g' ./configs/3-slice/sst2_both_mid_3_big.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_big

    # 50
    sed -i 's/"lambda": 10/"lambda": 50/g' ./configs/3-slice/sst2_both_mid_3_big.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_big

    # 100
    sed -i 's/"lambda": 50/"lambda": 100/g' ./configs/3-slice/sst2_both_mid_3_big.json
    python main_pipeline_llm_Both.py --seed $seed --configs 3-slice/sst2_both_mid_3_big

    sed -i 's/"lambda": 100/"lambda": 0.5/g' ./configs/3-slice/sst2_both_mid_3_big.json

done
