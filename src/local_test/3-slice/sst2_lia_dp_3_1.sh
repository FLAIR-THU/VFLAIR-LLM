for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_wo_3_1

    # 1e5
    sed -i 's/"epsilon": 50/"epsilon": 1e5/g' ./configs/3-slice/sst2_lia_dp_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_dp_3_1

    # 1e6
    sed -i 's/"epsilon": 1e5/"epsilon": 1e6/g' ./configs/3-slice/sst2_lia_dp_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_dp_3_1

    # 1e7
    sed -i 's/"epsilon": 1e6/"epsilon": 1e7/g' ./configs/3-slice/sst2_lia_dp_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_dp_3_1

    # 1e8
    sed -i 's/"epsilon": 1e7/"epsilon": 1e8/g' ./configs/3-slice/sst2_lia_dp_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_dp_3_1

    # 1e9
    sed -i 's/"epsilon": 1e8/"epsilon": 1e9/g' ./configs/3-slice/sst2_lia_dp_3_1.json
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_dp_3_1

    sed -i 's/"epsilon": 1e9/"epsilon": 50/g' ./configs/3-slice/sst2_lia_dp_3_1.json

done

for seed in {10,11,12,13}
    do
    python main_pipeline_llm_LIA_1.py --seed $seed --configs 3-slice/sst2_lia_wo_3_1
done