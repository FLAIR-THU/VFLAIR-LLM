for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_wo_3

    # 1e5
    sed -i 's/"epsilon": 50/"epsilon": 1e5/g' ./configs/3-slice/sst2_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_dp_3

    # 1e6
    sed -i 's/"epsilon": 1e5/"epsilon": 1e6/g' ./configs/3-slice/sst2_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_dp_3

    # 1e7
    sed -i 's/"epsilon": 1e6/"epsilon": 1e7/g' ./configs/3-slice/sst2_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_dp_3

    # 1e8
    sed -i 's/"epsilon": 1e7/"epsilon": 1e8/g' ./configs/3-slice/sst2_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_dp_3

    # 1e9
    sed -i 's/"epsilon": 1e8/"epsilon": 1e9/g' ./configs/3-slice/sst2_lia_dp_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_dp_3

    sed -i 's/"epsilon": 1e9/"epsilon": 50/g' ./configs/3-slice/sst2_lia_dp_3.json

done

for seed in {10,11,12,13}
    do
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_wo_3
done