for seed in {60,61,62,63,64,65}
    do
    # 0.5
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 1e-3
    sed -i 's/"lambda": 0.01/"lambda": 1e-3/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 1e-4
    sed -i 's/"lambda": 1e-3/"lambda": 1e-4/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 1e-5
    sed -i 's/"lambda": 1e-4/"lambda": 1e-5/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 5
    sed -i 's/"lambda": 1e-5/"lambda": 5/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 10
    sed -i 's/"lambda": 5/"lambda": 10/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 50
    sed -i 's/"lambda": 10/"lambda": 50/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    # 100
    sed -i 's/"lambda": 50/"lambda": 100/g' ./configs/3-slice/sst2_lia_mid_3.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/sst2_lia_mid_3

    sed -i 's/"lambda": 100/"lambda": 0.5/g' ./configs/3-slice/sst2_lia_mid_3.json

done
