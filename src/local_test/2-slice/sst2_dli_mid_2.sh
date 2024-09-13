for seed in {60,61,62,63,64,65}
    do
    # 0.5
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 1e-5
    sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 0.005
    sed -i 's/"lambda": 0.00001/"lambda": 0.005/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 0.05
    sed -i 's/"lambda": 0.005/"lambda": 0.05/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 1e-6
    sed -i 's/"lambda": 0.05/"lambda": 1e-6/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 1e-7
    sed -i 's/"lambda": 1e-6/"lambda": 1e-7/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 1e-8
    sed -i 's/"lambda": 1e-7/"lambda": 1e-8/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    # 1e-9
    sed -i 's/"lambda": 1e-8/"lambda": 1e-9/g' ./configs/2-slice/sst2_dli_mid_2.json
    python main_pipeline_llm_new.py --seed $seed --configs 2-slice/sst2_dli_mid_2

    sed -i 's/"lambda": 1e-9/"lambda": 0.5/g' ./configs/2-slice/sst2_dli_mid_2.json


done
