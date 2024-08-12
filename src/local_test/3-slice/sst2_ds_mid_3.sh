for seed in {60,61,62,63,64,65}
    do
    # 0.5
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/sst2_ds_mid_3.json
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/sst2_ds_mid_3.json
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/sst2_ds_mid_3.json
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/sst2_ds_mid_3.json
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    # 1e-5
    sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/sst2_ds_mid_3.json
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    # 0.005
    sed -i 's/"lambda": 0.00001/"lambda": 0.005/g' ./configs/sst2_ds_mid_3.json
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    # 0.05
    sed -i 's/"lambda": 0.005/"lambda": 0.05/g' ./configs/sst2_ds_mid_3.json
    python main_pipeline_llm_3.py --seed $seed --configs sst2_ds_mid_3

    sed -i 's/"lambda": 0.05/"lambda": 0.5/g' ./configs/sst2_ds_mid_3.json

done
