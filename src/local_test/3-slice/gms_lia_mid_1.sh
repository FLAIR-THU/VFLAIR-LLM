for seed in {60,61,62,63,64,65}
    do
    # 0.5
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_mid_1

    # 1e-2
    sed -i 's/"lambda": 0.5/"lambda": 1e-2/g' ./configs/3-slice/gms_lia_mid_1.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_mid_1

    # 1e-4
    sed -i 's/"lambda": 1e-2/"lambda": 1e-4/g' ./configs/3-slice/gms_lia_mid_1.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_mid_1

    # 1e-6
    sed -i 's/"lambda": 1e-4/"lambda": 1e-6/g' ./configs/3-slice/gms_lia_mid_1.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_mid_1


    sed -i 's/"lambda": 1e-6s/"lambda": 0.5/g' ./configs/3-slice/gms_lia_mid_1.json

done
