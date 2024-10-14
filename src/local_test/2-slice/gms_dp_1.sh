for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_wo_1

    # 50
    # python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_1

    # 1e2
    sed -i 's/"epsilon": 50/"epsilon": 1e2/g' ./configs/2-slice/gms_mia_dp_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_1

    # 1e3
    sed -i 's/"epsilon": 1e2/"epsilon": 1e3/g' ./configs/2-slice/gms_mia_dp_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_1

    # 1e4
    sed -i 's/"epsilon": 1e3/"epsilon": 1e4/g' ./configs/2-slice/gms_mia_dp_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_1

    # 1e5
    sed -i 's/"epsilon": 1e4/"epsilon": 1e5/g' ./configs/2-slice/gms_mia_dp_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_1
    
    # 1e6
    sed -i 's/"epsilon": 1e5/"epsilon": 1e6/g' ./configs/2-slice/gms_mia_dp_1.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/gms_mia_dp_1

    sed -i 's/"epsilon": 1e6/"epsilon": 50/g' ./configs/2-slice/gms_mia_dp_1.json

done
