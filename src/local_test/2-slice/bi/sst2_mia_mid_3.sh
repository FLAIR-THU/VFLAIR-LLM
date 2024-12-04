for seed in {60,61,62,63,64,65}
    do
    # 0.5
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_mid_3

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/2-slice/bi/sst2_mia_mid_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_mid_3

    # 0.01
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/2-slice/bi/sst2_mia_mid_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_mid_3

    # 0.001
    sed -i 's/"lambda": 0.01/"lambda": 0.001/g' ./configs/2-slice/bi/sst2_mia_mid_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_mid_3

    # 0.0001
    sed -i 's/"lambda": 0.001/"lambda": 0.0001/g' ./configs/2-slice/bi/sst2_mia_mid_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_mid_3

    # 1e-5
    sed -i 's/"lambda": 0.0001/"lambda": 0.00001/g' ./configs/2-slice/bi/sst2_mia_mid_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/bi/sst2_mia_mid_3

    
    sed -i 's/"lambda": 0.00001/"lambda": 0.5/g' ./configs/2-slice/bi/sst2_mia_mid_3.json


done
