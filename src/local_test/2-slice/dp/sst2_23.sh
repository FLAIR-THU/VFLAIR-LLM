
for seed in 60 61 62 63 
    do

    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_2

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/dp/sst2_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_2

    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/2-slice/dp/sst2_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_2

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/dp/sst2_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_2

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/dp/sst2_2.json

    #############3
    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_3

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/dp/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_3

    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/2-slice/dp/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_3

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/dp/sst2_3.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_3

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/dp/sst2_3.json

done