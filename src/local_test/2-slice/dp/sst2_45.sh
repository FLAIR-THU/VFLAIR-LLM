
for seed in 60 61 62 63 
    do

    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_4

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/dp/sst2_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_4

    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/2-slice/dp/sst2_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_4

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/dp/sst2_4.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_4

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/dp/sst2_4.json

    #############3
    # 50
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_5

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/dp/sst2_5.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_5

    # 100
    sed -i 's/"epsilon": 70/"epsilon": 100/g' ./configs/2-slice/dp/sst2_5.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_5

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/dp/sst2_5.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/dp/sst2_5

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/2-slice/dp/sst2_5.json

done