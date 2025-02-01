
for seed in 60 61 62 63 
    do

    # 2
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/sst2_wo

    # 3
    sed -i 's/"local_encoders_num": 2/"local_encoders_num": 3/g' ./configs/2-slice/dp/sst2_wo.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/sst2_wo

    # 4
    sed -i 's/"local_encoders_num": 3/"local_encoders_num": 4/g' ./configs/2-slice/dp/sst2_wo.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/sst2_wo

    # 5
    sed -i 's/"local_encoders_num": 4/"local_encoders_num": 5/g' ./configs/2-slice/dp/sst2_wo.json
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/dp/sst2_wo

    sed -i 's/"local_encoders_num": 5/"local_encoders_num": 2/g' ./configs/2-slice/dp/sst2_wo.json

done