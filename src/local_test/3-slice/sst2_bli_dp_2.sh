# 500
sed -i 's/"epsilon": 50/"epsilon": 500/g' ./configs/sst2_bli_dp_2.json
python main_pipeline_llm_2.py --seed 64 --configs sst2_bli_dp_2
sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_bli_dp_2.json

for seed in {65,66}
    do
    python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_wo_2

    # 50
    python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_dp_2

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/sst2_bli_dp_2.json
    python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_dp_2

    # 80
    sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/sst2_bli_dp_2.json
    python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_dp_2

    # 100
    sed -i 's/"epsilon": 80/"epsilon": 100/g' ./configs/sst2_bli_dp_2.json
    python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_dp_2

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/sst2_bli_dp_2.json
    python main_pipeline_llm_2.py --seed $seed --configs sst2_bli_dp_2

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/sst2_bli_dp_2.json

done
