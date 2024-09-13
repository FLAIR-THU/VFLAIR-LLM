python main_pipeline_llm2.py --seed 61 --configs 2-slice/sst2_dli_wo_2

for seed in {60,62,63,64,65}
    do
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_wo_2

    # 50
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 80
    sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 100
    sed -i 's/"epsilon": 80/"epsilon": 100/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 5000
    sed -i 's/"epsilon": 500/"epsilon": 5000/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 10000
    sed -i 's/"epsilon": 5000/"epsilon": 10000/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 100000
    sed -i 's/"epsilon": 10000/"epsilon": 100000/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 50000
    sed -i 's/"epsilon": 100000/"epsilon": 50000/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2

    # 500000
    sed -i 's/"epsilon": 50000/"epsilon": 500000/g' ./configs/2-slice/sst2_dli_dp_2.json
    python main_pipeline_llm2.py --seed $seed --configs 2-slice/sst2_dli_dp_2


    sed -i 's/"epsilon": 500000/"epsilon": 50/g' ./configs/2-slice/sst2_dli_dp_2.json


done
