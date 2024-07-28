for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm3.py --seed $seed --configs lambada_wo_3

    # 50
    python main_pipeline_llm3.py --seed $seed --configs lambada_dp_3

    # 70
    sed -i 's/"epsilon": 50/"epsilon": 70/g' ./configs/lambada_dp_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_dp_3

    # 80
    sed -i 's/"epsilon": 70/"epsilon": 80/g' ./configs/lambada_dp_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_dp_3

    # 100
    sed -i 's/"epsilon": 80/"epsilon": 100/g' ./configs/lambada_dp_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_dp_3

    # 500
    sed -i 's/"epsilon": 100/"epsilon": 500/g' ./configs/lambada_dp_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_dp_3

    sed -i 's/"epsilon": 500/"epsilon": 50/g' ./configs/lambada_dp_3.json

done