for seed in 61 62 63 64 65
    do

    # 0.01
    sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 0.1
    sed -i 's/"epsilon": 0.01/"epsilon": 0.1/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 1
    sed -i 's/"epsilon": 0.1/"epsilon": 1/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    # 5
    sed -i 's/"epsilon": 1/"epsilon": 5/g' ./configs/2-slice/custext/alpaca.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/custext/alpaca

    sed -i 's/"epsilon": 5/"epsilon": 1/g' ./configs/2-slice/custext/alpaca.json
   
done
