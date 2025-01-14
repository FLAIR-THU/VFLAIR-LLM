for seed in 60 61 62
    do

    # wo
    python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/bi/gms_wo

    # # 50
    # sed -i 's/"epsilon": 1/"epsilon": 50/g' ./configs/2-slice/bi/gms_custext.json
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/bi/gms_custext

    # # 30
    # sed -i 's/"epsilon": 50/"epsilon": 30/g' ./configs/2-slice/bi/gms_custext.json
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/bi/gms_custext

    # # 1
    # sed -i 's/"epsilon": 30/"epsilon": 1/g' ./configs/2-slice/bi/gms_custext.json
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/bi/gms_custext

    # # 0.01
    # sed -i 's/"epsilon": 1/"epsilon": 0.01/g' ./configs/2-slice/bi/gms_custext.json
    # python main_pipeline_llm_MIA_attackonly.py --seed $seed --configs 2-slice/bi/gms_custext

    # sed -i 's/"epsilon": 0.01/"epsilon": 1/g' ./configs/2-slice/bi/gms_custext.json
done