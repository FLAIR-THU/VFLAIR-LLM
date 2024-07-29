   

for seed in {60,61,62,63,64,65}
    do

    # 0.001
    python main_pipeline_llm3.py --seed $seed --configs lambada_ad_3

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/lambada_ad_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_ad_3

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/lambada_ad_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_ad_3

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/lambada_ad_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_ad_3

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/lambada_ad_3.json
    python main_pipeline_llm3.py --seed $seed --configs lambada_ad_3


    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/lambada_ad_3.json
    
done
