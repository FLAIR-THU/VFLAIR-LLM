# 1e-5
sed -i 's/"lambda": 0.5/"lambda": 1e-5/g' ./configs/3-slice/yelp_lia_mid_3_small.json
python main_pipeline_llm_LIA.py --seed 60 --configs 3-slice/yelp_lia_mid_3_small

sed -i 's/"lambda": 1e-5/"lambda": 0.5/g' ./configs/3-slice/yelp_lia_mid_3_small.json

for seed in {61,62,63,64,65}
    do
    # 0.5
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_mid_3_small

    # 0.1
    sed -i 's/"lambda": 0.5/"lambda": 0.1/g' ./configs/3-slice/yelp_lia_mid_3_small.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_mid_3_small

    # 1e-2
    sed -i 's/"lambda": 0.1/"lambda": 0.01/g' ./configs/3-slice/yelp_lia_mid_3_small.json
    # python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_mid_3_small

    # 1e-3
    sed -i 's/"lambda": 0.01/"lambda": 1e-3/g' ./configs/3-slice/yelp_lia_mid_3_small.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_mid_3_small

    # 1e-4
    sed -i 's/"lambda": 1e-3/"lambda": 1e-4/g' ./configs/3-slice/yelp_lia_mid_3_small.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_mid_3_small

    # 1e-5
    sed -i 's/"lambda": 1e-4/"lambda": 1e-5/g' ./configs/3-slice/yelp_lia_mid_3_small.json
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/yelp_lia_mid_3_small

   
    sed -i 's/"lambda": 1e-5/"lambda": 0.5/g' ./configs/3-slice/yelp_lia_mid_3_small.json

done
