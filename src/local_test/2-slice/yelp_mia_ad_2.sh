# 0.1
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/yelp_mia_ad_2

# 1
sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/2-slice/yelp_mia_ad_2.json
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/yelp_mia_ad_2

# 5
sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/2-slice/yelp_mia_ad_2.json
python main_pipeline_llm_MIA.py --seed 60 --configs 2-slice/yelp_mia_ad_2

sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/2-slice/yelp_mia_ad_2.json

for seed in {61,62,63,64,65}
    do
    # 0.001
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_2

    # 0.01
    sed -i 's/"lambda": 0.001/"lambda": 0.01/g' ./configs/2-slice/yelp_mia_ad_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_2

    # 0.1
    sed -i 's/"lambda": 0.01/"lambda": 0.1/g' ./configs/2-slice/yelp_mia_ad_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_2

    # 1
    sed -i 's/"lambda": 0.1/"lambda": 1.0/g' ./configs/2-slice/yelp_mia_ad_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_2

    # 5
    sed -i 's/"lambda": 1.0/"lambda": 5.0/g' ./configs/2-slice/yelp_mia_ad_2.json
    python main_pipeline_llm_MIA.py --seed $seed --configs 2-slice/yelp_mia_ad_2

    sed -i 's/"lambda": 5.0/"lambda": 0.001/g' ./configs/2-slice/yelp_mia_ad_2.json

done
