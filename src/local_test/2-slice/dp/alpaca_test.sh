# python main_pipeline_llm_MIA_attackonly.py --seed 60 --configs 2-slice/dp/alpaca_wo

# 500
# python main_pipeline_llm_MIA_attackonly.py --seed 60 --configs 2-slice/dp/alpaca_dp

# 100
sed -i 's/"epsilon": 500/"epsilon": 100/g' ./configs/2-slice/dp/alpaca_dp.json
python main_pipeline_llm_MIA_attackonly.py --seed 60 --configs 2-slice/dp/alpaca_dp

# 50
sed -i 's/"epsilon": 100/"epsilon": 50/g' ./configs/2-slice/dp/alpaca_dp.json
python main_pipeline_llm_MIA_attackonly.py --seed 60 --configs 2-slice/dp/alpaca_dp

sed -i 's/"epsilon": 50/"epsilon": 100/g' ./configs/2-slice/dp/alpaca_dp.json
