#!/bin/bash
#SBATCH --job-name alpaca_ob          # 任务名叫 example
#SBATCH --gres gpu:a100:3                # 每个子任务都用一张 A100 GPU
#SBATCH --time 2-12:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output exp_result/alpaca_ob.out
#SBATCH --mem 160000MB


for seed in 6
    do 


    python main_pipeline_llm.py --prefix "ob_test" --seed $seed --configs 2-slice/ob/alpaca_test

    sed -i 's/"lr": 5e-5/"lr": 1e-5/g' ./configs/2-slice/ob/alpaca_test.json
    python main_pipeline_llm.py --prefix "ob_test" --seed $seed --configs 2-slice/ob/alpaca_test
    sed -i 's/"lr": 1e-5/"lr": 5e-5/g' ./configs/2-slice/ob/alpaca_test.json
    
    sed -i 's/"lr": 5e-5/"lr": 1e-4/g' ./configs/2-slice/ob/alpaca_test.json
    python main_pipeline_llm.py --prefix "ob_test" --seed $seed --configs 2-slice/ob/alpaca_test
    sed -i 's/"lr": 1e-5/"lr": 1e-4/g' ./configs/2-slice/ob/alpaca_test.json
    

    #####3
    sed -i 's/"w_cluster_close": 0.3/"w_cluster_close": 0.1/g' ./configs/2-slice/ob/alpaca_test.json
    
    python main_pipeline_llm.py --prefix "ob_test" --seed $seed --configs 2-slice/ob/alpaca_test

    sed -i 's/"lr": 5e-5/"lr": 1e-5/g' ./configs/2-slice/ob/alpaca_test.json
    python main_pipeline_llm.py --prefix "ob_test" --seed $seed --configs 2-slice/ob/alpaca_test
    sed -i 's/"lr": 1e-5/"lr": 5e-5/g' ./configs/2-slice/ob/alpaca_test.json
    
    sed -i 's/"lr": 5e-5/"lr": 1e-4/g' ./configs/2-slice/ob/alpaca_test.json
    python main_pipeline_llm.py --prefix "ob_test" --seed $seed --configs 2-slice/ob/alpaca_test
    sed -i 's/"lr": 1e-5/"lr": 1e-4/g' ./configs/2-slice/ob/alpaca_test.json
    
    
    sed -i 's/"w_cluster_close": 0.1/"w_cluster_close": 0.3/g' ./configs/2-slice/ob/alpaca_test.json
    

done
