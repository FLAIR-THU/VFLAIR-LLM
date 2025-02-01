for seed in 60 61 62 63
    do 
    python main_pipeline_llm.py --prefix "obcluster" --seed $seed --configs 2-slice/ob/cluster/gms_250
done 
