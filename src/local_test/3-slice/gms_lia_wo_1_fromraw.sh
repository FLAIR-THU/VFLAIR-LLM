
for seed in {60,61,62,63,64,65}
    do
    python main_pipeline_llm_LIA.py --seed $seed --configs 3-slice/gms_lia_wo_1_fromraw
done
