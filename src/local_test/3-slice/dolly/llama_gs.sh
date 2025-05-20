#!/bin/bash




python main_pipeline_llm.py --prefix "pmc_dolly_gs" --attack_only 0 --save_model 0 --seed 60 --configs 3-slice/dolly/llama_wo

python main_pipeline_llm.py --prefix "pmc_dolly_gs" --attack_only 0 --save_model 0 --seed 60 --configs 3-slice/dolly/llama_gs_96

