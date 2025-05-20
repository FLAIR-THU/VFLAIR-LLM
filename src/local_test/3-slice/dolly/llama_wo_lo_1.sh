#!/bin/bash

# seed96 new lr1e-4


# seed96 model
python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 96 --configs 3-slice/dolly/llama_gs_95_lo16_1

# seed97 model
python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_95_lo16_97


# seed97 model
python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 96 --configs 3-slice/dolly/llama_gs_96_lo16_97
