#!/bin/bash

# seed96 new lr1e-4

python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 96 --configs 3-slice/dolly/llama_wo_lo_16_1


# python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 96 --configs 3-slice/dolly/llama_gs_97_lo16_1


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 96 --configs 3-slice/dolly/llama_gs_98_lo16_1
