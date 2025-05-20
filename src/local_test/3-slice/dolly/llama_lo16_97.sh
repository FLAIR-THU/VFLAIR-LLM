#!/bin/bash

python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_wo_lo16_97


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_95_lo16_97


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_96_lo16_97


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_97_lo16_97


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_98_lo16_97
