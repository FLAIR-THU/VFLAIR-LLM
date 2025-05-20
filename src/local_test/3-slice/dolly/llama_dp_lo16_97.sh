#!/bin/bash



# python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_dp_500_lo16_97


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_dp_100_lo16_97


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_dp_70_lo16_97


python main_pipeline_llm.py --prefix "pmc_dolly_lo_r16" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_dp_50_lo16_97
