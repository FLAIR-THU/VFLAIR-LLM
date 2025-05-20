#!/bin/bash



python main_pipeline_llm.py --prefix "pmc_dolly_va" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_95_va

python main_pipeline_llm.py --prefix "pmc_dolly_va" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_96_va

python main_pipeline_llm.py --prefix "pmc_dolly_va" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_97_va

python main_pipeline_llm.py --prefix "pmc_dolly_va" --attack_only 0 --save_model 0 --seed 97 --configs 3-slice/dolly/llama_gs_98_va
