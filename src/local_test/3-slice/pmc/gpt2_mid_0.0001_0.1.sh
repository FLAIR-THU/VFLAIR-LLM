#!/bin/bash

# train + inference + attack
python main_pipeline_llm.py --prefix "pmc_mid_2" --attack_only 0 --save_model 1 --seed 64 --configs 3-slice/pmc/gpt2_mid_0.0001_0.1
python main_pipeline_llm.py --prefix "pmc_mid_2" --attack_only 0 --save_model 1 --seed 64 --configs 3-slice/pmc/gpt2_mid_0.00001_0.1
