#!/bin/bash

# untrained gpt2 + untrained ad
python main_pipeline_llm.py --prefix "pmc_mid_2" --attack_only 0 --save_model 1 --seed 60 --configs 3-slice/pmc/gpt2_mid_2