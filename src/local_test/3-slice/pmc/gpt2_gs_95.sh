#!/bin/bash

# trained gpt2 + dp
python main_pipeline_llm.py --prefix "pmc_gs" --attack_only 0 --save_model 0 --seed 60 --configs 2-slice/gs/alpaca