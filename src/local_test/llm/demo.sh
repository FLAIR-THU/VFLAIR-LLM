#!/bin/bash

YOUR_PREFIX="exp1"
YOUR_CONFIG_FILE="/configs/llm_configs/vanilla/3-slice/codealpaca_codellama"

python main_pipeline_llm.py --prefix "$YOUR_PREFIX" --attack_only 0 --save_model 1 --seed 60 --configs "$YOUR_CONFIG_FILE"
