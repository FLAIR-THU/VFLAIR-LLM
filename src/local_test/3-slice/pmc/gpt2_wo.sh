#!/bin/bash
python main_pipeline_llm.py --prefix "pmc_lora" --attack_only 0 --save_model 1 --seed 60 --configs 3-slice/pmc/gpt2_wo
# python main_pipeline_llm.py --prefix "pmc" --save_model 1 --seed 61 --configs 3-slice/pmc/gpt2
# python main_pipeline_llm.py --prefix "pmc" --save_model 1 --seed 62 --configs 3-slice/pmc/gpt2
# python main_pipeline_llm.py --prefix "pmc" --save_model 1 --seed 63 --configs 3-slice/pmc/gpt2
