#!/bin/bash
python main_pipeline_llm.py --prefix "origin" --save_model 0 --seed 60 --configs 3-slice/pmc/gpt2_origin
python main_pipeline_llm.py --prefix "origin" --save_model 0 --seed 60 --configs 3-slice/pmc/llama_origin
