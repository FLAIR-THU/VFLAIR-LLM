#!/bin/bash
# Define the config files to process
configs=(
  "3-slice/pmc/gpt2_gs_95"
  "3-slice/pmc/gpt2_gs_96"
  "3-slice/pmc/gpt2_gs_97"
  "3-slice/pmc/gpt2_gs_98"
)

# Then run the commands for each config
for config in "${configs[@]}"; do
  echo "Processing config: $config"
  python main_pipeline_llm.py --prefix "pmc_lora" --attack_only 0 --save_model 0 --seed 60 --configs "$config"
done

# Then run the commands for each config
for config in "${configs[@]}"; do
  echo "Processing config: $config"
  python main_pipeline_llm.py --prefix "pmc_lora" --attack_only 0 --save_model 0 --seed 80 --configs "$config"
done

# Then run the commands for each config
for config in "${configs[@]}"; do
  echo "Processing config: $config"
  python main_pipeline_llm.py --prefix "pmc_lora" --attack_only 0 --save_model 0 --seed 97 --configs "$config"
done

echo "All runs completed"