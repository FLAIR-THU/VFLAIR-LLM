#!/bin/bash
# Define the config files to process
configs=(
  "3-slice/pmc/gpt2_gs_90"
  "3-slice/pmc/gpt2_gs_96"
  "3-slice/pmc/gpt2_gs_97"
  "3-slice/pmc/gpt2_gs_98"
)

for config in "${configs[@]}"; do
    python main_pipeline_llm.py --prefix "gs" --attack_only 0 --save_model 0 --seed 60 --configs "$config"
done

configs=(
  "3-slice/pmc/gpt2_gs_50"
  "3-slice/pmc/gpt2_gs_70"
  "3-slice/pmc/gpt2_gs_90"
  "3-slice/pmc/gpt2_gs_95"
  "3-slice/pmc/gpt2_gs_96"
  "3-slice/pmc/gpt2_gs_97"
  "3-slice/pmc/gpt2_gs_98"
)
# Define the attack seeds to iterate over
attack_seeds=(2 3 4 1)

# Loop through each attack seed
for seed in "${attack_seeds[@]}"; do
  echo "Running with attack_seed: $seed"
  
  # First, update all config files with the current attack_seed
  for config in "${configs[@]}"; do
    config_file="./configs/${config}.json"
    if [ -f "$config_file" ]; then
      sed -i "s/\"attack_seed\": [0-9]/\"attack_seed\": $seed/g" "$config_file"
    else
      echo "Config file not found: $config_file"
      exit 1
    fi
  done
  
  # Then run the commands for each config
  for config in "${configs[@]}"; do
    echo "Processing config: $config"
    python main_pipeline_llm.py --prefix "gs" --attack_only 1 --save_model 0 --seed 60 --configs "$config"
  done
done

echo "All runs completed"