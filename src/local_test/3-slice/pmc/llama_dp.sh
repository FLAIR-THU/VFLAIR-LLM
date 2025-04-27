#!/bin/bash


# Define the attack seeds to iterate over
attack_seeds=(10)

# Define the config files to process
configs=(
  "3-slice/pmc/llama_dp_50"
  # "3-slice/pmc/llama_dp_70"

)

# Loop through each attack seed
for seed in "${attack_seeds[@]}"; do
  echo "Running with attack_seed: $seed"
  
  # First, update all config files with the current attack_seed
  for config in "${configs[@]}"; do
    config_file="./configs/${config}.json"
    if [ -f "$config_file" ]; then
      sed -i "s/\"attack_seed\":[0-9]/\"attack_seed\":$seed/g" "$config_file"
      sed -i "s/\"attack_seed\": [0-9]/\"attack_seed\": $seed/g" "$config_file"

    else
      echo "Config file not found: $config_file"
      exit 1
    fi
  done
  
  # Then run the commands for each config
  for config in "${configs[@]}"; do
    echo "Processing config: $config"
    python main_pipeline_llm.py --prefix "llama_dp" --attack_only 1 --save_model 0 --seed 10 --configs "$config"
  done
done

echo "All runs completed"

