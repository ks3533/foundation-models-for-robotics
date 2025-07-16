#!/bin/bash

python_script="Code/LiteLLM/main.py"

batch_size=5

# Define combined argument sets
configs=(
  "-b test"
)

cd ..

# Loop through each config and run the Python script
for config in "${configs[@]}"; do
  echo "Running: python $python_script $config"
  for ((i = 1; i <= batch_size; i++)); do
      echo "Iteration $i"
      python "$python_script" "$config"
  done
done