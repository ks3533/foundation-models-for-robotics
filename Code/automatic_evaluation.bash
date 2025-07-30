#!/bin/bash

python_script="Code/LiteLLM/main.py"

batch_size=20

# Define combined argument sets
# ran once:
#  "-b vision_o4 -v -p"
#  "-b json_vision_o4 -v -j -p"
# ran zero times / o3 auth problems
#  "-b vision_o3 -v -m o3 -p"
#  "-b json_vision_o3 -v -j -m o3 -p"
configs=(
  "-b no_vision_o4 -p"
  "-b vision_gpt-4o -v -m gpt-4o -p"
  "-b json_vision_gpt-4o -v -j -m gpt-4o -p"
  "-b no_vision_gpt-4o -m gpt-4o -p"
)

cd ..

# Loop through each config and run the Python script
for config in "${configs[@]}"; do
  echo "Running: python $python_script $config"
  for ((i = 1; i <= batch_size; i++)); do
      echo "Iteration $i"
      # shellcheck disable=SC2086
      timeout 5m python "$python_script" $config
  done
done