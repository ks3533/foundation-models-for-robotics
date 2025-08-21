#!/bin/bash

python_script="Code/LiteLLM/main.py"

batch_size=10

# Define combined argument sets

configs=(
  "-b vision_o4-mini -v -p"
  "-b vision_o4-mini_reasoning -v -R -p"
  "-b json_vision_o4-mini -v -j -p"
  "-b json_vision_o4-mini_reasoning -v -j -R -p"
  "-b no_vision_o4-mini -p"
  "-b no_vision_o4-mini_reasoning -R -p"
  "-b vision_legacy_o4-mini --vision-legacy -p"
  "-b vision_legacy_o4-mini_reasoning --vision-legacy -R -p"
  "-b vision_legacy_picture_every_tool_call_o4-mini --vision-legacy -s -p"
  "-b vision_legacy_picture_every_tool_call_o4-mini_reasoning --vision-legacy -s -R -p"
  
  "-b vision_gpt-5 -v -m gpt-5 -p"
  "-b vision_gpt-5_reasoning -v -m gpt-5 -R -p"
  "-b json_vision_gpt-5 -v -j -m gpt-5 -p"
  "-b json_vision_gpt-5_reasoning -v -j -m gpt-5 -R -p"
  "-b no_vision_gpt-5 -m gpt-5 -p"
  "-b no_vision_gpt-5_reasoning -m gpt-5 -R -p"

  "-b vision_gpt-5-nano -v -m gpt-5-nano -p"
  "-b vision_gpt-5-nano_reasoning -v -m gpt-5-nano -R -p"
  "-b json_vision_gpt-5-nano -v -j -m gpt-5-nano -p"
  "-b json_vision_gpt-5-nano_reasoning -v -j -m gpt-5-nano -R -p"
  "-b no_vision_gpt-5-nano -m gpt-5-nano -p"
  "-b no_vision_gpt-5-nano_reasoning -m gpt-5-nano -R -p"
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