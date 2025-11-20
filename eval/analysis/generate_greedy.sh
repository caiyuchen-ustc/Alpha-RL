#!/bin/bash
# ================================================================
# This script reproduces the greedy decoding experiment described
# in the Appendix of the paper:
# 
#   "On Predictability of Reinforcement Learning Dynamics 
#    for Large Language Models"
#
# It performs greedy decoding using the trained model to
# generate reasoning traces on the math dataset.
# ================================================================

current_file_path=$(dirname "$(realpath "$0")")
parent_dir=$(dirname "$current_file_path")
export PYTHONPATH="$parent_dir:$PYTHONPATH"
echo "Parent directory added to PYTHONPATH: $parent_dir"

python generate_greedy.py \
  --model_path "$parent_dir/models/DAPO/global-step-27" \
  --output_prefix ./generate_greedy
