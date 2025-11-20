#!/bin/bash
# Visualize TSNE of rank-1 U vectors from RL Trained model across multiple steps.

current_file_path=$(dirname "$(realpath "$0")")
parent_dir=$(dirname "$current_file_path")
export PYTHONPATH="$parent_dir:$PYTHONPATH"
echo "Parent directory added to PYTHONPATH: $parent_dir"



# Define paths and other parameters
model_file_path="$parent_dir/models"  # Path to model files
reasoning_acc_path="$parent_dir/output/eval/models"
echo "Evaluated reasoning files PATH: $reasoning_acc_path"

python AlphaPredVector.py \
    --reasoning_acc_path "$reasoning_acc_path" \
    --model_file_path "$parent_dir/models" \
    --start 1 \
    --end 9 \
    --rl_algorithm DAPO \
    --save_dir "$current_file_path/AlphaPredFig" \
    --y_predict 1.7


