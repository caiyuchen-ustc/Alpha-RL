#!/bin/bash
# Visualize TSNE of rank-1 U vectors from RL Trained model across multiple steps.

current_file_path=$(dirname "$(realpath "$0")")
parent_dir=$(dirname "$current_file_path")
export PYTHONPATH="$parent_dir:$PYTHONPATH"
echo "Parent directory added to PYTHONPATH: $parent_dir"

python visualize_rank1_u_tsne.py \
    --model_file_path "$parent_dir/models" \
    --start 1 \
    --end 27 \
    --output_dir ./TSNE_rank1_u_per_submodule
