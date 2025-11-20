#!/bin/bash
current_file_path=$(dirname "$(realpath "$0")")
echo "Current file directory: $current_file_path"
export PYTHONPATH="$current_file_path:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0,1,2,3
for i in {0..27..1}; do
    echo "Running iteration $i"
    python -m reasoning_eval \
    --model_name_or_path "$current_file_path/models/DAPO-step-$i" \
    --data_name "math" \
    --temperature 0.6 \
    --start_idx 0 \
    --end_idx -1 \
    --n_sampling 4 \
    --k 1 \
    --split test \
    --max_tokens 30000 \
    --seed 0 \

    echo "Iteration $i completed successfully!"
done
