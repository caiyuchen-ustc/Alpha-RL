#!/bin/bash
current_file_path=$(dirname "$(realpath "$0")")
parent_dir=$(dirname "$current_file_path")
echo "Parent directory: $parent_dir"
export PYTHONPATH="$parent_dir:$PYTHONPATH"
# Example usage of embedding_shift.py
python embedding_shift.py \
  --base_model_path "$parent_dir/models/DAPO-step-0" \
  --trained_model_path "$parent_dir/models/DAPO-step-27" \
  --input_file "$parent_dir/output/eval/models/DAPO-step-27/math/test_t0.6_k1.jsonl" \
  --output_dir ./figures


