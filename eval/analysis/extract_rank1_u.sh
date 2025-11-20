#!/bin/bash
# This script extracts the first column of U from SVD components for all steps.

current_file_path=$(dirname "$(realpath "$0")")
parent_dir=$(dirname "$current_file_path")
export PYTHONPATH="$parent_dir:$PYTHONPATH"
echo "Parent directory added to PYTHONPATH: $parent_dir"

python extract_rank1_u.py \
    --model_file_path "$parent_dir/models" \
    --start 1 \
    --end 27 \
    --rl_algorithm DAPO \


# python extract_rank1_u.py \
#     --model_file_path "/project/ugmathllm/caiyuchen/verl/ckptsss/RL-GSPO/qwen3-8B-GSPO" \
#     --start 1 \
#     --end 24 \
#     --rl_algorithm GSPO \

# python extract_rank1_u.py \
#     --model_file_path "/project/ugmathllm/caiyuchen/spiral/oat-output/model/Spiral" \
#     --start 1 \
#     --end 23 \
#     --rl_algorithm Spiral \

