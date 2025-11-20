#!/bin/bash

# Get the current directory of the script
current_file_path=$(dirname "$(realpath "$0")")
parent_dir=$(dirname "$current_file_path")
export PYTHONPATH="$parent_dir:$PYTHONPATH"
echo "Parent directory added to PYTHONPATH: $parent_dir"


# Define paths and other parameters
model_file_path="$parent_dir/models"  # Path to model files
reasoning_acc_path="$parent_dir/output/eval/models"
echo "Evaluated reasoning files PATH: $reasoning_acc_path"
save_dir="$current_file_path/PLS_rank1_u_per_submodule"  # Directory to save PLS plots
top_k=10

# Run the Python script
python AlphaPLS.py \
    --reasoning_acc_path "$reasoning_acc_path" \
    --model_file_path "$model_file_path" \
    --start 1 \
    --end 27 \
    --rl_algorithm DAPO \
    --save_dir "$save_dir" \
    --top_k "$top_k"

# Run the Python script
# python AlphaPLS.py \
#     --reasoning_acc_path "/project/ugmathllm/caiyuchen/Alpha-RL/Alpha-RL/eval/output/RL-GSPO/qwen3-8B-GSPO" \
#     --model_file_path "/project/ugmathllm/caiyuchen/verl/ckptsss/RL-GSPO/qwen3-8B-GSPO" \
#     --start 2\
#     --end 24 \
#     --rl_algorithm GSPO \
#     --save_dir "$save_dir" \
#     --top_k "$top_k"

# python AlphaPLS.py \
#     --reasoning_acc_path "/project/ugmathllm/caiyuchen/Alpha-RL/Alpha-RL/eval/output/RL-GSPO/qwen3-8B-GSPO" \
#     --model_file_path "/project/ugmathllm/caiyuchen/verl/ckptsss/RL-GSPO/qwen3-8B-GSPO" \
#     --start 2\
#     --end 24 \
#     --rl_algorithm GSPO \
#     --save_dir "$save_dir" \
#     --top_k "$top_k"

# python AlphaPLS.py \
#     --reasoning_acc_path "/project/ugmathllm/caiyuchen/Alpha-RL/Alpha-RL/eval/output/model/Spiral" \
#     --model_file_path "/project/ugmathllm/caiyuchen/spiral/oat-output/model/Spiral" \
#     --start 2\
#     --end 23 \
#     --rl_algorithm Spiral \
#     --save_dir "$save_dir" \
#     --top_k "$top_k"

# python AlphaPLS.py \
#     --reasoning_acc_path "/project/ugmathllm/caiyuchen/Alpha-RL/Alpha-RL/eval/output/DAPO/Qwen3-14B" \
#     --model_file_path "/project/ugmathllm/caiyuchen/verl/ckpts/DAPO/Qwen3-14B" \
#     --start 2\
#     --end 12 \
#     --rl_algorithm DAPO14B \
#     --save_dir "$save_dir" \
#     --top_k "$top_k"