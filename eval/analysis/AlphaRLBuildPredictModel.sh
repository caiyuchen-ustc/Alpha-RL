current_file_path=$(dirname "$(realpath "$0")")
parent_dir=$(dirname "$current_file_path")
export PYTHONPATH="$parent_dir:$PYTHONPATH"
echo "Parent directory added to PYTHONPATH: $parent_dir"

python AlphaRLBuildPerdictModel.py \
  --step_model_path $parent_dir/models \
  --ckpt_step 9 \
  --rl_algorithm DAPO \
  --device cpu \
  --Only_Rank_1_Space True \

# python AlphaRLBuildPerdictModel.py \
#   --step_model_path /project/ugmathllm/caiyuchen/verl/ckptsss/RL-GSPO/qwen3-8B-GSPO \
#   --ckpt_step 11 \
#   --rl_algorithm GSPO \
#   --device cpu

# python AlphaRLBuildPerdictModel.py \
#   --step_model_path /project/ugmathllm/caiyuchen/spiral/oat-output/model/Spiral \
#   --ckpt_step 9 \
#   --rl_algorithm Spiral \
#   --device cpu

# python AlphaRLBuildPerdictModel.py \
#   --step_model_path /project/ugmathllm/caiyuchen/verl/ckpts/DAPO/Qwen3-14B \
#   --ckpt_step 12 \
#   --rl_algorithm DAPO14B \
#   --device cpu
