python svd.py \
  --base_model_path ./models/DAPO-step-0 \
  --models_root ./models \
  --rl_algorithm DAPO \
  --start_step 1 \
  --end_step 27

# python svd.py \
#   --base_model_path /project/ugmathllm/xxuca/models/raw/Qwen3-8B-Base \
#   --models_root /project/ugmathllm/caiyuchen/verl/ckptsss/RL-GSPO/qwen3-8B-GSPO \
#   --rl_algorithm GSPO \
#   --start_step 17 \
#   --end_step 28


# python svd.py \
#   --base_model_path /project/ugmathllm/xxuca/models/raw/Qwen3-4B-Base \
#   --models_root /project/ugmathllm/caiyuchen/spiral/oat-output/model/Spiral \
#   --rl_algorithm Spiral \
#   --start_step 1 \
#   --end_step 23

# python svd.py \
#   --base_model_path /project/ugmathllm/xxuca/models/raw/Qwen3-14B-Base \
#   --models_root /project/ugmathllm/caiyuchen/verl/ckpts/DAPO/Qwen3-14B \
#   --rl_algorithm DAPO14B \
#   --start_step 1 \
#   --end_step 25 \
#   --device 'cpu'
