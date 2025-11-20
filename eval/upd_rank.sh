python upd_rank.py \
  --base_model_path ./models/DAPO-step-0 \
  --step_model_path ./models \
  --svd_base_path ./models \
  --start_step 1 \
  --end_step 27 \
  --rank 1 \
  --alpha 1 \
  --device cpu


# python upd_rank.py \
#   --base_model_path /project/ugmathllm/xxuca/models/raw/Qwen3-4B-Base \
#   --step_model_path "/project/ugmathllm/caiyuchen/spiral/oat-output/model/Spiral" \
#   --svd_base_path "/project/ugmathllm/caiyuchen/spiral/oat-output/model/Spiral" \
#   --start_step 1 \
#   --end_step 24 \
#   --rl_algorithm Spiral \
#   --rank 1 \
#   --alpha 1 \
#   --device cpu
