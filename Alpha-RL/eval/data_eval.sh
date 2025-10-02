#!/bin/bash
# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments. Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.


nvcc -V

source /home/
echo $PATH


CUDA_VISIBLE_DEVICES=0,1,2,3
for i in {27..27..1}; do
    echo "Running iteration $i"

    python Data_eval.py \
    --model_name_or_path "" \
    --data_name "aime24" \
    --prompt_type "qwen-instruct" \
    --temperature 0.6 \
    --start_idx 0 \
    --end_idx -1 \
    --start_penalty_length 0 \
    --n_sampling 4 \
    --k 1 \
    --output_dir "" \
    --Iterations "$i" \
    --split test \
    --max_tokens 30000 \
    --seed 0 \
    --top_p 1

    echo "Iteration $i completed successfully!"
done
