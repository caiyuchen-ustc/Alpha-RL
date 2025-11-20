
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import importlib.util
import os

import math
import argparse
import vllm.envs as envs
import random
import time
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_path)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
print("parent_dir:", parent_dir)
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb

import sympy as sp
from sympy import simplify, Eq, sympify, Pow
from sympy.parsing.latex import parse_latex

import sympy as sp
from sympy import simplify, Eq, sympify, Pow
from sympy.parsing.latex import parse_latex
# envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

def parse_list(arg):
    return arg.split(',')


def apply_template(tokenizer, question):
    txt = tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return txt


def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./models/DAPO-step-27", help="model dir")
    parser.add_argument('--n_sampling', type=int, default = 1, help="n for sampling")
    parser.add_argument("--k", type=int, default = 1, help ="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default = f"{parent_dir}/data",type=str)
    parser.add_argument('--data_name', type=str, default="math")
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    
    return args


# def apply_qwen_math_template(question: str):
#     return (
#         "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
#         + question
#         + "<|im_end|>\n<|im_start|>assistant\n"
#     )

# def apply_r1_template(question: str):
#     return (
#         "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
#         "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
#         + question
#         + "\nAssistant: <think>"
#     )


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    
    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_tokens, 
                                     n=factor,
                                     )

        
    llm = LLM(model=model_name_or_path, 
        tensor_parallel_size=len(available_gpus), 
        trust_remote_code=True, 
        gpu_memory_utilization=0.6,
        )

    print(args.data_name)
    examples = load_data(args.data_name, args.split, args.data_dir)

    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    out_file_prefix = f'{args.split}_t{args.temperature}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}.jsonl'
    
    if os.path.exists(out_file):
        print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
    
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # 准备所有prompts
    prompt_batch = []
    for idx, example in tqdm(enumerate(examples)):
        question = parse_question(example, args.data_name)
        question_prompt = '\nPlease reason step by step, and put your final answer within \\boxed{}'
        question =  question + question_prompt
        cur_prompt = apply_template(tokenizer=tokenizer, question=question)
        prompt_batch.append(cur_prompt)
    
    print("Sample prompt:")
    print(prompt_batch[0])
    batch_size = len(prompt_batch)
    
    file_outputs = []
    correct_cnt = 0
    
    for cur_generation_epoch in range(generation_epoch):
        print(f"🔄{cur_generation_epoch + 1}/{generation_epoch}")

        for batch_start in tqdm(range(0, len(prompt_batch), batch_size)):
            batch_end = min(batch_start + batch_size, len(prompt_batch))
            current_batch_prompts = prompt_batch[batch_start:batch_end]
            current_batch_examples = examples[batch_start:batch_end]

            try:
                completions = llm.generate(current_batch_prompts, sampling_params)
                

                for i, completion in enumerate(completions):
                    global_idx = batch_start + i
                    d = current_batch_examples[i]
                    question = parse_question(d, args.data_name)
                    generated_responses = [completion.outputs[j].text for j in range(len(completion.outputs))]
                    
                    if cur_generation_epoch == 0:
                        result = {
                            "question": question,
                            "generated_responses": generated_responses,
                        }
                        if "id" in d:
                            result["id"] = d["id"]
                        if "source" in d:
                            result["source"] = d["source"]
                        file_outputs.append(result)
                    else:
                        file_outputs[global_idx]['generated_responses'] += generated_responses
                        
            except Exception as e:
                print(f"❌")
                raise e


    pass_at_k_list = []
    k = args.k
    
    for i in tqdm(range(len(examples))):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = file_outputs[i]['generated_responses']
        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]

        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list
        
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
    
    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        count = 0
        for d in tqdm(file_outputs, "Writing..."):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100 == 0:
                f.flush()
        acc = correct_cnt/ len(examples)
        f.write(json.dumps({'acc':acc}, ensure_ascii=False))
        f.flush()
    os.rename(temp_out_file, out_file)
    
    print(f"🎯 Accuracy: {correct_cnt / len(examples):.4f}")


    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"🎯 Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"🎯 Pass@1: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)
