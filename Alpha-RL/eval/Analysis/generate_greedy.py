import sys
import os
import argparse
sys.path.append('')

from utils.data_loader import load_data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="1", help="1")
parser.add_argument("--output_prefix", type=str, default="", help="")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cdevice = 0

model_path = ''
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": cdevice}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====== 函数 ======
def parse_question(example, data_name=None):
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            return example[key].strip()
    return ""

def apply_template(tokenizer, question):
    return tokenizer.apply_chat_template(
        [{"content": question, "role": "user"}],
        tokenize=False,
        add_generation_prompt=True,
    )

for data in ["math",'aime24','aime25','minerva','gpqa','gsm8k']:
    examples = load_data(data, "test", "")
    prompt_batch = []

    for idx, example in tqdm(enumerate(examples), desc=""):
        question = parse_question(example, "math")

        question_prompt = '\nPlease reason step by step, and put your final answer within \\boxed{}'
        cur_prompt = question + question_prompt
        prompt_batch.append(cur_prompt)

    results = []
    for prompt in tqdm(prompt_batch[:10], desc=""):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=6000,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0])
        results.append({
            "prompt": prompt,
            "full_text": generated_text
        })

    output_file = f""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
