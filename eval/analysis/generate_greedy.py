import sys
import os
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ set project root ------------------
current_file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_file_path))  # 上两级
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import load_data  # 确保可以import

# ------------------ args ------------------
parser = argparse.ArgumentParser(description="Greedy decoding with template and average perplexity calculation.")
parser.add_argument("--gpu", type=str, default="0", help="GPU id to use.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
parser.add_argument("--output_prefix", type=str, default="./outputs", help="Output directory prefix.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cdevice = 0

# ------------------ load model ------------------
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map={"": cdevice}
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------ helper functions ------------------
def parse_question(example):
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            return example[key].strip()
    return ""

def apply_template(tokenizer, question):
    """
    Apply the model's chat template to a single question.
    """
    return tokenizer.apply_chat_template(
        [{"content": question, "role": "user"}],
        tokenize=False,
        add_generation_prompt=True,
    )

def calculate_ppl(model, tokenizer, results, device):
    """
    Calculate perplexity for a list of texts.
    Only on generated text (without prompt).
    """
    ppl_list = []

    for result in results:
        generated_text = result["full_text"]
        prompt_text = result["prompt"]  # 问题 + 提示

        # tokenize
        tokenized = tokenizer(generated_text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)

        # 只计算生成文本的 loss，mask 掉 prompt
        prompt_len = len(tokenizer(prompt_text)["input_ids"])
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # -100 表示忽略计算 loss
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            ppl = torch.exp(loss).item()
            ppl_list.append(ppl)

    avg_ppl = sum(ppl_list) / len(ppl_list)
    return avg_ppl

# ------------------ main ------------------
examples = load_data("math", "test")
prompt_batch = []

# Step 1: prepare prompt with template
for example in tqdm(examples, desc="Preparing prompts"):
    question = parse_question(example)
    question_prompt = "\nPlease reason step by step, and put your final answer within \\boxed{}"
    full_question = question + question_prompt
    templated_prompt = apply_template(tokenizer, full_question)
    prompt_batch.append(templated_prompt)

# Step 2: generate outputs
results = []
for prompt in tqdm(prompt_batch[:10], desc="Generating"):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=6000,
        do_sample=False,  # greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0])
    results.append({
        "prompt": prompt,
        "full_text": generated_text
    })

# Step 3: save results
os.makedirs(args.output_prefix, exist_ok=True)
output_file = os.path.join(args.output_prefix, "greedy_decode_results.json")
avg_ppl = calculate_ppl(model, tokenizer, results, model.device)
print(f"Average perplexity on RL model generated text: {avg_ppl:.4f}")
results.append({"avg_ppl":avg_ppl})

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


print(f"Results saved to: {output_file}")



