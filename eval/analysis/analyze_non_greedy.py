import sys
import os
import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ helper functions ------------------
def compute_perplexity(logits, input_ids, start_pos=0):
    seq_len = input_ids.shape[1]
    if seq_len <= start_pos + 1:
        return float("nan")

    pred_logits = logits[0, start_pos:seq_len-1, :].float()  # (checked_len, vocab)
    target_ids = input_ids[0, start_pos+1:seq_len]          # (checked_len,)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    target_log_probs = log_probs[range(target_ids.shape[0]), target_ids]
    nll = -target_log_probs.mean().item()
    ppl = float(torch.exp(torch.tensor(nll)))
    return ppl

def check_greedy_decisions(tokenizer, model, pre_text, text, default_skip_tokens=100):
    pre_inputs = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False) if pre_text else None
    actual_skip_tokens = pre_inputs['input_ids'].shape[1] if pre_inputs is not None else default_skip_tokens

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    start_pos = min(actual_skip_tokens, seq_len - 2)
    if start_pos >= seq_len - 1:
        return 0.0, float("nan")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # perplexity for generated tokens only
    ppl = compute_perplexity(logits, input_ids, start_pos=start_pos)

    # non-greedy ratio
    check_logits = logits[0, start_pos:seq_len-1, :].cpu()
    max_tokens = torch.argmax(check_logits, dim=-1)
    actual_tokens = input_ids[0, start_pos+1:seq_len].cpu()
    total_checked = actual_tokens.shape[0]
    mismatch_mask = actual_tokens != max_tokens
    mismatch_count = int(mismatch_mask.sum().item())
    ratio = mismatch_count / total_checked if total_checked > 0 else 0.0

    return ratio, ppl

# ------------------ main ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze non-greedy decisions and perplexity.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model.")
    parser.add_argument("--greedy_json", type=str, required=True, help="Path to RL greedy decoding JSON file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.greedy_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    ratios, ppls = [], []

    for result in tqdm(results[:-1], desc="Analyzing"):
        
        ratio, ppl = check_greedy_decisions(
            tokenizer=tokenizer,
            model=model,
            pre_text=result["prompt"],
            text=result["full_text"],
            default_skip_tokens=100
        )
        ratios.append(ratio)
        ppls.append(ppl)

    overall_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    valid_ppls = [p for p in ppls if not (p is None or (isinstance(p, float) and (p != p)))]
    mean_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("nan")
    print("--------------------------------------------------------------------------------------------------------")
    print(f"Overall non-greedy ratio for RL model's greedy-generated text (evaluated on base model): {overall_ratio:.2%}")
    print(f"Mean perplexity of RL model's greedy-generated text (evaluated on base model): {mean_ppl:.4f}")
    print("--------------------------------------------------------------------------------------------------------")
