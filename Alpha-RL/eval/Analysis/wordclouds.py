import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

def check_greedy_decisions(tokenizer, model, pre_text, text, default_skip_tokens=100, context_window=20):
    pre_inputs = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)
    actual_skip_tokens = pre_inputs['input_ids'].shape[1] if pre_text else default_skip_tokens

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    start_pos = min(actual_skip_tokens, seq_len - 2)
    if start_pos >= seq_len - 1:
        return 0.0, [], []

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    check_logits = logits[0, start_pos:-1, :]
    max_tokens = torch.argmax(check_logits, dim=-1).cpu()
    actual_tokens = input_ids[0, start_pos+1:].cpu()

    total_checked = len(actual_tokens)
    non_greedy_tokens = []
    position_percents = []

    for idx_in_range, current_pos in enumerate(range(start_pos, seq_len - 1)):
        if actual_tokens[idx_in_range].item() != max_tokens[idx_in_range].item():
            token_str = tokenizer.decode([actual_tokens[idx_in_range].item()]).strip()
            if token_str:
                non_greedy_tokens.append(token_str)
            relative_pos = (current_pos / seq_len) * 100
            position_percents.append(relative_pos)

    ratio = len(non_greedy_tokens) / total_checked if total_checked > 0 else 0
    return ratio, non_greedy_tokens, position_percents

if __name__ == "__main__":
    input_file = ""
    model_path = ""

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="float16",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    all_tokens, all_positions, all_ratios = [], [], []

    for result in tqdm(results, desc="Analyzing greedy decisions"):
        ratio, tokens, positions = check_greedy_decisions(
            tokenizer=tokenizer,
            model=model,
            pre_text=result["prompt"],
            text=result["full_text"],
            default_skip_tokens=100,
            context_window=30
        )
        all_ratios.append(ratio)
        all_tokens.extend(tokens)
        all_positions.extend(positions)

    overall_ratio = sum(all_ratios) / len(all_ratios) if all_ratios else 0
    print(f"Overall non-greedy ratio: {overall_ratio:.2%}")
    print(f"Unique non-greedy tokens: {len(set(all_tokens))}")
    print(f"Total non-greedy tokens: {len(all_tokens)}")

    # 词云
    token_counts = Counter(all_tokens)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate_from_frequencies(token_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Non-Greedy Token WordCloud", pad=10)
    plt.tight_layout()
    plt.savefig("non_greedy_wordcloud.svg", dpi=300)
    plt.show()

    # 位置分布
    bins = [i*10 for i in range(11)]
    plt.figure(figsize=(8, 5))
    plt.hist(all_positions, bins=bins, edgecolor="black", alpha=0.75, color="#4C72B0")
    plt.xticks(bins)
    plt.xlabel("Position in Text (%)")
    plt.ylabel("Frequency")
    plt.title("Non-Greedy Token Position Distribution", pad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("non_greedy_position_distribution.svg", dpi=3000)
    plt.show()
