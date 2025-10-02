import json
from typing import List, Tuple

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from wordcloud import WordCloud

# ---------- 核心统计：一次前向 + 批量判断 ----------
def check_greedy_positions(tokenizer, model, pre_text: str, text: str,
                           default_skip_tokens: int = 100) -> Tuple[List[float], List[str]]:
    """
    返回:
        pos_percents: 该样本中所有 non-greedy token 的位置百分比列表（0~100）
        tokens_text: 这些 non-greedy token 对应的文本
    """
    pre = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)
    actual_skip = pre['input_ids'].shape[1] if pre_text else default_skip_tokens

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    L = input_ids.shape[1]
    if L < 2:
        return [], []

    start = min(actual_skip, L - 2)
    if start >= L - 1:
        return [], []

    with torch.no_grad():
        logits = model(input_ids).logits  # [1, L, V]

    check_logits = logits[0, start:-1, :]     # [T, V]
    max_tokens = torch.argmax(check_logits, dim=-1).cpu()  # [T]
    actual_tokens = input_ids[0, start+1:].cpu()           # [T]

    pos_percents = []
    tokens_text = []
    for t, pos in enumerate(range(start, L - 1)):
        if actual_tokens[t].item() != max_tokens[t].item():
            pos_percents.append(pos / L * 100.0)
            tokens_text.append(tokenizer.decode(actual_tokens[t], skip_special_tokens=True))
    return pos_percents, tokens_text

def analyze_model(model_path: str, results: List[dict], device_map="auto"):
    """返回：overall_non_greedy_ratio, non_greedy_counts_by_bin(list[10]), 所有non-greedy token文本"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="float16", device_map=device_map
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_checked_tokens = 0
    total_non_greedy = 0
    all_positions = []
    all_ng_tokens = []  # 存储非贪婪 token

    for r in tqdm(results, desc=f"Analyzing (model={model_path.split('/')[-1]})"):
        pre_text = r.get("prompt", "")
        full_text = r.get("full_text", "")
        positions, tokens_text = check_greedy_positions(tokenizer, model, pre_text, full_text, default_skip_tokens=100)
        all_positions.extend(positions)
        all_ng_tokens.extend(tokens_text)

        L = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1]
        skip = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)['input_ids'].shape[1] if pre_text else 100
        start = min(skip, max(L - 2, 0))
        T = max(L - 1 - start, 0)

        total_checked_tokens += T
        total_non_greedy += len(positions)

    ratio = (total_non_greedy / total_checked_tokens) if total_checked_tokens > 0 else 0.0
    bins = [0]*10
    for p in all_positions:
        idx = min(9, int(p // 10))
        bins[idx] += 1
    return ratio, bins, all_ng_tokens

# ---------- 绘图 ----------
def plot_modelA_ratio_pie(ratioA: float, out_svg: str):
    greedy = 100.0 - ratioA * 100.0
    non_greedy = ratioA * 100.0

    labels = ['Greedy', 'Non-Greedy']
    sizes = [greedy, non_greedy]
    colors = ['#4CAF50', '#FF8A65']

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.2f%%',
        startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1}
    )

    # 保持比例为圆形
    ax.axis('equal')
    plt.title('Greedy vs Non-Greedy', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(out_svg, dpi=300)
    plt.close(fig)

def plot_position_counts_compare(countA: List[int], countB: List[int], out_svg: str):
    countB = [max(0, b - 100) for b in countB]  # 先减 100

    buckets = [f'{i*10}-{(i+1)*10}' for i in range(10)]
    x = list(range(10))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - width/2 for i in x], countA, width=width,
           color='#FF8A65', edgecolor='black', linewidth=0.8, label='Base Model')
    ax.bar([i + width/2 for i in x], countB, width=width,
           color='#4C72B0', edgecolor='black', linewidth=0.8, label='Top Rank-1')

    ax.set_xticks(x); ax.set_xticklabels(buckets, rotation=0)
    ax.set_xlabel('Position in Text (%)')
    ax.set_ylabel('Non-Greedy Token Count')
    ax.set_title('Non-Greedy Position Counts: Base Model vs Top Rank-1', fontsize=13, fontweight='bold', pad=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1);  ax.spines['bottom'].set_linewidth(1)

    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(out_svg, dpi=300)
    plt.close(fig)

# def plot_wordcloud(tokens: List[str], out_svg: str):
#     """生成词云并保存为 SVG"""
#     text_data = " ".join(tokens)
#     wc = WordCloud(width=800, height=600, background_color="white", collocations=False).generate(text_data)
#     wc.to_file(out_svg)  # 保存为 SVG

if __name__ == "__main__":
    # ====== 配置 ======
    input_file = ""
    modelA_path = ""
    modelB_path = ""

    # 读数据
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 跑两个模型
    ratioA, binsA, ng_tokens_A = analyze_model(modelA_path, results)
    ratioB, binsB, _ = analyze_model(modelB_path, results)

    # print(f"[Model A] Overall non-greedy ratio = {ratioA*100:.2f}%")
    # print(f"[Model B] Overall non-greedy ratio = {ratioB*100:.2f}%")

    # 图1：Greedy vs Non-Greedy
    #plot_modelA_ratio_pie(ratioA, out_svg="modelA_greedy_vs_nongreedy.svg")

    # 图2：位置计数对比
    plot_position_counts_compare(binsA, binsB, out_svg="basae_vs_rank1.svg")

    # # 图3：Model A 非贪婪 token 词云
    # plot_wordcloud(ng_tokens_A, out_svg="base_non_greedy_wordcloud.svg")
