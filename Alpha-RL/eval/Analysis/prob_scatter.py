import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns  # 可选，美化
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

def get_token_probs(model, tokenizer, pre_text, text, default_skip_tokens=100):
    """
    返回从 start_pos 开始到倒数第二个 token 为止，
    每个位置真实下一个 token 的概率列表。
    """
    pre_inputs = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)
    skip_tokens = pre_inputs['input_ids'].shape[1] if pre_text else default_skip_tokens

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    attn_mask = inputs.attention_mask.to(model.device) if "attention_mask" in inputs else None

    seq_len = input_ids.shape[1]
    start_pos = min(skip_tokens, seq_len - 2)
    if start_pos >= seq_len - 1:
        return []

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

    # logits[t] 预测的是 token_{t+1}
    check_logits = logits[0, start_pos:-1, :]                 # [T, V]
    actual_tokens = input_ids[0, start_pos+1:]                # [T]

    probs = torch.softmax(check_logits, dim=-1)               # [T, V]
    gathered = probs.gather(dim=-1, index=actual_tokens.unsqueeze(-1)).squeeze(-1)
    return gathered.cpu().tolist()

def scatter_plot(p1, p2, out_path="prob_scatter.svg", m1="Model1", m2="Model2"):
    plt.figure(figsize=(6.5,6.5))

    # 使用颜色映射：点的颜色随 x 概率变化
    sc = plt.scatter(p1, p2, 
                     c=p1,                # 按 model1 概率上色
                     cmap="viridis",      # 颜色渐变
                     s=8, alpha=0.4,      # 点更小更透明
                     edgecolors="none")

    # 添加颜色条
    # cbar = plt.colorbar(sc)
    # cbar.set_label(f"Token Probability", fontsize=12)

    # 参考线 y=x
    plt.plot([0,1],[0,1], 'k--', linewidth=1.2, alpha=0.7)

    # 标签 & 标题
    plt.xlabel(f"{m1} Probability")
    plt.ylabel(f"{m2} Probability")
    plt.title("Token Probability Comparison", fontsize=15)

    # 美化边框和网格
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    sns.despine()  # 去掉上和右的边框（需要 seaborn）

    # 坐标范围
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    # ===== 路径配置 =====
    input_file   = 
    model1_path  = 
    model2_path  = 

    # ===== 加载模型 & tokenizer（用同一个 tokenizer 保证对齐）=====
    model1 = AutoModelForCausalLM.from_pretrained(model1_path, torch_dtype="float16", device_map="auto")
    model2 = AutoModelForCausalLM.from_pretrained(model2_path, torch_dtype="float16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== 数据读取 =====
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    p1_all, p2_all = [], []
    a=0
    for r in tqdm(results, desc="Comparing probabilities"):
        pre_text = r.get("prompt", "")
        text     = r.get("full_text", "")

        p1 = get_token_probs(model1, tokenizer, pre_text, text)
        p2 = get_token_probs(model2, tokenizer, pre_text, text)

        # 对齐：取 min 长度
        L = min(len(p1), len(p2))
        p1_all.extend(p1[:L])
        p2_all.extend(p2[:L])
        a+=1
        if a==20:
            break


    # ===== 画图 =====
    scatter_plot(p1_all, p2_all, out_path="rlooprob_scatter.png",
                 m1="RL updated", m2="Base Model")
    print("Saved figure to prob_scatter.svg")
