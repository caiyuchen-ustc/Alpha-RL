import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import json
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.linewidth": 1,
})

input_file = ""
with open(input_file, "r", encoding="utf-8") as f:
    results = json.load(f)

aa = "".join(res["full_text"] for res in results[:]) 


model1_path = ""
model2_path = ""

model1 = AutoModelForCausalLM.from_pretrained(model1_path, device_map="cpu")
model2 = AutoModelForCausalLM.from_pretrained(model2_path, device_map="cpu")

tok1 = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)
tok2 = AutoTokenizer.from_pretrained(model2_path, trust_remote_code=True)

# ---------------- tokenization ----------------
ids1 = tok1(aa, return_tensors="pt", add_special_tokens=False)["input_ids"][0].unique().tolist()
ids2 = tok2(aa, return_tensors="pt", add_special_tokens=False)["input_ids"][0].unique().tolist()

tokens1 = [tok1.decode([i]) for i in ids1]
tokens2 = [tok2.decode([i]) for i in ids2]


emb_matrix1 = model1.get_input_embeddings().weight.detach().cpu()
emb_matrix2 = model2.get_input_embeddings().weight.detach().cpu()

emb1 = emb_matrix1[ids1]
emb2 = emb_matrix2[ids2]

all_embeddings = torch.cat([emb1, emb2], dim=0).numpy()
labels = np.array([0]*len(emb1) + [1]*len(emb2))  # 0=Base, 1=DIST

pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(all_embeddings)

coords1 = embeddings_2d[:len(emb1)]
coords2 = embeddings_2d[len(emb1):]


common_tokens = set(tokens1) & set(tokens2)
print("Common tokens:", len(common_tokens))

pairs = []
for t in common_tokens:
    i1 = tokens1.index(t)
    i2 = tokens2.index(t)
    pairs.append((coords1[i1], coords2[i2], t))


color_base = "#EE1127"  
color_dist = "#5880F8"  
plt.figure(figsize=(9, 6))
plt.scatter(coords1[:,0], coords1[:,1], 
            edgecolors=color_base, facecolors="none", alpha=0.9, 
            s=30, marker="o", label="BASE")

plt.scatter(coords2[:,0], coords2[:,1], 
            edgecolors=color_dist, facecolors="none", alpha=0.9, 
            s=20, marker="^", label="DAPO")


for (p1, p2, tok) in pairs:
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c="gray", alpha=0.3, linewidth=0.5)

plt.legend(fontsize=24, handletextpad=0.3)  
# plt.tick_params(axis='both', which='major', labelsize=24)  
# plt.title("Projection of Embeddings Before and After Dr.GRPO", fontsize=20)
# plt.xlabel("t-SNE Dim 1", fontsize=20)
# plt.ylabel("t-SNE Dim 2", fontsize=20)
plt.tight_layout()
plt.gca().tick_params(axis='x', which='both', labelbottom=False)
plt.gca().tick_params(axis='y', which='both', labelleft=False)
plt.savefig("DAPO.svg", bbox_inches="tight")
plt.savefig("DAPO.pdf", dpi=300, bbox_inches="tight")
plt.show()
