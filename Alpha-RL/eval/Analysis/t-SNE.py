import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from functools import reduce
import os, csv, sys, matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np



def load_dicts(base_path, start=1, end=28):
    all_dicts = []
    for i in range(start, end + 1):
        file_path = os.path.join(base_path, f"global_step_{i}", "first_u_vectors.pt")
        if not os.path.exists(file_path):
            print(f"⚠️ Not found: {file_path}")
            continue
        data = torch.load(file_path, map_location="cpu")
        all_dicts.append((i, data))
    return all_dicts

def get_common_keys(all_dicts):
    keys_list = [set(d.keys()) for _, d in all_dicts]
    common_keys = reduce(lambda a, b: a & b, keys_list)
    return sorted(list(common_keys))

plt.rcParams.update({
    "pdf.fonttype": 42, 
    "ps.fonttype": 42,
    "font.size": 9,   
    "axes.linewidth": 0.6,
})

def _clamp(x, lo, hi): return max(lo, min(hi, x))

def visualize_tsne_per_key(all_dicts, common_keys, output_dir="tsne_first_u_per_key"):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "_versions.txt"), "w") as f:
        import sklearn, torch
        f.write(f"python: {sys.version}\n")
        f.write(f"sklearn: {sklearn.__version__}\n")
        f.write(f"matplotlib: {matplotlib.__version__}\n")
        f.write(f"torch: {torch.__version__}\n")

    for key in common_keys:
        print(key)
        if "21" not in key:
            continue
        else:
            vectors, labels = [], []
            for step, d in all_dicts:
                vec = d[key].numpy()
                vectors.append(vec)
                labels.append(step)
            vectors = np.asarray(vectors)
            labels  = np.asarray(labels)

            n_samples = vectors.shape[0]
            if n_samples < 3:
                print(f"⚠️ Skip {key}, only {n_samples} samples"); continue


            X = StandardScaler().fit_transform(vectors)

            pca = PCA(n_components=2, random_state=42)
            Xp  = pca.fit_transform(X)

            perplexity = _clamp(n_samples // 3, 5, 30)
            tsne = TSNE(
                n_components=2,
                random_state=42,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
                n_iter=1000,
                metric="euclidean",
            )
            Y = tsne.fit_transform(Xp)


            fig, ax = plt.subplots(figsize=(6, 5))  
            order = np.argsort(labels)  
            Yo, Lo = Y[order], labels[order]


            from matplotlib.collections import LineCollection
            segs = np.stack([Yo[:-1], Yo[1:]], axis=1)
            lc = LineCollection(segs, colors="0.5", linewidths=0.6, alpha=0.7)
            ax.add_collection(lc)

            # 仅散点，无额外标记
            sc = ax.scatter(
                Yo[:, 0], Yo[:, 1],
                c=Lo, cmap="viridis", s=72, alpha=0.85, edgecolors="none"
            )

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.grid(alpha=0.15, linewidth=0.4)


            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            ticks = np.linspace(Lo.min(), Lo.max(), num=min(6, len(np.unique(Lo))), dtype=int)
            cbar.set_ticks(sorted(set(ticks)))
            cbar.set_ticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])


            plt.tight_layout()
            pdf_path = os.path.join(output_dir, f"{key.replace('.', '_')}.svg")
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved {pdf_path}")


if __name__ == "__main__":
    base_path = ""
    all_dicts = load_dicts(base_path, start=1, end=27)
    common_keys = get_common_keys(all_dicts)
    print(f"Found {len(common_keys)} common keys")
    visualize_tsne_per_key(all_dicts, common_keys, output_dir="")
