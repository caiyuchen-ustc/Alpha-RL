import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from functools import reduce


plt.rcParams.update({
    "pdf.fonttype": 42,  
    "ps.fonttype": 42,
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "font.family": "Arival"
})

plt.style.use("seaborn-v0_8") 

def load_dicts(base_path, start=1, end=27):

    all_dicts = []
    for i in range(start, end + 1):
        file_path = os.path.join(base_path, f"global_step_{i}", "first_u_vectors.pt")
        if not os.path.exists(file_path):
            continue
        data = torch.load(file_path, map_location="cpu")
        all_dicts.append((i, data))
    return all_dicts


def get_common_keys(all_dicts):

    keys_list = [set(d.keys()) for _, d in all_dicts]
    return sorted(list(reduce(lambda a, b: a & b, keys_list)))


def pls_regression_visualize(vectors, accuracies_raw, accuracies_scaled, key,
                             save_dir="pls_plots", r2_threshold=0.7):
    os.makedirs(save_dir, exist_ok=True)

    X_min = vectors.min(axis=0)
    X_max = vectors.max(axis=0)
    X_norm = (vectors - X_min) / (X_max - X_min + 1e-8)
    X_scaled = np.log(X_norm + 1e-6)

    pls = PLSRegression(n_components=1)
    projected = pls.fit_transform(X_scaled, accuracies_scaled.reshape(-1, 1))[0]


    reg = LinearRegression().fit(projected, accuracies_scaled)
    r2 = r2_score(accuracies_scaled, reg.predict(projected))

    if r2 < r2_threshold:
        return None


    fig, ax = plt.subplots(1, 1, figsize=(8, 6))


    x_proj = projected[:, 0]
    x_sqrt = (x_proj - x_proj.min()) 
    x_norm = (x_sqrt - x_sqrt.min()) / (x_sqrt.max() - x_sqrt.min() + 1e-8)

    sc = ax.scatter(
        x_norm, accuracies_raw,
        c=accuracies_raw, cmap="viridis",
        s=55, marker="D", alpha=0.85
    )


    x_line = np.linspace(x_proj.min(), x_proj.max(), 100)
    x_line_sqrt = x_line
    x_line_norms = (x_line_sqrt ) / (x_line_sqrt.max()  + 1e-8)
    
    x_line_norm = (x_line_norms - x_line_norms.min()) / (x_line_norms.max() - x_line_norms.min() + 1e-8)
    
    ax.plot(x_line_norm, reg.predict(x_line.reshape(-1, 1)), "r--", lw=2)

    cbar = fig.colorbar(sc, ax=ax, label="         ")
    cbar.ax.tick_params(labelsize=22)

    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.set_ticks([])
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{key.replace('.', '_')}.svg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved {save_path}, R²={r2:.3f}")
    
    return r2


def run_pls_all_modules(base_path, y, start=1, end=27,
                        save_dir="pls_plots", r2_threshold=0.7,
                        r2_savefile="r2_results.json", top_k=10):

    all_dicts = load_dicts(base_path, start=start, end=end)
    common_keys = get_common_keys(all_dicts)
    print(f"🔍 Found {len(common_keys)} common keys")

    y_scaled =y
    r2_results = {}

    for key in common_keys:
        vectors = [d[key].numpy() for step, d in all_dicts if step <= len(y)]
        if len(vectors) < 3:
            continue

        r2 = pls_regression_visualize(
            np.array(vectors), y[:len(vectors)], y_scaled[:len(vectors)], key,
            save_dir=save_dir, r2_threshold=r2_threshold
        )
        if r2 is not None:
            r2_results[key] = float(r2)

    with open(r2_savefile, "w") as f:
        json.dump(r2_results, f, indent=2)
    print(f"📑 R² results saved to {r2_savefile} (scaled accuracies)")

    # 打印 Top-K
    sorted_r2 = sorted(r2_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔥 Top {top_k} modules by R²:")
    for i, (key, r2) in enumerate(sorted_r2[:top_k], 1):
        print(f"{i:2d}. {key:<55} R² = {r2:.3f}")

    return r2_results



def save_r2_latex_table(r2_results, save_path="r2_table.tex", top_k=10):

    sorted_r2 = sorted(r2_results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    with open(save_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l c}\n")
        f.write("\\toprule\n")
        f.write("Module & $R^2$ \\\\\n")
        f.write("\\midrule\n")
        for key, r2 in sorted_r2:
            f.write(f"{key} & {r2:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Top-{top_k} modules ranked by $R^2$ from PLS regression.}}\n")
        f.write("\\label{tab:r2_topk}\n")
        f.write("\\end{table}\n")
    print(f"📑 LaTeX table saved to {save_path}")


if __name__ == "__main__":
    base_path = ""
    y = np.array([
    ])

    r2_results = run_pls_all_modules(
        base_path, y, start=1, end=27,
        save_dir="", r2_threshold=0,
        r2_savefile=os.path.join(base_path, "", ""),
        top_k=10
    )

    save_r2_latex_table(
        r2_results,
        save_path=os.path.join(base_path, "", ""),
        top_k=1
    )
