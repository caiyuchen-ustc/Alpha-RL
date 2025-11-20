import os
import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from functools import reduce
import argparse
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.linewidth": 0.6,
})

def load_dicts(model_file_path, start=1, end=28):
    """
    Load the saved U[:, 0] vectors along with the flip status for each module from multiple steps.
    
    Args:
        model_file_path (str): Path to the folder containing model checkpoints.
        start (int): The starting step index.
        end (int): The ending step index.

    Returns:
        list of tuples: Each tuple contains step index and the corresponding dictionary with U[:, 0] and flip status.
    """
    all_dicts = []
    for i in range(start, end + 1):
        file_path = os.path.join(model_file_path, f"DAPO-step-{i}", "rank1_u_vectors.pt")
        if not os.path.exists(file_path):
            print(f"⚠️ Not found: {file_path}")
            continue
        data = torch.load(file_path, map_location="cpu")
        all_dicts.append((i, data))
    return all_dicts
def get_common_keys(all_dicts):
    """
    Get the common keys present in all dictionaries across the steps.
    
    Args:
        all_dicts (list of tuples): List of tuples where each tuple contains step index and corresponding dictionary.

    Returns:
        list: Sorted list of common keys.
    """

    keys_list = [set(d['first_u'].keys()) for _, d in all_dicts]
    common_keys = reduce(lambda a, b: a & b, keys_list)
    all_keys = reduce(lambda a, b: a | b, keys_list)
    non_common_keys = list(all_keys - common_keys)
    if non_common_keys:
        print(f"⚠️ Non-common keys across steps: {non_common_keys}")
    return sorted(list(common_keys))

def _clamp(x, lo, hi):
    """
    Clamp a value between a minimum and maximum value.
    
    Args:
        x (float): The value to be clamped.
        lo (float): The minimum value.
        hi (float): The maximum value.

    Returns:
        float: The clamped value.
    """
    return max(lo, min(hi, x))

def visualize_tsne_per_key(all_dicts, common_keys, output_dir="tsne_first_u_per_key"):
    """
    Visualize t-SNE for each common key (submodule) across all steps.
    
    Args:
        all_dicts (list of tuples): List of tuples containing step index and the dictionary with first U vectors.
        common_keys (list): List of common keys across steps (modules).
        output_dir (str): Directory to save t-SNE plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save system versions used for reproducibility
    with open(os.path.join(output_dir, "_versions.txt"), "w") as f:
        import sklearn
        f.write(f"python: {sys.version}\n")
        f.write(f"sklearn: {sklearn.__version__}\n")
        f.write(f"matplotlib: {matplotlib.__version__}\n")
        f.write(f"torch: {torch.__version__}\n")

    for key in common_keys:

        vectors_raw = []
        labels_raw = []
        for step, d in all_dicts:
            # Retrieve U[:, 0] vector and flipped status from the data
            vec = d['first_u'][key].numpy()
            vectors_raw.append(vec)
            labels_raw.append(step)

        vectors_raw = np.asarray(vectors_raw)
        labels_raw = np.asarray(labels_raw)
        n_samples = len(vectors_raw)

        if n_samples < 3:
            print(f"⚠️ Skip {key}, only {n_samples} samples")
            continue

        # Standardize the vectors
        X = StandardScaler().fit_transform(vectors_raw)

        # Apply PCA for dimensionality reduction (to 2D)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(X)

        # Apply t-SNE for further dimensionality reduction
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

        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 5))
        order = np.argsort(labels_raw)
        Yo, Lo = Y[order], labels_raw[order]

        ax.set_title(f"t-SNE: {key.replace('_','.')}", pad=4)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        # Plot line segments connecting points
        segs = np.stack([Yo[:-1], Yo[1:]], axis=1)
        lc = LineCollection(segs, colors="0.5", linewidths=0.6, alpha=0.7)
        ax.add_collection(lc)

        # Scatter plot for the points
        sc = ax.scatter(
            Yo[:, 0], Yo[:, 1],
            c=Lo, cmap="viridis", s=72, alpha=0.85, edgecolors="none"
        )

        # Remove unnecessary axes and grid styling
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(alpha=0.15, linewidth=0.4)

        # Layout and saving the plot
        plt.tight_layout()
        pdf_path = os.path.join(output_dir, f"{key.replace('.', '_')}.svg")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved {pdf_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSNE visualization of rank1 U vectors per key.")
    parser.add_argument("--model_file_path", type=str, required=True, help="Base path containing step directories.")
    parser.add_argument("--start", type=int, default=1, help="Start step index.")
    parser.add_argument("--end", type=int, default=27, help="End step index.")
    parser.add_argument("--output_dir", type=str, default="TSNE_rank1_u_per_submodule", help="Directory to save TSNE plots.")
    args = parser.parse_args()

    all_dicts = load_dicts(args.model_file_path, start=args.start, end=args.end)
    common_keys = get_common_keys(all_dicts)
    print(f"Found {len(common_keys)} common keys")
    visualize_tsne_per_key(all_dicts, common_keys, output_dir=args.output_dir)
