import os
import json
import torch
import argparse
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

def parse_args():
    parser = argparse.ArgumentParser(description="PLS Regression and visualization script.")
    parser.add_argument("--reasoning_acc_path", type=str, required=True, help="Evaluated reasoning files.")
    parser.add_argument("--model_file_path", type=str, required=True, help="Base path containing model files.")
    parser.add_argument("--start", type=int, default=1, help="Start step index.")
    parser.add_argument("--end", type=int, default=27, help="End step index.")
    parser.add_argument("--rl_algorithm", type=str, default="DAPO", help="rl_algorithm")
    parser.add_argument("--save_dir", type=str, default="pls_plots", help="Directory to save PLS plots.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top modules to display in results.")
    return parser.parse_args()

def load_dicts(model_file_path, start=1, end=28, rl_algorithm="DAPO"):
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
        file_path = os.path.join(model_file_path, f"{rl_algorithm}-step-{i}", "rank1_u_vectors.pt")
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

def pls_regression_visualize(vectors, accuracies_scaled, key, save_dir="pls_plots"):
    """
    Perform PLS regression and visualize the results.

    Args:
        vectors (ndarray): The vectors to use for PLS regression.
        accuracies_raw (ndarray): The raw accuracies to plot.
        accuracies_scaled (ndarray): The scaled accuracies used for regression.
        key (str): The module name to label the plot.
        save_dir (str): The directory to save the plot.

    Returns:
        float: The R² score of the regression, or None if R² is below the threshold.
    """
    os.makedirs(save_dir, exist_ok=True)
    pls = PLSRegression(n_components=1)
    projected = pls.fit_transform(vectors, accuracies_scaled.reshape(-1, 1))[0]
    reg = LinearRegression().fit(projected, accuracies_scaled)
    r2 = r2_score(accuracies_scaled, reg.predict(projected))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc = ax.scatter(
        projected[:, 0], accuracies_scaled,
        c=accuracies_scaled, cmap="viridis", s=55, marker="D", alpha=0.85
    )
    x_line = np.linspace(projected.min(), projected.max(), 100)
    ax.plot(x_line, reg.predict(x_line.reshape(-1, 1)), "r--", lw=2)
    cbar = fig.colorbar(sc, ax=ax, label=" ")
    cbar.ax.tick_params(labelsize=22)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{key.replace('.', '_')}.svg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved {save_path}, R²={r2:.3f}")
    return r2


def run_pls_all_modules(base_path, y, start=1, end=27, rl_algorithm='DAPO', save_dir="pls_plots", top_k=10):
    all_dicts = load_dicts(base_path, start=start, end=end, rl_algorithm=rl_algorithm)
    common_keys = get_common_keys(all_dicts)
    print(f"🔍 Found {len(common_keys)} common keys")

    y_scaled = y
    r2_results = {}

    for key in common_keys:
        vectors = []
        for step, d in all_dicts:
            vec = d['first_u'][key].numpy()
            vectors.append(vec)
        r2 = pls_regression_visualize(
            np.array(vectors), y[:len(vectors)], key,
            save_dir=save_dir
        )
        if r2 is not None:
            r2_results[key] = float(r2)
    sorted_r2 = sorted(r2_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔥 Top {top_k} modules by R²:")
    for i, (key, r2) in enumerate(sorted_r2[:top_k], 1):
        print(f"{i:2d}. {key:<55} R² = {r2:.3f}")
    return r2_results


def main():
    args = parse_args()
    y = []
    for i in range(1, 28):
        file_path = f"{args.reasoning_acc_path}/{args.rl_algorithm}-step-{i}/math/test_t0.6_k1.jsonl"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    try:
                        data = json.loads(last_line) 
                        acc = data.get("acc")
                        if acc is not None:
                            y.append(acc)
                        else:
                            print(f"⚠️ No 'acc' field in {file_path}")
                    except json.JSONDecodeError:
                        print(f"⚠️ Error decoding JSON in {file_path}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    y_array = np.array(y)
    print("y array:", y_array)

    r2_results = run_pls_all_modules(
        args.model_file_path, y_array, start=args.start, end=args.end, rl_algorithm=args.rl_algorithm,
        save_dir=args.save_dir, top_k=args.top_k
    )


if __name__ == "__main__":
    main()
