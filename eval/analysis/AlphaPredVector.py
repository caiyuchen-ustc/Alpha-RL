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

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from functools import reduce

def parse_args():
    parser = argparse.ArgumentParser(description="PLS Regression and visualization script.")
    parser.add_argument("--reasoning_acc_path", type=str, required=True, help="Evaluated reasoning files.")
    parser.add_argument("--model_file_path", type=str, required=True, help="Base path containing model files.")
    parser.add_argument("--start", type=int, default=1, help="Start step index.")
    parser.add_argument("--end", type=int, default=27, help="End step index.")
    parser.add_argument("--rl_algorithm", type=str, default="DAPO", help="rl_algorithm")
    parser.add_argument("--save_dir", type=str, default="pls_plots", help="Directory to save PLS plots.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top modules to display in results.")
    parser.add_argument("--y_predict", type=float, required=True, help="The predicted value (or values) to be used for regression.")
    return parser.parse_args()

def load_dicts(model_file_path, start=1, end=28,rl_algorithm='DAPO'):
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
        file_path = os.path.join(model_file_path, f"{rl_algorithm}-step-{i}", "rank1_au_vectors.pt")
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
    keys_list = [set(d['first_au'].keys()) for _, d in all_dicts]
    common_keys = reduce(lambda a, b: a & b, keys_list)
    all_keys = reduce(lambda a, b: a | b, keys_list)
    non_common_keys = list(all_keys - common_keys)
    if non_common_keys:
        print(f"⚠️ Non-common keys across steps: {non_common_keys}")
    return sorted(list(common_keys))

def pls_regression_predict_and_visualize(vectors, accuracies_scaled, y_predict, key, save_dir="pls_plots"):
    """
    Perform PLS regression, visualize the results, and predict the vector for a given y_predict (predicted accuracy).

    Args:
        vectors (ndarray): The vectors to use for PLS regression.
        accuracies_scaled (ndarray): The scaled accuracies used for regression.
        y_predict (float): The predicted value to be used for regression (i.e., target accuracy).
        key (str): The module name to label the plot.
        save_dir (str): The directory to save the plot.

    Returns:
        float: The R² score of the regression.
        ndarray: The predicted vector corresponding to the given y_predict accuracy.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the PLS model
    pls = PLSRegression(n_components=1)
    
    # Perform PLS regression and get the projected data
    projected = pls.fit_transform(vectors, accuracies_scaled.reshape(-1, 1))[0]
    
    # Perform linear regression on the projected data
    reg = LinearRegression().fit(projected, accuracies_scaled)
    
    # Calculate the R² score
    r2 = r2_score(accuracies_scaled, reg.predict(projected))

    # Visualize the PLS regression results
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
    
    # Save the plot
    save_path = os.path.join(save_dir, f"{key.replace('.', '_')}.svg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved {save_path}, R²={r2:.3f}")
    
    # Predict the vector for the given y_predict accuracy
    predicted_projected = (y_predict - reg.intercept_) / reg.coef_
    
    # Transform the predicted projected data back to the original vector space
    predicted_vector = pls.inverse_transform(predicted_projected.reshape(1, -1))

    return r2, predicted_vector  # Return R² score and the predicted vector

def run_pls_all_modules(base_path, y, start=1, end=27, rl_algorithm ='DAPO',save_dir="pls_plots", y_predict=0.78):
    """
    Run PLS regression and generate predictions for all submodules in the specified step range.

    Args:
        base_path (str): Path to the folder containing model checkpoints.
        y (ndarray): The accuracies to use for regression.
        start (int): The starting step index.
        end (int): The ending step index.
        save_dir (str): Directory to save the PLS plots.
        y_predict (float): The predicted value (accuracy) for which the corresponding vector is needed.

    Returns:
        dict: Dictionary containing R² results for each module and the predicted vectors for each submodule.
    """
    # Load the models and their corresponding data
    all_dicts = load_dicts(base_path, start=start, end=end,rl_algorithm=rl_algorithm)
    
    # Get the common keys (submodules) across all models in the specified range
    common_keys = get_common_keys(all_dicts)
    print(f"🔍 Found {len(common_keys)} common keys")

    y_scaled = y 
    r2_results = {}
    predicted_vectors = {}

    for key in common_keys:
        vectors = []
        accuracies_raw = []  # To store raw accuracies for visualization

        for step, d in all_dicts:
            vec = d['first_au'][key].numpy()  # Ensure to get the correct key (e.g., 'first_au')
            vectors.append(vec)
            acc = d.get('acc', None)  # Assuming accuracy is stored in 'acc'
            if acc is not None:
                accuracies_raw.append(acc)

        vectors = np.array(vectors)  # Convert the list of vectors to a numpy array
        accuracies_raw = np.array(accuracies_raw)

        # Perform the prediction and visualization
        r2, predicted_vector = pls_regression_predict_and_visualize(
            vectors, y_scaled, y_predict, key, save_dir
        )
        predicted_vectors[key] = predicted_vector
        if r2 is not None:
            r2_results[key] = float(r2)
    sorted_r2 = sorted(r2_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔥 From ckpt-{start} to ckpt-{end} Top 10 modules by R²:")
    for i, (key, r2) in enumerate(sorted_r2[:10], 1):
        print(f"{i:2d}. {key:<55} R² = {r2:.3f}")

    return r2_results, predicted_vectors

def main():
    args = parse_args()
    y = []
    for i in range(args.start, args.end + 1):  # Use start and end args to control range
        file_path = f"{args.reasoning_acc_path}/{args.rl_algorithm}-step-{i}/math/test_t0.6_k4.jsonl"
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

    r2_results, predicted_vectors = run_pls_all_modules(
        args.model_file_path, y_array, start=args.start, end=args.end, rl_algorithm=args.rl_algorithm,
        save_dir=args.save_dir, y_predict=args.y_predict
    )
    # Optionally, you can save the predicted vectors to a file
    save_pred_path = os.path.join(args.model_file_path, f"{args.rl_algorithm}-step-{i}", "predicted_vectors.pt")
    torch.save(predicted_vectors, save_pred_path)

    print(f"✅ Predicted vectors saved to {save_pred_path}")

if __name__ == "__main__":
    main()
