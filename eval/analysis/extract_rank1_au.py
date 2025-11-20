import os
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity

def extract_first_au_from_svd(svd_file, device="cpu"):
    """
    Extract the first column of U from the SVD components.

    Args:
        svd_file (str): Path to the SVD components file (e.g., "svd_components.pt").
        device (str): Device to use (default is CPU), can be "cpu" or "cuda".

    Returns:
        dict: Dictionary containing U[:, 0]*S[0] vectors for each module.
    """
    if not os.path.exists(svd_file):
        print(f"❌ Not found: {svd_file}")
        return {}

    svd_components = torch.load(svd_file, map_location=device)
    first_au_dict = {}
    for layer_key, layer_svd in svd_components.items():
        for key, value in layer_svd.items():
            if key.endswith("_U"):
                module_name = key.replace("_U", "")

                U = value.to(device)
                S = layer_svd[key.replace("_U", "_S")]
                if U.size(0) == 0:
                    continue
                first_au_dict[f"{layer_key}_{module_name}"] = U[:, 0].clone() * S[0].item()
    return first_au_dict


def compare_and_flip_signs(first_au_dict, last_U_vectors):
    """
    Compare U[:, 0] vectors between the current step and the previous step. 
    If the cosine similarity is less than 0, flip the sign of the current U[:, 0].

    Args:
        first_au_dict (dict): Dictionary of U[:, 0] vectors for the current step.
        last_U_vectors (dict): Dictionary of U[:, 0] vectors from the previous step.

    Returns:
        tuple: Updated first_au_dict and a flipped dictionary indicating whether the sign was flipped.
    """
    flipped_dict = {}

    for key, cur_vec in first_au_dict.items():
        flipped = False
        if key in last_U_vectors:
            prev_vec = last_U_vectors[key]
            cosine_sim = cosine_similarity(cur_vec.cpu().numpy().reshape(1, -1), prev_vec.cpu().numpy().reshape(1, -1))[0][0]
            
            if cosine_sim < 0:
                cur_vec = -cur_vec
                flipped = True
                print(f"⚠️ Cosine similarity = {cosine_sim:.4f} | Layer: {key} | Flipping sign")
        flipped_dict[key] = flipped
        first_au_dict[key] = cur_vec
    return first_au_dict, flipped_dict



def save_first_u_and_flipped(first_au_dict, flipped_dict, output_file):
    """
    Save the extracted U[:, 0] vectors and the flip status to an output file.

    Args:
        first_au_dict (dict): Dictionary containing U[:, 0] vectors for each module.
        flipped_dict (dict): Dictionary containing the flip status (True/False) for each module.
        output_file (str): Path to save the output.
    """
    torch.save({'first_au': first_au_dict, 'flipped': flipped_dict}, output_file)
    print(f"✅ Saved {output_file}")


def process_all_steps(model_file_path, start, end, rl_algorithm, device="cpu"):
    """
    Process all model steps from start to end and extract, compare, and save U[:, 0] vectors.

    Args:
        model_file_path (str): Path to the folder containing model checkpoints.
        start (int): Starting step index.
        end (int): Ending step index.
        device (str): Device to use (default is CPU), can be "cpu" or "cuda".
    """
    last_U_vectors = {}  # Dictionary to store U[:, 0] vectors from the previous step

    for i in range(end, start - 1, -1):  # Reverse loop from end to start
        step_dir = os.path.join(model_file_path, f"{rl_algorithm}-step-{i}")
        svd_file = os.path.join(step_dir, "svd_components.pt")
        output_file = os.path.join(step_dir, "rank1_au_vectors.pt")
        print(f"\n===== Processing step {i} =====")
        
        first_au_dict = extract_first_au_from_svd(svd_file, device)
        first_au_dict, flipped_dict = compare_and_flip_signs(first_au_dict, last_U_vectors)
        save_first_u_and_flipped(first_au_dict, flipped_dict, output_file)
        last_U_vectors.update(first_au_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first column of U from SVD components.")
    parser.add_argument("--model_file_path", type=str, required=True, help="Base path containing step directories.")
    parser.add_argument("--start", type=int, default=1, help="Start step index.")
    parser.add_argument("--end", type=int, default=27, help="End step index.")
    parser.add_argument("--rl_algorithm", type=str, default="DAPO", help="rl_algorithm")
    args = parser.parse_args()

    process_all_steps(args.model_file_path, args.start, args.end, args.rl_algorithm)
