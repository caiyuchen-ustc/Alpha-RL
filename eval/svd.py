import os
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

@torch.no_grad()
def save_svd_components(base_model_path, models_root, rl_algorithm, start_step=0, end_step=27, device="cuda"):
    """
    Compute SVD of parameter updates between base model and rl models,
    and save results to the same folder as each step model.

    Args:
        base_model_path (str): Base model path (e.g., ./dapomodels/DAPO-step-0)
        models_root (str): Directory containing step models (e.g., ./dapomodels)
        start_step (int): Starting step index
        end_step (int): Ending step index
        device (str): "cuda" or "cpu"
    """
    print(f"🧩 Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    base_model.eval()

    for step in tqdm(range(start_step, end_step + 1), desc="Processing steps"):
        model_path = os.path.join(models_root, f"{rl_algorithm}-step-{step}")
        
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping step {step}: {model_path} not found.")
            continue

        print(f"\n🔹 Comparing base model with step {step}: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()

        svd_components = {}

        for layer_idx, (base_layer, new_layer) in enumerate(zip(base_model.model.layers, model.model.layers)):
            print(f"\n🔹 Computing SVD for layer {layer_idx}")
            layer_svd = {}

            # Self-attention weights
            for name, param in new_layer.self_attn.named_parameters():
                if name.endswith(".weight"):
                    ref_param = base_layer.self_attn.get_parameter(name)
                    # import pdb
                    # pdb.set_trace()
                    diff = (param - ref_param).to('cuda')
                    if diff.dim() == 2:
                        if device == 'cpu':
                            U, S, Vt = torch.linalg.svd(diff, full_matrices=False)
                            # diff_cpu = diff.detach().cpu().numpy()
                            # U, S, Vt = np.linalg.svd(diff_cpu, full_matrices=False)
                            # U = torch.from_numpy(U)
                            # S = torch.from_numpy(S)
                            # Vt = torch.from_numpy(Vt)
                        else:
                            U, S, Vt = torch.linalg.svd(diff, full_matrices=False)
                        layer_svd[f"self_attn_{name}_U"] = U.cpu()
                        layer_svd[f"self_attn_{name}_S"] = S.cpu()
                        layer_svd[f"self_attn_{name}_Vt"] = Vt.cpu()
                        print(f"  [Self-Attn] {name} | shape={param.shape} | SVD done, rank={len(S)}")

            # MLP weights
            for name, param in new_layer.mlp.named_parameters():
                if name.endswith(".weight"):
                    ref_param = base_layer.mlp.get_parameter(name)
                    diff = (param - ref_param).to('cuda')
                    if diff.dim() == 2:
                        if device == 'cpu':
                            U, S, Vt = torch.linalg.svd(diff, full_matrices=False)
                            # diff_cpu = diff.detach().cpu().numpy()
                            # U, S, Vt = np.linalg.svd(diff_cpu, full_matrices=False)
                            # U = torch.from_numpy(U)
                            # S = torch.from_numpy(S)
                            # Vt = torch.from_numpy(Vt)
                        else:
                            U, S, Vt = torch.linalg.svd(diff, full_matrices=False)
                        layer_svd[f"mlp_{name}_U"] = U.cpu()
                        layer_svd[f"mlp_{name}_S"] = S.cpu()
                        layer_svd[f"mlp_{name}_Vt"] = Vt.cpu()
                        print(f"  [MLP] {name} | shape={param.shape} | SVD done, rank={len(S)}")

            svd_components[f"layer_{layer_idx}"] = layer_svd

        save_path = os.path.join(model_path, "svd_components.pt")
        torch.save(svd_components, save_path)
        
        
        print(f"✅ SVD components saved: {save_path}")
        del model, svd_components
        torch.cuda.empty_cache()
        gc.collect()

    print("\n🎉 All SVD decompositions completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute SVD between base and step models.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model (e.g., ./dapomodels/DAPO-step-0)")
    parser.add_argument("--models_root", type=str, required=True, help="Directory containing all step models (e.g., ./dapomodels)")
    parser.add_argument("--rl_algorithm", type=str, required=True, help="rl_algorithm")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=27)
    parser.add_argument("--device", type=str, default='cuda', help="rl_algorithm")
    args = parser.parse_args()

    save_svd_components(
        base_model_path=args.base_model_path,
        models_root=args.models_root,
        rl_algorithm = args.rl_algorithm,
        start_step=args.start_step,
        end_step=args.end_step,
        device = args.device
    )
