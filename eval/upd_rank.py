import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

@torch.no_grad()
def reconstruct_rank_k(base_model_path, step_model_path, svd_base_path, start_step, end_step, rl_algorithm,rank=1,alpha=1, device="cuda"):
    """Reconstruct models using top-k SVD components."""
    print(f"Loading base config and tokenizer from {base_model_path} ...")
    config = AutoConfig.from_pretrained(base_model_path)
    dtype = config.torch_dtype or torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for step in tqdm(range(start_step, end_step + 1), desc="Processing steps"):
        print(f"\n=== Step {step} ===")
        norm_k_sum = 0
        norm_sum = 0
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype).to(device)
        step_model_dir = os.path.join(step_model_path, f"{rl_algorithm}-step-{step}")
        step_model = AutoModelForCausalLM.from_pretrained(step_model_dir, torch_dtype=dtype).to(device)

        svd_file = os.path.join(svd_base_path, f"{rl_algorithm}-step-{step}", "svd_components.pt")
        if not os.path.exists(svd_file):
            print(f"⚠️  SVD file not found: {svd_file}, skipping step {step}")
            continue
        svd_components = torch.load(svd_file, map_location=device)

        output_dir = os.path.join(svd_base_path, f"{rl_algorithm}-step-{step}", f"rank_{rank}")
        os.makedirs(output_dir, exist_ok=True)
        for layer_idx, (layer_base, layer_step) in enumerate(zip(base_model.model.layers, step_model.model.layers)):
            layer_svd = svd_components.get(f"layer_{layer_idx}", {})

            print(f"\n🔹 Processing layer {layer_idx}")

            # Self-attention
            for (name_base, param_base), (name_step, param_step) in zip(layer_base.self_attn.named_parameters(),
                                                                        layer_step.self_attn.named_parameters()):
                if name_base.endswith(".weight"):
                    key_U = f"self_attn_{name_base}_U"
                    key_S = f"self_attn_{name_base}_S"
                    key_Vt = f"self_attn_{name_base}_Vt"
                    if key_U in layer_svd:
                        U = layer_svd[key_U].to(device, dtype=dtype)
                        S = layer_svd[key_S].to(device, dtype=dtype)
                        Vt = layer_svd[key_Vt].to(device, dtype=dtype)
                        U_k = U[:, :rank]
                        S_k = S[:rank]
                        Vt_k = Vt[:rank, :]
                        update = U @ torch.diag(S) @ Vt
                        update_k = U_k @ torch.diag(S_k) @ Vt_k
                        update_norm = torch.norm(update.data)
                        update_k_norm = torch.norm(update_k.data)
                        param_base.data += alpha * (update_norm/update_k_norm) * update_k

                        print(f"  [Self-Attn] {name_base} | rank={rank} | update_norm={update_norm:.4f}")
                        
                        norm_k_sum += update_k_norm
                        norm_sum += update_norm

            # MLP
            for (name_base, param_base), (name_step, param_step) in zip(layer_base.mlp.named_parameters(),
                                                                        layer_step.mlp.named_parameters()):
                if name_base.endswith(".weight"):
                    key_U = f"mlp_{name_base}_U"
                    key_S = f"mlp_{name_base}_S"
                    key_Vt = f"mlp_{name_base}_Vt"
                    if key_U in layer_svd:
                        U = layer_svd[key_U].to(device, dtype=dtype)
                        S = layer_svd[key_S].to(device, dtype=dtype)
                        Vt = layer_svd[key_Vt].to(device, dtype=dtype)
                        U_k = U[:, :rank]
                        S_k = S[:rank]
                        Vt_k = Vt[:rank, :]
                        update = U @ torch.diag(S) @ Vt
                        update_k = U_k @ torch.diag(S_k) @ Vt_k
                        update_norm = torch.norm(update.data)
                        update_k_norm = torch.norm(update_k.data)
                        param_base.data += alpha * (update_norm/update_k_norm) * update_k

                        print(f"  [Self-Attn] {name_base} | rank={rank} | update_norm={update_k_norm:.4f}")

                        norm_k_sum += update_k_norm
                        norm_sum += update_norm

        print('---------------------------------------------')
        print(f" Norm[Top - {rank}] / Norm[Top - 100%] = {norm_k_sum/norm_sum:.4f} ")
        print('---------------------------------------------')
        base_model.to("cpu")
        base_model.save_pretrained(output_dir, torch_dtype=dtype, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Saved rank-{rank} model at {output_dir}")

        del base_model, step_model, svd_components
        torch.cuda.empty_cache()
        gc.collect()

    print("🎉 All steps processed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--step_model_path", type=str, required=True, help="Directory containing DAPO-step-i models")
    parser.add_argument("--svd_base_path", type=str, required=True, help="Directory containing SVD .pt files")
    parser.add_argument("--start_step", type=int, default=1)
    parser.add_argument("--end_step", type=int, default=27)
    parser.add_argument("--rl_algorithm", type=str, required=True, help="rl_algorithm")
    parser.add_argument("--rank", type=int, default=1, help="Top-k rank for SVD reconstruction")
    parser.add_argument("--alpha", type=int, default=1, help="Scale Factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    reconstruct_rank_k(
        args.base_model_path,
        args.step_model_path,
        args.svd_base_path,
        args.start_step,
        args.end_step,
        args.rl_algorithm,
        rank=args.rank,
        alpha=args.alpha,
        device=args.device
    )
