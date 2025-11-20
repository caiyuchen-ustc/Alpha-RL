import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

@torch.no_grad()
def AlphaRLBuildPerdictModel(step_model_path, predict_ckpt_step=10, rl_algorithm='DAPO', device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n=== Step {predict_ckpt_step} ===")
    norm_k_sum = 0
    norm_sum = 0

    step_model_dir = os.path.join(step_model_path, f"{rl_algorithm}-step-{predict_ckpt_step}")
    print(f"Loading base config and tokenizer from {step_model_dir} ...")
    step_model = AutoModelForCausalLM.from_pretrained(step_model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(step_model_dir)
    


    svd_file = os.path.join(step_model_path, f"{rl_algorithm}-step-{predict_ckpt_step}", "svd_components.pt")
    predict_file = os.path.join(step_model_path, f"{rl_algorithm}-step-{predict_ckpt_step}", "predicted_vectors.pt")
    Flippingign_file = os.path.join(step_model_path, f"{rl_algorithm}-step-{predict_ckpt_step}", "rank1_au_vectors.pt")
    if not os.path.exists(svd_file):
        print(f"⚠️  SVD file not found: {svd_file}, skipping step {predict_ckpt_step}")

    svd_components = torch.load(svd_file, map_location=device)
    predict_pt = torch.load(predict_file, weights_only=False)

    Flippingign = torch.load(Flippingign_file)
    output_dir = os.path.join(step_model_path, f"{rl_algorithm}-step-{predict_ckpt_step}", f"AlphaPredictModel")
    os.makedirs(output_dir, exist_ok=True)
    for layer_idx, layer_step in enumerate(step_model.model.layers):
        layer_svd = svd_components.get(f"layer_{layer_idx}", {})

        print(f"\n🔹 Processing layer {layer_idx}")

        # Self-attention
        for (name_base, param_base), in zip(layer_step.self_attn.named_parameters()):
            if name_base.endswith(".weight"):
                key_Vt = f"self_attn_{name_base}_Vt"
                if key_Vt in layer_svd:
                    Vt = layer_svd[key_Vt].to(device)
                    Vt_k = Vt[:1, :]
                    Flip_sign = Flippingign['flipped'][f'layer_{layer_idx}_self_attn_{name_base}']
                    Predict_au = torch.tensor(predict_pt[f'layer_{layer_idx}_self_attn_{name_base}']).float()

                    update = Predict_au.T @ Vt_k
                    if Flip_sign:
                        update = -update
                    update_norm = torch.norm(update)
                    param_base.data += update

                    print(f"  [Self-Attn] {name_base} | update_norm={update_norm:.4f}")
                    
        # MLP
        for (name_base, param_base), in zip(layer_step.mlp.named_parameters()):
                key_Vt = f"mlp_{name_base}_Vt"
                if key_Vt in layer_svd:
                    Vt = layer_svd[key_Vt].to(device)
                    Vt_k = Vt[:1, :]
                    Flip_sign = Flippingign['flipped'][f'layer_{layer_idx}_mlp_{name_base}']
                    Predict_au = torch.tensor(predict_pt[f'layer_{layer_idx}_mlp_{name_base}']).float()

                    update = Predict_au.T @ Vt_k
                    if Flip_sign:
                        update = -update
                    update_norm = torch.norm(update)
                    param_base.data += update

                    print(f"  [MLP] {name_base} | update_norm={update_norm:.4f}")

    step_model.to("cpu")
    step_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved Alpha Predict model at {output_dir}")

    torch.cuda.empty_cache()
    gc.collect()

    print("🎉processed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_model_path", type=str, required=True, help="Directory containing DAPO-step-i models")
    parser.add_argument("--ckpt_step", type=int, default=1)
    parser.add_argument("--rl_algorithm", type=str, default="DAPO", help="rl_algorithm")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    AlphaRLBuildPerdictModel(
        args.step_model_path,
        args.ckpt_step,
        args.rl_algorithm,
        device=args.device
    )
