import os
import torch

def extract_first_v_from_svd(svd_file, output_file, device="cpu"):
    """从svd_components.pt中提取每个模块V的第一行，并保存"""
    if not os.path.exists(svd_file):
        print(f"❌ Not found: {svd_file}")
        return

    svd_components = torch.load(svd_file, map_location=device)
    first_v_dict = {}

    for layer_key, layer_svd in svd_components.items():
        for key, value in layer_svd.items():
            # import pdb
            # pdb.set_trace()
            if key.endswith("_U"):
                module_name = key.replace("_U", "")
                S = layer_svd[module_name + "_S"].to(device)  # [r]
                Vt = layer_svd[module_name + "_Vt"].to(device)  # [r, n]

                U = value.to(device)

                if U.size(0) == 0:
                    print(f"⚠️ {layer_key} {key} empty, skipping.")
                    continue
                print(key)
                print(U.shape)
                norm = torch.norm(U@torch.diag(S)@Vt)
                first_row = U[:, 0].clone()
                print(first_row)
                first_v_dict[f"{layer_key}_{module_name}"] = first_row

    torch.save(first_v_dict, output_file)
    print(f"✅ Saved {output_file}")


def process_all_steps(base_path, start=1, end=27):
    device = torch.device("cpu")

    for i in range(start, end + 1):
        step_dir = os.path.join(base_path, f"global_step_{i}")
        svd_file = os.path.join(step_dir, "svd_components.pt")
        output_file = os.path.join(step_dir, "first_u_vectors.pt")

        print(f"\n===== Processing step {i} =====")
        extract_first_v_from_svd(svd_file, output_file, device)


if __name__ == "__main__":
    base_path = ""
    process_all_steps(base_path, start=1, end=27)
