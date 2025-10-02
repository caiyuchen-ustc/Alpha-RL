import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def reconstruct_and_save_rank1(model1_path, model2_path, svd_components_base_path, start_step, end_step):
    print("Loading base config and tokenizer...")
    original_config = AutoConfig.from_pretrained(model1_path)
    target_dtype = original_config.torch_dtype
    tokenizer = AutoTokenizer.from_pretrained(model1_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device ='cpu'
    print(f"Using device: {device}")
    norm=0

    count=0
    svg_norm=0
    svg_count=0
    for top_k in range(0, 2, 10):
        for global_step in range(start_step, end_step + 1):
            print(f'\n===== Processing top_k: {top_k} =====')
            print("Loading original models...")
            model1 = AutoModelForCausalLM.from_pretrained(
                model1_path,
                torch_dtype=target_dtype,
                device_map=device  
            )
            model2_paths = os.path.join(model2_path)
            model2 = AutoModelForCausalLM.from_pretrained(
                model2_paths,
                torch_dtype=target_dtype,
                device_map=device 
            )
            print("Original models loaded successfully")
            try:
                print(f"\n----- Processing global_step {global_step} -----")

                if "aa" in model2_path:
                    svd_file = os.path.join(svd_components_base_path, f"global_step_{global_step}", 'svd_components.pt')
                    output_dir = os.path.join(svd_components_base_path, f"global_step_{global_step}", f'rank_1')

                else:
                    svd_file = os.path.join(svd_components_base_path,'svd_components.pt')
                    output_dir = os.path.join(svd_components_base_path, f'rank_1')


                if not os.path.exists(svd_file):
                    print(f"SVD file not found: {svd_file} - skipping")
                    continue


                os.makedirs(output_dir, exist_ok=True)

                svd_components = torch.load(svd_file, map_location=device)

                print("Starting model parameter updates...")

                for layer_idx, (layer1, layer2) in enumerate(zip(model1.model.layers, model2.model.layers)):
                    print(f"  Processing layer {layer_idx}")
                    layer_svd = svd_components.get(f'layer_{layer_idx}', {})
                    if not layer_svd:
                        print(f"  No SVD data for layer {layer_idx} - skipping")
                        continue


                    for (name1, param1), (name2, param2) in zip(layer1.self_attn.named_parameters(), 
                                                               layer2.self_attn.named_parameters()):

                        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
                        
                        if name1.endswith('.weight'):
                            key_U = f'self_attn_{name1}_U'
                            key_S = f'self_attn_{name1}_S'
                            key_Vt = f'self_attn_{name1}_Vt'
                            
                            if key_U in layer_svd and key_S in layer_svd and key_Vt in layer_svd:

                                U = layer_svd[key_U].to(device, dtype=target_dtype)
                                S = layer_svd[key_S].to(device, dtype=target_dtype)
                                Vt = layer_svd[key_Vt].to(device, dtype=target_dtype)


                                k = max(1, int(len(S) * top_k / 100))

                                print(f"    Using top {k}/{len(S)} components for self_attn.{name1}")


                                U_top = U[:, :k]
                                S_top = S[:k]
                                V_top = Vt[:k, :]

                                S_matrix = torch.diag(S_top)
                                USV_top = U_top @ S_matrix @ V_top

                                norm_factor = torch.norm(U@torch.diag(S)@Vt)
                                count+=1
                                USV_top_norm = torch.norm(USV_top)
                                norm+=USV_top_norm/norm_factor

                                svg_norm +=S_top.sum()/S.sum()
                                param1.data = param1.data + 1* (norm_factor/USV_top_norm)*USV_top
                                print(f"    Updated model1 self_attn.{name1} parameters")


                    for (name1, param1), (name2, param2) in zip(layer1.mlp.named_parameters(), 
                                                               layer2.mlp.named_parameters()):

                        assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
                        
                        if name1.endswith('.weight'):
                            key_U = f'mlp_{name1}_U'
                            key_S = f'mlp_{name1}_S'
                            key_Vt = f'mlp_{name1}_Vt'
                            
                            if key_U in layer_svd and key_S in layer_svd and key_Vt in layer_svd:

                                U = layer_svd[key_U].to(device, dtype=target_dtype)
                                S = layer_svd[key_S].to(device, dtype=target_dtype)
                                Vt = layer_svd[key_Vt].to(device, dtype=target_dtype)

                                k = max(1, int(len(S) * top_k / 100))

                                print(f"    Using top {k}/{len(S)} components for mlp.{name1}")


                                U_top = U[:, :k]
                                S_top = S[:k]
                                V_top = Vt[:k, :]


                                S_matrix = torch.diag(S_top)
                                USV_top = U_top @ S_matrix @ V_top 

                                norm_factor = torch.norm(U@torch.diag(S)@Vt) 
                                USV_top_norm = torch.norm(USV_top)
                                print('---------')
                                print(norm_factor)
                                print(USV_top_norm)
                                print('---------')
                                param1.data = param1.data + 1* (norm_factor/USV_top_norm)*USV_top
                                norm+=USV_top_norm/norm_factor

                                count+=1

                                print(f"    Updated model1 mlp.{name1} parameters")

                model1 = model1.to('cpu') 
                model1.save_pretrained(
                    output_dir,
                    torch_dtype=target_dtype,
                    safe_serialization=True
                )
                # tokenizer.save_pretrained(output_dir)
                # print(f"Successfully saved rank-1 model for global_step {global_step} (top_k={top_k})")

            except KeyboardInterrupt:
                print(f"\nUser interrupted - skipping global_step {global_step}")
                continue
            except Exception as e:
                print(f"\nError processing global_step {global_step}: {str(e)}")
                continue
            finally:
                torch.cuda.empty_cache() 


            del model1, model2, svd_components
            torch.cuda.empty_cache()  
            print(f"Released GPU memory after top_k={top_k}\n")
    print(norm/count)
    print("All processes completed")




model1_path = ""
model2_path = ""
svd_components_base_path = ""


start_step = 1
end_step = 1
reconstruct_and_save_rank1(model1_path, model2_path, svd_components_base_path, start_step, end_step)
