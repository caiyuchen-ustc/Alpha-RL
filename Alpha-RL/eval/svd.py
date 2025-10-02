import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from tqdm import tqdm

def save_svd_components(model1_path, model2_path, base_output_path, start_step=9, end_step=10):

    os.makedirs(base_output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model1_path)
    model1 = AutoModelForCausalLM.from_pretrained(model1_path)
    model1.to('cuda')


    print(start_step)
    print(end_step)
    for global_step in tqdm(range(start_step, end_step + 1)):

        if "aa" in model2_path:
            current_model2_path = model2_path.format(i=global_step)
            current_output_path = os.path.join(base_output_path, f"global_step_{global_step}")

            os.makedirs(current_output_path, exist_ok=True)
        else:
            current_model2_path = model2_path
            current_output_path = model2_path

        print(f" {current_model2_path}")
        model2 = AutoModelForCausalLM.from_pretrained(current_model2_path)

        svd_components = {}
        with torch.no_grad():
            for layer_idx, (layer1, layer2) in enumerate(zip(model1.model.layers, model2.model.layers)):
                layer_svd = {}
                for name, param in layer2.self_attn.named_parameters():
                    if name.endswith('.weight'):
                        print(f"self_attn.{name}")
                        ref_param = layer1.self_attn.get_parameter(name)
                        weight_update = param - ref_param
                        weight_update.to('cuda')
                        if weight_update.dim() >= 2:
                            U, S, Vt = torch.linalg.svd(weight_update, full_matrices=False)
                            weight_update.cpu()
                            layer_svd[f'self_attn_{name}_U'] = U.cpu()
                            layer_svd[f'self_attn_{name}_S'] = S.cpu()
                            layer_svd[f'self_attn_{name}_Vt'] = Vt.cpu()
                            

                for name, param in layer2.mlp.named_parameters():
                    if name.endswith('.weight'):
                        print(f"    mlp.{name}")
                        ref_param = layer1.mlp.get_parameter(name)
                        weight_update = param - ref_param
                        weight_update.to('cuda')
                        if weight_update.dim() >= 2:
                            U, S, Vt = torch.linalg.svd(weight_update, full_matrices=False)
                            weight_update.cpu()
                            layer_svd[f'mlp_{name}_U'] = U.cpu()
                            layer_svd[f'mlp_{name}_S'] = S.cpu()
                            layer_svd[f'mlp_{name}_Vt'] = Vt.cpu()
                

                svd_components[f'layer_{layer_idx}'] = layer_svd
        

        torch.save(svd_components, os.path.join(current_output_path, 'svd_components.pt'))
        



model1_path = ""
model2_path = ""
base_output_path = ""

start_step=27
end_step=27
print(start_step)
save_svd_components(model1_path, model2_path, base_output_path, start_step, end_step)
