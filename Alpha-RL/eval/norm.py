import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from tqdm import tqdm

def save_svd_components(model1_path, model2_path, base_output_path, start_step=9, end_step=10):

    norm = 0
    count = 0
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

        model2 = AutoModelForCausalLM.from_pretrained(current_model2_path)
        model2.to('cuda')


        with torch.no_grad():
            for layer_idx, (layer1, layer2) in enumerate(zip(model1.model.layers, model2.model.layers)):

                for name, param in layer2.self_attn.named_parameters():
                    if name.endswith('.weight'):

                        ref_param = layer1.self_attn.get_parameter(name)
                        weight_update = param - ref_param
                        norm += torch.norm(weight_update)
                        count += 1

                for name, param in layer2.mlp.named_parameters():
                    if name.endswith('.weight'):
                        ref_param = layer1.mlp.get_parameter(name)
                        weight_update = param - ref_param
                        norm += torch.norm(weight_update)
                        count += 1
    print(norm/count)  
    return norm/count

# 设置路径
model1_path = ""
model2_path = ""
base_output_path  = ""
# 执行保存
start_step=1
end_step=1

save_svd_components(model1_path, model2_path, base_output_path, start_step, end_step)
