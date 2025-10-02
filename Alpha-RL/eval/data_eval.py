
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import sys
import math
sys.path.append('')
import argparse
import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb


import sympy as sp
from sympy import simplify, Eq, sympify, Pow
from sympy.parsing.latex import parse_latex

class OBJudge:
    def __init__(self):

        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8  # Default precision for comparison

    def split_by_comma(self, expr: str):
        # Splits expressions by commas outside of brackets
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char in ["(", "["]:
                in_bracket_num += 1
            elif char in [")", "]"]:
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())   
        
        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        # Translates plus-minus signs into separate expressions
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)
        
        return new_expr_list
    
    def judge(self, expression1, expression2, precision=1e-8):
        # Judge if two expressions are equal (expression1 is considered as the Ground Truth)
        # Default precision is a list for supporting multiple expressions
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except:
            return False
        if expression1 == expression2:
            # print("Exactly equal")
            return True
        
        # Remove Chinese characters from the string, as answers like "yes" or "no" in Chinese have been considered
        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)
        
        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # Set up a list for allowed errors
        if len(precision) <= 1:
            precision = precision * len(temp_list1)
        
        if len(temp_list1) != len(temp_list2):
            return False

        # Check if elements in both lists can be paired and are equal
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]

            for item2 in temp_list2:
                if self.is_equal(item1, item2):
                    temp_list1.remove(item1)
                    temp_list2.remove(item2)
                    precision.remove(self.precision)
                    break
            else:
                # If no match was found, return False
                return False

        # If all elements are matched, return True
        return True
    
    def is_interval(self, expr):
        # Checks if an expression is an interval
        return expr.startswith(("(", "[")) and expr.endswith((")", "]"))

    def sympy_sub_pi(self, expression_sympy):
        # Replaces the symbol for pi in sympy expressions with its numerical value
        return expression_sympy.subs(self.pi, math.pi)
    
    def is_equal(self, expression1, expression2):
        # Default first expression is ground truth. Check if expressions are equal in different aspects
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            # print("Equivalent natively")
            return True

        # First check if both are intervals
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    # print("Interval equivalent")
                    return True
            except:
                return False

        # Then check for numerical equality
        try:
            if self.numerical_equal(expression1, expression2):
                # print("Numerically equivalent")
                return True
        except:
            pass
        
        # Then check if expressions are mathematically equal
        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
                # print("Expression equivalent")
                return True
        except:
            pass
            
        # Lastly, check for equation equality
        try:
            if self.equation_equal(expression1, expression2):
                # print("Equation equivalent")
                return True
        except:
            pass
            
        return False

    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):
        # Check if two numerical values are equal within an allowed error range
        # Includes possible percentage cases
        reference = float(expression1)
        prediction = float(expression2)
        
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        
        for item in gt_result:
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    def expression_equal(self, exp1, exp2):
        # Check if two expressions are mathematically equivalent
        # Extract expression and use sympy for equivalence checking
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()
        
        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print(f"These two numbers cannot be calculated by the current computer for: \"{str(expr1_sym)}\" and \"{str(expr2_sym)}\"")
                        return False

                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except:
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)
                    num_value = simplified_expr.evalf()
                    return abs(num_value) < 1e-3
                except:
                    return False

    def equation_equal(self, expression1, expression2):
        # Check if two equations are mathematically equivalent
        # Simplify equations and use sympy for equivalence checking
        def simplify_equation(latex_eq):
            lhs, rhs = latex_eq.split('=')
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)
            equation = Eq(lhs_expr, rhs_expr)
            simplified_eq = simplify(equation.lhs - equation.rhs)
            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        if (division_result_1.is_Integer and division_result_1 != 0) or (division_result_2.is_Integer and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        # Check if two intervals are mathematically equivalent
        def compare_two_interval(inter1, inter2):
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False
            
            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True
            
        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")
            
            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):
        # Preprocess expressions to extract and replace special symbols
        def extract_boxed_content(latex_str):
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    raise ValueError("Mismatched braces in LaTeX string.")

            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str
                
            return results
        
        def sepcial_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]
            
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression
        
        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2
    
    def can_compute_power(self, expr):
        # Checks if a power expression can be computed
        if isinstance(expr, Pow):
            base, exp = expr.as_base_exp()
            if base.is_number and exp.is_number:
                MAX_EXP = 1000  # Adjust based on computing environment
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True  # Not a power expression, can compute

# 新增一个增强版的正确性检查函数
def enhanced_check_is_correct(generated_answer, gt_ans):
    """
    增强版的正确性检查函数，结合原有的check_is_correct和新的OBJudge
    """
    # 首先尝试原有的检查方法
    try:
        original_result = check_is_correct(generated_answer, gt_ans)
        if original_result:
            return True
    except:
        pass
    
    # 如果原有方法失败或返回False，尝试使用OBJudge
    try:
        obj_judge = OBJudge()
        enhanced_result = obj_judge.judge(str(gt_ans), str(generated_answer), 1e-8)
        return enhanced_result
    except Exception as e:
        print(f"Enhanced judge failed: {e}")
        return False

# envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

def parse_list(arg):
    return arg.split(',')


def apply_template(tokenizer, question):
    txt = tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return txt


def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="/project/ugmathllm/caiyuchen/LIMO/eval/data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--repetition_penalty", default=1, type=float)
    parser.add_argument("--start_penalty_length", default=0, type=int)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--QwQ", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./previous500", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    parser.add_argument("--Iterations", default=1, type=int)
    # 新增参数：是否使用增强版评估器
    parser.add_argument("--use_enhanced_judge", action="store_true", help="Use enhanced OBJudge for evaluation")
    # parser.add_argument("--use_qwen_check", action="store_true")
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
    print(f"current stop list: {args.stop}")
    return args


def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join("/project/ugmathllm/caiyuchen/LIMO/eval/prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format

def truncate_text_by_percentage(text, percentage):

    if percentage >= 100:
        return text
    
    paragraphs = [p for p in re.split(r'\n\n', text)]
    
    clean_paragraphs = [p for p in paragraphs if p.strip()]
    
    if not clean_paragraphs:
        return text
    
    total_paragraphs = len(clean_paragraphs)
    target_count = math.ceil(total_paragraphs * percentage / 100)
    target_count = max(1, target_count)
    selected_paragraphs = clean_paragraphs[:target_count]
    truncated_text = "\n\n".join(selected_paragraphs) + "\n\n"
    
    return truncated_text

def truncate_text_by_repently_0_percentage(repently_0text, text, percentage, tokenizer):
    if percentage >= 100:
        return text
    
    # 分割文本为段落并清理空段落
    paragraphs = [p for p in re.split(r'\n\n', repently_0text) if p.strip()]
    
    if not paragraphs:
        return text
    
    # 计算需要保留的段落数
    total_paragraphs = len(paragraphs)
    target_count = max(1, math.ceil(total_paragraphs * percentage / 100))
    
    # 截取指定数量的段落
    selected_paragraphs = paragraphs[:target_count]
    truncated_text = "\n\n".join(selected_paragraphs) + "\n\n"
    
    # 使用tokenizer统计token数量，跳过特殊token
    truncated_tokens = tokenizer.encode(truncated_text, add_special_tokens=False)
    token_count = len(truncated_tokens)
    result = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    return result



def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    print(f"使用增强版评估器: {args.use_enhanced_judge}")
    
    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_tokens, 
                                     n=factor,
                                     #repetition_penalty=args.repetition_penalty,
                                     #min_p=1,
                                     #frequency_penalty = args.frequency_penalty
                                     )
    data_names = ['math','aime24','aime25','minerva','gpqa','gsm8k']
    # data_names = ['math']

    is_large_model = "32" in args.model_name_or_path
        
    if is_large_model:
        llm = LLM(model=model_name_or_path, 
            tensor_parallel_size=len(available_gpus), 
            trust_remote_code=True, 
            gpu_memory_utilization=0.7,   
            swap_space=64              
            )
    else:
        llm = LLM(model=model_name_or_path, 
            tensor_parallel_size=len(available_gpus), 
            trust_remote_code=True, 
            gpu_memory_utilization=0.96, 
            )
    for data_name in tqdm(data_names):
        args.data_name = data_name
        print(args.data_name)
        examples = load_data(args.data_name, args.split, args.data_dir)

        if args.end_idx == -1:
            args.end_idx = len(examples)
        examples = examples[args.start_idx:args.end_idx]

        dt_string = datetime.now().strftime("%m-%d_%H-%M")
        model_name = "/".join(args.model_name_or_path.split("/")[-3:])
        out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}'
        out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_{args.Iterations}st{args.start_penalty_length}_r{args.repetition_penalty}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'
        
        if os.path.exists(out_file):
            print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
        
        os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)
        os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}', exist_ok=True)
        

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 准备所有prompts
        prompt_batch = []
        for idx, example in tqdm(enumerate(examples)):
            question = parse_question(example, args.data_name)
            question_prompt = '\nPlease reason step by step, and put your final answer within \\boxed{}'

            question =  question + question_prompt
            cur_prompt = apply_template(tokenizer=tokenizer, question=question)
            prompt_batch.append(cur_prompt)
        
        print("Sample prompt:")
        print(prompt_batch[0])
        batch_size = len(prompt_batch)
        
        file_outputs = []
        correct_cnt = 0
        
        for cur_generation_epoch in range(generation_epoch):
            print(f"🔄{cur_generation_epoch + 1}/{generation_epoch}")

            for batch_start in tqdm(range(0, len(prompt_batch), batch_size)):
                batch_end = min(batch_start + batch_size, len(prompt_batch))
                current_batch_prompts = prompt_batch[batch_start:batch_end]
                current_batch_examples = examples[batch_start:batch_end]
                

                try:
                    completions = llm.generate(current_batch_prompts, sampling_params)
                    

                    for i, completion in enumerate(completions):
                        global_idx = batch_start + i
                        d = current_batch_examples[i]
                        question = parse_question(d, args.data_name)
                        generated_responses = [completion.outputs[j].text for j in range(len(completion.outputs))]
                        
                        if cur_generation_epoch == 0:

                            result = {
                                "question": question,
                                "generated_responses": generated_responses,
                            }
                            if "id" in d:
                                result["id"] = d["id"]
                            if "source" in d:
                                result["source"] = d["source"]
                            file_outputs.append(result)
                        else:
                            # 后续轮次：追加结果
                            file_outputs[global_idx]['generated_responses'] += generated_responses
                            
                except Exception as e:
                    print(f"❌")

                    raise e


        pass_at_k_list = []
        k = args.k
        
        for i in tqdm(range(len(examples)), "检查正确性..."):
            d = examples[i]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
            generated_responses = file_outputs[i]['generated_responses']
            generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
            
            # 🔥 关键修改：使用增强版的正确性检查函数
            if args.use_enhanced_judge:
                is_correct_list = [enhanced_check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            else:
                is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            
            is_correct = any(is_correct_list)
            if is_correct:
                correct_cnt += 1
            file_outputs[i]['generated_answers'] = generated_answers
            file_outputs[i]['gold_answer'] = gt_ans
            file_outputs[i]['is_correct'] = is_correct
            file_outputs[i]['answers_correctness'] = is_correct_list
            
            if len(is_correct_list) > 1:
                correct_answers = sum(is_correct_list)
                n = len(generated_answers)
                if correct_answers > 0:
                    if n - correct_answers < k:
                        pass_at_k = 1
                    else:
                        pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                    pass_at_k_list.append(pass_at_k)
                else:
                    pass_at_k_list.append(0)
        
        # 保存结果
        temp_out_file = out_file + ".tmp"
        with open(temp_out_file, 'w', encoding='utf-8') as f:
            count = 0
            for d in tqdm(file_outputs, "写入结果文件..."):
                f.write(json.dumps(d, ensure_ascii=False))
                f.write("\n")
                count += 1
                if count % 100 == 0:
                    f.flush()
            acc = correct_cnt/ len(examples)
            f.write(json.dumps({'acc':acc}, ensure_ascii=False))
            f.flush()
        os.rename(temp_out_file, out_file)
        
        print(f"📊 正确数量 / 总数量: {correct_cnt}/{len(examples)}")
        print(f"🎯 准确率: {correct_cnt / len(examples):.4f}")

        if pass_at_k_list:
            average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
            print(f"🎯 Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
        else:
            print(f"🎯 Pass@1: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer(args)
