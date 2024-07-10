import argparse
import json
import re
import time
from sglang import function, gen, RuntimeEndpoint
import math
import threading
from collections import Counter
from evaluate.eval_deepseek.python_executor import *
from transformers import AutoTokenizer

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' or (l == '.' and start == True):
            start = True
            out.append(l)
        else:
            if start:
                break
        
    out = reversed(out)
    return ''.join(out)

def extract_answer(completion):
    try:
        result = re.findall(r'\\boxed\{(\d+\.?\d*)\}', completion)
        if not len(result):
            result =naive_parse(completion)
            result = int(float(result))
        else:
            result = result[-1]
            result = int(float(result))
            if result < 0:
                result = "Invalid"
    except Exception as e:
        print(e)
        result = "Invalid"
    return result

def convert_int(answer):
    try:
        answer = int(float(answer))
        if answer < 0:
            return "Invalid"
        return answer
    except Exception as e:
        print(e)
        return "Invalid"

def grade_answer(pred, answer):
    answer = int(answer)
    if pred == answer:
        return 1
    return 0

def get_prompts_jsonl(args):
    test_cases = read_jsonl(args.input_path)
    prompts = []
    for test in test_cases:
        prompts.append(test["problem"])
    return prompts, test_cases

def get_prompts_json(args):
    test_cases = json.load(open(args.input_path, "r"))
    prompts = []
    for test in test_cases:
        prompts.append(test["problem"])
    return prompts, test_cases

def extract_code(text):
    if not text.strip().endswith("```"):
        return ""
    if text.startswith("```python"):
        text = "hey\n" + text
    blocks = [block.split("```", 1)[0].strip() for block in text.split("```python") if '```' in block]
    blocks = [block for block in blocks if block]
    if not blocks:
        return ""
    code = []
    for block in blocks[:-1]:
        for line in block.split("\n"):
            if line.startswith("    ") or line.startswith("import") or line.startswith("def "):
                code.append(line)
            elif 'print(' not in line:
                code.append(line)
    code = "\n".join(code) + "\n" + blocks[-1]
    return code.strip()

@function
def tora_sampling(s, id, question, sampling_num, max_tokens, ground_truth_answer, temperature):
    prompt = f"{question}" + "\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
    # prompt = f"{question}" + "\nPlease integrate natural language reasoning with programs to solve the above problem, and put your final numerical answer within \\boxed{}.\nNote that the intermediary calculations may be real numbers, but the final numercal answer would always be an integer."
    s += prompt
    forks = s.fork(sampling_num)
    forks += gen("code", max_tokens=max_tokens, temperature=temperature, stop="```output")
    answers = []
    executor = PythonExecutor(get_answer_from_stdout=True)
    codes = []
    texts = []
    for state in forks:
        text = state.text()
        code_text = text.replace(prompt, "").split("```output")[0]
        code = extract_code(code_text)
        codes.append(code)
        texts.append(text)
    batch_results = executor.batch_apply(codes)
    for (exec_result, metadata), code, text in zip(batch_results, codes, texts):
        pred = convert_int(exec_result.strip())
        acc = 0
        if grade_answer(pred, ground_truth_answer):
            acc = 1
        answers.append({'answer':pred, 'code':code, 'text':text, 'accuracy':acc, 'exe_result':exec_result})
        
    answer_for_the_question = {"id":id, "question": question, "model_answer":answers, "ground_truth_answer": ground_truth_answer}
    return answer_for_the_question


@function
def text_sampling(s, id, question, sampling_num, max_tokens, ground_truth_answer, temperature):
    promt = f"{question}" + "Please reason step by step, and put your final answer within \\boxed{}. Although the intermeidate steps can be non-integers. The final result will always be integer."
    s += promt
    forks = s.fork(sampling_num)
    forks += gen("answer", max_tokens=max_tokens, temperature=temperature)
    answers = []
    for state in forks:
        text = state.text()
        answer = extract_answer(text)
        acc = 0
        if grade_answer(answer, ground_truth_answer):
            acc = 1
        answers.append({'answer':answer, 'text':text, "accuracy":acc})
    answer_for_the_question = {"id":id, "question": question, "model_answer":answers, "ground_truth_answer": ground_truth_answer}
    return answer_for_the_question


def majority_vote(answers):
    candidates = []
    for item in answers:
        answer = item["answer"]
        if isinstance(answer, int) and answer >= 0:
            candidates.append(answer)
    counter = Counter(candidates)
    most_common = counter.most_common(1)
    if most_common:
        return most_common[0][0]
    else:
        return None


def main(args):
    if "jsonl" in args.input_path:
        prompts, test_examples = get_prompts_jsonl(args)
    else:
        prompts, test_examples = get_prompts_json(args)
    input_list_dict = []
    for i, prompt in enumerate(prompts):
        input_list_dict.append({"id": i, "question": prompt, "sampling_num": args.sampling_num, "max_tokens": args.max_tokens, "ground_truth_answer": test_examples[i]["answer"], "temperature": args.temperature})
    if args.method == "program_only":
        states = tora_sampling.run_batch(input_list_dict, backend=RuntimeEndpoint(args.policy_host), num_threads=args.num_threads, progress_bar=True)
    if args.method == "text_only":
        states = text_sampling.run_batch(input_list_dict, backend=RuntimeEndpoint(args.policy_host), num_threads=args.num_threads, progress_bar=True)
    results = []
    correct_num = 0
    for s, truth in zip(states, test_examples):
        answer = s.ret_value
        answer["correctness"] = grade_answer(majority_vote(answer["model_answer"]), answer["ground_truth_answer"])
        results.append(answer)
        correct_num += answer["correctness"]
    with open(args.info_path, "a") as file:
        file.write(f"T-{args.temperature}_N-{args.sampling_num}, Method-{args.method},  Accuracy: {correct_num / len(test_examples) * 100.0}\n")
    json.dump(results, open(args.output_path, "w"), indent=4)



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--sampling_num', type=int, default=1)
    args_parser.add_argument('--method', type=str, required=True)
    args_parser.add_argument('--max_tokens', type=int, default=1024)
    args_parser.add_argument('--info_path', type=str, required=True)
    args_parser.add_argument('--policy_host', type=str, default="http://localhost:30100")
    args_parser.add_argument('--num_threads',  type=int, required=True)
    args_parser.add_argument('--temperature', type=float, default=0)
    args = args_parser.parse_args()
    main(args)