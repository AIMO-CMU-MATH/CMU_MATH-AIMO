APIKEY = ""
MODEL = ""
from openai import OpenAI
import json
import argparse
import re
import time
from evaluate.data_processing.answer_extraction import *
from evaluate.eval_deepseek.python_executor import *
client = OpenAI(api_key = APIKEY)



def few_shot_example_gpt(question):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=1.0,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You are a helpful math expert, you will be asked to solve a math problem using sympy-based python program. Your answer should strictly follow the example format."},
                {"role": "system", "name": "example_user", "content": "Please use python to solve the problems below, and put your final answer within \\boxed{}. The answer will be a nonnegative integer, your python code should exactly print a nonegative integer.\n Problem: Tom, Dorothy, and Sammy went on a vacation and agreed to split the costs evenly. During their trip Tom paid $ $105$ , Dorothy paid $ $125$ , and Sammy paid $ $175$ . In order to share the costs equally, Tom gave Sammy $t$ dollars, and Dorothy gave Sammy $d$ dollars. What is $t-d$"},
                {"role": "system", "name": "example_assistant", "content": "```python\nfrom sympy import symbols, Eq, solve\n\ndef calculate_difference():\n    \"\"\"Tom, Dorothy, and Sammy went on a vacation and agreed to split the costs evenly. During their trip Tom paid $ $105$ , Dorothy paid $ $125$ , and Sammy paid $ $175$ . In order to share the costs equally, Tom gave Sammy $t$ dollars, and Dorothy gave Sammy $d$ dollars. What is $t-d$\"\"\"\n    t, d = symbols('t d')\n\n    # Tom gave Sammy t dollars\n    equation_1 = Eq(105 + t, 125 + 175 - t)\n\n    # Dorothy gave Sammy d dollars\n    equation_2 = Eq(125 + d, 105 + 175 - d)\n\n    # Solve the equations\n    solution = solve((equation_1, equation_2), (t, d))\n\n    return solution[t] - solution[d]\n\nresult = calculate_difference()\nprint(result)\n```\n\n```output\n20\n```\nThe difference between the amount Tom gave Sammy and the amount Dorothy gave Sammy is $\\boxed{20}$ dollars."},
                {"role": "system", "name": "example_user", "content": "Please use python to solve the problems below, and put your final answer within \\boxed{}. The answer will be a nonnegative integer, your python code should exactly print a nonegative integer.\n Problem: Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$"},
                {"role": "system", "name": "example_assistant", "content": "```python\nfrom sympy import symbols, solve, simplify\n\ndef sum_of_initial_values():\n    \"\"\"Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$\"\"\"\n    x, a, b, c, d, e, f = symbols('x a b c d e f')\n\n    # Define the quadratic polynomials\n    P = 2*x**2 + a*x + b\n    Q = -2*x**2 + c*x + d\n\n    # Set up the system of equations\n    eq1 = P.subs(x, 16) - 54\n    eq2 = P.subs(x, 20) - 53\n    eq3 = Q.subs(x, 16) - 54\n    eq4 = Q.subs(x, 20) - 53\n\n    # Solve the system of equations\n    solution = solve((eq1, eq2, eq3, eq4), (a, b, c, d))\n\n    # Calculate P(0) + Q(0)\n    sum_initial_values = simplify(P.subs({x: 0, **solution}) + Q.subs({x: 0, **solution}))\n\n    return sum_initial_values\n\nresult = sum_of_initial_values()\nprint(result)\n```\n\n```output\n116\n```\nThe value of $P(0) + Q(0)$ is $\\boxed{116}$."},
                {"role": "user", "content": "Please use python to solve the problems below, and put your final answer within \\boxed{}. The answer will be a nonnegative integer, your python code should exactly print a nonegative integer.\n" + f" Problem: {question}"},
            ]
            )
        return response.choices[0].message.content
    except Exception as e:
        time.sleep(10)
        return None

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

def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)

def extract_answer(completion):
    try:
        result = re.findall(r'\\boxed\{(\d+\.?\d*)\}', completion)
        if not len(result):
            result =naive_parse(completion)
            result = int(result)
        else:
            result = result[-1]
            result = int(result)
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

def main(args):
    data = json.load(open(args.input_path, "r"))
    executor = PythonExecutor(get_answer_from_stdout=True)
    completed_sample = []
    it = 0
    solution_set = {}
    try_set = {}
    with open(args.output_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            completed_sample.append(json_object)
            if json_object["problem"] not in solution_set:
                solution_set[json_object["problem"]] = json_object["accuracy"]
                try_set[json_object["problem"]] = 0
            else:
                solution_set[json_object["problem"]] += json_object["accuracy"]
            try_set[json_object["problem"]] += 1
    for item in data:
        problem = item["problem"]
        answer = extract_answer(item["solution"])
        try:
            answer = int(answer)
        except Exception as e:
            continue
        for j in range(args.max_sampling_num):
            if problem not in solution_set:
                solution_set[problem] = 0
            if problem not in try_set:
                try_set[problem] = 0
            if solution_set[problem] >= args.correct_sampling_num or try_set[problem] >= args.max_sampling_num:
                break  
            for i in range(10):
                sampled_solution = few_shot_example_gpt(problem)
                if sampled_solution is not None:
                    break
                print(f"try again {i}")
            try:
                sampled_solution = "```python" + sampled_solution.split("```python")[1]
                code = extract_code(sampled_solution.split("```output")[0])
            except Exception as e:
                code = ""
            result, _ = executor.batch_apply([code])[0]
            result = convert_int(result.strip())
            accuracy = 0
            try_set[problem] += 1
            if grade_answer(result, answer):
                accuracy = 1
                solution_set[problem] += 1
            ouput_item = {"accuracy": accuracy, "prediction":result, "answer": answer, "problem": problem, "solution_idx":j, "sampled_solution":sampled_solution}
            with open(args.output_path, "a") as file:
                json.dump(ouput_item, file)
                file.write('\n')
        


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--max_sampling_num', type=int, default=1)
    args_parser.add_argument('--correct_sampling_num', type=int, default=1)
    args = args_parser.parse_args()
    main(args)