import argparse
import copy
import json
import os
import pickle
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

from arc_prompt import create_new_archive, get_prompt, get_self_reflection_prompt

client = openai.OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

from utils import random_id, format_arc_data, eval_solution, list_to_string, calculate_fitness

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `transform`. This function should take a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation that works for both example and test inputs. Make sure that the transform function is capable of handling both example and test inputs effectively, reflecting the learned transformation rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
generating_new_agents = True


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=1024, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        chat_log,
        model,
        temperature=0.8
):
    response = client.chat.completions.create(
        model=model,
        messages=chat_log,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gemma2', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        code_output = False

        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
        for key in output_fields_and_description:
            if 'answer' in key:
                output_fields_and_description[key] = f"Your {key}. ONLY return a string of list[list[int]]. DO NOT return anything else."
            elif 'code' in key:
                output_fields_and_description[key] = f"Your {key}. Don't write tests in your Python code, ONLY return the `transform` function. DO NOT return anything else. (It will be tested later.)"
                code_output = True
        system_prompt = ROLE_DESC(self.role) + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue

            if isinstance(content, list):
                try:
                    content = list_to_string(content)
                except:
                    pass

            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and generating_new_agents:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)


class AgentSystem():
    def __init__(self, examples, test_iuput) -> None:
        self.examples = examples
        self.test_iuput = test_iuput

    def run_examples_and_get_feedback(self, code):
        examples = self.examples

        correct_examples = []
        wrong_examples = []

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('feedback', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}"), correct_examples, wrong_examples
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code."), correct_examples, wrong_examples

        transform = local_vars['transform']

        feedback = ""

        for idx, example in enumerate(examples):
            input_grid = example['input']
            output_grid = example['output']
            try:
                transformed_grid = transform(input_grid)
            except Exception as e:
                return gen_output("Error during function execution: {e}"), correct_examples, wrong_examples

            if transformed_grid == output_grid:
                feedback += f"Your transform function generates a CORRECT answer in Example {idx}!\n\n"
                correct_examples.append(example)
            else:
                try:
                    transformed_grid = list_to_string(transformed_grid)
                except:
                    pass
                feedback += f"Your transform function generates a WRONG answer in Example {idx}!\nExpect: See above Example {idx} output.\nYou got: {transformed_grid}\nObserve the Example {idx} carefully!\n\n"
                wrong_examples.append(example)

        return gen_output(feedback), correct_examples, wrong_examples

    def get_test_output_from_code(self, code):
        test_input = self.test_iuput

        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        gen_output = lambda msg: Info('answer', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}")
        if 'transform' not in local_vars:
            return gen_output("Function 'transform' not found in the code.")

        transform = local_vars['transform']
        try:
            transform_output = transform(test_input)
            transform_output = list_to_string(transform_output)
        except Exception as e:
            return gen_output("Error during function execution: {e}")

        return gen_output(transform_output)


def search(cmd_line_args):
    file_path = os.path.join(cmd_line_args.save_dir, f"{cmd_line_args.archive_name}_run_archive.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = create_new_archive()
        start = 0

    for solution in archive:
        if 'fitness' in solution:
            continue

        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            score_list = evaluate_forward_fn(cmd_line_args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        calculated_fitness = calculate_fitness(score_list)
        solution['fitness'] = calculated_fitness

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    for n in range(start, cmd_line_args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt, prompt = get_prompt(archive)
        chat_log = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            llm_latest_response = get_json_response_from_gpt_reflect(chat_log, cmd_line_args.model)

            review_and_correct, correct_using_examples = get_self_reflection_prompt(archive[-1] if n > 0 else None)
            # self_reflection 1
            chat_log.append({"role": "assistant", "content": str(llm_latest_response)})
            chat_log.append({"role": "user", "content": review_and_correct})
            llm_latest_response = get_json_response_from_gpt_reflect(chat_log, cmd_line_args.model)
            # self_reflection 2
            chat_log.append({"role": "assistant", "content": str(llm_latest_response)})
            chat_log.append({"role": "user", "content": correct_using_examples})
            llm_latest_response = get_json_response_from_gpt_reflect(chat_log, cmd_line_args.model)
        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            continue

        score_list = []
        for _ in range(cmd_line_args.debug_max):
            try:
                score_list = evaluate_forward_fn(cmd_line_args, llm_latest_response["code"])
                if np.mean(score_list) < 0.01 and generating_new_agents:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                chat_log.append({"role": "assistant", "content": str(llm_latest_response)})
                chat_log.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    llm_latest_response = get_json_response_from_gpt_reflect(chat_log, cmd_line_args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        if not score_list:
            continue

        calculated_fitness = calculate_fitness(score_list)
        llm_latest_response['fitness'] = calculated_fitness
        llm_latest_response['generation'] = n + 1

        if 'debug_thought' in llm_latest_response:
            del llm_latest_response['debug_thought']
        if 'reflection' in llm_latest_response:
            del llm_latest_response['reflection']
        archive.append(llm_latest_response)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(cmd_line_args):
    file_path = os.path.join(cmd_line_args.save_dir, f"{cmd_line_args.archive_name}_run_archive.json")
    eval_file_path = str(os.path.join(cmd_line_args.save_dir, f"{cmd_line_args.archive_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    current_idx = 0
    while (current_idx < len(archive)):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if current_idx < len(eval_archive):
            current_idx += 1
            continue
        sol = archive[current_idx]
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        try:
            score_list = evaluate_forward_fn(cmd_line_args, sol["code"])
        except Exception as e:
            print(e)
            continue
        calculated_fitness = calculate_fitness(score_list)
        sol['test_fitness'] = calculated_fitness
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

        current_idx += 1


def evaluate_forward_fn(cmd_line_args, code_being_judged):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(code_being_judged, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    if generating_new_agents:
        arc_dir = cmd_line_args.val_data_path
    else:
        arc_dir = cmd_line_args.test_data_path
    print(arc_dir)
    with open(arc_dir, 'rb') as pickle_file:
        arc_data_queue = pickle.load(pickle_file)

    print(f"problem length: {len(arc_data_queue) * cmd_line_args.n_repreat}")
    max_workers = min(len(arc_data_queue) * cmd_line_args.n_repreat, cmd_line_args.max_workers) if cmd_line_args.multiprocessing else 1

    agent_task_queue = []
    for arc_data in arc_data_queue:
        task_str, examples, test_input = format_arc_data(arc_data)
        taskInfo = Info('task', 'User', task_str, -1)
        agent_task_queue.extend([(AgentSystem(examples, test_input), taskInfo, arc_data)] * cmd_line_args.n_repreat)

    def run_function(agent_task_queue):
        agent, taskInfo, arc_data = agent_task_queue
        function_response = agent.forward(taskInfo)
        origin_res = function_response
        try:
            if isinstance(function_response, Info):
                function_response = function_response.content
            if isinstance(function_response, str):
                function_response = eval(function_response)
            hard_score = eval_solution(function_response, arc_data, soft_eval=False)
            return hard_score
        except Exception as e:
            # print(e)
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        score_list = list(tqdm(executor.map(run_function, agent_task_queue), total=len(agent_task_queue)))

    print("acc:", calculate_fitness(score_list))
    return score_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data_path', type=str, default='sampled_arc_val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='sampled_arc_test_data.pkl')
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--archive_name', type=str, default='arc_llama3.1_results')
    parser.add_argument('--n_generation', type=int, default=25)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='llama3.1',
                        choices=['mistral-nemo', 'gemma2', 'llama3.1'])

    cmd_line_args = parser.parse_args()
    # search
    generating_new_agents = True
    search(cmd_line_args)

    # evaluate
    generating_new_agents = False
    evaluate(cmd_line_args)
