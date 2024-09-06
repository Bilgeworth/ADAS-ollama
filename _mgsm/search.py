import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

from mgsm_prompt import create_new_archive, get_prompt, get_self_reflection_prompt

client = openai.OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

from utils import get_all_examples, random_id, calculate_fitness, score_mgsm

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""
ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

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
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
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
        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY an integer. DO NOT return anything other than the integer answer." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # construct input infos text
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
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
    def __init__(self) -> None:
        pass


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
            n -= 1
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
            n -= 1
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
        current_idx += 1
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

    # set seed 0 for valid set
    examples = get_all_examples()
    random.seed(cmd_line_args.shuffle_seed)
    random.shuffle(examples)

    if generating_new_agents:
        examples = examples[:cmd_line_args.valid_size] * cmd_line_args.n_repreat
    else:
        examples = examples[cmd_line_args.valid_size:cmd_line_args.valid_size + cmd_line_args.test_size] * cmd_line_args.n_repreat

    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]

    print(f"problem length: {len(examples)}")
    max_workers = min(len(examples), cmd_line_args.max_workers) if cmd_line_args.multiprocessing else 1

    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1)
        task_queue.append(taskInfo)

    agentSystem = AgentSystem()

    score_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    for q_idx, function_response in enumerate(results):
        try:
            if isinstance(function_response, Info):
                extracted_answer = function_response.content
            else:
                extracted_answer = function_response
            correct_answer = answers[q_idx]
            correct = score_mgsm(correct_answer, extracted_answer)
        except Exception as e:
            score_list.append(0)
            continue

        score_list.append(1 if correct else 0)
    print(f"acc: {calculate_fitness(score_list)}")
    return score_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--archive_name', type=str, default="mgsm_llama3.1_results")
    parser.add_argument('--n_generation', type=int, default=30)
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
