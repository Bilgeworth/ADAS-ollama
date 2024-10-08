import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import openai
import pandas
from tqdm import tqdm

client = openai.OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

from mmlu_utils import format_multichoice_question, random_id, calculate_fitness

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
    json_dict = json.loads(content, strict=False)
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
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
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


def evaluate(cmd_line_args):
    eval_file_path = cmd_line_args.eval_file_path
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            test_entries = json.load(json_file)
    else:
        raise AssertionError(f"File {eval_file_path} does not exist.")

    for sol in test_entries:
        print(f"{sol['name']}")
        score_list = evaluate_forward_fn(cmd_line_args, sol['code'])
        sol['test_fitness_GPQA'] = calculate_fitness(score_list)

    # Step 5: Save the test entries
    with open(eval_file_path, 'w') as json_file:
        json.dump(test_entries, json_file, indent=4)


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

    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # set seed 0 for valid set
    df = pandas.read_csv(cmd_line_args.data_filename)
    random.seed(cmd_line_args.shuffle_seed)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)

    if generating_new_agents:
        examples = examples[:cmd_line_args.valid_size] * cmd_line_args.n_repreat
    else:
        examples = examples[cmd_line_args.valid_size:cmd_line_args.valid_size + cmd_line_args.test_size] * cmd_line_args.n_repreat

    questions = [format_multichoice_question(example) for example in examples]
    answers = [LETTER_TO_INDEX[example['Answer']] for example in examples]

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
            if isinstance(function_response, str) and function_response in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[function_response]
            elif 'A)' in function_response:
                predicted_idx = 0
            elif 'B)' in function_response:
                predicted_idx = 1
            elif 'C)' in function_response:
                predicted_idx = 2
            elif 'D)' in function_response:
                predicted_idx = 3
            elif isinstance(function_response, list):
                try_res = function_response[1]
                predicted_idx = LETTER_TO_INDEX[try_res.content]
            elif function_response.content in LETTER_TO_INDEX:
                predicted_idx = LETTER_TO_INDEX[function_response.content]
            elif 'A)' in function_response.content:
                predicted_idx = 0
            elif 'B)' in function_response.content:
                predicted_idx = 1
            elif 'C)' in function_response.content:
                predicted_idx = 2
            elif 'D)' in function_response.content:
                predicted_idx = 3
            else:
                print(f"error in q {q_idx}")
                score_list.append(0)
                continue
        except Exception as e:
            score_list.append(0)
            continue

        if predicted_idx == answers[q_idx]:
            score_list.append(1)
        else:
            score_list.append(0)
    print(f"acc: {calculate_fitness(score_list)}")
    return score_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="dataset/mmlu.csv")
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repreat', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--eval_file_path', type=str, default='')
    parser.add_argument('--model',
                        type=str,
                        default='llama3.1',
                        choices=['mistral-nemo', 'gemma2', 'llama3.1'])

    cmd_line_args = parser.parse_args()

    generating_new_agents = False
    evaluate(cmd_line_args)
