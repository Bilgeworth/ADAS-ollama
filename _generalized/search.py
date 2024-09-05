import argparse
import copy
import json
import os
import pickle
import backoff
import numpy as np
import openai

from tqdm import tqdm
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from system_prompt_library import get_init_archive, get_prompt, get_reflexion_prompt
from data_tools import format_data, eval_solution, bootstrap_confidence_interval, random_id, list_to_string

######################################################### API Setup #########################################################
# Initialize OpenAI client
client = openai.OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama',
)

# Low temp short response, the initial response
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def api_call_initial(
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
    assert not json_dict is None
    return json_dict

# High temp long response, used for followup attempts
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def api_call_followup(
        msg_list,
        model,
        temperature=0.8
):
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict

######################################################### Agent Setup #########################################################
# Define a named tuple for storing information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instruction for JSON response
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n"""

SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `current_function`. This function should take a single input, and return the single output. You should make sure that you implement a version of the code that works for both example and test inputs. Make sure that the current_function function is capable of handling both example and test inputs effectively, reflecting the learned rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

# Agent template
class LLMAgentBase():
    """
    Attributes:
    """
    # Adds a function to the agent to initialize it with the given fields i.e. exampleAgent = agent(output_fields, agent_name, role, model, temperature)
    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='llama3.1', temperature=0.5) -> None:
        
        # Pipe the inputs into the given agent instance being made
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = random_id()

    # Generates the prompts for the agent
    def generate_prompt(self, input_infos, instruction) -> str:
        # Assume no code output
        code_output = False

        # Grab the list of output fields and their descriptions
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
        
        # Special instructions for answer and code fields, wrapped around the given entry from above and reinserted
        for key in output_fields_and_description:
            if 'answer' in key:
                output_fields_and_description[key] = f"Your {key}. ONLY return a string of list[list[int]]. DO NOT return anything else."
            elif 'code' in key:
                output_fields_and_description[key] = f"Your {key}. Don't write tests in your Python code, ONLY return the `current_function` function. DO NOT return anything else. (It will be tested later.)"
                code_output = True
        
        # Construct the system prompt
        system_prompt = ROLE_DESC(self.role) + FORMAT_INST(output_fields_and_description)

        # Clear the text from the last loop, iterate through input info, if its in the Info tuple, unpack it
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            
            # If the content is a list, convert it to a string
            if isinstance(content, list):
                try:
                    content = list_to_string(content)
                except:
                    pass
            
            # If the author is the same as the agent, append a note to the author
            if author == self.__repr__():
                author += ' (yourself)'
            # If the field name is task, wrap that field with Your Task: and new lines
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'

            # Otherwise just pass the input_info to input_infos_text
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        # Make the prompt with the input_infos_text, the untouched instruction, and whether to do code_output
        prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
        return system_prompt, prompt

    # Make the json to send to the OpenAI API client, send it, and handle the response
    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        # For the current agent instance, generate the prompt
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        
        # Try to get the json response from the LLM
        try:
            response_json = {}
            response_json = api_call_initial(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # Max context error, for example llama3.1 can keep track of 128k tokens at a time
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # Pad the response_json with empty strings if the response_json is less than the output_fields
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            # Remove any extra fields from the response_json
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]

        # Create a list of Info tuples to return
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    # Adds a function to the agent to return the name and id i.e. agent.__repr__()
    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    # Makes the agent callable as a function i.e. agent(input_infos, instruction, iteration_idx)
    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

######################################################### Agent Setup #########################################################
# Agent System class
class AgentSystem():
    # When agent system is initialized, pass its inputs into the given agent system instance
    def __init__(self, examples, test_iuput) -> None:
        self.examples = examples
        self.test_iuput = test_iuput

    # Pass the example list into the agent system instance
    def run_examples_and_get_feedback(self, code):
        examples = self.examples

        correct_examples = []
        wrong_examples = []

        # If the code is the Info tuple, assign the author and content to the agent system instance
        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        # Construct an Info tuple from the msg
        gen_output = lambda msg: Info('feedback', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        # Try the code with local variables only, if it fails, return an error message alongside what succeeded and failed
        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}"), correct_examples, wrong_examples
        
        # The current_function is the actual function that will be evaluated

        # If the current_function function is not in the local variables, return an error message
        if 'current_function' not in local_vars:
            return gen_output("Function 'current_function' not found in the code."), correct_examples, wrong_examples
        
        # Pull the current_function function from the local variables
        current_function = local_vars['current_function']

        feedback = ""

        # Run the inputs through the current_function function and compare them to the expected outputs, return the feedback
        for idx, example in enumerate(examples):
            # Unpack the input and output grids from the example
            input_grid = example['input']
            output_grid = example['output']

            # Try to run the input grid through the current_function function
            try:
                output_grid = current_function(input_grid)
            except Exception as e:
                return gen_output("Error during function execution: {e}"), correct_examples, wrong_examples

            # If the current_function function returns the correct output grid, return a correct message
            if output_grid == output_grid:
                feedback += f"Your current_function generates a CORRECT answer in Example {idx}!\n\n"
                correct_examples.append(example)
            else:
                try:
                    output_grid = list_to_string(output_grid)
                except:
                    pass
                feedback += f"Your current_function generates a WRONG answer in Example {idx}!\nExpect: See above Example {idx} output.\nYou got: {output_grid}\nObserve the Example {idx} carefully!\n\n"
                wrong_examples.append(example)

        # Return the feedback, correct examples, and wrong examples
        return gen_output(feedback), correct_examples, wrong_examples

    # Called in the prompts, lets debaters and judges evaluate the code
    def get_test_output_from_code(self, code):
        test_input = self.test_iuput

        # if the code is the Info tuple, assign the author and content to the agent system instance
        if isinstance(code, Info):
            author = code.author
            code = code.content
        else:
            author = None

        # construct an Info tuple from the msg
        gen_output = lambda msg: Info('answer', f"{author}'s code evaluator" if author else "code evaluator", msg, -1)

        # try the code with local variables only, if it fails, return an error message
        local_vars = {}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return gen_output(f"Error during code execution: {e}")
        if 'current_function' not in local_vars:
            return gen_output("Function 'current_function' not found in the code.")

        current_function = local_vars['current_function']
        try:
            current_function_output = current_function(test_input)
            current_function_output = list_to_string(current_function_output)
        except Exception as e:
            return gen_output("Error during function execution: {e}")

        return gen_output(current_function_output)

######################################################### Setup #########################################################
# The main search function, taking the command line arguments
def search(args):
    # Set the file path to the save directory and the expression name
    file_path = os.path.join(args.save_dir, f"{args.folder}_run_archive.json")
    
    # If the run archive already exists and is valid, load it
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            start = archive[-1]['generation']
        else:
            start = 0
    else:
        archive = get_init_archive()
        start = 0

    # For each solution in the archive, evaluate it
    for solution in archive:
        # Skip if fitness already calculated
        if 'fitness' in solution:
            continue

        # For the current solution, set the generation to initial
        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        # Grab the code from the solution, run it through evaluate forward, stick it in accuracy list
        try:
            acc_list = run_and_score(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue
        
        # Calculate the fitness string from the accuracy list and the function grabbed from utils.py
        fitness_str = bootstrap_confidence_interval(acc_list)
        solution['fitness'] = fitness_str

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    # For N generations, chosen by command line, reflect on on the result of the last attempt and try again
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        # Grab the prompt from the archive, package it for the LLM
        system_prompt, prompt = get_prompt(archive)
        # Build the chat history to build off of
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        # Try to get two responses from the LLM, the first has it identify issues with the solution, the second has it try again
        try:
            next_solution = api_call_followup(msg_list, args.model)

            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            # Reflexion 1 "Where did we go wrong?"
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = api_call_followup(msg_list, args.model)
            # Reflexion 2 "With this in mind, try again"
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = api_call_followup(msg_list, args.model)
        except Exception as e:
            print("Error while LLM was generating a new solution:")
            print(e)
            continue

        acc_list = []
        # Underscore means we dont use the variable value, we just want to loop debug_max times
        for _ in range(args.debug_max):
            # Try to evaluate the solution, if it fails, try again
            try:
                acc_list = run_and_score(args, next_solution["code"])
                # If the accuracy list is all 0, raise an exception
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE:
                    raise Exception("All 0 accuracy")
                break
            except Exception as e:
                print("During evaluation:")
                print(e)
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = api_call_followup(msg_list, args.model)
                except Exception as e:
                    print("Error while LLM was generating a new solution:")
                    print(e)
                    continue
                continue

        # If there is no accuracy list after the loop, exit the for loop
        if not acc_list:
            continue

        # Calculate the fitness string from the accuracy list and the function grabbed from utils.py, prepare to go to the next solution
        fitness_str = bootstrap_confidence_interval(acc_list)
        next_solution['fitness'] = fitness_str
        next_solution['generation'] = n + 1

        # Clear out the thought and debug fields and append the next solution to the archive
        if 'debug_thought' in next_solution:
            del next_solution['debug_thought']
        if 'reflection' in next_solution:
            del next_solution['reflection']
        archive.append(next_solution)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

# The evaluate functions, grades without making new solutions
def evaluate(args):
    # load archive
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)
    eval_archive = []
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as json_file:
            eval_archive = json.load(json_file)

    # Run the current solutions through the evaluation function and grade fitness
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
            acc_list = run_and_score(args, sol["code"])
        except Exception as e:
            print(e)
            continue
        fitness_str = bootstrap_confidence_interval(acc_list)
        sol['test_fitness'] = fitness_str
        eval_archive.append(sol)

        # save results
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)

        current_idx += 1

# Evaluate the fitness of the given solution for the given task, this one is specialized to ARC, needs generalized
def run_and_score(args, forward_str):
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    
    namespace = {}
    # Try the solution with global variables and the solutions namespace
    exec(forward_str, globals(), namespace)
    # If there is more than one function in the namespace, raise an exception
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    # Grab the function from the namespace
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # Choose the folder to load the data from
    if SEARCHING_MODE:
        arc_dir = args.val_data_path
    else:
        arc_dir = args.test_data_path
    print(arc_dir)
    with open(arc_dir, 'rb') as pickle_file:
        data_queue = pickle.load(pickle_file)
    
    # The data queue is grabbed from the pickle, the number of iterations is calculated, and the max workers is set 
    print(f"Number of iterations to do: {len(data_queue) * args.n_repreat}")
    max_workers = min(len(data_queue) * args.n_repreat, args.max_workers) if args.multiprocessing else 1

    agent_task_queue = []
    # For each data in the queue, format and add it to the agent task queue
    for data in data_queue:
        task_str, examples, test_input = format_data(data)
        taskInfo = Info('task', 'User', task_str, -1)
        agent_task_queue.extend([(AgentSystem(examples, test_input), taskInfo, data)] * args.n_repreat)

    # For the agent task, call the forward function and evaluate the solution
    def call_forward(agent_task_queue):
        agent, taskInfo, data = agent_task_queue
        res = agent.forward(taskInfo) # Response
        try:
            # If the response is an Info tuple, unpack it
            if isinstance(res, Info):
                res = res.content
            # If the response is a string, evaluate it
            if isinstance(res, str):
                res = eval(res)
            # Evaluate the response and get a score
            hard_score = eval_solution(res, data, soft_eval=False)
            return hard_score
        except Exception as e:
            print(e)
            return 0

    # For each agent task in the queue, call_forward
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        acc_list = list(tqdm(executor.map(call_forward, agent_task_queue), total=len(agent_task_queue)))
    
    # Return the accuracy list
    print("acc:", bootstrap_confidence_interval(acc_list))
    return acc_list

# If this file is called directly from the command line, parse the arguments and run the search and evaluate functions
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='_arc')
    parser.add_argument('--val_data_path', type=str, default='sampled_arc_val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='sampled_arc_test_data.pkl')
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--expr_name', type=str, default='arc_llama3.1_results')
    parser.add_argument('--n_generation', type=int, default=25)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--model',
                        type=str,
                        default='llama3.1',
                        choices=['mistral-nemo', 'gemma2', 'llama3.1'])

    args = parser.parse_args()
    # Create new solutions
    SEARCHING_MODE = True
    search(args)

    # Test existing solutions
    SEARCHING_MODE = False
    evaluate(args)
