

utility_code = [{
"""
```python
from collections import namedtuple, Union
import numpy as np
import json
import random
import string
import openai
import backoff

client = openai.OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama',
)

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

def random_id(length=4):
    characters = string.ascii_letters + string.digits
    random_id = ''.join(random.choices(characters, k=length))
    return random_id

def list_to_string(list_2d):
    sublists_as_strings = [f"[{','.join(map(str, sublist))}]" for sublist in list_2d]
    return f"[{','.join(sublists_as_strings)}]"

# Format instructions for LLM response
def FORMAT_INST(request_keys):
    return f\"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n\"""

# Role description for the LLM
def ROLE_DESC(role):
    return f"You are a {role}."

# Instruction for the code function
CODE_INST = \"""You should write a function called 'current_function' which takes a single input, and returns the single output. You should ensure that you implement a version of the code that works for both example and test inputs.\"""

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def api_call_initial(msg, model, system_message, temperature=0.5):
    \"""
    Function to get JSON response from GPT model.

    cmd_line_args:
    - msg (str): The user message.
    - model (str): The model to use.
    - system_message (str): The system message.
    - temperature (float): Sampling temperature.

    Returns:
    - dict: The JSON response.
    \"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message['content']
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    \"""
    Base class for an LLM agent.

    Attributes:
    - output_fields (list): Fields expected in the output.
    - agent_name (str): Name of the agent.
    - role (str): Role description for the agent.
    - model (str): Model to be used.
    - temperature (float): Sampling temperature.
    - id (str): Unique identifier for the agent instance.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gemma2', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        An example of a generated prompt:
        You are a helpful assistant.
        
        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will be given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # Instruction: 
        Please think step by step and then solve the task by writing the code.

        \"""
        code_output = False

        # construct system prompt
        output_fields_and_description = {key: f"Your {key}." for key in self.output_fields}
        for key in output_fields_and_description:
            if 'answer' in key:
                output_fields_and_description[key] = f"Your {key}. ONLY return a string of list[list[int]]. DO NOT return anything else."
            elif 'code' in key:
                output_fields_and_description[key] = f"Your {key}. Don't write tests in your Python code, ONLY return the `current_function` function. DO NOT return anything else. (It will be tested later.)"
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
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + "# Instruction: \n" + instruction + "\n\n" + (CODE_INST if code_output else '')
        return system_prompt, prompt 

    def query(self, input_infos: list[Info], instruction: str, iteration_idx=-1) -> list[Info]:
        \"""
        Queries the LLM with provided input information and instruction.

        cmd_line_args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.

        Returns:
        - output_infos (list[Info]): Output information.
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = api_call_initial(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    def __call__(self, input_infos: list[Info], instruction: str, iteration_idx=-1) -> list[Info]:
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    def __init__(self, examples: list[dict], test_input: list[list[int]]) -> None:
        \"""
        Initializes the AgentArchitecture with examples and a test input.
        
        cmd_line_args:
            examples (List[Dict[str, List[List[int]]]]): A list of dictionaries, where each dictionary contains an 'input' and 'output'.
                - 'input' (List[List[int]]): A 2D list representing the input grid.
                - 'output' (List[List[int]]): A 2D list representing the expected output grid for the corresponding input.
            test_input (List[List[int]]): The input grid for the test problem, which is a 2D list. The agent's task is to determine the correct output for this input.
        
        Note:
            You are free to use these data in any way that aids in solving the task.
        \"""
        self.examples = examples
        self.test_input = test_input
    
    def run_examples_and_get_feedback(self, code):
        \"""
        Runs provided code on examples and gets feedback. This is very useful to provide feedback to the generated current_function code.

        cmd_line_args:
        - code (Info/str): The CODE to evaluate.

        Returns:
        - Info: Feedback on the code whether it works on examples or not. The content is a string summarizing the success and failure on examples.
        - list(list[dict[str, list[list[int]]]]): list of Correct Examples
        - list(list[dict[str, list[list[int]]]]): list of Wrong Examples
            - keys for both correct and wrong example dict:
            - 'input' (list[list[int]]): A 2D list representing the input grid.
            - 'output' (list[list[int]]): A 2D list representing the expected output grid for the corresponding input.

        An example of feedback Info content:
        "Your current_function generates a WRONG answer in Example 0!
        Expect: xxx
        You got: yyy
        Observe the Example 0 carefully!

        Your current_function generates a CORRECT answer in Example 1!
        ..."
        \"""
        examples = self.examples

        #... (code to run the provided code on examples and get feedback)
            
        # return feedback, correct_examples, wrong_examples

    def get_test_output_from_code(self, code):
        \"""
        Gets the output from the code on the test input.

        cmd_line_args:
        - code (Info/str): The code to evaluate.

        Returns:
        - Info: Output on the test input with the provided code, which is the answer to the task.
        \"""
        test_input = self.test_input

        #... (code to run the provided code on the test input and get the output)
            
        # return current_function_output

    \"""
    Fill in your code here.
    \"""
    def forward(self, taskInfo) -> Union[Info, str, list[list[int]]]:
        \"""
        Placeholder method for processing task information.

        cmd_line_args:
        - taskInfo (Info): Task information.

        Returns:
        - Answer (Union[Info, str, list[list[int]]]): Your FINAL answer. Return either a named tuple Info or a string of answer or a list[list[int]].
        \"""
        pass


#... Code to implement the evaluation of the agent's performance
```
"""
}]