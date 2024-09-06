import copy
from collections import namedtuple
import random
import string

from data_tools import format_data, eval_solution, calculate_fitness, random_id, list_to_string
from api import api_call_initial, api_call_followup

######################################################### Agent Setup #########################################################
# Define a named tuple for storing information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instruction for JSON response
ROLE_DESC = lambda role: f"You are a {role}.\n\n"
FORMAT_INST = lambda request_keys: f"""# Output Format:\nReply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a WELL-FORMED JSON object!\n"""

SYSTEM_MSG = ""
CODE_INST = "You will write code to solve this task by creating a function named `current_function`. This function should take a single input, and return the single output. You should make sure that you implement a version of the code that works for both example and test inputs. Make sure that the current_function function is capable of handling both example and test inputs effectively, reflecting the learned rules from the Examples inputs and outputs."

PRINT_LLM_DEBUG = False
generating_new_agents = True


# Generate a random ID
def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


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
            if "maximum context length" in str(e) and generating_new_agents:
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
    def __init__(self, examples, test_input) -> None:
        self.examples = examples
        self.test_input = test_input

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
        test_input = self.test_input

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