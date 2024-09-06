import json

from example_data import example_list
from base_code import utility_code

# Gets inserted into base prompt where [EXAMPLE] is
EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return answer
"""
}

# Have the LLM list its chain of thought, and then output the final answer
COT_code = {
    "thought": "Directly formatting the output can be challenging. A good practice is to allow the LLM to write the code and then evaluate it to generate the output. This ensures that the output is derived from executable code, improving reliability.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Instruction for the Chain-of-Thought (CoT) approach with code generation
    cot_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instantiate a new LLM agent specifically for CoT with code output
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    cot_agent = LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent')
    
    # Get the CoT agent's response, which includes both thinking steps and code
    thinking, code = cot_agent([taskInfo], cot_instruction)
    
    # Evaluate the generated code to get the output
    answer = self.get_test_output_from_code(code)
    
    # Return the final output derived from the code execution
    return answer
    """
}

# Attempt 5 high temperature solutions, combining passing solutions
COT_split_brain = {
    "thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling. Note that we need to collect only the ones that pass the examples, preventing the context length from becoming too long.",
    "name": "Self-Consistency with Chain-of-Thought",
    "code": """def forward(self, taskInfo):
    # Instruction for step-by-step reasoning and code generation
    cot_instruction = "Please think step by step and then solve the task by writing the code."
    N = 5  # Number of CoT agents
    
    # Initialize multiple CoT agents with a higher temperature for varied reasoning
    cot_agents = [LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent', temperature=0.7) for _ in range(N)]

    # Instruction for final decision-making based on collected reasoning and answers
    final_decision_instruction = "Given all the above solutions, reason over them carefully and provide a final answer by writing the code."
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)
    
    possible_answers = []
    
    # Collect reasoning and answers from each CoT agent
    for i in range(N):
        thinking, code = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.extend([thinking, code])
    
    # Make a final decision based on all collected reasoning and answers
    thinking, code = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    answer = self.get_test_output_from_code(code)
    
    return answer
    """
}

# Give the LLM a loop and a feedback function it can call, tries again 3 times
self_reflection = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. After each answer, testing on the examples to provide feedback, and the LLM uses insights from previous attempts and feedback to refine its answer. It is very good practice to use `self.run_examples_and_get_feedback` to get feedback. One should consider trying to use this feedback in future agent design.",
    "name": "Self-Refine (self_reflection)",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning and code generation
    cot_initial_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you went wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    
    # Instantiate a Chain-of-Thought (CoT) agent
    cot_agent = LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent')
    
    N_max = 3  # Maximum number of attempts
    
    # Initial attempt
    thinking, code = cot_agent([taskInfo], cot_initial_instruction, 0)
    
    # Iteratively refine the answer based on feedback
    for i in range(N_max):
        # Get feedback by testing the code on examples
        feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)  
        
        # Add feedback to the inputs for the next iteration
        attempt = [thinking, code, feedback]

        # Reflect on previous attempts and refine the answer
        # Only consider the latest attempts to control context length. You can try to increase the N_max.
        # The input to LLMAgentBase should be a list of Info.
        thinking, code = cot_agent([taskInfo] + attempt, cot_reflect_instruction, i + 1)  

    # Get the final answer after refinement
    answer = self.get_test_output_from_code(code)
    return answer
    """
}

# Use the LLM to generate multiple solutionsm, have them debate 2 rounds, then make a final decision
LLM_debate = {
    "thought": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.",
    "name": "LLM Debate",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning and code generation
    debate_initial_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instruction for debating and updating the solution based on other agents' solutions
    debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer by writing the code."
    
    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    debate_agents = [LLMAgentBase(['thinking', 'code'], 'Debate Agent', temperature=0.6, role=role) for role in ['Puzzle Game Designer', 'Expert Logician']]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer by writing the code."
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)

    max_round = 2  # Maximum number of debate rounds
    all_results = [[] for _ in range(max_round)]
    
    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, code = debate_agents[i]([taskInfo], debate_initial_instruction)
                answer = self.get_test_output_from_code(code)
            else:
                input_infos = [taskInfo] + all_results[r-1]
                thinking, code = debate_agents[i](input_infos, debate_instruction)
                answer = self.get_test_output_from_code(code)
            all_results[r].extend([thinking, answer])
    
    # Make the final decision based on all debate results and solutions
    thinking, code = final_decision_agent([taskInfo] + all_results[max_round-1], final_decision_instruction)
    answer = self.get_test_output_from_code(code)
    return answer
    """
}

# Have the LLM generate a solution, then another trying to be different from previous, so on so forth 3 times, grab 2 best solutions based on fitness, then let agents debate and pick between them
Outside_the_box = {
    "thought": "Similar to Quality-Diversity methods, allowing the LLM to generate multiple diverse and interesting solutions could be beneficial.",
    "name": "Quality-Diversity",
    "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning and code generation
    cot_initial_instruction = "Please think step by step and then solve the task by writing the code."
    
    # Instruction for generating another interesting way to solve the task based on previous attempts
    cot_QD_instruction = "Given previous attempts, try to come up with another interesting way to solve the task by writing the code."
    
    # Initialize the Chain-of-Thought (CoT) agent
    cot_agent = LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent')

    # Instruction for final decision-making based on all solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer by writing the code."
    final_decision_agent = LLMAgentBase(['thinking', 'code'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3  # Maximum number of attempts
    Outside_the_box_inputs = [taskInfo]  # Initialize inputs with the task information

    possible_answers = []
    
    # Generate multiple diverse solutions
    # Different from generating multiple answers through repeated questioning, we generate interestingly new solutions based on previous attempts
    for i in range(N_max):
        # Generate a solution based on the instruction (initial or Outside_the_box)
        # Also control the context length.
        thinking, code = cot_agent(qd_inputs[-3:], cot_initial_instruction if i == 0 else cot_QD_instruction, i)
        # Get feedback by testing the code on examples
        feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
        # Add the solution to inputs for the next iteration
        Outside_the_box_inputs.extend([thinking, code, feedback])  
        # Collect all possible answers
        possible_answers.append({
            'thinking': thinking,
            'code': code,
            'feedback': feedback,
            'correct_count': len(correct_examples)
        })

    # Sort the possible answers based on the number of correct examples in descending order
    sorted_answers = sorted(possible_answers, key=lambda x: x['correct_count'], reverse=True)
    
    # Select the top solutions (e.g., top 2 solutions)
    top_solutions = sorted_answers[:2]

    # Prepare inputs for the final decision agent
    final_inputs = [taskInfo] + [item for solution in top_solutions for item in [solution['thinking'], solution['code'], solution['feedback']]]

    # Make the final decision based on all solutions
    thinking, code = final_decision_agent(final_inputs, final_decision_instruction)
    answer = self.get_test_output_from_code(code)
    return answer
    """
}

# The default system prompt, usually gets overwritten
system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

# The prompt for the agent builder itself
base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the given benchmark. 

The end result is a prompt library that contains the code and instructions for a given task or subtask to be completed by an LLM agent. The entries should have a descriptive name that allows the LLM to understand the role of the agent and the task it is expected to perform. The code should be complete and ready to be used in the final system.

# An example task from the benchmark:

## Task Overview:
You will be given some number of paired example inputs and outputs. The outputs were produced by executing code with the inputs. There will be an input without a paired output. 

Your task is to generate llm agents that can produce code that can generate the output for the given input. The code should be able to generate the output for the test input as well.

The code only needs to be unambiguous and applicable to the example inputs and the test input. It doesn't need to work for all possible inputs. Observe the examples carefully, imagine the execution flow visually, and try to find the pattern.

"""

for i, example in enumerate(example_list):
    input_value = example[0]
    output_value = example[1]
    base += f"### Example {i+1}:\ninput = {input_value}\noutput = {output_value}\n\n"

base +="""

Analyze the code flow based on the provided Examples and determine what the output should be for the unpaired input.

"""

base += f"## Utility Code:\n{utility_code}\n\n"

base +="""

# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".
The "generation" number indicates the sequential order of attempts made in designing the architecture. Each generation represents a distinct iteration or version, reflecting the evolution and refinement of the design.

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture. 
Finally, the last key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. DON'T try to use some function that doesn't exisit. In forward(), you need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. 
Also, it might be helpful to set the LLM’s role and temperature to further control the LLM’s response. It is also helpful to allow chain-of-thought thinking in addition to your required output fields. Note that the LLMAgentBase() will automatically parse the output and return a list of “Info”. And when you query LLMAgentBase(), it takes in a list of "Info". DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.
In this domain, because you have been given training examples, you could choose to test your proposed solution against those training examples to see if it is correct. One example way to use this domain’s API is:
```
possible_answers = []
# got some new answers
thinking, code = ...
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
# collect possible answers
possible_answers.append({
    'thinking': thinking,
    'code': code,
    'feedback': feedback,
    'correct_count': len(correct_examples)
})
# Sort the possible answers based on the number of correct examples in descending order
sorted_answers = sorted(possible_answers, key=lambda x: x['correct_count'], reverse=True)

# Select the top solutions (e.g., top 3 solutions)
top_solutions = sorted_answers[:3]

# And then use the top_solutions anyway you want.
# One example is to use it for final decision
final_inputs = [taskInfo] + [item for solution in top_solutions for item in [solution['thinking'], solution['code'], solution['feedback']]]
# Make a final decision based on the top solutions
thinking, code = final_decision_agent(final_inputs, final_decision_instruction)
```

## WRONG Implementation examples:
Here are some mistakes you may make:
1. This is WRONG: ```
thinking, code = code_generator_agent([taskInfo] + rules, code_generation_instruction)
feedback_info = verifier_agent([taskInfo, Info('thinking', 'Code Generator Agent', thinking, 0), Info('code', 'Code Generator Agent', code, 0)], verification_instruction)
```
First, it is wrong to use "Info('thinking', 'Code Generator Agent', thinking, 0)". The returned "thinking" and "code" is already Info.
Second, directly let a agent provide feedback to a generated code is not a good practice. You could use `self.run_examples_and_get_feedback(code)` to get feedback first, as it will test the code on examples. And then you could use this feedback anyway you like, including providing it to another critic agent.

2. This is WRONG: ```
# Debugging: Log the generated code
print('Generated Code:', dynamic_memory['code'])
feedback_info = verifier_agent([taskInfo, Info('thinking', 'Code Generator Agent', thinking, 0), Info('code', 'Code Generator Agent', code, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
First, the len(feedback_info) will not work. The only way to check how many examples are passed in the generated code is to get feedback from `self.run_examples_and_get_feedback` and check the correct_count.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.

3. This is WRONG: ```
# Verify the transformation rules using the feedback mechanism
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(rules)
```
`self.run_examples_and_get_feedback()` can only take in CODE input.

5. This is WRONG: ```
for thinking, code in module_solutions:
    # Implement actual integration logic
    integrated_solution.append(code)  # Placeholder
```
You should not put any unfinished code or placeholder in your code.

6. This is WRONG: ```
answer = code
return answer
```
You should always use self.get_test_output_from_code() to run the code and return the output answer.

7. This is Wrong: ```
queried_solutions = []
query_infos = active_query_agent(input_infos, active_query_instruction)
query_thinking, query_code = query_infos[0], query_infos[1]
# THIS IS VERY WRONG! YOU DON'T NEED TO MAKE INFO YOURSELF IN MOST CASES. JUST CONCATENATE THE PREVIOUS OUTPUTS!
queried_solutions.append(Info('queried_solution', 'Active Query Agent', {'thinking': query_thinking.content, 'code': query_code.content}, iteration_idx=solution.iteration_idx))
```
Correct way will be: `queried_solutions.extend([query_thinking, query_code])`

8. This is Wrong: ```# Generate sub-goals using the manager agent
manager_agent = LLMAgentBase(['thinking', 'sub_goals'], 'Manager Agent')
thinking, sub_goals = manager_agent([taskInfo], manager_instruction)
sub_goals_content = sub_goals.content[:len(actor_agents)]
```
Info.content is a string. You can not slice them. What you can do is to require more output fields like `LLMAgentBase(['thinking', 'subgoal1', 'subgoal2', ...], 'Manager Agent')`.
If you want to give multiple infos into other agents, please use "queried_solutions + [query_thinking, query_code]".

9. This is Wrong: ```# Collect solutions that pass at least one example
possible_answers = []
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
if len(correct_examples) > 0:  
    possible_answers.extend([thinking, code, feedback])

10. This is NOT recommended: ```
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
if len(correct_examples) == len(self.examples):
    break  # If all examples are correct, break the loop
```
Even if all examples are correct, it doesn't guarantee that the code is correct. Don't break the loop in your agent.

# Sort the possible answers based on the number of correct examples in descending order
sorted_answers = sorted(possible_answers, key=lambda x: x.content['correct_count'], reverse=True)
```
You can not sort in this way. The correct method is: ```# Collect solutions that pass at least one example
feedback, correct_examples, wrong_examples = self.run_examples_and_get_feedback(code)
if len(correct_examples) > 0:  
    possible_answers.append({
        'code': code,
        'feedback': feedback,
        'correct_count': len(correct_examples)
    })
sorted_answers = sorted(possible_answers, key=lambda x: x['correct_count'], reverse=True)
```

DON'T make those mistakes.

# Your task
You are deeply familiar with prompting techniques and the agent works from the literature. Your goal is to maximize the specified performance metrics by proposing interestingly new agents.
Observe the discovered agents carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative when thinking about the next interesting agent to try. You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
Use the knowledge from the archive and inspiration from academic literature to propose the next interesting agentic system design.
THINK OUTSIDE THE BOX.
"""

# Compare current solution to existing, reflect on interestingness, implementation mistakes, and suggest improvements
review_and_correct = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

# Make another attempt to improve the code
correct_using_examples = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""

# Assemble placeholder replacements into a prompt to send to the LLM
def get_prompt(current_archive):
    archive_str = ",\n".join([json.dumps(solution) for solution in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))

    return system_prompt, prompt

# Initialize a new archive
def create_new_archive():
    return [COT_code, self_reflection, LLM_debate, COT_split_brain, Outside_the_box]

# Build a prompt to reflect on and pass it to the caller
def get_self_reflection_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = review_and_correct.replace("[EXAMPLE]", prev_example_str) if prev_example else review_and_correct.replace("[EXAMPLE]", "")
    return r1, correct_using_examples
