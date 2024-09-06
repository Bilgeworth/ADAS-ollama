import os
import json
import numpy as np

from prompt_library import create_new_archive, get_prompt, get_self_reflection_prompt
from data_tools import calculate_fitness
import run_and_score
from api import api_call_followup

#1. `score_agents` function:
#   - Loads the archive file where progress is saved for the given arguments.
#   - For each solution in the archive:
#     - If the last entry in the archive has a generation, start from there; otherwise, start from generation 0.
#     - Run the code in the solution and debug if there are any errors.
#     - If a valid score is received, store the fitness and append the latest solution to the archive.
#
#2. `generate_agents` function:
#   - Loads the archive file where progress is saved for the given arguments.
#   - If the archive exists, load it; otherwise, initialize a new archive.
#   - If the archive has entries, find the last generation of agents.
#   - If the current generation is less than the desired number of generations:
#     - Generate a new prompt from the archive.
#     - Use the prompt to make API calls to generate responses from the LLM (Language Model).
#     - Perform reflections on the generated responses to identify mistakes and attempt to fix them.
#     - Repeat this process for the desired number of generations.

# Score any existing solutions, debugging if errors, and save the results
def score_agents(cmd_line_args):
    archive = []
    
    # Use the input arguments to find where the archive would be saved
    file_path = os.path.join(cmd_line_args.save_dir, f"{cmd_line_args.folder}_run_archive.json")
    
    # If the archive exists, load it
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)    

    # For each existing agent in the archive, evaluate it
    for solution in archive:
        # Skip if fitness already calculated
        if 'fitness' in solution:
            continue

        score_list = []

        if archive and "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
            current_agents_generation = archive[-1]['generation']
        else:
            current_agents_generation = 0
            solution['generation'] = 0            
        
        print(f"============Generation {current_agents_generation}=================")
        # Grab the code from the solution, run it through evaluate forward, stick it in score list
        try:
            score_list = run_and_score(cmd_line_args, solution["code"])
            if np.mean(score_list) < 0.01:
                raise Exception("No valid scores")
        except Exception as e:
            print("Error running generated code:")
            print(e)
            llm_latest_response = solution["code"]
            system_prompt, prompt = get_prompt(archive)
            # assemble prompts into json
            chat_log = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            for m in range(cmd_line_args.debug_max):
                print(f"Debugging attempt {m + 1} of {cmd_line_args.debug_max}")
                chat_log.append({"role": "assistant", "content": str(llm_latest_response)})
                chat_log.append({"role": "user", "content": f"Error running generated code:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    llm_latest_response = api_call_followup(chat_log, cmd_line_args.model)
                except Exception as e:
                    print("Error with LLM API call or reponse format:")
                    print(e)
                    continue
                

        # If there is no accuracy list after the loop [everything errored out], exit the for lboop
        if not score_list:
            print("No valid scores within debug_max")
        else:
            # Calculate the fitness from the score list
            fitness = calculate_fitness(score_list)
            print("fitness:", calculate_fitness(score_list))
            solution['fitness'] = fitness
            solution['generation'] += 1

            # Clear out the thought and debug fields and append the next solution to the archive
            if 'debug_thought' in llm_latest_response:
                del llm_latest_response['debug_thought']
            if 'reflection' in llm_latest_response:
                del llm_latest_response['reflection']
            archive.append(llm_latest_response)

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)
    
# Picking up on the last entry in the archive, create new generations, evaluate them, and add them to the archive
def generate_agents(cmd_line_args):
    archive = []
    
    # Use the input arguments to find where the archive would be saved
    file_path = os.path.join(cmd_line_args.save_dir, f"{cmd_line_args.folder}_run_archive.json")
    
    # If the archive exists, load it, else initialize a new one
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
    else:
        archive = create_new_archive()
    
    # If the archive has entries find the last generation of agents
    if archive and "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
        current_agents_generation = archive[-1]['generation']
    else:
        current_agents_generation = 0
    
    if current_agents_generation < cmd_line_args.n_generation:
        for generation_index in range(current_agents_generation, cmd_line_args.n_generation):
            print(f"============Generation {generation_index}=================")
            # Grab the prompt from the archive
            system_prompt, prompt = get_prompt(archive)

            # assemble prompts into json
            chat_log = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            # Try to get three responses from the LLM, the first has it attempt a solution, the second has it list mistakes, third attempts to fix mistakes
            try:
                llm_latest_response = api_call_followup(chat_log, cmd_line_args.model)

                review_and_correct, correct_using_examples = get_self_reflection_prompt(archive[-1] if generation_index > 0 else None)
                # self_reflection 1 "Where did we go wrong?"
                chat_log.append({"role": "assistant", "content": str(llm_latest_response)})
                chat_log.append({"role": "user", "content": review_and_correct})
                llm_latest_response = api_call_followup(chat_log, cmd_line_args.model)
                # self_reflection 2 "With this in mind, try again"
                chat_log.append({"role": "assistant", "content": str(llm_latest_response)})
                chat_log.append({"role": "user", "content": correct_using_examples})
                llm_latest_response = api_call_followup(chat_log, cmd_line_args.model)

                if 'debug_thought' in llm_latest_response:
                    del llm_latest_response['debug_thought']
                if 'reflection' in llm_latest_response:
                    del llm_latest_response['reflection']
                archive.append(llm_latest_response)

                # save results
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as json_file:
                    json.dump(archive, json_file, indent=4)
                    
            except Exception as e:
                print("Error with LLM API call or reponse format:")
                print(e)
                continue