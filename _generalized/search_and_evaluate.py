import os
import json
import numpy as np

from system_prompt_library import get_init_archive, get_prompt, get_reflexion_prompt
from data_tools import format_data, eval_solution, calculate_fitness, random_id, list_to_string
import run_and_score
from api import api_call_initial, api_call_followup

# TODO:
# - Set search functionality to extend existing agents to desired number of generations, i.e. agent 7 has 1 generation, we want 5 for each, it goes back and makes more
#

######################################################### Setup #########################################################
def search_and_evaluate(args):
    archive = []
    
    # Use the input arguments to find where the archive would be saved
    file_path = os.path.join(args.save_dir, f"{args.folder}_run_archive.json")
    
    # If the archive exists, load it
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
    
        if args.mode == "search":
            if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
                current_agents_generation = archive[-1]['generation']
            else:
                current_agents_generation = 0
    
    # If the archive doesn't exist and you want to make new agents, intitialize an archive
    elif args.mode == "search":
        archive = get_init_archive()

    # For each existing agent in the archive, evaluate it
    for solution in archive:
        # Skip if fitness already calculated
        if 'fitness' in solution:
            continue

        # Calculate the fitness of the exisiting agent
        solution['generation'] = "initial"
        print(f"============Initial Archive: {solution['name']}=================")
        # Grab the code from the solution, run it through evaluate forward, stick it in score list
        try:
            score_list = run_and_score(args, solution["code"])
        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue
        
        # Calculate the fitness from the score list
        fitness = calculate_fitness(score_list)
        solution['fitness'] = fitness

        # save results
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)
    
    # Picking up on the last entry in the archive, create new generations, evaluate them, and add them to the archive
    if args.mode == "search":
        for n in range(current_agents_generation, args.n_generation):
            print(f"============Generation {n + 1}=================")
            # Grab the prompt from the archive
            system_prompt, prompt = get_prompt(archive)

            # assemble prompts into json
            msg_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            # Try to get two responses from the LLM, the first has it attempt a solution, the second has it try again
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

            score_list = []
            # Underscore means we dont use the variable value, we just want to loop debug_max times
            for _ in range(args.debug_max):
                # Try to evaluate the solution, if it fails, try again
                try:
                    score_list = run_and_score(args, next_solution["code"])
                    # If the accuracy list is all 0, raise an exception
                    if np.mean(score_list) < 0.01 and args.mode == "search":
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

            # If there is no accuracy list after the loop [everything errored out], exit the for n loop
            if not score_list:
                continue

            # Calculate the fitness string from the accuracy list and the function grabbed from utils.py, prepare to go to the next solution
            fitness_str = calculate_fitness(score_list)
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
