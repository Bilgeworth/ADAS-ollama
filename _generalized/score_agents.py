import os
import json
import numpy as np

from prompt_library import get_prompt
import run_and_score
from api import api_call_followup

#   - Loads the archive file where progress is saved for the given arguments.
#   - For each solution in the archive:
#     - If the last entry in the archive has a generation, start from there; otherwise, start from generation 0.
#     - Run the code in the solution and debug if there are any errors.
#     - If a valid score is received, store the fitness and append the latest solution to the archive.

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
    
