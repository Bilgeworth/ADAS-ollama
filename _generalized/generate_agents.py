import os
import json

from prompt_library import get_prompt, get_self_reflection_prompt
from prompt_library import default_agents
from api import api_call_followup

#   - Loads the archive file where progress is saved for the given arguments.
#   - If the archive exists, load it; otherwise, initialize a new archive.
#   - If the archive has entries, find the last generation of agents.
#   - If the current generation is less than the desired number of generations:
#     - Generate a new prompt from the archive.
#     - Use the prompt to make API calls to generate responses from the LLM (Language Model).
#     - Perform reflections on the generated responses to identify mistakes and attempt to fix them.
#     - Repeat this process for the desired number of generations.

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
        archive = default_agents
    
    # If the archive has entries find the last generation of agents
    if archive and "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
        current_agents_generation = archive[-1]['generation']
    else:
        current_agents_generation = 0
    
    if current_agents_generation < cmd_line_args.n_generation:
        for generation_index in range(current_agents_generation, cmd_line_args.n_generation):
            agent_name = archive[-1].get('name', 'Unknown Agent')
            print(f"============Generation {generation_index}: {agent_name}=================")
            # Grab the prompt from the archive
            system_prompt, prompt = get_prompt(archive)

            # assemble prompts into json
            chat_log = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            # Try to get three responses from the LLM, the first has it attempt a solution, the second has it list mistakes, third attempts to fix mistakes
            try:
                print("Sending API call with chat log:", chat_log)
                llm_latest_response = api_call_followup(chat_log, cmd_line_args.model)             
                print("Received response:", llm_latest_response)

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