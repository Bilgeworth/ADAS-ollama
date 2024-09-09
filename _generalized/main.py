import argparse
from score_agents import score_agents 
from generate_agents import generate_agents

# If this file is called directly from the command line, parse the arguments and run the search and evaluate functions
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='search', choices=['search', 'evaluate'])

    parser.add_argument('--model',
                        type=str,
                        default='gemma2',
                        choices=['mistral-nemo', 'gemma2', 'llama3.1'])

    parser.add_argument('--folder', type=str, default='_generalized')

    parser.add_argument('--archive_name', type=str)
    parser.add_argument('--data_path', type=str)

    parser.add_argument('--n_generation', type=int, default=25)
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)

    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)

    parser.add_argument('--save_dir', type=str)

    cmd_line_args = parser.parse_args()

    # Set the dependent defaults if not provided
    if not cmd_line_args.archive_name:
        cmd_line_args.archive_name = f"{cmd_line_args.folder}/{cmd_line_args.model}_results"
    if not cmd_line_args.data_path:
        cmd_line_args.data_path = f"{cmd_line_args.folder}/benchmark_data/data.pkl"
    if not cmd_line_args.save_dir:
        cmd_line_args.save_dir = f"{cmd_line_args.folder}/results/"

    generating_new_agents = False
    
    if cmd_line_args.mode == 'search':
        generating_new_agents = True   
        generate_agents(cmd_line_args)

    score_agents(cmd_line_args)