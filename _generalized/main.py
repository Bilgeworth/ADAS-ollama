import argparse
from search_and_evaluate import generate_agents, score_agents

# If this file is called directly from the command line, parse the arguments and run the search and evaluate functions
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='search', choices=['search', 'evaluate'])

    parser.add_argument('--model',
                        type=str,
                        default='llama3.1',
                        choices=['mistral-nemo', 'gemma2', 'llama3.1'])

    parser.add_argument('--folder', type=str, default='_example')
    parser.add_argument('--archive_name', type=str, default={parser.folder}+{parser.model}+'_results')
    parser.add_argument('--data_path', type=str, default={parser.folder}+'data.pkl')

    parser.add_argument('--n_generation', type=int, default=25)
    parser.add_argument('--n_repreat', type=int, default=5)
    parser.add_argument('--reflect_max', type=int, default=3)
    parser.add_argument('--debug_max', type=int, default=3)

    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=32)

    parser.add_argument('--save_dir', type=str, default={parser.folder}+'results/')

    cmd_line_args = parser.parse_args()

    generating_new_agents = False
    
    if cmd_line_args.mode == 'search':
        generating_new_agents = True   
        generate_agents(cmd_line_args)

    score_agents(cmd_line_args)