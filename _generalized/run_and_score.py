
import pickle
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

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
    if args.mode == 'search':
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
        score_list = list(tqdm(executor.map(call_forward, agent_task_queue), total=len(agent_task_queue)))
    
    # Return the accuracy list
    print("acc:", calculate_fitness(score_list))
    return score_list