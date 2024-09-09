import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from agent import AgentSystem, format_data, eval_solution, Info
import numpy as np

# Evaluate the fitness of the given solution for the given task
def run_and_score(cmd_line_args, code_being_judged):
    score_list=run_code(cmd_line_args,code_being_judged)
    return score_list

def run_code(cmd_line_args,code_being_judged):
    namespace = {}

    exec(code_being_judged, globals(), namespace)

    function_being_run = get_function_from_namespace(namespace)

    setattr(AgentSystem, "function_to_run", function_being_run)

    data_file = cmd_line_args.data_path
    data_queue = load_data_from_pickle(data_file)

    max_workers = calculate_max_workers(cmd_line_args, data_queue)
    agent_task_queue = create_agent_task_queue(cmd_line_args, data_queue)
    score_list = run_task_threads(agent_task_queue, max_workers)

    return score_list

def get_function_from_namespace(namespace):   
    function_list = list(namespace.keys())
    if len(function_list) != 1:
        raise AssertionError(f"{len(function_list)} functions in namespace. Please only provide 1")
    function_being_run = namespace[function_list[0]]
    if not callable(function_being_run):
        raise AssertionError(f"{function_being_run} is not callable")
    return function_being_run

def load_data_from_pickle(data_file):
    with open(data_file, 'rb') as pickle_file:
        data_queue = pickle.load(pickle_file)
    return data_queue

def calculate_max_workers(cmd_line_args, data_queue):
    max_workers = min(len(data_queue) * cmd_line_args.n_repreat, cmd_line_args.max_workers) if cmd_line_args.multiprocessing else 1
    return max_workers

def create_agent_task_queue(cmd_line_args, data_queue):
    agent_task_queue = []
    for data in data_queue:
        task_str, examples, test_input = format_data(data)
        taskInfo = Info('task', 'User', task_str, -1)
        agent_task_queue.extend([(AgentSystem(examples, test_input), taskInfo, data)] * cmd_line_args.n_repreat)
    return agent_task_queue

def run_task_threads(agent_task_queue, max_workers):
    def run_function(agent_task_queue):
        agent, taskInfo, data = agent_task_queue
        function_response = agent.function_to_run(taskInfo)
        try:
            if isinstance(function_response, Info):
                function_response = function_response.content
            if isinstance(function_response, str):
                function_response = eval(function_response)
            hard_score = eval_solution(function_response, data, soft_eval=False)
            return hard_score
        except Exception as e:
            print(e)
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        score_list = list(tqdm(executor.map(run_function, agent_task_queue), total=len(agent_task_queue)))
    return score_list

def calculate_fitness(score_list, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the confidence interval of the score list using bootstrap confidence intervals.
    Also returns the median of the bootstrap means.
    
    cmd_line_args:
    - score_list: 1D list of score floats.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    score_list = np.array(score_list)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(score_list, size=len(score_list), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
