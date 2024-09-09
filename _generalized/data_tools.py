import concurrent.futures
import numpy as np

TASK_OVERVIEW = """You will be given some number of paired example inputs and outputs. The outputs were produced by applying code to the inputs. In addition to the paired example inputs and outputs, there is also one test input without a known output. Your task is to determine the code applied to the examples to create the answer.

The code only needs to be unambiguous and applicable to the example inputs and the test input. It doesn't need to work for all possible inputs. Observe the examples carefully, and try to find the pattern.
"""

def list_to_string(list_2d):
    sublists_as_strings = [f"[{','.join(map(str, sublist))}]" for sublist in list_2d]
    return f"[{','.join(sublists_as_strings)}]"

def format_data(benchmark_data, direct=False):
    task_str = TASK_OVERVIEW

    task_demo_str = ''
    # Get task demo string
    task_demo_str += '## Examples:\n\n'
    for i, demo in enumerate(benchmark_data['data_library']):
        task_demo_str += f'### Example {i}:\n'
        task_demo_str += f'input = {list_to_string(demo["input"])}\n'
        task_demo_str += f'output = {list_to_string(demo["output"])}\n\n'

    # Get task test string
    task_test_str = ''
    for testcase in benchmark_data['test']:
        task_test_str += '## Test Problem:\n'
        task_test_str += f'Given input:\n {list_to_string(testcase["input"])}\n\n'
        task_test_str += f'Analyze the provided Examples and determine what code should be applied to the Test Problem.'

    task_str += task_demo_str + task_test_str

    return task_str, benchmark_data['data_library'], benchmark_data['test'][0]['input']


def compare_to_expected(expected_output, received_output):
    if not received_output:
        return 0

    score = 0
    if expected_output == received_output:
        score += 1

    return score


def eval_algo(solve_fn, benchmark_data, soft_eval=False):
    # Calculate percentage of test cases done correctly
    testcases = benchmark_data['test']
    scores = []
    for testcase in testcases:
        input = testcase['input']
        output = testcase['output']
        gen_output = None
        # Run solve_fn with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(solve_fn, input)
                try:
                    gen_output = future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    future.cancel()
            except:  # if the function does not work
                continue
        # Check if correct output
        if soft_eval:
            score = compare_to_expected(output, gen_output)
        else:
            score = 1 if output == gen_output else 0
        scores.append(score)
    return np.mean(scores)


def eval_solution(output, benchmark_data, soft_eval=False):
    if not output:
        return 0

    solution = benchmark_data['test'][0]['output']
    if soft_eval:
        score = compare_to_expected(solution, output)
    else:
        score = 1 if output == solution else 0
    return score