<h1 align="center">
  <img src="misc/art_fig.png" width="200" /></a><br>
  <b>Automated Design of Agentic Systems</b><br>
</h1>

<p align="center">
  <a href="https://github.com/ShengranHu/ADAS/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/sematic?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2408.08435"><img src="https://img.shields.io/badge/arXiv-2408.08435-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <a href="https://www.shengranhu.com/ADAS/"><img src="https://img.shields.io/badge/-Website-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
  <a href="https://twitter.com/shengranhu/status/1825555341922480322"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
</p>

In this work, we describe a newly forming research area **A**utomated **D**esign of **A**gentic **S**ystems (**ADAS**), which aims to *automatically create powerful agentic system designs, including inventing novel building blocks and/or combining them in new ways.*


We present a simple yet effective ADAS algorithm named **Meta Agent Search** to demonstrate that **agents can invent novel and powerful agent designs**. In Meta Agent Search, a "meta" agent iteratively *programs* interesting new agents in code based on previous discoveries.


<p align="center">
<img src="misc/algo.png"/></a><br>
</p>

## Setup
```bash
conda create -n adas python=3.11
conda activate adas
pip install -r requirements.txt

# provide your OpenAI API key
export OPENAI_API_KEY="YOUR KEY HERE"
```

## Running Instructions

### Running Meta Agent Search

To run experiments for each domain, navigate to its respective folder. The code in each folder is self-contained. Launch experiments using the `search.py` script located in each domain's folder.

```bash
python {DOMAIN}/search.py
```

Replace `{DOMAIN}` with the specific domain folder name {`_arc`, `_drop`, `_mgsm`, ...} to run the experiment for.

### Customizing Meta Agent Search for New Domains

You can easily adapt the code to search for new domains. To do so, follow these steps:

1. Modify the `evaluate_forward_fn()` function and adjust any necessary formatting prompts (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/search.py#L89)) in the `search.py` file. 

2. Consider adding additional basic functions for the meta agent to utilize during the design process (similar to [this line](https://github.com/ShengranHu/ADAS/blob/main/_arc/search.py#L161)).

3. Update the domain-specific information within the prompts to match the requirements of your new domain (e.g. [this line](https://github.com/ShengranHu/ADAS/blob/main/_mmlu/mmlu_prompt.py#L229)).

4. Run the search and evaluation on your new domain.

### Safety Consideration
> [!WARNING]  
> The code in this repository involves executing untrusted model-generated code. We strongly advise users to be aware of this safety concern. While it is highly unlikely that model-generated code will perform overtly malicious actions in our current settings and with the models we use, such code may still act destructively due to limitations in model capability or alignment. By using this repository, you acknowledge and accept these risks.


## Citing
If you find this project useful, please consider citing:
```
@article{hu2024ADAS,
title={Automated Design of Agentic Systems},
author={Hu, Shengran and Lu, Cong and Clune, Jeff},
journal={arXiv preprint arXiv:2408.08435},
year={2024}
}
```


## Use Case and How to Use This Repository

This repository is designed for the automated design of agentic systems. The primary use case involves defining benchmarks and iteratively generating and optimizing agents to achieve the best performance for a given task.

### Defining a Benchmark

To define a new benchmark, follow these steps:

1. **Create a Benchmark Folder**: Create a new folder in the repository starting with an underscore (`_`). For example, `_new_benchmark`.

2. **Add a `search.py` Script**: Inside the new benchmark folder, add a `search.py` script. This script should define the evaluation function and the search process for the benchmark.

### Running the Search

Once you have defined a benchmark, you can run the search process to generate and optimize agents for that benchmark.

1. **Navigate to the Repository**: Open a terminal and navigate to the root directory of the repository.

2. **Run the Search Script**: Execute the `search.py` script for the benchmark. For example, if your benchmark folder is `_new_benchmark`, run:
   ```bash
   python _new_benchmark/search.py
   ```

### Process Overview

1. **Initialization**: The search script initializes the environment and loads any necessary data or models.

2. **Agent Generation**: The script iteratively generates new agents by creating or modifying code based on the defined prompts and evaluation criteria.

3. **Evaluation**: Each generated agent is evaluated using the specified evaluation function. The performance score is recorded.

4. **Optimization**: The script evolves the prompt setups based on the performance scores, iteratively improving the agents.

5. **Output**: The optimized agents and their performance scores are saved to an output JSON file for the given task.

### Example

Here is an example of how to define and run a benchmark:

1. **Create a Folder**: `_example_benchmark`
2. **Add `search.py`**:
   ```python
   import argparse
   import os
   import pickle
   import logging
   from tqdm import tqdm

   # Configure logging
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

   def evaluate_forward_fn(agent_code):
       try:
           compiled_code = compile(agent_code, '<string>', 'exec')
           exec(compiled_code)
           if 'agent_classify' in locals():
               accuracy = agent_classify()
               logging.info(f"Agent accuracy: {accuracy}")
               return accuracy
           else:
               logging.error("'agent_classify' function not found in generated code.")
               return 0
       except SyntaxError as e:
           logging.error(f"Syntax error in generated code: {e}")
           return 0
       except Exception as e:
           logging.error(f"Error during evaluation: {e}")
           return 0

   def search(args):
       logging.info("Starting search...")
       with open(args.val_data_path, 'rb') as file:
           val_data = pickle.load(file)
       logging.info(f"Loaded validation data: {val_data}")

       results = []

       for i in tqdm(range(args.n_generation), desc="Generating agents"):
           logging.info(f"Generating agent {i+1}/{args.n_generation}...")
           agent_code = "def agent_classify(): return 0.95"  # Example agent code
           score = evaluate_forward_fn(agent_code)
           logging.info(f"Agent {i+1} score: {score}")
           results.append({'agent': i+1, 'score': score})

       # Save results to a file
       results_path = os.path.join(args.save_dir, 'results.json')
       with open(results_path, 'w') as file:
           json.dump(results, file)

       logging.info(f"Search completed. Results saved to {results_path}.")

   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument('--val_data_path', type=str, default='sampled_arc_val_data.pkl')
       parser.add_argument('--n_generation', type=int, default=25)
       parser.add_argument('--model', type=str, default='llama3.1', choices=['mistral-nemo', 'gemma2', 'llama3.1'])
       args = parser.parse_args()

       # Modify save_dir and expr_name to include the model name
       args.save_dir = f"results_{args.model}/"
       args.expr_name = f"arc_results_{args.model}"

       # Create the save directory if it doesn't exist
       os.makedirs(args.save_dir, exist_ok=True)

       # Start the search
       search(args)
   ```

3. **Run the Script**:
   ```bash
   python _example_benchmark/search.py
   ```

By following these steps, you can define new benchmarks and use this repository to automatically generate and optimize agents for various tasks.