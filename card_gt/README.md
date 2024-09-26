# Introduction
These are the codes for evaluating the quality of generated tabular data of LLMs.


# Usage
* Generated tabular data: Given the benchmark dataset with causal graph and tabular data in " data/sim_lu/100", the corresponding generated tabular data should be saved in directory: "synthetic/sim_lu/100/qwen_100i.csv".  
* Evaluation results: the evaluated results will be saved in "results/qwen_100i" according to different LLMs.

# Example
* causal skeleton: python card_gt/eva_skeleton_llm.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105  
* d-sepearation: python card_gt/eva_ci_sets_llm.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105  
* causal direction: python card_gt/eva_bcd_llm.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105  
* causal graphs: python card_gt/eva_cdir_lingam_llm.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105 
* interventional inference: python card_gt/eva_intervention_llm.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105
* counterfactural inference: python card_gt/eva_counterfactural_llm.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105


## Measuring Decision-making ability of LLM: 

### Generating questions and answers for causal inference tasks
Open task_cf_intv.ipynb and run code the block for generating questions and answers for causal inference tasks.

* Generating intervention inference questions and answers 
* Generating counterfactual inference questions and answers

### Applying LLMs to the causal inference tasks
**Intervention inferece**
```python llm_table_graph_inf.py --llm llama  --max_new_tokens 1000 --sim_seed 10  --input_type table --max_table_rows 50 --batch_size 1 ```

**Counterfactual inferece**
```python llm_table_graph_cf.py --llm llama  --max_new_tokens 1000 --sim_seed 10  --input_type table --max_table_rows 50 --batch_size 1 ```

* max_new_tokens: number of maximum generated tokens
* sim_seed: the index of causal graphs
* input_type: input data type for inference, 'table' or 'graph' 
* max_table_rows: the number of rows of input tables
* batch_size: parameter for calling LLMs

### Evaluation of causal inference results
Open task_cf_intv.ipynb and run code the block for evaluating on causal inference tasks.
* Evaluation of intervention inference resutls
* Evaluation of counterfactual inference resutls
