# Introduction
These are the soource codes used for in-context generating tabular data given a benchmark dataset.


# Parameters
The prompts are given  
--seed: the random seed for reproducing the results  
--sim_seed: the random seed used for finding the benchmark datasets, of which the value is from 100 to 109.  
--cm: the name of the causal mechanism, e.g., 'lu' is for linear uniform distributions. Used together with sim_seed for finding the benchmark dataset.  
--bt: how many times the llm is called for generating text.  
--max_table_rows: the number of rows of the benchmark data which are used as in-context learning.  
--max_new_tokens: the number of token for text generation.  
--prow_num: the number of rows that llms are required for generating tabular data.  
--llm: the name of llms.  

e.g., 
```python Qwen/Qwen2-7B-Instruct/tab_eval.py --llm qwen --sim_seed 103 --max_table_rows 100 --bt 10 --max_new_tokens 10000 --prow_num 100 ```
