# Introduction
This is a repository for paper ["CARD-GT: Rethinking Language Model Capabilities for Causal Reasoning and Decision-Making Using Causal Graphs and Tabular Data"]().
It includes benchmarking tasks from the perspectives of causal graph reasoning, knowledge discovery, and decision-making.

```
├── task_performance_correlation.ipynb  <- correlation analysis of the performance of LLMs on different tasks
├── task_cf_intv.ipynb               <- Generating benchmark data for causal graph reasoning, knowledge discovery, and decision-making tasks,  it also evaluates the results of decision-making task
├── eva_causal_graph_adj_cdir.py        <- Causal graph reasoning: evaluate adjacency matrix or dag estimation with causal  graphs as input
├── eva_causal_graph_dsep.py            <- Causal graph reasoning: evaluate d-separation estimation with causal graphs as input
├── eva_causal_table_dsep.py            <- Knowledge discovery: evaluate conditional independence estimation with tables as input
├── eva_causal_table_cdir.py            <- Knowledge discovery: evaluate causal direction estimation with tables as input
│
│
├── data/              <- Benchmark data
│   │
│   ├── graph/         <- Causal graph reasoning tasks: questions and answers
│   └── table/         <- Knolwedge discovery and decision making: questions and answers
│   
├── results/           <- responses of LLms
│   │
│   ├── llama/         
│   ├── ...
│   └── table/     
│   
├── src/                   <- Source code
│   │
│   ├── llms_graph_adj.py      <- using llms for adjacency matrix estimation with causal graphs as input
│   ├── llms_graph_dsep.py      <- using llms for d-separation estimation with causal graphs as input
│   ├── llms_graph_cdir.py      <- using llms for causal direction estimation with causal graphs as input
│   ├── llms_table_dsep.py      <- using llms for conditional independence estimation with tables as input
│   ├── llms_table_cdir.py      <- using llms for causal direction estimation with tables as input
│   ├── llm_table_graph_inf.py  <- using llms for intervention inference tasks
│   ├── llm_table_graph_cf.py   <- using llms for counterfactual inference tasks
│   └── utils.py                <- util functions
│   
```
# Benchmark for Causal Reasoning and Decision-Making

##  Measuring causal graph reasoning ability of LLM
### Adjacency matrix estimation 

**Applying LLMs to the task**
```
cd card_gt
python llm_graph_cdir.py --llm llama --sim_seed 1 --bt 1 --max_new_tokens 1000 --task_type graph_adj 
```

* sim_seed: the index of causal graphs
* bt: bootstrapping times
* max_new_tokens: number of maximum generated tokens
* task_type: task type can be 'graph_cdir' for causal direction estimation and 'graph_adj' for adjacency matrix estimation


**Evaluation**
```
cd card_gt  
python eva_causal_graph_adj_cdir.py --cm lu --llm llama --task_type graph_adj
```
* cm: the causal mechanism used for generating tables
* llm: the evaluated LLM 
* input_type: the task type for the evaluation, 'graph_cdir' for causal direction estimation and 'graph_adj' for adjacency matrix estimation

### D-separation estimation 

**Generating benchmark data**
Open task_cf_intv.ipynb and run code the block for generating questions and answers.
* Causal graph reasoning: Generating D-separation questions

**Applying LLMs to the task**
```
cd card_gt
python llm_graph_dsep.py --llm llama  --max_new_tokens 1000 --sim_seed 2  --input_type graph
```

* max_new_tokens: number of maximum generated tokens
* sim_seed: the index of causal graphs
* input_type: input data type for inference, 'table' or 'graph' 

**Evaluation**

```
cd card_gt  
python eva_causal_graph_dsep.py --cm lu --llm llama --input_type graph
```
* cm: the causal mechanism used for generating tables
* llm: the evaluated LLM 
* input_type: input data type for inference, 'table' or 'graph' 


### Causal direction estimation 

**Applying LLMs to the task**
```
cd card_gt
python llm_graph_cdir.py --llm llama --sim_seed 1 --bt 1 --max_new_tokens 10000 --task_type graph_cdir
```

* sim_seed: the index of causal graphs
* bt: bootstrapping times
* max_new_tokens: number of maximum generated tokens
* task_type: task type can be 'graph_cdir' for causal direction estimation and 'graph_adj' for adjacency matrix estimation

**Evaluation**
```
cd card_gt  
python eva_causal_graph_adj_cdir.py --cm lu --llm llama --task_type graph_cdir
```
* cm: the causal mechanism used for generating tables
* llm: the evaluated LLM 
* input_type: the task type for the evaluation, 'graph_cdir' for causal direction estimation and 'graph_adj' for adjacency matrix estimation


##  Measuring knowledge discovery ability of LLM

### Conditional independence estimation 

**Generating benchmark data**
Open task_cf_intv.ipynb and run code the block for generating questions and answers.
* Knowledge discovery: Generating conditional independence questions and answers

**Applying LLMs to the task**
```
cd card_gt  
python src/llm_table_desp.py --llm llama  --max_new_tokens 10 --sim_seed 10  --input_type table --max_table_rows 50 --batch_size 1
```

* max_new_tokens: number of maximum generated tokens
* sim_seed: the index of causal graphs
* input_type: input data type for inference, 'table' or 'graph' 
* max_table_rows: the number of rows of input tables
* batch_size: parameter for calling LLMs

**Evaluation**
``` 
cd card_gt  
python eva_causal_table_cdir.py --cm lu --llm llama --input_type table
```
* cm: the causal mechanism used for generating tables
* llm: the evaluated LLM 
* input_type: input data type for inference, 'table' or 'graph' 

### Causal direction estimation 

**Generating benchmark data**
Open task_cf_intv.ipynb and run code the block for generating questions and answers.
* Knowledge discovery: Generating causal direction questions
All the answers to the question is yes.

**Applying LLMs to the task**
```
cd card_gt  
python src/llm_table_cdir.py --llm llama  --max_new_tokens 10 --sim_seed 10  --input_type table --max_table_rows 50 --batch_size 1
```

* max_new_tokens: number of maximum generated tokens
* sim_seed: the index of causal graphs
* input_type: input data type for inference, 'table' or 'graph' 
* max_table_rows: the number of rows of input tables
* batch_size: parameter for calling LLMs

**Evaluation**
``` 
cd card_gt  
python eva_causal_table_cdir.py --cm lu --llm llama --input_type table
```

## Measuring Decision-making ability of LLM

### Generating questions and answers for causal inference tasks
Open task_cf_intv.ipynb and run code the block for generating questions and answers for causal inference tasks.

* Generating intervention inference questions and answers 
* Generating counterfactual inference questions and answers

### Applying LLMs to the causal inference tasks
**Intervention inferece**
```
cd card_gt  
python src/llm_table_graph_inf.py --llm llama  --max_new_tokens 1000 --sim_seed 10  --input_type table --max_table_rows 50 --batch_size 1 
```

**Counterfactual inferece**
```
cd card_gt  
python src/llm_table_graph_cf.py --llm llama  --max_new_tokens 1000 --sim_seed 10  --input_type table --max_table_rows 50 --batch_size 1 
```

* max_new_tokens: number of maximum generated tokens
* sim_seed: the index of causal graphs
* input_type: input data type for inference, 'table' or 'graph' 
* max_table_rows: the number of rows of input tables
* batch_size: parameter for calling LLMs

### Evaluation of causal inference results
Open task_cf_intv.ipynb and run code the block for evaluating on causal inference tasks.
* Evaluation of intervention inference resutls
* Evaluation of counterfactual inference resutls
