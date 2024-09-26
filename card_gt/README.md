# Introduction
This is a repository for paper ["CARD-GT: Rethinking Language Model Capabilities for Causal Reasoning and Decision-Making Using Causal Graphs and Tabular Data"]().
It includes benchmarking tasks from the perspectives of causal graph reasoning, knowledge discovery, and decision-making.

```
├── .github              
│
├── task_cf_intv.ipynb               <- Generating benchmark data for causal graph reasoning, knowledge discovery, and decision-making tasks,
│                                           <- it also evaluates the results of decision-making task
├── eva_causal_table_cdir.py         <- Knowledge discovery: evaluate causal direction estimation of 

├── data/              <- Benchmark data
│   ├── graph/         <- Causal graph reasoning tasks: questions and answers
│   └── table/         <- Knolwedge discovery and decision making: questions and answers
│   
├── results/           <- responses of LLms
│   ├── llama/         
│   ├── ...
│   └── table/         
│   
├── src/                   <- Source code
│   │
│   ├── llms_table_cdir.py          <- using llms for causal direction estimation with tables as input
│   ├── llm_table_graph_inf.py          <- using llms for intervention inference tasks
│   ├── llm_table_graph_cf.py           <- using llms for counterfactual inference tasks
│   ├── utils.py                        <- util functions
│   └── ...                   <- Q1: Downloaded mistral source code
│   
```


##  Measuring knowledge discovery ability of LLM

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
