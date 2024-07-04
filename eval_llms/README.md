# Introduction
These are the codes for evaluating the quality of generated tabular data of LLMs.


# Usage
* Generated tabular data: Given the benchmark dataset with causal graph and tabular data in " data/sim_lu/100", the corresponding generated tabular data should be saved in directory: "synthetic/sim_lu/100/qwen_100i.csv".  
* Evaluation results: the evaluated results will be saved in "results/qwen_100i" according to different LLMs.

# Example
* causal skeleton: python eval_llms/eva_skeleton.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105  
* d-sepearation: python eval_llms/eva_ci_sets.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105  
* causal direction: python eval_llms/eva_bcd.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105  
* causal graphs: python eval_llms/eva_cdir_lingam.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105 
* interventional inference: python eval_llms/eva_intervention.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105
* counterfactural inference: python eval_llms/eva_counterfactural.py --cm lu --sz 400 --llm qwen_100i --seed_sim 105