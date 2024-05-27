# CauTabBench

![alt text](https://github.com/TURuibo/CauTabBench/blob/main/benchmark.png "Overview")

This repository is used for generating benchmark datasets and evaluating tabular synthesis models on high-order structural causal information. The paper (Causality for Tabular Data Synthesis: A High-Order Structure Causal Benchmark Framework) is available at URL.

## Structure of the repository

process_sim_datasets.py: the code for generating benchmark datasets  
|- benchmark: source code for evaluation  
|- data: the benchmark data used for training baseline methods   
|- synthetic: the generated data by training baseline methods  
|- utils: modified code based on other repositories  
|- result: benchmarking results  


## Installation

Install pytorch

`pip install torch torchvision torchaudio`

Install other packages

`pip install -r requirements.txt`

## Usage

Settings:  
* --cm: causal mechanisms: lg, lu, sg, nn
* --sz: sample size
* --bt: bootstrapping times
* --seed: the random seed for runing the experiments

Generating benchmark datasets  
`python process_sim_dataset.py --seed 100 --cm lg`  

Evaluate baseline methods with different high-order structure metrics  
`python benchmark/eva_skeleton.py --seed 100 --cm lg --sz 15000 --bt 10`  
`python benchmark/eva_ci_sets.py --seed 100 --cm lg --sz 15000 `  
`python benchmark/eva_bcd.py --seed 100 --cm lg --sz 15000 --bt 10`  
`python benchmark/eva_cdir_lingam.py --seed 100 --sz 15000 --bt 10`  
`python benchmark/eva_intervention.py --seed 100 --cm lg --sz 1000`  
`python benchmark/eva_intervention_amm.py --seed 100 --cm lg --sz 1000`  
`python benchmark/eva_counterfactual.py --seed 100 --cm lg --sz 1000`  
`python benchmark/eva_counterfactual_anm.py --seed 100 --cm lg --sz 1000`  

Implementation in folder "utils" is modified based on other repositories:
* acyclic_graph_generator.py and causal_mechanisms.py are modified based on [cdt](https://github.com/FenTechSolutions/CausalDiscoveryToolbox).  
* process_dataset.py and utils.py are modified based on [tabsyn](https://github.com/amazon-science/tabsyn).  

