
import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
import time


import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--seed',type=int, default=29)   
    parser.add_argument('--sim_seed',type=int, default=109)   
    parser.add_argument('--cm',type=str, default='lu')   
    parser.add_argument('--bt',type=int, default=10)   
    parser.add_argument('--max_table_rows',type=int, default=100)   
    parser.add_argument('--max_new_tokens',type=int, default=10000)   
    parser.add_argument('--prow_num',type=int, default=100)
    parser.add_argument('--llm',type=str, default='null')

    args = parser.parse_args()
    return args

def get_dag_table(dataname,seed_sim):
    adj_path = cwd+f'/data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
    graph_np = pd.read_csv(adj_path)
    graph_np = graph_np.iloc[:10,:10]
    dag_gt = graph_np.to_numpy()

    data_train_path = cwd+f'/data/sim_{dataname}/{seed_sim}/train.csv'
    data_train = pd.read_csv(data_train_path)
    data_train = data_train.iloc[:,:10]
    return dag_gt,data_train  

def get_table(dataname,seed_sim):
    adj_path = cwd+f'/data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
    graph_np = pd.read_csv(adj_path)
    graph_np = graph_np.iloc[:10,:10]
    adj_gt = graph_np.to_numpy() + graph_np.to_numpy().T

    data_train_path = cwd+f'/data/sim_{dataname}/{seed_sim}/train.csv'
    data_train = pd.read_csv(data_train_path)
    data_train = data_train.iloc[:,:10]
    return adj_gt,data_train  

def get_adj(dag_gt):
    return dag_gt + dag_gt.T


def get_subset_markdown_table(data,max_table_rows):
    row_index = np.random.randint(len(data), size=max_table_rows)
    df = data.loc[row_index]
    markdown_data = df.to_markdown(index=False)
    return markdown_data    
        

def graph_to_text(dag_gt):
    heads, tails = np.where(dag_gt == 1)
    nrow,ncol = dag_gt.shape
    # causal graph nodes
    prompt_dag = 'A causal graph has nodes '
    for i in range(nrow-1):
        prompt_dag += f'V{i}, '
    prompt_dag += f'and V{i+1}. And its edges are '

    # causal graph edges
    for i in range(len(heads)-1):
        prompt_dag += f'V{heads[i]} -> V{tails[i]}, '
    prompt_dag += f'and V{heads[i+1]} -> V{tails[i+1]}. '
    
    return prompt_dag

def get_prompt(causal_graph_data,task):
    messages = [
        {"role": "user", 
         "content": f"You are reasoning over causal graphs. {causal_graph_data}. {task}"},]
    return messages

def get_prompt_text2adj(response):
    messages = [
        {"role": "user", 
         "content": f"You are extracting neighbor information from given responses. The response is: {response}.  The result in the format: Neighbors of XXX are XXX, ... where XXX should be replaced by nodes."},]
    return messages

if __name__ == "__main__":
    args = get_args()
    seed =args.seed
    dataname = args.cm
    iteration_prompt = args.bt
    max_table_rows = args.max_table_rows
    max_new_tokens = args.max_new_tokens
    prow_num = args.prow_num
    llm = args.llm
    seed_sim = args.sim_seed
    model_id = "Qwen/Qwen2-7B-Instruct"

    np.random.seed(seed)
    
    # Start time
    start_time = time.time()
    print(f"loading seed {seed_sim}  type {dataname} dataset.")

    dag_gt,data_train = get_dag_table(dataname,seed_sim)
    adj_gt = get_adj(dag_gt) 

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    with open(cwd+f'/result/{llm}/causal_dag_response_i1_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'w') as file:
        file.write('')
    with open(cwd+f'/result/{llm}/causal_dag_response_i2_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'w') as file:
        file.write('')
    with open(cwd+f'/result/{llm}/causal_dag_prompt_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'w') as file:
        file.write('')
    with open(cwd+f'/result/{llm}/causal_dag_adj_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'w') as file:
        file.write('')


    for i in range(int(iteration_prompt)):
        causal_graph_text = graph_to_text(dag_gt)  

        # causal graph reasoning questions
        task = 'What are the neighbors of each node in the causal graph?'

        messages = get_prompt(causal_graph_text,task)
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        
        messages = get_prompt_text2adj(response)
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=max_new_tokens)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response_adj = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        with open(cwd+f'/result/{llm}/causal_dag_response_i1_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'a') as file:
            file.write(f'bt{i} response:\n {response}\n')

        with open(cwd+f'/result/{llm}/causal_dag_response_i2_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'a') as file:
            file.write(f'bt:{i} response:\n {response_adj}\n')

        with open(cwd+f'/result/{llm}/causal_dag_prompt_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'a') as file:
            file.write(f'bt:{i} prompt:\n {messages}\n')        

        with open(cwd+f'/result/{llm}/causal_dag_adj_{dataname}{seed_sim}_out{max_new_tokens}.txt', 'a') as file:
            file.write(f'bt:{i} adj:\n {adj_gt}\n')
        

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time using time module: {elapsed_time:.6f} seconds")
