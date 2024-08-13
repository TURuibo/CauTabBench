import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
import time
import numpy as np
import pandas as pd
import torch
import transformers
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
    seed_sim = args.sim_seed
    iteration_prompt = args.bt
    max_table_rows = args.max_table_rows
    prow_num = args.prow_num
    max_new_tokens = args.max_new_tokens
    llm = args.llm
    

    if llm == 'gemma':
        model_id = "google/gemma-2-9b-it"
    elif llm == 'mistral':
        model_id = "mistral_models/Mistral-7B-Instruct-v0.1"
    elif llm == 'mixtral':
        model_id = "mistral_models/Mixtral-8x7B-Instruct-v0.1"
    elif llm == 'qwen':
        model_id = "Qwen/Qwen2-7B-Instruct"

    np.random.seed(seed)
    start_time = time.time()
    print(f"loading seed {seed_sim}, dataset {dataname}, model {llm}.")

    dag_gt,data_train = get_dag_table(dataname,seed_sim)
    adj_gt = get_adj(dag_gt) 

    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [pipe.tokenizer.eos_token_id,
                   pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
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
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9)
        #     temperature=0.7,
        #     top_k=50,
        #     top_p=0.95
        # )
        response = outputs[0]["generated_text"][-1]["content"]
        print(response)
        
        messages = get_prompt_text2adj(response)
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        response_adj = outputs[0]["generated_text"][-1]["content"]
        print(response_adj)
                
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


