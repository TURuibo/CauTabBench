import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import argparse
import pandas as pd
import numpy as np

import torch
import transformers


def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')
    # General configs
    parser.add_argument('--seed',type=int, default=29)   
    parser.add_argument('--sim_seed',type=int, default=109)   
    parser.add_argument('--cm',type=str, default='lu')   
    parser.add_argument('--bt',type=int, default=10)   
    parser.add_argument('--max_table_rows',type=int, default=10)   
    parser.add_argument('--batch_size',type=int, default=10)
    parser.add_argument('--max_new_tokens',type=int, default=10000)   
    parser.add_argument('--prow_num',type=int, default=100)
    parser.add_argument('--llm',type=str, default='null')
    parser.add_argument('--temperature',type=float, default=0.6)  
    parser.add_argument('--top_p',type=float, default=0.9) 
    parser.add_argument('--input_type',type=str, default='graph')   
    parser.add_argument('--result_path',type=str, default='/result')   
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

    for i in range(len(heads)-1):
        prompt_dag += f'V{heads[i]} -> V{tails[i]}, '
    prompt_dag += f'and V{heads[i+1]} -> V{tails[i+1]}. '
    
    return prompt_dag


def get_qwen_response(model_id,message_ls,batch_size,max_new_tokens,temperature,top_p,seed):
        np.random.seed(seed)
    
        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            )
        
        pipe.tokenizer.padding_side = "left"
        
        outputs = pipe(
            message_ls,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p)
        
        response_ls =[]
        for i in range(len(outputs)):
            response_ls.append(outputs[i][0]["generated_text"][-1]["content"])
        return response_ls

def get_mistral_response(model_id,message_ls,batch_size,max_new_tokens,temperature,top_p,seed):
    np.random.seed(seed)
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        )
    
    pipe.tokenizer.padding_side = "left"
    terminators = [pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    outputs = pipe(
        message_ls,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p)
    
    response_ls =[]
    for i in range(len(outputs)):
        response_ls.append(outputs[i][0]["generated_text"][-1]["content"])
    return response_ls


def get_graph_prompt_list(causal_graph_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over causal graphs. {causal_graph_data}. {task} "
          }] for task in task_ls
          ]
    return messages

def get_table_prompt_list(tabular_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over tables. Columns represent different nodes and rows represent different sampels. The data are {tabular_data}. {task} You have to give the answer in the response by yourself."
          }] for task in task_ls
        ]
    return messages

def get_prompt_graph2adj_list(response_ls,task_ls):
    messages = [
        [{
            "role": "user", 
            "content": f"The question is {task}. The response is: {response}. Given the question and the response, summarize the answer as yes or no based on the response. If the response indicates the answer cannot be determined, summarize the answer as unknown."
            }] for response,task in zip(response_ls,task_ls)
            ]
    return messages

def get_example_prompt(tabular_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over tables. Columns represent different nodes and rows represent different sampels. The data are {tabular_data}. {task} Answer the question with only yes or no."
          }] for task in task_ls
        ]
    return messages
