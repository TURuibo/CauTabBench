import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
import time
import numpy as np
import torch
import transformers

from src.utils import *


def create_output_fiels(dataname,seed_sim):
    with open(cwd+f'/result/{llm}/causal_dag_response_i1_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')
    with open(cwd+f'/result/{llm}/causal_dag_response_i2_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')
    with open(cwd+f'/result/{llm}/causal_dag_prompt_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')
    with open(cwd+f'/result/{llm}/causal_dag_adj_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')

def save_results(dataname,seed_sim,response,response_adj,messages,adj_gt):
    with open(cwd+f'/result/{llm}/causal_dag_response_i1_{dataname}{seed_sim}.txt', 'a') as file:
        file.write(f'response:\n {response}\n')

    with open(cwd+f'/result/{llm}/causal_dag_response_i2_{dataname}{seed_sim}.txt', 'a') as file:
        file.write(f'response:\n {response_adj}\n')

    with open(cwd+f'/result/{llm}/causal_dag_prompt_{dataname}{seed_sim}.txt', 'a') as file:
        file.write(f'prompt:\n {messages}\n')        

    with open(cwd+f'/result/{llm}/causal_dag_adj_{dataname}{seed_sim}.txt', 'a') as file:
        file.write(f'adj:\n {adj_gt}\n')


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
    temperature = args.temperature
    top_p = args.top_p
    create_output_fiels(dataname,seed_sim)
    
    if llm == 'gemma':
        model_id = "google/gemma-2-9b-it"
    elif llm == 'llama':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    

    causal_graph_text = graph_to_text(dag_gt)  

    # causal graph reasoning questions
    task = 'What are the neighbors of each node in the causal graph?'

    messages = get_graph_prompt(causal_graph_text,task)
    if llm == 'qwen':
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p)
    else:
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p)
    response = outputs[0]["generated_text"][-1]["content"]
    print(response)
    
    messages = get_prompt_graph2adj(response)
    if llm == 'qwen':
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p)
    else:
        outputs = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
    response_adj = outputs[0]["generated_text"][-1]["content"]
    print(response_adj)
    save_results(dataname,seed_sim,response,response_adj,messages,adj_gt)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time using time module: {elapsed_time:.6f} seconds")


