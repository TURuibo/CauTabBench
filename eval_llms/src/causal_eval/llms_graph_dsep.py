import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
import time
import numpy as np
import torch
import transformers
import argparse
from src.utils import *



def load_task(graph_id,input_type):
    with open(f'./benchmark/{input_type}/{graph_id}_questions.txt', 'r') as file:
        questions_ls = file.readlines()
    return questions_ls


def create_output_fiels(dataname,seed_sim,result_path,prefix):
    with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')
    # with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_i2_{dataname}{seed_sim}.txt', 'w') as file:
    #     file.write('')


def save_results(dataname,seed_sim,response_ls,response_adj_ls,result_path,prefix,questions):
    # with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_i1_{dataname}{seed_sim}.txt', 'a') as file:
    #     for response,question in zip(response_ls,questions):
    #         response = response.replace('\n', '')
    #         file.write(f'{{"question":\"{question.rstrip()}\", "answer": \"{response}\"}}\n')
            # file.write(f'Question is {question}\n Answer is {response}\n')
    
    with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_{dataname}{seed_sim}.txt', 'a') as file:
        for response_adj,question in zip(response_adj_ls,questions):
            file.write(f'{{"question":\"{question.rstrip()}\", "answer": \"{response_adj}\"}}\n')
            # if response_adj[-1:] == '\n':
            #     file.write(f'{response_adj}')        
            # else:
            #     file.write(f'{response_adj}\n')        


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
    result_path = args.result_path
    input_type = args.input_type
    
    if llm == 'gemma': model_id = "google/gemma-2-9b-it"
    elif llm == 'llama': model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif llm == 'mistral': model_id = "mistral_models/Mistral-7B-Instruct-v0.1"
    elif llm == 'mixtral': model_id = "mistral_models/Mixtral-8x7B-Instruct-v0.1"
    elif llm == 'qwen': model_id = "Qwen/Qwen2-7B-Instruct"

    np.random.seed(seed)
    start_time = time.time()
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        )
    terminators = [pipe.tokenizer.eos_token_id,
                   pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    graph_id = seed_sim
    create_output_fiels(dataname,graph_id,result_path,input_type)

    dag_gt,data_train = get_dag_table(dataname,graph_id)
    adj_gt = get_adj(dag_gt) 

    causal_graph_text = graph_to_text(dag_gt)  
    # causal graph reasoning questions
    questions_ls  = load_task(graph_id,input_type)
    
    if len(questions_ls) < 10:
        batch_size = len(questions_ls)
    else: 
        batch_size = 10
    
    if llm == 'gemma':
        batch_size = 1
    
    message_ls= get_graph_prompt_list(causal_graph_text,questions_ls)
    pipe.tokenizer.padding_side = "left"
    
    if llm == 'qwen' or llm == 'gemma' :
        outputs = pipe(
            message_ls,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p)
    else:
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
    
    messages_ls = get_prompt_graph2adj_list(response_ls,questions_ls)
    max_new_tokens = 10
    if llm == 'qwen' or llm == 'gemma':
        outputs = pipe(
            messages_ls,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p)
    else:
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        outputs = pipe(
            messages_ls,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p)
    response_adj_ls=[]
    for i in range(len(outputs)):
        response_adj_ls.append(outputs[i][0]["generated_text"][-1]["content"])
    save_results(dataname,seed_sim,response_ls,response_adj_ls,result_path,input_type,questions_ls)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time using time module: {elapsed_time:.6f} seconds")


