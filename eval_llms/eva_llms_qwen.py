
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
    parser.add_argument('--cm',type=str, default='lu')   
    parser.add_argument('--bt',type=int, default=10)   
    parser.add_argument('--max_table_rows',type=int, default=100)   
    parser.add_argument('--max_new_tokens',type=int, default=10000)   
    parser.add_argument('--prow_num',type=int, default=100)
    parser.add_argument('--llm',type=str, default='null')

    args = parser.parse_args()
    return args


def get_table(dataname,seed_sim):
    adj_path = cwd+f'/data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
    graph_np = pd.read_csv(adj_path)
    graph_np = graph_np.iloc[:10,:10]
    adj_gt = graph_np.to_numpy() + graph_np.to_numpy().T

    data_train_path = cwd+f'/data/sim_{dataname}/{seed_sim}/train.csv'
    data_train = pd.read_csv(data_train_path)
    data_train = data_train.iloc[:,:10]
    return adj_gt,data_train  


def get_subset_markdown_table(data,max_table_rows):
    row_index = np.random.randint(len(data), size=max_table_rows)
    df = data.loc[row_index]
    markdown_data = df.to_markdown(index=False)
    return markdown_data    
        

def get_prompt(prow_num, markdown_data):
    messages = [
        {"role": "user", 
         "content": f"You are a tabular synthetic data generation model. \
            Your goal is to produce samples which mirrors the given examples in causal structure and data distributions but also produce as diverse samples as possible.  \
            DO NOT COPY THE EXAMPLES but realistic new and diverse samples. \
            You keep generating next {prow_num} rows of tabular data without skipping or omitting any rows with .... \
            I will give you examples: {markdown_data}. "},]
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

    model_id = "Qwen/Qwen2-7B-Instruct"

    np.random.seed(seed)

    for seed_sim in range(100,105):
        # Start time
        start_time = time.time()
        print(f"loading seed {seed_sim}  type {dataname} dataset.")
        adj_gt,data_train = get_table(dataname,seed_sim)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)


        with open(cwd+f'/result/{llm}/eva_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'w') as file:
            file.write('')

        with open(cwd+f'/result/{llm}/prompt_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'w') as file:
            file.write('')


        for i in range(int(iteration_prompt)):
            markdown_data = get_subset_markdown_table(data_train, max_table_rows)  # get 20 rows from the whole table
            messages = get_prompt(prow_num,markdown_data)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(response)
            
            with open(cwd+f'/result/{llm}/eva_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'a') as file:
                file.write(f'{response}\n')
            with open(cwd+f'/result/{llm}/prompt_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'a') as file:
                file.write(f'{markdown_data}\n')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time using time module: {elapsed_time:.6f} seconds")
