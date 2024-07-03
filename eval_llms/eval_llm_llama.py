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
    parser.add_argument('--cm',type=str, default='lu')   
    parser.add_argument('--bt',type=int, default=10)   
    parser.add_argument('--max_table_rows',type=int, default=100)   
    parser.add_argument('--max_new_tokens',type=int, default=8192)   
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
    messages= [
            {"role": "system",
            "content": "You are a tabular synthetic data generation model."},
            {"role": "user",
            "content": f"Your goal is to produce samples which mirrors the given examples in causal structure and data distributions\
                    but also produce as diverse samples as possible. \
                    I will give you examples: {markdown_data}. \
                    You keep generating the next {prow_num} rows of tabular data untill the last row. DONOT omit any data with .... \
                    DO NOT COPY THE EXAMPLES but realistic new and diverse samples."},]
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

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    np.random.seed(seed)

    for seed_sim in range(100,105):
        # Start time
        start_time = time.time()
        print(f"loading seed {seed_sim}  type {dataname} dataset.")
        adj_gt,data_train = get_table(dataname,seed_sim)
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        with open(cwd+f'/result/{llm}/eva_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'w') as file:
            file.write('')

        with open(cwd+f'/result/{llm}/prompt_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'w') as file:
            file.write('')


        for i in range(int(iteration_prompt)):
            markdown_data = get_subset_markdown_table(data_train, max_table_rows)  # get 20 rows from the whole table
            messages = get_prompt(prow_num,markdown_data)

            terminators = [pipeline.tokenizer.eos_token_id,
                           pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            
            outputs = pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            
            response = outputs[0]["generated_text"][-1] ['content']
            print(response)
            
            with open(cwd+f'/result/{llm}/eva_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'a') as file:
                file.write(f'{response}\n')
            
            with open(cwd+f'/result/{llm}/prompt_{dataname}{seed_sim}_prow{prow_num}_in{max_table_rows}_out{max_new_tokens}.txt', 'a') as file:
                file.write(f'{markdown_data}\n')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time using time module: {elapsed_time:.6f} seconds")
