import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
import time
from src.utils import *


def load_task(graph_id,input_type):
    with open(f'./benchmark/{input_type}/cdir_{graph_id}_questions.txt', 'r') as file:
        questions_ls = file.readlines()
    return questions_ls


def create_output_fiels(dataname,seed_sim,llm,result_path,prefix):
    with open(cwd+f'{result_path}/{llm}/{prefix}_cdir_response_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')
    

def save_results(dataname,seed_sim,llm,response_ls,result_path,prefix,questions):   
    with open(cwd+f'{result_path}/{llm}/{prefix}_cdir_response_{dataname}{seed_sim}.txt', 'a') as file:
        for response,question in zip(response_ls,questions):
            response = response.replace('\n', '')
            file.write(f'{{"question":\"{question.rstrip()}\", "answer": \"{response}\"}}\n')
    


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
    batch_size = args.batch_size
    
    graph_id = seed_sim
    start_time = time.time()
    dag_gt,data_train = get_dag_table(dataname,graph_id)
    adj_gt = get_adj(dag_gt) 

    table_text = get_subset_markdown_table(data_train, max_table_rows)  # get 20 rows from the whole table
    questions_ls  = load_task(graph_id,input_type)
    message_ls = get_example_prompt(table_text,questions_ls)
    
    create_output_fiels(dataname,graph_id,llm,result_path,input_type)
    if llm == 'gemma': 
        model_id = "google/gemma-2-9b-it"
        response_ls = get_qwen_response(model_id,message_ls,batch_size,max_new_tokens,temperature,top_p,seed)
    elif llm == 'qwen': 
        model_id = "Qwen/Qwen2-7B-Instruct"
        response_ls = get_qwen_response(model_id,message_ls,batch_size,max_new_tokens,temperature,top_p,seed)
    elif llm == 'llama': 
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        response_ls = get_mistral_response(model_id,message_ls,batch_size,max_new_tokens,temperature,top_p,seed)
    elif llm == 'mistral': 
        model_id = "mistral_models/Mistral-7B-Instruct-v0.1"
        response_ls = get_mistral_response(model_id,message_ls,batch_size,max_new_tokens,temperature,top_p,seed)
    elif llm == 'mixtral': 
        model_id = "mistral_models/Mixtral-8x7B-Instruct-v0.1"
        response_ls = get_mistral_response(model_id,message_ls,batch_size,max_new_tokens,temperature,top_p,seed)
    
    save_results(dataname,seed_sim,llm,response_ls,result_path,input_type,questions_ls)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time using time module: {elapsed_time:.6f} seconds")


