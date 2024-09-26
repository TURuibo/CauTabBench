import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
import re
import argparse
import pandas as pd
import numpy as np
import torch
import transformers
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed',type=int, default=29)   
    parser.add_argument('--cm',type=str, default='lg')   
    parser.add_argument('--bt',type=int, default=10)   
    parser.add_argument('--n_nodes',type=int, default=51)   

    parser.add_argument('--seed_sim',type=int, default=101)   
    parser.add_argument('--llm',type=str, default='null')   
    parser.add_argument('--n_nodes',type=int, default=10)   
    parser.add_argument('--n_prt',type=int, default=2)   
    parser.add_argument('--sz',type=int, default=19019)   
    parser.add_argument('--noise_coeff',type=float, default=0.4)   
    parser.add_argument('--input_type',type=str, default='graph')   
    parser.add_argument('--task_type',type=str, default='null')   
    parser.add_argument('--max_table_rows',type=int, default=10)   
    parser.add_argument('--batch_size',type=int, default=10)
    parser.add_argument('--max_new_tokens',type=int, default=10000)   
    parser.add_argument('--prow_num',type=int, default=100)
    parser.add_argument('--temperature',type=float, default=0.6)  
    parser.add_argument('--top_p',type=float, default=0.9) 
    parser.add_argument('--result_path',type=str, default='/result')       
    args = parser.parse_args()

    return args


#####  Prompts for graph adjacency matrix estimation #### 

def get_graph_prompt(causal_graph_data,task):
    messages = [
        {"role": "user", 
         "content": f"You are reasoning over causal graphs. {causal_graph_data}. {task}"},]
    return messages


def get_task(task_type):
        if task_type == 'graph_cdir':
            task = 'What are the parents of each node in the causal graph?'
        elif task_type == 'graph_adj':
            task = 'What are the neighbors of each node in the causal graph?'
        return task

def get_prompt_graph_tasks(response,task_type):
    if task_type == 'graph_cdir':
        messages = [
        {"role": "user", 
         "content": f"You are extracting children of nodes from given responses. The response is: {response}. You first summarize the children of each node in the format: Children of XXX are XXX, ... where XXX should be replaced by nodes."},]
    elif task_type == 'graph_adj':
        messages = [
        {"role": "user", 
         "content": f"You are extracting neighbor information from given responses. The response is: {response}.  The result in the format: Neighbors of XXX are XXX, ... where XXX should be replaced by nodes."},]
    return messages


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


def get_dag_table(dataname,seed_sim):
    adj_path = cwd+f'/data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
    graph_np = pd.read_csv(adj_path)
    nrow,_ = graph_np.shape

    graph_np = graph_np.iloc[:nrow-1,:nrow-1]
    dag_gt = graph_np.to_numpy()

    data_train_path = cwd+f'/data/sim_{dataname}/{seed_sim}/train.csv'
    data_train = pd.read_csv(data_train_path)
    data_train = data_train.iloc[:,:nrow-1]
    return dag_gt,data_train  


def get_adj(dag_gt):
    return dag_gt + dag_gt.T


def get_subset_markdown_table(data,max_table_rows):
    row_index = np.random.randint(len(data), size=max_table_rows)
    df = data.loc[row_index]
    markdown_data = df.to_markdown(index=False)
    return markdown_data    
        

def graph_to_text(dag_gt):
    heads, tails = np.where(dag_gt == 1)
    nrow,_ = dag_gt.shape
    # causal graph nodes
    prompt_dag = 'A causal graph has nodes '
    for i in range(nrow-1):
        prompt_dag += f'V{i}, '
    prompt_dag += f'and V{i+1}. And its edges are '

    for i in range(len(heads)-1):
        prompt_dag += f'V{heads[i]} -> V{tails[i]}, '
    prompt_dag += f'and V{heads[i+1]} -> V{tails[i+1]}. '
    
    return prompt_dag

#### Prompt for experiments of graph d-separation estimation #### 

def get_graph_prompt_list(causal_graph_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over causal graphs. {causal_graph_data}. {task} "
          }] for task in task_ls
          ]
    return messages


def get_prompt_graph2adj_list(response_ls,task_ls):
    messages = [
        [{
            "role": "user", 
            "content": f"The question is {task}. The response is: {response}. Given the question and the response, summarize the answer as yes or no based on the response. The answer is in the format, the answer is XXX, where XXX is either yes or no or unknown."
            }] for response,task in zip(response_ls,task_ls)
            ]
    return messages



def get_qwen_response(model_id,message_ls,batch_size,temperature,top_p,seed,max_new_tokens):
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


def get_qwen_response_refinement(model_id,message_ls,batch_size,temperature,top_p,seed,max_new_tokens):
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
        message_ls =get_causal_inference_prompt_refinement(response_ls)
        response_ls = get_qwen_response(model_id,message_ls,batch_size,temperature,top_p,seed,max_new_tokens=50)

        return response_ls


def get_mistral_response_refinement(model_id,message_ls,batch_size,temperature,top_p,seed,max_new_tokens):
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
    
    message_ls =get_causal_inference_prompt_refinement(response_ls)
    response_ls = get_mistral_response(model_id,message_ls,batch_size,temperature,top_p,seed,max_new_tokens=50)

    return response_ls


def get_mistral_response(model_id,message_ls,batch_size,temperature,top_p,seed,max_new_tokens):
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


def get_table_prompt_list(tabular_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over tables. Columns represent different nodes and rows represent different sampels. The data are {tabular_data}. {task} You have to give the answer in the response by yourself."
          }] for task in task_ls
        ]
    return messages

def get_example_prompt(tabular_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over tables. Columns represent different nodes and rows represent different sampels. The data are {tabular_data}. {task} Answer the question with only yes or no."
          }] for task in task_ls
        ]
    return messages

def get_causal_inference_prompt(tabular_data,graph_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over tables and causal graphs and then answer causal inference questions. Columns represent different nodes and rows represent different sampels. The data are {tabular_data}. The causal graph is {graph_data}. {task}. You must provide a fload number as the final results."
          }] for task in task_ls
        ]
    return messages

def get_cf_prompt(tabular_data,graph_data,task_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are reasoning over tables and causal graphs and then answer causal inference questions. Columns represent different nodes and rows represent different sampels. The data are {tabular_data}. The causal graph is {graph_data}. {task}. You must provide a fload number as the final results."
          }] for task in task_ls
        ]
    return messages

def get_causal_inference_prompt_refinement(response_ls):
    messages = [
        [{"role": "user", 
          "content": f"You are extracting the final answer from a response of a model. The response is {response}. Answer in the format that The final answer is XXX, where XXX is the results as a float number."
          }] for response in response_ls
        ]
    return messages


# Evaluate Counterfactual and intervention inference

def extract_yes_no(text):
    # Use regular expressions to find occurrences of "yes" or "no" in the text
    yes_no_responses = re.findall(r'\b(yes|no)\b', text, flags=re.IGNORECASE)
    
    # Normalize the responses to lower case (or capitalize as needed)
    normalized_responses = [response.capitalize() for response in yes_no_responses]
    
    return normalized_responses


def remove_quotes(input_str):
    # Find the start of the "answer" field
    start_index = input_str.find('"answer": "') + len('"answer": "')
    
    # Find the end of the answer field (next double quote after the value)
    # end_index = input_str.find('"', start_index)
    
    # Extract the text after "answer": 
    answer_text = input_str[start_index:].replace('"', '')
    
    return input_str[:start_index] + answer_text[:-2] +'"}'

def remove_quotes_in_answer(input_str):
    # Regex pattern to match the "answer" field content and remove quotes inside it
    modified_str = re.sub(r'("answer":\s*")([^"]*)"', lambda m: m.group(1) + m.group(2).replace('"', '') + '"', input_str)
    return modified_str


def one_hot_encode_to_boolean(number, num_classes):
    """
    One-hot encode a number as boolean array.
    
    Args:
    - number: The number to encode.
    - num_classes: The total number of classes.
    
    Returns:
    - one_hot_bool: The one-hot encoded boolean array.
    """
    one_hot_bool = np.zeros(num_classes, dtype=bool)
    one_hot_bool[number] = True
    return ~one_hot_bool


def cf_gen(gcm, causal_model, inv_dim, inv_data, obs_data):
    samples = gcm.counterfactual_samples(causal_model,
                                         {inv_dim: lambda y: inv_data},
                                         observed_data=obs_data)
    return samples.to_numpy()

def interv_gen(gcm, causal_model,inv_dim,inv_data,sz=1000):
    samples = gcm.interventional_samples(causal_model,
                                        {inv_dim: lambda y: inv_data},
                                        num_samples_to_draw=sz)
    return samples.to_numpy()


def train_scm_model(adj_df,data_df):
    causal_graph = nx.from_numpy_array(adj_df.to_numpy(), create_using=nx.DiGraph)
    causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)
    nrow,_ = adj_df.shape
    data = data_df.iloc[:,:nrow]
    data.dropna(inplace=True)

    data.columns = list(causal_graph.nodes)
    dag,nodes = adj2dag(adj_df.to_numpy())
    cg_nodes = list(causal_graph.nodes)

    gcm.auto.assign_causal_mechanisms(causal_model, data)

    for ind,node in enumerate(list(cg_nodes)):
        if len(dag.get_parents(nodes[ind])) == 0 :
            causal_model.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        else:     
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

    gcm.fit(causal_model, data)
    return gcm,causal_model,causal_graph

def mae_mean_cf(gcm_gt, causal_model_gt,gcm_syn, causal_model_syn,data_df,sz=1000,n_nodes=51):
    intervention_ls = np.random.randn(n_nodes)*5
    mae_dims = []
    for inv_dim in range(n_nodes):
        for itvn in intervention_ls:
            index = np.arange(0,len(data_df.iloc[:,0]))
            np.random.shuffle(index)
            AE = np.abs(np.mean(cf_gen(gcm_gt, causal_model_gt,inv_dim,itvn,data_df.iloc[index[:sz]]),axis=0)-\
                        np.mean(cf_gen(gcm_syn, causal_model_syn,inv_dim,itvn,data_df.iloc[index[:sz]]),axis=0))
            MAE_i = np.mean(AE[one_hot_encode_to_boolean(inv_dim,n_nodes)])
            mae_dims.append(MAE_i)
    return np.mean(mae_dims)

def adj2dag(adj_df):
    G = nx.from_numpy_array(adj_df, create_using=nx.DiGraph)
    nodes = ini_nodes(adj_df)
    dag = Dag(nodes)
    for i,j in list(G.edges()):
        dag.add_directed_edge(nodes[i], nodes[j])
    return dag,nodes

def ini_nodes(adj_df):
    nodes = []
    for i in range(len(adj_df[0,:])):
        nodes.append(GraphNode(str(i)))
    return nodes