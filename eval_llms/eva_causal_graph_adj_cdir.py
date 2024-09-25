import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
from src.causal_eval.helper import get_adjacency_matrice
import pandas as pd
import numpy as np
from cdt.metrics import precision_recall
from utils.utils import get_args


def get_f1(precision, recall):
    f1 = 2*precision*recall/(precision+recall)
    return f1

def test(adj_llm,adj_gt):
    _, curve = precision_recall(adj_llm, adj_gt)
    precision,recall= curve[1]
    f1 = get_f1(precision, recall)
    return f1,precision,recall

def get_dag_gt(dataname,seed_sim):
    adj_path = f'./data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
    graph_np = pd.read_csv(adj_path)
    nrow,_ = graph_np.shape
    graph_np = graph_np.iloc[:nrow-1,:nrow-1]
    dag_gt = graph_np.to_numpy()
    return dag_gt

def get_adj_gt(dataname,seed_sim):
    adj_path = f'./data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
    graph_np = pd.read_csv(adj_path)
    nrow,_ = graph_np.shape
    graph_np = graph_np.iloc[:nrow-1,:nrow-1]
    adj_gt = graph_np.to_numpy() + graph_np.to_numpy().T
    return adj_gt



if __name__ == "__main__":
    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lu 
    llm = args.llm # qwen
    task_type = args.task_type
    f1_gs, precision_gs,recall_gs  = [], [], []

    for seed_sim in range(100,110):
        print(seed_sim)
        if task_type == 'graph_adj':
            adj_gt = get_adj_gt(dataname,seed_sim)
        elif task_type =='graph_cdir':
            adj_gt = get_dag_gt(dataname,seed_sim)
        f1_ls, precision_ls,recall_ls  = [], [], []
        file_dir = f'./eval_llms/result/{llm}/{task_type}_answer_{dataname}{seed_sim}.txt'
        n_nodes,_= adj_gt.shape
        adj_llm = get_adjacency_matrice(file_dir,n_nodes)

       
        
        f1, precision,recall = test(adj_llm,adj_gt)
        if np.isnan(f1):
            pass
        else:
            f1_gs.append(f1)
            precision_gs.append( precision)
            recall_gs.append(recall)
    
    print(f'&${np.mean(f1_gs):.2f}\pm{np.std(f1_gs):.2f}$ &${np.mean(precision_gs):.2f}\pm{np.std(precision_gs):.2f}$ &$ {np.mean(recall_gs):.2f}\pm{np.std(recall_gs):.2f}$')