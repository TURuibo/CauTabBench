import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
from src.tab_gen.helper import get_adjacency_matrice
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

def get_adj_gt(dataname,seed_sim):
    adj_path = f'./data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
    graph_np = pd.read_csv(adj_path)
    graph_np = graph_np.iloc[:10,:10]
    adj_gt = graph_np.to_numpy() + graph_np.to_numpy().T
    return adj_gt



if __name__ == "__main__":
    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lu 
    llm = args.llm # qwen
    bt = args.bt  # 5

    llm='qwen'
    dataname='lu'
    f1_gs, precision_gs,recall_gs  = [], [], []

    for seed_sim in range(100,105):
        f1_ls, precision_ls,recall_ls  = [], [], []
        max_new_tokens= 1000
        file_dir = f'./eval_llms/results/{llm}/causal_dag_response_i2_{dataname}{seed_sim}_out{max_new_tokens}.txt'
        adj_llm_ls = get_adjacency_matrice(file_dir)
        adj_gt = get_adj_gt(dataname,seed_sim)
        for bt_i in range(bt):
            f1, precision,recall = test(adj_llm_ls[bt_i],adj_gt)
            f1_ls.append(f1)
            precision_ls.append(precision)
            recall_ls.append(recall)
        f1_gs.append(np.mean(f1_ls))
        precision_gs.append(np.mean(precision_ls))
        recall_gs.append(np.mean(recall_ls))
    
    print(f'&${np.mean(f1_gs):.2f}\pm{np.std(f1_gs):.2f}$, &${np.mean(precision_gs):.2f}\pm{np.std(precision_gs):.2f}$, &$ {np.mean(recall_gs):.2f}\pm{np.std(recall_gs):.2f}$\\\')

