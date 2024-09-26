import os,sys
cwd = os.path.abspath(os.path.curdir)
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.append(cwd)  # workplace
import numpy as np
import pandas as pd
from src.utils import get_args,interv_gen,train_scm_model

def get_gt_intervention_distribution(gcm_gt, causal_model_gt,sim_seed,sz=1000):
    np.random.seed(77)
    intervention_ls = np.random.randn(51)*5

    with open(f'./data/table/intervention_{sim_seed}_gt.txt', 'w') as file:
        for inv_dim in range(51):
            itvn = intervention_ls[inv_dim]
            res = np.mean(interv_gen(gcm_gt, causal_model_gt,inv_dim,itvn,sz),axis=0)
            res_ph = get_back_v(inv_dim,itvn,res)
            v_o = np.random.randint(0, high=51, size=1, dtype=int)[0]
            r = res_ph[v_o]
            # for r in res_ph:
            file.write(f'{r[0]}\n')
       
def get_back_v(inv_dim,itvn,res):
    res_ph = np.zeros((51,1))
    res_ph[inv_dim] = itvn
    for i in range(9):
        if i <inv_dim:
            j = i
        else: 
            j = i+1
        res_ph[j] = res[i]
    return res_ph

if __name__ == "__main__":

    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lg 

    np.random.seed(seed)
    
    dataname= 'sim_' + dataname # lg lu sg nn
    
    mae_m = []
    for sindex in range(1,11):
        data_path = parent_dir+f'/data/{dataname}/{sindex}/generated_graph_data.csv'
        data_df = pd.read_csv(data_path)
        adj_path = parent_dir+f'/data/{dataname}/{sindex}/generated_graph_target.csv'
        adj_df = pd.read_csv(adj_path)
        gcm_gt ,causal_model_gt, causal_graph_gt = train_scm_model(adj_df,data_df)

        get_gt_intervention_distribution(gcm_gt, causal_model_gt,sindex)
