import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import numpy as np
import pandas as pd
import networkx as nx
from cdt.metrics import precision_recall
from causallearn.search.ConstraintBased.PC import pc
import os
from causallearn.utils.cit import kci
from joblib import Parallel, delayed
from utils.utils import get_args


def skeleton_diff(adj_gt, adj_pred):
    return np.sum(np.abs(adj_pred-adj_gt))


def run_pc_nonlinear(data,total_sz,sz,bt_i):
    np.random.seed(bt_i)
    index = np.random.randint(total_sz, size=sz)
    datai = data.iloc[index,:]
    causallearn_cg=pc(datai.to_numpy(),0.05, kci,kernelX='Gaussian',show_progress=False)
    return causallearn_cg


def run_pc_linear(data,total_sz,sz,bt_i):
    np.random.seed(bt_i)
    index = np.random.randint(total_sz, size=sz)
    datai = data.iloc[index,:]
    causallearn_cg=pc(datai.to_numpy(),show_progress=False)
    return causallearn_cg


def test(causallearn_cg,adj_gt):
    causallearn_cg.to_nx_graph()
    causallearn_cg.to_nx_skeleton()
    adj_causallearn = nx.to_numpy_array(causallearn_cg.nx_skel,  weight=None)
    skel_diff_cl = skeleton_diff(adj_gt, adj_causallearn)
    
    _, curve = precision_recall(adj_causallearn, adj_gt)
    precision,recall= curve[1]
    return skel_diff_cl,precision,recall
    

def metric_eva(data,adj_gt,dataname,bt,sz,seed):
    np.random.seed(seed)
    total_sz = len(data.iloc[:,0])
    skel_diff_causallearn=[]
    precision_ls=[]
    recall_ls=[]
    if dataname == 'sg' or dataname == 'nn': 
        causallearn_cg_ls  = Parallel(n_jobs=-2)(delayed(run_pc_nonlinear)(data,total_sz,sz,bt_i) for bt_i in range(bt))    # [causallearn_cg_ls,...]
    else:
        causallearn_cg_ls  = Parallel(n_jobs=-2)(delayed(run_pc_linear)(data,total_sz,sz,bt_i) for bt_i in range(bt))    # [causallearn_cg_ls,...]
    
    for i in range(bt):
        skel_diff_causallearn_i,precision_i,recall_i =test(causallearn_cg_ls[i],adj_gt)    
        skel_diff_causallearn.append(skel_diff_causallearn_i)
        precision_ls.append(precision_i)
        recall_ls.append(recall_i)
    return skel_diff_causallearn,precision_ls,recall_ls


def eva_skel_ratio(dataname = 'lg',m_name = 'tabsyn',bt=10,sz=1000,seed=7):
    
    skel_syn_ls = []
    f1_syn_ls=[]
    precision_syn_ls = []
    recall_syn_ls = []

    for seed_sim in range(100,110): # [100, 102,104,106,108]: ##['1','2','29']: [100, 102,104,106,108]
        
        data_sim_path = f'./synthetic/sim_{dataname}/{seed_sim}/{m_name}.csv'
        data_sim = pd.read_csv(data_sim_path)
        data_sim = data_sim.iloc[:,:data_sim.shape[1]-1]
       
        adj_path = f'./data/sim_{dataname}/{seed_sim}/generated_graph_target.csv'
        graph_np = pd.read_csv(adj_path)
        graph_np = graph_np.iloc[:data_sim.shape[1],:data_sim.shape[1]]
        adj_gt = graph_np.to_numpy() + graph_np.to_numpy().T

        skel_diff_syn,precision_syn,recall_syn = metric_eva(data_sim,adj_gt,dataname,bt=bt,sz=sz,seed=seed)
        f1_syn=[]
        for i in range(len(recall_syn)):
            f1_i = 2*precision_syn[i]*recall_syn[i]/(precision_syn[i]+recall_syn[i])
            if np.isnan(f1_i):
                pass
            else:
                f1_syn.append(f1_i)
        
        skel_syn_ls.append(np.mean(skel_diff_syn))        
        precision_syn_ls.append(np.mean(precision_syn))
        recall_syn_ls.append(np.mean(recall_syn))
        f1_syn_ls.append(np.mean(f1_syn))
       
    with open(f'./result/eva_{dataname}.txt', 'a') as file:
        file.write(f'{m_name}'+\
                    f'\t & ${np.mean(skel_syn_ls):.2f}'+'\pm '+f'{np.std(skel_syn_ls):.2f}$'+ \
                    f'\t & ${np.mean(f1_syn_ls):.2f}'+'\pm '+f'{np.std(f1_syn_ls):.2f}$'+ \
                    f'\t & ${np.mean(precision_syn_ls):.2f}'+'\pm '+f'{np.std(precision_syn_ls):.2f}$'+ \
                    f'\t & ${np.mean(recall_syn_ls):.2f}'+'\pm '+f'{np.std(recall_syn_ls):.2f}$\\\ '+ '\n')
        


if __name__ == "__main__":

    args = get_args()
    seed =args.seed
    dataname = args.cm
    sz =args.sz
    bt = args.bt

    m_name_ls= ['real','tabsyn','stasy','tabddpm','codi','great','ctgan','tvae'] 
    
    print(dataname,bt,sz)

    with open(f'./result/eva_{dataname}.txt', 'w') as file:
        file.write('\t& adj  '+ '\t & f1 '+ '\t & precision '+ '\t& recall \\\ \n')

    for i in range(len(m_name_ls)):
        print(f'evaluating {m_name_ls[i]}')
        eva_skel_ratio(dataname = dataname,m_name = m_name_ls[i],bt=bt,sz=sz)
        