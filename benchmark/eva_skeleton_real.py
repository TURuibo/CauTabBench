import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace
import sys
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from cdt.metrics import precision_recall
from cdt.metrics import get_CPDAG,SHD
from cdt.data import AcyclicGraphGenerator
from causallearn.search.ConstraintBased.PC import pc
import os
from causallearn.utils.cit import kci
from joblib import Parallel, delayed

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
    if dataname == '_sg' or dataname == '_nn': 
        causallearn_cg_ls  = Parallel(n_jobs=-2)(delayed(run_pc_nonlinear)(data,total_sz,sz,bt_i) for bt_i in range(bt))    # [causallearn_cg_ls,...]
    else:
        causallearn_cg_ls  = Parallel(n_jobs=-2)(delayed(run_pc_linear)(data,total_sz,sz,bt_i) for bt_i in range(bt))    # [causallearn_cg_ls,...]
    
    for i in range(bt):
        skel_diff_causallearn_i,precision_i,recall_i =test(causallearn_cg_ls[i],adj_gt)    
        skel_diff_causallearn.append(skel_diff_causallearn_i)
        precision_ls.append(precision_i)
        recall_ls.append(recall_i)
    return skel_diff_causallearn,precision_ls,recall_ls


def eva_skel_ratio(dataname = '_lg',m_name = 'tabsyn',bt=10,sz=1000,seed=7):
    
    skel_syn_ls = []
    f1_syn_ls=[]
    precision_syn_ls = []
    recall_syn_ls = []

    # for seed_sim in [0]: # [100, 102,104,106,108]: ##['1','2','29']: [100, 102,104,106,108]
    seed_sim = 0
    data_path = f'./synthetic/{dataname}/{seed_sim}/real.csv'
    
    data_real = pd.read_csv(data_path)
    if dataname == 'beijing':
        data_real = data_real.drop(columns=['year','month','day','hour','cbwd'])
        data_real = data_real.dropna(how='any')
    elif dataname == 'magic':
        data_real = data_real.dropna(how='any')
        data_real = data_real.drop(columns=['class'])
    elif dataname == 'jm':
        data_real = data_real.drop(columns=['defects'])
        data_real = data_real.dropna(how='any')
    elif dataname == 'parkinsons':
        data_real = data_real.dropna(how='any')
    elif dataname == 'house':
        data_real = data_real.dropna(how='any')
        data_real = data_real.drop(columns=['binaryClass'])

    cg_real = pc(data_real.to_numpy(),show_progress=False)

    cg_real.to_nx_graph()
    cg_real.to_nx_skeleton()
    adj_gt = nx.to_numpy_array(cg_real.nx_skel,  weight=None)

    data_sim_path = f'./synthetic/{dataname}/{seed_sim}/{m_name}.csv'

    # Method 2: Using os.path.isfile()
    if os.path.isfile(data_sim_path): 
        data_sim = pd.read_csv(data_sim_path)
    
        if dataname == 'beijing':
            data_sim = data_sim.drop(columns=['year','month','day','hour','cbwd'])
        elif dataname == 'magic':
            data_sim = data_sim.drop(columns=['class'])
        elif dataname == 'jm':
            data_sim = data_sim.drop(columns=['defects'])
        elif dataname == 'house':
            data_sim = data_sim.dropna(how='any')
            data_sim = data_sim.drop(columns=['binaryClass'])

            

        skel_syn_ls,precision_syn_ls,recall_syn_ls = metric_eva(data_sim,adj_gt,dataname,bt=bt,sz=sz,seed=seed)
        f1_syn_ls=[]
        for i in range(len(precision_syn_ls)):
            f1_i = 2*precision_syn_ls[i]*recall_syn_ls[i]/(precision_syn_ls[i]+recall_syn_ls[i])
            if np.isnan(f1_i):
                pass
            else:
                f1_syn_ls.append(f1_i)

        
        with open(f'./result/eva_real_{dataname}.txt', 'a') as file:
            file.write(f'{m_name}'+\
                        f'\t &  ${np.mean(skel_syn_ls):.2f}'+'\pm '+f'{np.std(skel_syn_ls):.2f}$ '+ \
                        f'\t & ${np.mean(f1_syn_ls):.2f}'+'\pm '+f'{np.std(f1_syn_ls):.2f}$ '+ \
                        f'\t & ${np.mean(precision_syn_ls):.2f}'+'\pm '+f'{np.std(precision_syn_ls):.2f}$'+ \
                        f'\t & ${np.mean(recall_syn_ls):.2f}'+'\pm '+f'{np.std(recall_syn_ls):.2f}$ '+ '\n')
    else:
        print("File does not exist")

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 2 :
        dataname=sys.argv[1]  # _lg; _lu; _sg; _nn
    elif len(sys.argv) == 3 :
        dataname=sys.argv[1]  # _lg; _lu; _sg; _nn
        sz=int(sys.argv[2])  # number of samples for pc 
    elif len(sys.argv) == 4 :
        dataname=sys.argv[1] # _lg; _lu; _sg; _nn
        sz=int(sys.argv[2])  # number of samples for pc 
        bt=int(sys.argv[3])  # bootstrapping times
    else: 
        dataname='_lg'
        bt=10
        sz=50
    m_name_ls=['real','tabsyn','stasy','tabddpm','codi','great','ctgan','tvae']
    
    print(dataname,bt,sz)

    with open(f'./result/eva_real_{dataname}.txt', 'w') as file:
        file.write('\t\tadj  '+ '\t\tf1 '+ '\t\tprecision '+ '\t\trecall \n')

    for i in range(len(m_name_ls)):
        print(f'evaluating {m_name_ls[i]}')
        eva_skel_ratio(dataname = dataname,m_name = m_name_ls[i],bt=bt,sz=sz)
        