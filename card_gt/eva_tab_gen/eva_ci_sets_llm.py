import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import networkx as nx
import pandas as pd
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Dag import Dag
from causallearn.utils.GraphUtils import GraphUtils
import copy
import numpy as np
from causallearn.utils.cit import CIT

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from joblib import Parallel, delayed
import time

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from src.utils import get_args,adj2dag

def find_collider(dag,nodes,i,j):
    return list(set([int(n.get_name()) for n in dag.get_children(nodes[i])])&set([int(n.get_name()) for n in dag.get_children(nodes[j])]))


def get_num_nodes(adj_df):
    num_nodes = len(adj_df[0,:])
    return num_nodes


def get_sets(adj_df):
    dag,nodes = adj2dag(adj_df)

    gu= GraphUtils()
    direction_set =[]
    independent_set =[]
    conditional_independent_set =[]
    collider_set=[]

    columns = [str(i) for i in range(len(nodes))]
    dsep_df_real = pd.DataFrame([], columns=columns)
    collider_df = pd.DataFrame([], columns=columns)

    for i in range(len(nodes)):
        dsep_df_real.loc[i] = [[] for i in range(len(nodes))]

    for i in range(10):
        collider_df.loc[i] = [[] for i in range(10)]


    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):        
            res = gu.get_sepset(x=nodes[i],y=nodes[j],graph=dag)
            if res is None:
                direction_set.append((i,j))
                dsep_df_real.iloc[i,j] = -1 # dependent
            elif len(res)>0:
                dsep_df_real.iloc[i,j] = [int(r.get_name()) for r in res] # conditional independent
                conditional_independent_set.append((i,j,[int(r.get_name()) for r in res]))
            else:
                dsep_df_real.iloc[i,j] = 1  # independent
                independent_set.append((i,j))

    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):        
            res = find_collider(dag,nodes,i,j)
            if len(res) > 0 :
                collider_set.append((i,j,res))
            collider_df.iloc[i,j] = res # conditional independent
    
    conditional_dependent_set = get_conditional_dependent_set(conditional_independent_set)

    return direction_set,independent_set,conditional_independent_set,collider_set,conditional_dependent_set


def get_conditional_dependent_set(conditional_independent_set):
    conditional_independent_set_copy = copy.deepcopy(conditional_independent_set)
    if len(conditional_independent_set_copy) == 0:
        return []
    else: 
        [ (ci_i[0],ci_i[1],ci_i[2].pop(0)) for ci_i in conditional_independent_set_copy]
        return conditional_independent_set_copy
    
def check_d_sep(adj_df,test_set):
    G = nx.from_numpy_array(adj_df, create_using=nx.DiGraph)
    nodes = list(G.nodes)
    res = []
    for i in range(len(test_set)):
        res.append(nx.d_separated(G, 
                                set([nodes[test_set[i][0]]]), 
                                set([nodes[test_set[i][1]]]), 
                                set([nodes[index] for index in test_set[i][2]])) )
    return res


def check_minimal_dsep(adj_df,conditional_independent_set):
    G = nx.from_numpy_array(adj_df, create_using=nx.DiGraph)
    nodes = list(G.nodes)
    res = []
    for i in range(len(conditional_independent_set)):
        res.append(nx.is_minimal_d_separator(G, 
                                nodes[conditional_independent_set[i][0]], 
                                nodes[conditional_independent_set[i][1]], 
                                set([nodes[index] for index in conditional_independent_set[i][2]])) )
    return res

def eva_ci(conditional_independent_set,data,dataname,sz):
    np.random.seed(100)
    # index = np.random.randint(len(data.to_numpy()[:,0]), size=sz)
    index = np.random.choice(len(data.to_numpy()[:,0]), sz, replace=False)
    if dataname == 'lg' or dataname == 'lu':
        fisherz_obj = CIT(data.to_numpy()[index,:], "fisherz") # construct a CIT instance with data and method name
    else:
        fisherz_obj = CIT(data.to_numpy()[index,:], "kci") # construct a CIT instance with data and method name kci
    p_ls=[]
    for i in range(len(conditional_independent_set)):
        X,Y,S = conditional_independent_set[i]
        pValue = fisherz_obj(X, Y, np.array(S))
        p_ls.append(pValue)
    return p_ls

def compute_p_vals(conditional_independent_set,dataname,sim_seed,m_name,sz):
    data_path = f'./synthetic/sim_{dataname}/{sim_seed}/{m_name}.csv'   
    data = pd.read_csv(data_path)
    data = data.iloc[:,:10]
    data.dropna(inplace=True)

    # data_path = f'./data/sim_{dataname}/{sim_seed}/generated_graph_data.csv'
    # data_df = pd.read_csv(data_path)
    # n_rows,_ = data.shape
    # n_rows_totoal,_ = data_df.shape
    # index = np.random.randint(n_rows_totoal, size=(n_rows))
    # data = data_df.iloc[index,:10]
    # sz = n_rows

    p_real = eva_ci(conditional_independent_set,data,dataname,sz)
    
    return p_real


def run_cit(dataname,m_name,sz,sim_seed):
    adj_path = f'./data/sim_{dataname}/{sim_seed}/generated_graph_target.csv'
    adj_df = pd.read_csv(adj_path)
    _,_,conditional_independent_set,_,conditional_dependent_set = get_sets(adj_df.to_numpy())    
    p_ls_dsep = compute_p_vals(conditional_independent_set,dataname,sim_seed,m_name,sz)
    p_ls_dcon = compute_p_vals(conditional_dependent_set,dataname,sim_seed,m_name,sz)
    return p_ls_dsep,p_ls_dcon

def get_label_pred(dataname,sim_seed,m_name,sz):
    p_ls_dsep = []
    p_ls_dcon = []
    wrapped_res  = Parallel(n_jobs=-2)(delayed(run_cit)(dataname,m_name,sz,sindex) for sindex in range(100,sim_seed))    # [causallearn_cg_ls,...]
    for res_i in wrapped_res:
        p_ls_dsep_i, p_ls_dcon_i = res_i
        p_ls_dsep += p_ls_dsep_i
        p_ls_dcon += p_ls_dcon_i

    pred = np.array(p_ls_dsep+p_ls_dcon)
    labels = np.ones(len(pred))
    labels[len(p_ls_dsep):] =  0
    return labels, pred

if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    seed =args.seed
    dataname = args.cm  # lg 
    sz =args.sz # 15000
    sim_seed = args.seed_sim  # 101
    llm = args.llm # null
    
    print(dataname, llm)

    np.random.seed(seed)
    result_roc_path = f'./result/{llm}/eva_ci_auc_{dataname}_{sim_seed}.txt'
    
    with open(result_roc_path, 'w') as file:
        file.write('\n')
    
    fig, ax_roc= plt.subplots(1, 1, figsize=(6, 6))
    ax_roc.set_title(f"ROC curves ({dataname})")

    m_name = llm
    labels, preds = get_label_pred(dataname,sim_seed=sim_seed,m_name = m_name,sz=sz)
    roc = RocCurveDisplay.from_predictions(labels, preds, name=m_name,ax=ax_roc)
    auc = roc.roc_auc

    plt.legend()
    plt.savefig(f'./result/{llm}/{dataname}.pdf')
    print(f'{auc:.3f}')
    with open(result_roc_path, 'a') as file:
        file.write(f'{auc:.3f}')
    
    print("--- %s seconds ---" % (time.time() - start_time))


