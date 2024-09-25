import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from eval_llms.eva_tab_gen.eva_ci_sets_llm import *
from utils.utils import get_args

def one_hot_encode_to_boolean(number, num_classes=51):
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


def cf_gen(gcm, causal_model, inv_data, obs_data):
    samples = gcm.counterfactual_samples(causal_model,inv_data,observed_data=obs_data)
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
    print(nrow)
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

def get_gt_intervention_distribution(gcm_gt, causal_model_gt,sim_seed,sz=1000):
    np.random.seed(77)
    intervention_ls = np.random.randn(51)*5

    with open(f'/Users/ruibo/Documents/Codes/CauTabBench/eval_llms/data/table/intervention_{sim_seed}_gt.txt', 'w') as file:
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
        data_path = f'/Users/ruibo/Documents/Codes/CauTabBench/data/{dataname}/{sindex}/generated_graph_data.csv'
        data_df = pd.read_csv(data_path)
        adj_path = f'/Users/ruibo/Documents/Codes/CauTabBench/data/{dataname}/{sindex}/generated_graph_target.csv'
        adj_df = pd.read_csv(adj_path)
        gcm_gt ,causal_model_gt, causal_graph_gt = train_scm_model(adj_df,data_df)

        get_gt_intervention_distribution(gcm_gt, causal_model_gt,sindex)
