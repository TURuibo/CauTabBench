import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from card_gt.eva_tab_gen.eva_ci_sets_llm import *
from utils.utils import get_args

def one_hot_encode_to_boolean(number, num_classes=10):
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

    data = data_df.iloc[:,:10]
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

def mae_mean(gcm_gt, causal_model_gt,gcm_syn, causal_model_syn,sz=1000):
    intervention_ls = np.random.randn(10)*5
    mae_dims = []
    for inv_dim in range(10):
        for itvn in intervention_ls:
            AE = np.abs(np.mean(interv_gen(gcm_gt, causal_model_gt,inv_dim,itvn,sz),axis=0)-np.mean(interv_gen(gcm_syn, causal_model_syn,inv_dim,itvn,1000),axis=0))
            MAE_i = np.mean(AE[one_hot_encode_to_boolean(inv_dim)])
            mae_dims.append(MAE_i)
    return np.mean(mae_dims)


if __name__ == "__main__":

    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lg 
    sz =args.sz # 15000
    bt = args.bt  # 10
    llm = args.llm # null
    sim_seed = args.seed_sim  # 101

    np.random.seed(seed)
    
    dataname= 'sim_' + dataname # lg lu sg nn
    model = llm
    
    result_path = f'./result/{llm}/eva_itvn_err_{dataname}_mean.txt'
    with open(result_path, 'w') as file:
        file.write(f'{llm}\n')   
            

    mae_m = []
    for sindex in range(100,sim_seed):
        data_path = f'./data/{dataname}/{sindex}/generated_graph_data.csv'
        data_df = pd.read_csv(data_path)
        adj_path = f'./data/{dataname}/{sindex}/generated_graph_target.csv'
        adj_df = pd.read_csv(adj_path)
        gcm_gt ,causal_model_gt, causal_graph_gt = train_scm_model(adj_df,data_df)

        syn_path = f'./synthetic/{dataname}/{sindex}/{model}.csv'
        syn_df = pd.read_csv(syn_path)
        gcm_syn ,causal_model_syn, causal_graph_syn = train_scm_model(adj_df,syn_df)
        mae_m.append(mae_mean(gcm_gt, causal_model_gt,gcm_syn, causal_model_syn,sz))

    with open(result_path, 'a') as file:
        file.write(f'& ${np.mean(mae_m)*100:.2f} \pm {np.std(mae_m)*100:.1f}$\n')

