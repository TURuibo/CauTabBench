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



def mae_mean_cf(gcm_gt, causal_model_gt,gcm_syn, causal_model_syn,data_df,sz=1000):
    intervention_ls = np.random.randn(10)*5
    mae_dims = []
    np.random.seed(77)

    for inv_dim in range(10):
        v_o = np.random.randint(0, high=10, size=1, dtype=int)[0]
        intervention_ls = np.random.randn(10)*5
        itvn = intervention_ls[inv_dim]
        data = data_df.iloc[:1,:10]
        data.columns = list(causal_graph_gt.nodes)

        AE = np.abs(np.mean(cf_gen(gcm_gt, causal_model_gt,inv_dim,itvn,data),axis=0)-\
                    np.mean(cf_gen(gcm_syn, causal_model_syn,inv_dim,itvn,data),axis=0))
        
        r = AE[v_o]  
        mae_dims.append(r)
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
    
    result_path = f'./result/{llm}/eva_cf_err_{dataname}_mean.txt'
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
        
        # Reference 
        # n_rows,_ = syn_df.shape
        # n_rows_totoal,_ = data_df.shape
        # index = np.random.randint(n_rows_totoal, size=(n_rows))
        # syn_df = data_df.loc[index,:]

        gcm_syn ,causal_model_syn, causal_graph_syn = train_scm_model(adj_df,syn_df)
        
        test_path = f'./synthetic/{dataname}/{sindex}/test.csv'
        data = pd.read_csv(test_path)
        data = data.iloc[:,:10]
        data.columns = list(causal_graph_gt.nodes)
        mae_m.append(mae_mean_cf(gcm_gt, causal_model_gt,gcm_syn, causal_model_syn,data,sz))
    
    print(f'& ${np.mean(mae_m)} \pm {np.std(mae_m)}$\n')
    with open(result_path, 'a') as file:
        file.write(f'& ${np.mean(mae_m)} \pm {np.std(mae_m)}$\n')
