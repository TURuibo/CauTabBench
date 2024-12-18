import os,sys
cwd = os.path.abspath(os.path.curdir)
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))

sys.path.append(cwd)  # workplace
import numpy as np
import pandas as pd
from utils.utils import get_args
from src.utils import get_args,train_scm_model,mae_mean_cf


if __name__ == "__main__":

    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lg 
    sz =args.sz # 15000
    n_nodes = args.n_nodes
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
        data_path = parent_dir+f'/data/{dataname}/{sindex}/generated_graph_data.csv'
        data_df = pd.read_csv(data_path)
        adj_path = parent_dir+f'/data/{dataname}/{sindex}/generated_graph_target.csv'
        adj_df = pd.read_csv(adj_path)
        gcm_gt ,causal_model_gt, causal_graph_gt = train_scm_model(adj_df,data_df)

        syn_path = parent_dir+f'/synthetic/{dataname}/{sindex}/{model}.csv'
        syn_df = pd.read_csv(syn_path)
        gcm_syn ,causal_model_syn, causal_graph_syn = train_scm_model(adj_df,syn_df)
        
        test_path = parent_dir+f'/synthetic/{dataname}/{sindex}/test.csv'
        data = pd.read_csv(test_path)
        data = data.iloc[:,:10]
        data.columns = list(causal_graph_gt.nodes)
        mae_m.append(mae_mean_cf(gcm_gt, causal_model_gt,gcm_syn, causal_model_syn,data,sz,n_nodes))
    
    with open(result_path, 'a') as file:
        file.write(f'& ${np.mean(mae_m)*100:.2f} \pm {np.std(mae_m)*100:.1f}$\n')
