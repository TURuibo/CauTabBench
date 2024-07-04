import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

from causallearn.search.FCMBased import lingam
import numpy as np
import pandas as pd
from cdt.metrics import precision_recall
from cdt.metrics import SHD
from utils.utils import get_args



args = get_args()
seed =args.seed # 29
dataname = args.cm  # lg 
sz =args.sz # 15000
bt = args.bt  # 10
llm = args.llm # null
sim_seed = args.seed_sim  # 101
m_name = llm    

np.random.seed(seed)

def str_out(res):
    return f'\t & ${np.mean(res):.2f} \pm {np.std(res):.2f}$'

out_path = f'./result/{llm}/eva_lingam_{dataname}_{sim_seed}.txt'

with open(out_path, 'w') as file:
    file.write('\t& adj  '+ '\t & f1 '+ '\t & precision '+ '\t& recall \\\ \n')

p,r,s,f= [],[],[],[]
print(dataname, m_name, bt, sz)
for sim_seed in range(100,sim_seed):
    data_path = f'./synthetic/sim_{dataname}/{sim_seed}/{m_name}.csv'
    adj_path = f'./data/sim_{dataname}/{sim_seed}/generated_graph_target.csv'
    prn = []
    rec = []
    shd=[]    
    f1 = []
    for bt_i in range(bt):
        np.random.seed(bt_i)
        data = pd.read_csv(data_path)
        data.dropna(inplace=True)
        data = data.to_numpy()[:,:10]

        index = np.random.randint(len(data[:,0]), size=sz)

        adj_causallearn = pd.read_csv(adj_path).to_numpy()[:10,:10]

        model = lingam.DirectLiNGAM()
        model.fit(data[index,:])
                
        _, curve = precision_recall(((model.adjacency_matrix_!=0).T)*1.0, adj_causallearn)
        precision,recall= curve[1]
        prn.append(precision)
        rec.append(recall)
        f1_i = 2*prn[-1]*rec[-1]/(prn[-1]+rec[-1])
        if np.isnan(f1_i):
            print('Non',f1_i,prn[-1],rec[-1])
        else:
            f1.append(f1_i)
        shd.append(SHD(((model.adjacency_matrix_!=0).T)*1.0, adj_causallearn))
    p.append(np.mean(prn))
    r.append(np.mean(rec))
    f.append(np.mean(f1))
    s.append(np.mean(shd))

with open(out_path, 'a') as file:
    file.write(f'{m_name}'+ str_out(s) + str_out(f) + str_out(p) + str_out(r)+ '\\\ \n')