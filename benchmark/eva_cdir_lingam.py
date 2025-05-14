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
seed =args.seed
np.random.seed(seed)
bt = args.bt
sz = args.sz
dataname = args.cm
m_name_ls=['real','tabsyn','stasy','tabddpm','codi','great','ctgan','tvae']


def str_out(res):
    return f'\t & ${np.mean(res):.2f} \pm {np.std(res):.2f}$'

with open(f'./result/eva_lingam_{dataname}.txt', 'w') as file:
    file.write('\t& adj  '+ '\t & f1 '+ '\t & precision '+ '\t& recall \\\ \n')

for m_name in m_name_ls:
    p,r,s,f= [],[],[],[]
    print(m_name)
    for sim_seed in range(100,110):
        data_path = f'./synthetic/sim_{dataname}/{sim_seed}/{m_name}.csv'
        adj_path = f'./data/sim_{dataname}/{sim_seed}/generated_graph_target.csv'
        prn = []
        rec = []
        shd=[]    
        f1 = []
        for bt_i in range(bt):
            np.random.seed(bt_i)
            data = pd.read_csv(data_path)
            data = data.to_numpy()[:,:data.shape[1]-1]
            index = np.random.randint(len(data[:,0]), size=sz)

            adj_causallearn = pd.read_csv(adj_path).to_numpy()[:data.shape[1],:data.shape[1]]

            model = lingam.DirectLiNGAM()
            model.fit(data[index,:])
            # res_dir.append(np.sum(((model.adjacency_matrix_!=0).T)*1.0-adj_causallearn))
                    
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

    with open(f'./result/eva_lingam_{dataname}.txt', 'a') as file:
        file.write(f'{m_name}'+ str_out(s) + str_out(f) + str_out(p) + str_out(r)+ '\\\ \n')