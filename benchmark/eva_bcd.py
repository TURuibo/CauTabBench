import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import networkx as nx
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Dag import Dag
from causallearn.utils.GraphUtils import GraphUtils
from cdt.causality.pairwise import RECI,IGCI,CDS
import copy
import pandas as pd
import numpy as np
from scipy import stats
from utils.utils import get_args

def ini_nodes(adj_df):
    nodes = []
    for i in range(len(adj_df[0,:])):
        nodes.append(GraphNode(str(i)))
    return nodes

def adj2dag(adj_df):
    G = nx.from_numpy_array(adj_df, create_using=nx.DiGraph)
    nodes = ini_nodes(adj_df)
    dag = Dag(nodes)
    for i,j in list(G.edges()):
        dag.add_directed_edge(nodes[i], nodes[j])
    return dag,nodes

def remove_edge(index_x, index_y,nodes, dag):
    dag_rm = copy.deepcopy(dag)
    dag_rm.remove_connecting_edge(nodes[index_x], nodes[index_y])
    return dag_rm

def remove_edge_wnode(node_x, node_y,dag):
    dag_rm = copy.deepcopy(dag)
    dag_rm.remove_connecting_edge(node_x, node_y)
    return dag_rm

def get_all_xy_edges(dag,nodes):
    x_ls = []
    y_ls = []
    for e in list(dag.get_graph_edges()):        
        index_x = int(e.get_node1().get_name())
        index_y = int(e.get_node2().get_name())
        x_ls.append(index_x)
        y_ls.append(index_y)
    dir = np.array([x_ls,y_ls])
    return dir.T

def get_eva_xy_dirs(dag,nodes):
    x_ls = []
    y_ls = []
    for e in list(dag.get_graph_edges()):        
        index_x = int(e.get_node1().get_name())
        index_y = int(e.get_node2().get_name())
        dag_rm = remove_edge(index_x, index_y,nodes, dag)
        # print(e.get_node1(),e.get_node2(),dag_rm.is_dseparated_from(nodes[index_x],nodes[index_y],set()))
        if dag_rm.is_dseparated_from(nodes[index_x],nodes[index_y],set()):
            x_ls.append(index_x)
            y_ls.append(index_y)
    dir = np.array([x_ls,y_ls])
    return dir.T

def test_dir(dataname,sim_seed,m_name,eva_xy_dirs, test_method,bt=10,sz=10000):
    res = []
    for i in range(len(eva_xy_dirs[:,0])):
        indx,indy = eva_xy_dirs[i,:]
        res.append(xy_yx_bt(dataname,sim_seed,m_name,indx,indy,bt,sz,test_method))
    return res

def xy_yx_bt(dataname,sim_seed,m_name,index_x, index_y, bt=10,sz =10000,  func=RECI):
    outputxy_ls= []
    xy_dir = 0
    for bt_i in range(bt):
        outputxy = xy_yx(dataname,sim_seed,m_name,index_x, index_y, bt_i, sz, func)
        outputxy_ls.append(outputxy)
       
    if np.mean(outputxy_ls) > 0:
        xy_dir = 1
    return xy_dir,-0.1


def xy_yx(dataname,sim_seed,m_name,index_x, index_y, bt_i=0, sz =10000,  func=RECI):
    data_path = f'./synthetic/sim_{dataname}/{sim_seed}/{m_name}.csv'   
    data = pd.read_csv(data_path)
    
    np.random.seed(bt_i)
    index = np.random.randint(len(data.iloc[:,0]), size=sz)
    data = data.iloc[index,:10].to_numpy()

    d = {'A': [], 'B': []}
    dfxy = pd.DataFrame(data=d)
    
    data_x = data[:,index_x]
    data_y = data[:,index_y]
    dfxy.loc[0]= [data_x,data_y]
    obj = func()
    outputxy = obj.predict(dfxy)
    
    return outputxy


def acc(res):
    pos = 0
    for res_i in res:
        if res_i[0] > 0:
            pos+=1 
    acc_res = pos/len(res)
    return acc_res


if __name__ == "__main__":
    
    args = get_args()
    seed =args.seed
    dataname = args.cm
    sz =args.sz
    bt =args.bt

    np.random.seed(seed)

    acc_reci_ds,acc_igci_ds,acc_cds_ds=[],[],[]
    print(dataname)
    result_cdir_path = f'./result/eva_dir_acc_{dataname}.txt'

    with open(result_cdir_path, 'w') as file:
        file.write(' & ref & TabSyn & STASY & TaDDPM &CoDi&GReaT& CTGAN & TVAE \\\ \hline\n') # & CoDi & GReaT
    
    m_name_ls=['real' ,'tabsyn','stasy','tabddpm','codi','great','ctgan','tvae']  # 'codi','great',

    acc_reci_ls,acc_igci_ls,acc_cds_ls=[],[],[]
    for m_name in m_name_ls:
        print(m_name)
        res_igci = []
        res_reci = []
        res_cds = []
        for sim_seed in range(100,110):
            adj_path = f'./data/sim_{dataname}/{sim_seed}/generated_graph_target.csv'
            adj_df = pd.read_csv(adj_path)
            dag,nodes = adj2dag(adj_df.to_numpy())
            xy_edges = get_all_xy_edges(dag,nodes)
            eva_xy_dirs =  get_eva_xy_dirs(dag,nodes)

            res_igci += test_dir(dataname,sim_seed,m_name,eva_xy_dirs,IGCI, bt=bt,sz=sz)
            res_reci += test_dir(dataname,sim_seed,m_name,eva_xy_dirs,RECI, bt=bt,sz=sz)
            res_cds += test_dir(dataname,sim_seed,m_name,eva_xy_dirs, CDS, bt=bt,sz=sz)
        
        acc_reci,acc_igci,acc_cds = acc(res_reci),acc(res_igci),acc(res_cds)
        acc_reci_ls.append(acc_reci)
        acc_igci_ls.append(acc_igci)
        acc_cds_ls.append(acc_cds)
    acc_reci_ds.append(acc_reci_ls)
    acc_igci_ds.append(acc_igci_ls)
    acc_cds_ds.append(acc_cds_ls)
        

    with open(result_cdir_path, 'a') as file:
        file.write(f'{dataname} RECI: ')
        for a in acc_reci_ds[0]:
            file.write(f'&${a:.3f}$')
        file.write('\\\ \n')  
        file.write(f'{dataname} IGCI: ')
        for a in acc_igci_ds[0]:
            file.write(f'&${a:.3f}$')
        file.write('\\\ \n')
        file.write(f'{dataname} CDS: ')
        for a in acc_cds_ds[0]:
            file.write(f'&${a:.3f}$')
        file.write('\\\ \n')  
