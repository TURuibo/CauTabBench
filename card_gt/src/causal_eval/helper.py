import re
import numpy as np
import networkx as nx
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd
import copy


def add_v_before_number(text):
    # This regex matches numbers that either don't have a 'V' before them, or aren't part of a 'V+number' pattern
    modified_text = re.sub(r'\b(?<!V)(\d+)\b', r'V\1', text)
    return modified_text

def get_adjacency_matrice(file_dir,n_nodes):
    pattern_node = r'V\d+'
    adj = np.zeros((n_nodes,n_nodes))

    # Open the file in read mode
    with open(file_dir, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Process the line (strip is used to remove the newline character)
            text = line.strip()
            text = add_v_before_number(text)

            line = re.sub(r'[^a-zA-Z0-9]', '', text)


            # Print the extracted matches

            neighbors = re.findall(pattern_node, line)
            for i in range(len(neighbors)):
                if i == 0:
                    ni = re.sub(r'[^0-9]', '', neighbors[i])
                    row_index = int(ni)
                else: 
                    ni = re.sub(r'[^0-9]', '', neighbors[i])
                    col_index = int(ni)
                    if row_index < n_nodes & col_index < n_nodes:
                        adj[row_index,col_index]  = 1
    
    return adj


def find_collider(dag,nodes,i,j):
    return list(set([int(n.get_name()) for n in dag.get_children(nodes[i])])&set([int(n.get_name()) for n in dag.get_children(nodes[j])]))


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

def get_conditional_dependent_set(conditional_independent_set):
    conditional_independent_set_copy = copy.deepcopy(conditional_independent_set)
    if len(conditional_independent_set_copy) == 0:
        return []
    else: 
        [ (ci_i[0],ci_i[1],ci_i[2].pop(0)) for ci_i in conditional_independent_set_copy]
        return conditional_independent_set_copy

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

    for i in range(len(nodes)):
        collider_df.loc[i] = [[] for i in range(len(nodes))]


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
