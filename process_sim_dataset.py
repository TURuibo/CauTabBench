import os,sys
import pathlib
from utils.acyclic_graph_generator import AcyclicGraphGenerator
import numpy as np
import pandas as pd
from utils.causal_mechanisms import normal_noise, uniform_noise

path = os.getcwd()
sys.path.append(path)
from utils.process_dataset import process_data
from utils.utils import get_args


if __name__ == "__main__":
    args = get_args()
    seed =args.seed
    sim_ls = args.cm
    n_nodes = args.n_nodes  # default: 10
    sz = args.sz  # default: 19019
    noise_coeff = args.noise_coeff  # default: 0.4
    n_prt = args.n_prt  # default: 2
    n_d = 2

    i = sim_ls
    np.random.seed(seed)
    if i == 'lg':
        generator = AcyclicGraphGenerator('linear', 
                                          initial_variable_generator=normal_noise, 
                                          noise_coeff=noise_coeff, 
                                          nodes=n_nodes, 
                                          npoints=sz,
                                          parents_max=n_prt,
                                          expected_degree=n_d)
    elif i == 'lu':
        generator = AcyclicGraphGenerator('linear',
                                          noise='uniform', 
                                          initial_variable_generator=uniform_noise, 
                                          noise_coeff=noise_coeff, 
                                          nodes=n_nodes, 
                                          npoints=sz,
                                          parents_max=n_prt,
                                          expected_degree=n_d)
    elif i == 'sg':
        generator = AcyclicGraphGenerator('sigmoid_add',
                                          noise='gaussian', 
                                          initial_variable_generator=normal_noise,
                                          noise_coeff=noise_coeff, 
                                          nodes=n_nodes, 
                                          npoints=sz,
                                          parents_max=n_prt,
                                          expected_degree=n_d)
    elif i == 'nn':
        generator = AcyclicGraphGenerator('nn',
                                          noise='gaussian', 
                                          initial_variable_generator=normal_noise, 
                                          noise_coeff=noise_coeff, 
                                          nodes=n_nodes, 
                                          npoints=sz,
                                          parents_max=n_prt,
                                          expected_degree=n_d)
    data, graph = generator.generate()
    
    path = pathlib.Path('./data/sim_'+i+f'/{seed}')
    path.mkdir(parents=True, exist_ok=True)
    generator.to_csv('./data/sim_'+i+f'/{seed}/generated_graph')
    
    data_path = './data/sim_'+i+f'/{seed}/generated_graph_data.csv'
    data_df = pd.read_csv(data_path)
    
    n, p = 1, .5  # number of trials, probability of each trial
    new_column_values = np.random.binomial(n, p, len(data_df.iloc[:,0]))
    data_df['target'] = new_column_values

    # Path where you want to save the CSV file
    file_path = './data/sim_'+i+f'/{seed}/generated_graph_data.csv'

    # Save the DataFrame as a CSV file
    data_df.to_csv(file_path, index=False)  # Set index=False to avoid writing row numbers as the first column
    print(seed)
    process_data('sim_'+i,seed)