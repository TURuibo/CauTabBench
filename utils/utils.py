import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed',type=int, default=29)   
    parser.add_argument('--cm',type=str, default='lg')   
    parser.add_argument('--bt',type=int, default=10)   
    parser.add_argument('--seed_sim',type=int, default=101)   
    parser.add_argument('--llm',type=str, default='null')   
    parser.add_argument('--n_nodes',type=int, default=10)   
    parser.add_argument('--n_prt',type=int, default=2)   

    parser.add_argument('--sz',type=int, default=19019)   

    parser.add_argument('--noise_coeff',type=float, default=0.4)   

    parser.add_argument('--input_type',type=str, default='graph')   
    
    parser.add_argument('--task_type',type=str, default='null')   
    parser.add_argument('--max_table_rows',type=int, default=10)   
    parser.add_argument('--batch_size',type=int, default=10)
    parser.add_argument('--max_new_tokens',type=int, default=10000)   
    parser.add_argument('--prow_num',type=int, default=100)
    parser.add_argument('--temperature',type=float, default=0.6)  
    parser.add_argument('--top_p',type=float, default=0.9) 
    parser.add_argument('--result_path',type=str, default='/result')       
    args = parser.parse_args()

    return args



    