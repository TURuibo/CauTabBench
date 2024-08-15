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
    parser.add_argument('--sz',type=int, default=15000)   
    parser.add_argument('--bt',type=int, default=10)   
    parser.add_argument('--seed_sim',type=int, default=101)   
    parser.add_argument('--llm',type=str, default='null')   

    parser.add_argument('--input_type',type=str, default='graph')   
    
    args = parser.parse_args()

    return args