import numpy as np
import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import pandas as pd
import numpy as np

from utils.utils import get_args
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
def save_results(dataname,seed_sim,response_ls,response_adj_ls,result_path,prefix,questions):
    with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_i1_{dataname}{seed_sim}.txt', 'a') as file:
        for response,question in zip(response_ls,questions):
            file.write(f'Question is {question}\n Answer is {response}\n')
    
    with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_i2_{dataname}{seed_sim}.txt', 'a') as file:
        for response_adj,question in zip(response_adj_ls,questions):
            file.write(f'{response_adj}\n')        

def create_output_fiels(dataname,seed_sim,result_path,prefix):
    with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_i1_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')
    with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_i2_{dataname}{seed_sim}.txt', 'w') as file:
        file.write('')

def load_gt_answer(graph_id,result_path,llm,prefix):
    with open(cwd+f'/eval_llms/data/{prefix}/{graph_id}_answers.txt', 'r') as file:
        answers = file.readlines()
    ans = []
    
    for i in range(len(answers)):
        if answers[i].strip().lower() == 'yes':
            ans.append(1)
        else:
            ans.append(0)
    return ans

def load_llm_answer(graph_id,result_path,llm,prefix):
    with open(cwd+f'/eval_llms{result_path}/{llm}/{prefix}_dsep_response_i2_lu{graph_id}.txt', 'r') as file:
        answers = file.readlines()
    
    with open(cwd+f'/eval_llms{result_path}/{llm}/{prefix}_dsep_response_i2_lu{graph_id}_preds.txt', 'w') as file:
        print(cwd+f'/eval_llms{result_path}/{llm}/{prefix}_dsep_response_i2_lu{graph_id}_preds.txt')
        file.write('')

    ans = []
    with open(cwd+f'/eval_llms/{result_path}/{llm}/{prefix}_dsep_response_i2_lu{graph_id}_preds.txt', 'a') as file:
        for i in range(len(answers)):
            if answers[i].lower().find('yes') != -1:
                ans.append(1)
                file.write('1\n')
            elif answers[i].lower().find('no') != -1:
                ans.append(0)
                file.write('0\n')
            else:
                rand_num = np.random.randint(2, size=1)[0]
                ans.append(rand_num)
                file.write(f'{rand_num}\n')
    return ans

if __name__ == "__main__":
    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lu 
    llm = args.llm # qwen
    input_type =  args.input_type
    bt = args.bt  # 5
    np.random.seed(seed)

    labels = []
    preds = []
    fig, ax_roc= plt.subplots(1, 1, figsize=(6, 6))
    ax_roc.set_title(f"ROC curves ({dataname})")
    for seed_sim in range(100,110):
        all_lables = load_gt_answer(seed_sim,'/results',llm,input_type)
        for i in all_lables:
            labels.append(i)
        all_preds = load_llm_answer(seed_sim,'/results',llm,input_type)
        for i in all_preds:
            preds.append(i)
        print(len(all_lables),len(all_preds))
        print('\n')
    
    roc = RocCurveDisplay.from_predictions(labels, preds, name=llm,ax=ax_roc)
    auc = roc.roc_auc
    plt.legend()
    plt.savefig(cwd+f'/eval_llms/results/{llm}/{dataname}.pdf')
    print(f'{auc:.3f}')
