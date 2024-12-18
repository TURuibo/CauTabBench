import numpy as np
import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

import numpy as np

from src.utils import get_args
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def load_gt_answer(graph_id,prefix):
    with open(cwd+f'/data/{prefix}/{graph_id}_answers.txt', 'r') as file:
        answers = file.readlines()
    ans = []
    
    for i in range(len(answers)):
        if answers[i].strip().lower() == 'yes':
            ans.append(1)
        else:
            ans.append(0)
    return ans

def load_llm_answer(graph_id,result_path,llm,prefix):
    with open(cwd+f'{result_path}/{llm}/{prefix}_dsep_response_lu{graph_id}.txt', 'r') as file:
        answers = file.readlines()

    ans = []
    for i in range(len(answers)):
        if answers[i].lower().find('yes') != -1:
            ans.append(1)
        elif answers[i].lower().find('no') != -1:
            ans.append(0)
        else:
            rand_num = np.random.randint(2, size=1)[0]
            ans.append(rand_num)
    return ans

if __name__ == "__main__":
    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lu 
    llm = args.llm # qwen
    input_type =  args.input_type
    np.random.seed(seed)

    labels = []
    preds = []
    fig, ax_roc= plt.subplots(1, 1, figsize=(6, 6))
    ax_roc.set_title(f"ROC curves ({dataname})")
    for seed_sim in range(1,11):            
        all_lables = load_gt_answer(seed_sim,input_type)
        for i in all_lables:
            labels.append(i)
        
        all_preds = load_llm_answer(seed_sim,'/result',llm,input_type)
        for i in all_preds:
            preds.append(i)
    
    roc = RocCurveDisplay.from_predictions(labels, preds, name=llm,ax=ax_roc)
    auc = roc.roc_auc
    plt.legend()
    plt.savefig(cwd+f'/result/{llm}/{dataname}.pdf')
    print(f'{auc:.3f}')
