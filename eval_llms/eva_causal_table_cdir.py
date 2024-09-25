import numpy as np
import os,sys
cwd = os.path.abspath(os.path.curdir)
sys.path.append(cwd)  # workplace

from utils.utils import get_args

def load_llm_answer(graph_id,result_path,llm,prefix):
    with open(cwd+f'/eval_llms{result_path}/{llm}/{prefix}_cdir_response_lu{graph_id}.txt', 'r') as file:
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



def acc(res):
    pos = 0
    for res_i in res:
        if res_i > 0:
            pos+=1 
    acc_res = pos/len(res)
    return acc_res


if __name__ == "__main__":
    args = get_args()
    seed =args.seed # 29
    dataname = args.cm  # lg 
    sz =args.sz # 15000
    bt = args.bt  # 10
    llm = args.llm  # null
    sim_seed = args.seed_sim   # 101
    input_type =  args.input_type


    labels = []
    preds = []
    np.random.seed(seed)
    for seed_sim in range(1,11):        
        all_preds = load_llm_answer(seed_sim,'/result',llm,input_type)
        for i in all_preds:
            preds.append(i)
    results = acc(preds)
    print(results)