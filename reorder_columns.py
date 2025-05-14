import pandas as pd

dataset = "lg"
m_name_ls= ['real','tabsyn','stasy','tabddpm','codi','great','ctgan','tvae'] 
for seed in range(100,110):
    for m in m_name_ls:
        data_path = f"synthetic/sim_{dataset}/{seed}/{m}.csv"
        df = pd.read_csv(data_path)
        if len(df.columns) > 11:
            df = df.loc[:,['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'target']]
        else:
            df = df.loc[:,['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'target']]
        df.to_csv(data_path, index=False)