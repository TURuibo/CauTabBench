import pandas as pd
from tabpfn import TabPFNRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


seeds = list(range(100,110))
models = ['tabsyn','stasy','tabddpm','codi','great','ctgan','tvae'] 
np.random.seed(0)

df_real = pd.read_csv("sachs_train.csv")
df_test = pd.read_csv("sachs_test.csv")
X_valid, y_valid = df_real.drop(["P38"],axis=1).values, df_real["P38"].values
X_test, y_test = df_test.drop(["P38"],axis=1).values, df_test["P38"].values

res_dict = {"seed": [], "model": [], "r2": [], "mae": [], "mse": []}

for m in models:
    for s in seeds:   
        print(m)
        print(s)
        df = pd.read_csv(f"synthetic/sachs/{s}/{m}.csv")
        best_mse = np.inf
        best_preds = None
        for i in tqdm(range(30)):
            df_sample = df.sample(10000)

            X, y = df_sample.drop(["P38"],axis=1).values, df_sample["P38"].values

            model = TabPFNRegressor()
            model.fit(X, y)
            valid_pred = model.predict(X_valid)
            valid_mse = mean_squared_error(valid_pred, y_valid)
            if valid_mse < best_mse:
                best_mse = valid_mse
                pred = model.predict(X_test)

        # pred = model.predict(X_test)
        res_dict["seed"].append(s)
        res_dict["model"].append(m)
        res_dict["r2"].append(r2_score(pred, y_test))
        res_dict["mae"].append(mean_absolute_error(pred, y_test))
        res_dict["mse"].append(mean_squared_error(pred, y_test))
        print(res_dict)


res_df = pd.DataFrame.from_dict(res_dict)
res_df.to_csv("tabpfn_res.csv", index=False)

