import pandas as pd
from tabpfn import TabPFNRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x.squeeze()
    

class SachsDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.X = df.drop(["P38"],axis=1).values
        self.y = df["P38"].values

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

        
epochs = 100
batch_size = 128

seeds = list(range(100,110))
models = ['tabsyn','stasy','tabddpm','codi','great','ctgan','tvae'] 
np.random.seed(0)


df_valid = pd.read_csv("sachs_train.csv")
df_test = pd.read_csv("sachs_test.csv")

valid_set = SachsDataset(pd.read_csv("sachs_train.csv"))
test_set = SachsDataset(pd.read_csv("sachs_test.csv"))

valid_dl = DataLoader(valid_set, batch_size=128, shuffle=False)
test_dl = DataLoader(valid_set, batch_size=128, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



res_dict = {"seed": [], "model": [], "r2": [], "mae": [], "mse": []}

for m in models:
    for s in seeds:   

        print(m)
        print(s)
        train_set = SachsDataset(pd.read_csv(f"synthetic/sachs/{s}/{m}.csv"))
        train_dl = DataLoader(train_set, batch_size=128, shuffle=True)
        model = MLP(train_set[0][0].shape[0], 64, 1).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
        best_model = None
        best_val_mse = np.inf

        for e in tqdm(range(epochs)):
            for X, y in train_dl:
                X = X.to(device).float()
                y = y.to(device).float()

                optimizer.zero_grad()
                preds = model(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                # print(f"Batch loss {loss.item()}")
            
            val_labels = np.array([])
            val_preds = np.array([])
            for i, (X, y) in enumerate(valid_dl):
                X = X.to(device).float()
                y = y.to(device).float()

                with torch.no_grad():
                    preds = model(X)
                    val_labels = np.concat([val_labels, y.cpu().numpy()])
                    val_preds = np.concat([val_preds, preds.cpu().numpy()])
            

            val_mse = mean_squared_error(val_preds, val_labels)
            if val_mse <= best_val_mse:
                best_model = copy.deepcopy(model)


        test_labels = np.array([])
        test_preds = np.array([])
        for i, (X, y) in enumerate(test_dl):
            X = X.to(device).float()
            y = y.to(device).float()

            with torch.no_grad():
                preds = model(X)
                test_labels = np.concat([test_labels, y.cpu().numpy()])
                test_preds = np.concat([test_preds, preds.cpu().numpy()])

        
        test_mse = mean_squared_error(test_preds, test_labels)
        test_r2 = r2_score(test_preds, test_labels)
        test_mae = mean_absolute_error(test_preds, test_labels)

        res_dict["seed"].append(s)
        res_dict["model"].append(m)
        res_dict["r2"].append(test_r2)
        res_dict["mae"].append(test_mae)
        res_dict["mse"].append(test_mse)
        print(res_dict)


res_df = pd.DataFrame.from_dict(res_dict)
res_df.to_csv("mlp_res.csv", index=False)
