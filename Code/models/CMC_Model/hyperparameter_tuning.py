import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,classification_report
from generateSplits import generateSplits
import pandas as pd
from trainModel import trainModel
from Dataset import ModelDataset
from model import Model
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

metadata = pd.read_csv("Datasets/BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv")

train_df,val_df = generateSplits(metadata,0.2,seed=42)

train_df = train_df[["pid","pCR","ER","PR","HER2"]].set_index("pid",drop=True)
val_df = val_df[["pid","pCR","ER","PR","HER2"]].set_index("pid",drop=True)


skf = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)

def evaluate(model:torch.nn.Module,val_loader:torch.utils.data.DataLoader):
    model.eval()
    model.to(device)
    
    y_score = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images,mols,labels in val_loader:
            images = images.to(device)
            mols = mols.to(device)
            logits = model(images,mols)
            scores = torch.nn.functional.softmax(logits,dim=1)[:,1]
            y_score.extend(scores.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    return roc_auc_score(y_true,y_score)
            
        


def objective(trial:optuna.Trial):
    
    optimisers = {"Adam":torch.optim.Adam,"SGD":torch.optim.SGD}
    lr = trial.suggest_float("lr",1e-5,1e-3,log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0.1, 0.01, 0.001, 0.0001])
    batch_size = trial.suggest_categorical("batch_size",[2,4,8,16])
    optimiser_name = trial.suggest_categorical("optimiser_name", ["Adam", "SGD"])
    optimiser_class = optimisers[optimiser_name]
    
    scores = []
    
    for train_index,val_index in skf.split(train_df, train_df["pCR"]):
        fold_train_df = train_df.iloc[train_index]
        fold_val_df = train_df.iloc[val_index]
        fold_train_dataset = ModelDataset(fold_train_df,class_samples={0:3,1:8})
        fold_train_loader = DataLoader(fold_train_dataset,batch_size=batch_size,shuffle=True)
        model = Model()
        optimiser = optimiser_class(model.parameters(),lr=lr,weight_decay=weight_decay)
        model = trainModel(model,fold_train_loader,optimiser,device=device,num_epochs=10)
        fold_val_dataset = ModelDataset(fold_val_df,class_samples={0:1,1:1})
        fold_val_loader = DataLoader(fold_val_dataset,batch_size=batch_size)
        score = evaluate(model,fold_val_loader)
        scores.append(score)
    avg_score = sum(scores)/len(scores)
    return avg_score



study = optuna.create_study()

study.optimize(objective,n_trials=5)