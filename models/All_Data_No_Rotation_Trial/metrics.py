

import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../Code",))
sys.path.append(path)
import pandas as pd
from model import Model
from Dataset import ModelDataset
from torch.utils.data import DataLoader
import torch
from generateSplits import generateSplits
import pandas as pd

device = torch.device("cpu")

metadata = pd.read_csv("Datasets/BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv")
Model = Model()

Model.to(device)

train_metadata = pd.read_json("models/All_Data_No_Rotation_Trial/train_metadata.json",orient="index")
val_metadata = pd.read_json("models/All_Data_No_Rotation_Trial/validation_metadata.json",orient="index")


from sklearn import metrics

#train_dataset = ModelDataset(train_metadata, class_samples={0.0:1,1.0:1})
val_dataset = ModelDataset(val_metadata, class_samples={0.0:1,1.0:1})

#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)

y_true = []
y_score = []
y_pred = []
Model.load_state_dict(torch.load("models/All_Data_No_Rotation_Trial/model_weights.pth"))

Model.eval()
with torch.no_grad():
    for images, mol, labels in val_loader:
        images = images.to(device)
        mol = mol.to(device)
        labels = labels.to(device)
        y_true.append(labels.item())
        logits:torch.Tensor = Model(images,mol)
        print(logits)
        score = torch.nn.functional.softmax(logits,dim=1)[:,1]
        pred = logits.argmax(dim=1).item()
        y_score.append(score)
        y_pred.append(pred)
import matplotlib.pyplot as plt

print(y_true,y_score)
fpr,tpr,threshold = metrics.roc_curve(y_true,y_score)
print(metrics.roc_auc_score(y_true,y_score))

plt.plot(fpr,tpr)
plt.show()

print(metrics.classification_report(y_true,y_pred))