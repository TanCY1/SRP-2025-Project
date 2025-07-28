
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../Code",))
sys.path.append(path)


import pandas as pd
from model import model
from Dataset import ModelDataset
from torch.utils.data import DataLoader
import torch
from generateSplits import generateSplits
import pandas as pd
from trainModel import trainModel

device = torch.device("cuda")

metadata = pd.read_csv("Datasets/BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv")
model = model()

model.to(device)

train_metadata = pd.read_json("models/100_Sample_Test_Trial/train_metadata.json",orient="index")
val_metadata = pd.read_json("models/100_Sample_Test_Trial/validation_metadata.json",orient="index")

train_dataset = ModelDataset(train_metadata, class_samples={0.0:1,1.0:1})
val_dataset = ModelDataset(val_metadata, class_samples={0.0:1,1.0:1})

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)

optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

model = trainModel(model,train_loader,optimiser,device,10)

torch.save(model.state_dict(),"models/100_Sample_Test_Trial/model_weights.pth")

y_true = []
y_score = []
model.eval()
with torch.no_grad():
    for images, mol, labels in val_loader:
        images = images.to(device)
        mol = mol.to(device)
        labels = labels.to(device)
        y_true.append(labels.item())
        logits:torch.Tensor = model(images,mol)
        print(logits)
        preds = logits.argmax(dim=1).item()
        y_score.append(preds)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
print(y_true,y_score)
fpr,tpr,threshold = roc_curve(y_true,y_score)

plt.plot(fpr,tpr)
plt.show()
