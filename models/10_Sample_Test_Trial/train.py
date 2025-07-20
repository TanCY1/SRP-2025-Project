
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../Code",))
sys.path.append(path)


import pandas as pd
from model import LitModel
from Dataset import ModelDataset
from torch.utils.data import DataLoader
from lightning import Trainer

train_metadata = pd.read_json("models/10_Sample_Test_Trial/train_metadata.json",orient="index")
val_metadata = pd.read_json("models/10_Sample_Test_Trial/validation_metadata.json",orient="index")

train_dataset = ModelDataset(train_metadata, class_samples={0.0:1,1.0:1})
val_dataset = ModelDataset(val_metadata, class_samples={0.0:1,1.0:1})

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)

model = LitModel()
trainer = Trainer(max_epochs=100,log_every_n_steps=1)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
