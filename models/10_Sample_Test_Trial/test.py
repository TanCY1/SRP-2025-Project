import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../Code",))
sys.path.append(path)


from model import LitModel
from lightning import Trainer
from torch.utils.data import DataLoader
from Dataset import ModelDataset
import pandas as pd
from generateSplits import generateSplits

model = LitModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=99-step=400.ckpt")

model.eval()

train_df, val_df = generateSplits(metadata=pd.read_csv("Datasets/BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv"),test_size=0.0,number_of_phases=3)
train_df = train_df[["pid","pCR","ER","PR","HER2"]].set_index("pid",drop=True)

loader = DataLoader(ModelDataset(train_df),batch_size=2,num_workers=0)

trainer = Trainer(max_epochs=10,log_every_n_steps=1)
trainer.test(model,dataloaders = loader)

