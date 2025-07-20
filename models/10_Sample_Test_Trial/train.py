
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__),"../..",))
sys.path.append(path)


import pandas as pd
from model import LitModel
from Dataset import ModelDataset

train_metadata = pd.read_json("models/10_Sample_Test_Trial/train_metadata.json",orient="index")
print(train_metadata)