from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import torch,os
from generateSamples import generateSamples
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ModelDataset(Dataset):
    def __init__(self,df:pd.DataFrame,dataset_path,class_samples={0:1,1:1},loading_bar=True):
        self.df = df
        self.pids = self.df.index
        self.processed_volumes=dict()
        self.entries = list()
        for pid in tqdm(self.pids,disable = not loading_bar):
            label = self.df.loc[pid,"pCR"]
            num_samples = class_samples[label]
            data = torch.tensor(np.array(generateSamples(pid,num_samples,dataset_path)))
            self.processed_volumes[pid] = data
            self.entries.extend([(pid, i) for i in range(num_samples)])
            
        print(f"Dataset initialised with {len(self.entries)} entries.")

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self,idx):
        pid,sample_idx = self.entries[idx]
        #print(self.processed_volumes[pid][sample_idx].shape)
        T0_volume,T3_volume = self.processed_volumes[pid][sample_idx]
        mols = torch.tensor(self.df.loc[pid,["HR","HER2"]].values,dtype=torch.float32,device=device)
        label = torch.tensor(self.df.loc[pid,"pCR"],dtype=torch.long,device=device)
        return T0_volume, T3_volume, mols, label
