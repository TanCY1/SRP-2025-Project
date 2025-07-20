from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from generateProcessedSamples import generateProcessedSamples

class ModelDataset(Dataset):
    def __init__(self, train_metadata:pd.DataFrame,class_samples:dict={0.0:1,1.0:1}):
        self.df = train_metadata
        self.pids = self.df.index.to_list()
        self.allProcessedSamples = dict()
        self.entries = list()
        for pid in self.pids:
            label = self.df.loc[pid,"pCR"]
            num_angles = class_samples[label]
            self.allProcessedSamples[pid] = generateProcessedSamples(pid,num_angles)
            self.entries.extend([(pid, i) for i in range(num_angles)])
        #print(self.entries)
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, angle_idx = self.entries[idx]
        image = self.allProcessedSamples[pid][angle_idx]
        image = torch.tensor(image,dtype=torch.float32)
        mols = torch.tensor(self.df.loc[pid,["ER","PR","HER2"]],dtype=torch.float32)
        label = torch.tensor(self.df.loc[pid,"pCR"],dtype=torch.long)
        return image, mols, label



