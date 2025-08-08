from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

if torch.cuda.is_available():
    from cupy_generateProcessedSamples import generateProcessedSamples
    device = torch.device("cuda")
else:
    from numpy_generateProcessedSamples import generateProcessedSamples
    device = torch.device("cpu")
class ModelDataset(Dataset):
    def __init__(self, metadata:pd.DataFrame,class_samples:dict={0.0:1,1.0:1},loading_bar=True,caching=False):
        self.df = metadata
        self.pids = self.df.index.to_list()
        self.allProcessedSamples = dict()
        self.entries = list()
        cache_dir = "cache"
        for pid in tqdm(self.pids,disable=not loading_bar):
            label = self.df.loc[pid,"pCR"]
            num_angles = class_samples[label]
            cache_path = os.path.join(cache_dir,f"{pid}_{num_angles}.npz")
            if os.path.exists(cache_path):
                data_np = np.load(cache_path)["data"]
                data = torch.tensor(data_np,device=device,dtype=torch.float32)
                
            else:
                if torch.cuda.is_available:
                    data = torch.from_dlpack(generateProcessedSamples(pid,num_angles)).to(torch.float32) # pyright: ignore[reportPrivateImportUsage]
                else:
                    data = torch.tensor(generateProcessedSamples(pid,num_angles),device=device,dtype=torch.float32)
                #os.makedirs(os.path.dirname(cache_path),exist_ok=True)
                np.savez_compressed(cache_path,data=data.cpu().numpy())
            self.allProcessedSamples[pid] = data
            #print(pid) if len(self.allProcessedSamples[pid])>3 else None
            self.entries.extend([(pid, i) for i in range(num_angles)])
            
        print(f"Dataset initialised with {len(self.entries)} entries.")
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, angle_idx = self.entries[idx]
        image = self.allProcessedSamples[pid][angle_idx]
        mols = torch.tensor(self.df.loc[pid,["ER","PR","HER2"]],dtype=torch.float32)
        label = torch.tensor(self.df.loc[pid,"pCR"],dtype=torch.long)
        return image, mols, label



