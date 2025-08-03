import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os
import torch

metadata = pd.read_csv("Datasets/BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv")

def getAcqData():
    data = dict()
    pattern = re.compile(r"^(?P<pid>ISPY1_\d+)_.+_vis(?P<vis>\d+)_acq(?P<acq>\d+)")
    for fname in os.listdir("Datasets/BreastDCEDL_spy1/spt1_dce"):
        match = pattern.search(fname)
        if match:
            groupdict = match.groupdict()
            pid = groupdict["pid"]
            vis = groupdict["vis"] 
            acq = groupdict["acq"]
            if vis!="1":
                raise
            if pid not in data:
                data[pid]=set()
            data[pid].add(acq)
    return data

acqData = getAcqData()

def generateSplits(metadata:pd.DataFrame,test_size,max_pids=None,seed=42):
    metadata = metadata[metadata["pid"].isin(pid for pid in acqData)]
    metadata = metadata.dropna(subset=["pCR","ER","PR","HER2"])
    if max_pids is not None:
        pids:pd.Series = metadata["pid"]
        if max_pids>len(pids):
            raise ValueError
        target_samples:pd.Series = (metadata["pCR"].value_counts(normalize=True)*max_pids).round().astype(int)
        
        # Ensure sum matches exactly
        diff = max_pids - target_samples.sum()
        if diff > 0:
            target_samples[target_samples.idxmax()] += diff
        elif diff < 0:
            target_samples[target_samples.idxmin()] -= diff
            
        metadata = pd.concat([v.sample(n=target_samples[k],replace=False,random_state=seed) for k,v in metadata.groupby("pCR")])
    if test_size == 0:
        return metadata,pd.DataFrame()
    
    stratify_vals = metadata["pCR"].to_numpy()
    
    train_pids,test_pids = train_test_split(
        metadata["pid"].to_numpy(),
        test_size = test_size,
        random_state=seed,
        stratify = stratify_vals
    )
    
    train_df = metadata[metadata["pid"].isin(train_pids)].copy()
    val_df  = metadata[metadata["pid"].isin(test_pids)].copy()

    return train_df,val_df
    
def main():
    train_df, val_df = generateSplits(metadata,0.2,max_pids=None, seed=42)
    #print(train_df.columns)
    train_df = train_df[["pid","pCR","ER","PR","HER2"]].set_index("pid",drop=True)
    val_df = val_df[["pid","pCR","ER","PR","HER2"]].set_index("pid",drop=True)
    #print(train_df["pCR"].value_counts())
    #print(val_df["pCR"].value_counts())
    train_df.to_json("models/test/train_metadata.json",orient="index",indent=4)
    val_df.to_json("models/test/validation_metadata.json",orient="index",indent=4)
    
if __name__ == "__main__":
    main()