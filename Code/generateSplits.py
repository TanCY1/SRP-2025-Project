import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

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

def generateSplits(metadata:pd.DataFrame,test_size,number_of_phases,max_pids=None,seed=42):
    
    pids = (pid for pid,acqSet in acqData.items() if len(acqSet)==number_of_phases)

    metadata = metadata[metadata["pid"].isin(pids)]
    
    if max_pids is not None:
        pids:pd.Series = metadata["pid"]
        if max_pids>len(pids):
            raise ValueError
        selectedPids = pids.sample(n=max_pids, random_state=seed)
        metadata = metadata[metadata["pid"].isin(selectedPids)]
    stratify_vals = metadata["pCR"].to_numpy()
    train_pids,test_pids = train_test_split(
        metadata["pid"].to_numpy(),
        test_size = test_size,
        stratify = stratify_vals
    )
    
    train_df = metadata[metadata["pid"].isin(train_pids)].copy()
    val_df  = metadata[metadata["pid"].isin(test_pids)].copy()

    return train_df,val_df
    
def main():
    train_df, val_df = generateSplits(metadata,0.2,number_of_phases=3,max_pids=10)
    #print(train_df.columns)
    train_df = train_df[["pid","pCR","ER","PR","HER2"]].set_index("pid",drop=True)
    val_df = val_df[["pid","pCR","ER","PR","HER2"]].set_index("pid",drop=True)
    train_df.to_json("models/10_Sample_Test_Trial/train_metadata.json",orient="index",indent=4)
    val_df.to_json("models/10_Sample_Test_Trial/validation_metadata.json",orient="index",indent=4)
    
if __name__ == "__main__":
    main()