import pandas as pd
from sklearn.model_selection import train_test_split

metadata = pd.read_csv("Data/BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv")


def generateSplits(metadata:pd.DataFrame,test_size,max_pids=None,seed=42):
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
    test_df  = metadata[metadata["pid"].isin(test_pids)].copy()

    return train_df,test_df
    
def main():
    train_df, test_df = generateSplits(metadata,0.2,max_pids=10)
    #print(train_df.columns)
    train_df = train_df[["pid","pCR","ER","PR","HER2"]]
    test_df = test_df[["pid","pCR","ER","PR","HER2"]]
    

if __name__ == "__main__":
    main()