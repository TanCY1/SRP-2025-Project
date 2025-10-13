from sklearn.model_selection import train_test_split
import pandas as pd

def generateSplits(df:pd.DataFrame,test_size,max_pids=None,seed=42) -> tuple[pd.DataFrame,pd.DataFrame]:
    if max_pids is not None:
        if len(df)<max_pids:
            raise ValueError("max_pids cannot be greater than length of df")
        _,df = train_test_split(df,
                                test_size = max_pids,
                                stratify=df["pCR"],
                                random_state=seed)
        
    train_df, test_df = train_test_split(df,
                                         test_size=test_size,
                                         stratify=df["pCR"],
                                         random_state=seed)
    return train_df,test_df


