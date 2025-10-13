from generateSplits import generateSplits,metadata
import pandas as pd 

t1,v1 = generateSplits(metadata,0.2,max_pids=None, seed=42)
t2,v2 = generateSplits(metadata,0.2,max_pids=None, seed=42)

print(t1.equals(t2),v1.equals(v2))

