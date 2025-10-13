import nibabel as nib

from viz import viz3D_with_slider

from cupy_generateProcessedSamples import getAcqData,stackPhases,getCentreOfMass
import matplotlib.pyplot as plt
import cupy as cp
from tqdm import tqdm
import numpy as np

acqData = getAcqData()

acqData = {pid:val for pid,val in acqData.items() if len(val)==4}

print(acqData)

signalCurves = []
for pid in tqdm(acqData):
    stackedPhases = stackPhases(pid,acqData).get()
    COM = cp.array(getCentreOfMass(pid))  # Replace with your own coordinates

    x,y,z = COM.round().astype(int).get()

    # Step 3: Extract signal over time (6 phases)
    signalCurves.append(stackedPhases[:, x, y, z])  # shape: (6,)
    del stackedPhases
    cp.get_default_memory_pool().free_all_blocks()
    
signal_curve = np.mean(signalCurves,axis=0)

print(signal_curve)
# Step 4: Plot it
plt.plot(range(4), signal_curve)
plt.title(f"Enhancement curve")
plt.xlabel("Phase")
plt.ylabel("Signal Intensity")
plt.grid(True)
plt.show()



#for six phases, use 0,2,5
#for four phases use 0,1,2


