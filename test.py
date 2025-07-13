from generateCroppedPhases import generateAllCroppedPhases,cropPhases,stackPhases,get_vis_and_acq,normaliseOnePhase
import nibabel as nib
from viz import viz3D,viz3D_with_slider
import numpy as np
from upscale import upscale as up
from upscale import sharpen_3d
import pandas as pd

croppedPhases =np.load("croppedPhases.npz")

metadata = pd.read_csv("BreastDCEDL_spy1_metadata.csv")

vis_and_acq_dict = get_vis_and_acq()

pid="ISPY1_1222"

stacked  = stackPhases(pid, "1", vis_and_acq_dict)

cropped1 = cropPhases(stacked,pid,metadata)

viz3D_with_slider(croppedPhases[f"{pid}_vis1"])



