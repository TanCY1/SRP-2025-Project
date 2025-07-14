import pandas as pd
import nibabel as nib
import re
import os
import numpy as np

#Start by making your active directory be Data\BreastDCEDL_spy1
#cd ../Data\BreastDCEDL_spy1

metadata = pd.read_csv("BreastDCEDL_spy1_metadata.csv")

def getAcqData():
    data = dict()
    pattern = re.compile(r"^(?P<pid>ISPY1_\d+)_.+_vis(?P<vis>\d+)_acq(?P<acq>\d+)")
    for fname in os.listdir("spt1_dce"):
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

def normaliseOnePhase(arr):
    # Normalise the data to range [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def stackPhases(pid,acqData):
    acqs = sorted(acqData[pid],key=int)
    volumes = []
    for acq in acqs:
        img:nib.nifti1.Nifti1Image = nib.load(f"spt1_dce/{pid}_spy1_vis1_acq{acq}.nii.gz")
        data = img.get_fdata()

        data = normaliseOnePhase(data)
        volumes.append(data)
    stacked = np.stack(volumes,axis=0)
    return stacked
