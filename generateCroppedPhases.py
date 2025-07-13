'''
This script processes DCE-MRI data for breast cancer patients, normalizing and cropping the images based on metadata.
'''




import pandas as pd
import numpy as np
#Start by making your active directory be Data\BreastDCEDL_spy1
#cd ../Data\BreastDCEDL_spy1

metadata = pd.read_csv("BreastDCEDL_spy1_metadata.csv")



import nibabel as nib
import re
img:nib.nifti1.Nifti1Image = nib.load("spt1_dce/ISPY1_1201_spy1_vis1_acq0.nii.gz")
import nibabel as nib
import matplotlib.pyplot as plt
import os
def get_vis_and_acq():
    data = dict()
    pattern = re.compile(r"^(?P<pid>ISPY1_\d+)_.+_vis(?P<vis>\d+)_acq(?P<acq>\d+)")
    for fname in os.listdir("spt1_dce"):
        match = re.search(pattern,fname)
        if match:
            groupdict = match.groupdict()
            pid = groupdict["pid"]
            vis = groupdict["vis"] 
            acq = groupdict["acq"]
            if pid not in data:
                data[pid] = dict()
                data[pid]["vis"] = dict()
            if vis not in data[pid]["vis"]:
                data[pid]["vis"][vis] = dict()
                data[pid]["vis"][vis]["acq"] = set()
            data[pid]["vis"][vis]["acq"].add(acq)
    return data

    
def vizMidPhase(data):
    vizMid()

    
def normaliseOnePhase(arr):
    # Normalise the data to range [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def stackPhases(pid,vis,vis_and_acq_dict):
    acqs = sorted(vis_and_acq_dict[pid]["vis"][vis]["acq"],key=int)
    volumes = []
    for acq in acqs:
        img:nib.nifti1.Nifti1Image = nib.load(f"spt1_dce/{pid}_spy1_vis{vis}_acq{acq}.nii.gz")
        data = img.get_fdata()

        data = normaliseOnePhase(data)
        volumes.append(data)
    stacked = np.stack(volumes,axis=0)
    return stacked

def cropPhases(stacked,pid,metadata,margin=6) -> np.ndarray:
    row = metadata.loc[metadata["pid"]==pid].iloc[0]
    x_start=int(row["voi_start_x"])
    y_start=int(row["voi_start_y"])
    z_start=int(row["voi_start_z"])
    x_end=int(row["voi_end_x"]) 
    y_end=int(row["voi_end_y"])
    z_end=int(row["voi_end_z"])
    cropShape = (
        stacked.shape[0],
        (z_end + margin) - (z_start - margin),
        (x_end + margin) - (x_start - margin),
        (y_end + margin) - (y_start - margin),
    )
    
    
    cropped = np.zeros(cropShape,dtype=stacked.dtype)
      
    x_start = x_start - margin
    x_end = x_end + margin
    y_start = y_start - margin
    y_end = y_end + margin
    z_start = z_start - margin
    z_end = z_end + margin
    
    #clipped
    clipped_x_start = max(0, x_start)
    clipped_x_end = min(stacked.shape[2], x_end)
    clipped_y_start = max(0, y_start)
    clipped_y_end = min(stacked.shape[3], y_end)
    clipped_z_start = max(0, z_start)
    clipped_z_end = min(stacked.shape[1], z_end)
    
    stackedSlices = (
        slice(None),
        slice(clipped_z_start,clipped_z_end),
        slice(clipped_x_start,clipped_x_end),
        slice(clipped_y_start,clipped_y_end),
    )
    
    padded_x_start = clipped_x_start - x_start
    padded_x_end = padded_x_start + (clipped_x_end-clipped_x_start)
    padded_y_start = clipped_y_start - y_start
    padded_y_end = padded_y_start + (clipped_y_end-clipped_y_start)
    padded_z_start = clipped_z_start - z_start
    padded_z_end = padded_z_start + (clipped_z_end-clipped_z_start)
    
    croppedSlices = (
        slice(None),
        slice(padded_z_start, padded_z_end),
        slice(padded_x_start, padded_x_end),
        slice(padded_y_start, padded_y_end),
    )

    print(f"Cropping data from ({z_start}, {x_start}, {y_start}) to ({z_end}, {x_end}, {y_end})")
    
    cropped[croppedSlices] = stacked[stackedSlices]
    return cropped

def getSlice(pid):
    img:nib.nifti1.Nifti1Image = nib.load(f"spt1_dce/{pid}_spy1_vis")

def generateAllCroppedPhases():
    out = dict()
    vis_and_acq_dict = get_vis_and_acq()
    for pid in vis_and_acq_dict:
        for vis in vis_and_acq_dict[pid]["vis"]:
            print(f"Processing {pid}_vis_{vis}")
            cropped = cropPhases(stackPhases(pid, vis, vis_and_acq_dict), pid, metadata)
            
            #transpose from (t,x,z,y) to (t,x,y,z)
            cropped = np.transpose(cropped,(0,1,3,2))
            
            out[f"{pid}_vis{vis}"] = cropped
    return out
    

def main():
    out = generateAllCroppedPhases()
    np.savez_compressed("croppedPhases.npz",**out)
    print("All cropped phases saved to croppedPhases.npz")
    
if __name__ == "__main__":
    main()
