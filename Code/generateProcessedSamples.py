import pandas as pd
import nibabel as nib
import re
import os
import numpy as np
from scipy.ndimage import affine_transform, center_of_mass, rotate
from viz import viz3D, viz3D_with_slider

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

def normalise(arr):
    # Normalise the data to range [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def stackPhases(pid,acqData):
    acqs = sorted(acqData[pid],key=int)
    volumes = []
    for acq in acqs:
        img:nib.nifti1.Nifti1Image = nib.load(f"Datasets/BreastDCEDL_spy1/spt1_dce/{pid}_spy1_vis1_acq{acq}.nii.gz")
        
        #data has to be transposed from x,z,y, to x,y,z
        data = np.transpose(img.get_fdata(),(0,2,1))
        data = normalise(data)
        volumes.append(data)
    stacked = np.stack(volumes,axis=0)
    return stacked

def getCentreOfMass(pid):
    mask = nib.load(f"Datasets/BreastDCEDL_spy1/spy1_mask/{pid}_spy1_vis1_mask.nii.gz")
    mask = np.transpose(mask.get_fdata(),(0,2,1))
    COM = center_of_mass(mask)
    return COM

def cropStackedPhases(data, point, target_shape):
    """
    Crop 4D array (t, x, y, z) around a 3D point in (x, y, z) dims,
    padding with zeros if needed. Does not crop the t dimension.
    
    Parameters:
    - data: np.ndarray with shape (t, x, y, z)
    - point: array-like with 3 floats/ints (x, y, z)
    - target_shape: array-like with 3 ints (target_x, target_y, target_z)
    
    Returns:
    - cropped_data: np.ndarray with shape (t, target_x, target_y, target_z)
    """
    point = np.round(point).astype(int)  # Round to nearest integer
    shape = np.array(data.shape[1:])  # ignore t dimension
    target_shape = np.array(target_shape)

    half = target_shape // 2
    extra = target_shape % 2  # 1 if odd, 0 if even

    start = point - half
    end = point + half + extra

    # Padding if crop goes beyond the array bounds
    pad_before = np.maximum(-start, 0)
    pad_after = np.maximum(end - shape, 0)
    pad_width = [(0, 0)]  # no padding on t dimension
    pad_width += list(zip(pad_before, pad_after))
    
    padded = np.pad(data, pad_width, mode="constant", constant_values=0)

    # Adjust start after padding (shift start by padding)
    start = np.maximum(start, 0)

    # Create slices for cropping (t is fully included)
    slices = (slice(None),)  # full slice for t
    slices += tuple(slice(start[i], start[i] + target_shape[i]) for i in range(3))

    return padded[slices]


def rotateStackedPhasesInSaggitalPlane(pid:str,stackedPhases,n_samples:int):
    #COM = getCentreOfMass(pid)
    angleUnit = 360/n_samples
    rotatedStackedPhases = []
    for step in range(n_samples):
        angle = angleUnit*step
        stackedPhasesAfterRotation = []
        for phase in stackedPhases:
            rotatedPhase = rotate(phase,angle,(1,2),reshape=False)
            stackedPhasesAfterRotation.append(rotatedPhase)
        print(f"rotated {angle}")
        stackedPhasesAfterRotation = np.stack(stackedPhasesAfterRotation,axis=0)
        rotatedStackedPhases.append(stackedPhasesAfterRotation)
    rotatedStackedPhases = np.stack(rotatedStackedPhases,axis=0)
    
    #shape is (angles,t,x,y,z)
    return rotatedStackedPhases



def generateProcessedSamples(pid,n_samples):
    stackedPhases = stackPhases(pid,acqData)
    COM = getCentreOfMass(pid)
    croppedStackedPhases = cropStackedPhases(stackedPhases,COM,(16,182,182))
    rotatedStackedPhases = rotateStackedPhasesInSaggitalPlane(pid,croppedStackedPhases,n_samples)
    croppedRotatedStackedPhases=[]
    for stackedPhase in rotatedStackedPhases:
        point = (np.array(stackedPhase.shape)//2)[1:]
        croppedRotatedStackedPhases.append(cropStackedPhases(stackedPhase,point,(16,128,128)))
    croppedRotatedStackedPhases = np.stack(croppedRotatedStackedPhases,axis=0)
    return croppedRotatedStackedPhases

 




