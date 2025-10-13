Dataset_Path = r"e:\SRP\ISPY2-Data-Collector\ISPY2_T0_T3_DCE"

import os,sys

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, center_of_mass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def cropOneVolume(path):
    vol = nib.load(os.path.join(path,"ISPY2 VOLSER uni-lateral cropped original DCE.nii.bz2"))
    mask = nib.load(os.path.join(path,"ISPY2 VOLSER uni-lateral cropped Analysis Mask.nii.bz2"))
    spacing = vol.header.get_zooms()
    vol = zoom(vol.get_fdata(), np.array(spacing)/np.array((1,1,1,1)), order=1)
    mask = zoom(mask.get_fdata(), np.array((spacing[2],spacing[0],spacing[1]))/np.array((1,1,1)), order=0)
    COM = center_of_mass(mask)
    vol = vol.transpose((3,2,0,1))
    cropped = cropStackedPhases(vol,COM,(16,182,182))
    return cropped

def normalise(arr):
    # Normalise the data to range [0, 1]
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def saveAsNumpy(Id,path):
    T0 = np.asarray([normalise(phase) for phase in cropOneVolume(os.path.join(Dataset_Path,Id,"ISPY2_MRI_T0"))],dtype=np.float32)
    T3 = np.asarray([normalise(phase) for phase in cropOneVolume(os.path.join(Dataset_Path,Id,"ISPY2_MRI_T3"))],dtype=np.float32)
    np.savez_compressed(os.path.join(path,f"{Id}.npz"),T0=T0,T3=T3)
    tqdm.write(f"Saved {Id}")



    
def main():
    save_path = r"ISPY2_T0_T3_DCE_npz"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for Id in os.listdir(Dataset_Path)[300:]:
            if not os.path.exists(os.path.join(save_path,f"{Id}.npz")):
                futures.append(executor.submit(saveAsNumpy,Id,save_path))
        with tqdm(total=len(futures), position=0) as pbar:
            for future in as_completed(futures):
                pbar.update(1)

if __name__ == "__main__":
    main()

