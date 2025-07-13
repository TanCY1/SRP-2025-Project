from generateCroppedPhases import generateAllCroppedPhases,get_vis_and_acq,cropPhases
import pandas as pd
import numpy as np
from viz import viz, vizMid,viz3D,viz3D_with_slider
from scipy.ndimage import affine_transform,rotate,center_of_mass
import nibabel as nib


#cd ../Data\BreastDCEDL_spy1

metadata = pd.read_csv("BreastDCEDL_spy1_metadata.csv")



croppedPhases =np.load("croppedPhases.npz")
viz_and_acq_dict = get_vis_and_acq()

def getCentreOfMass(pid):
    mask = nib.load(f"spy1_mask/{pid}_spy1_vis1_mask.nii.gz").get_fdata()
    cropped = cropPhases(np.array([mask,]),pid,metadata,margin=6)
    
    #transpose from (t,x,z,y) to (t,x,y,z)
    cropped = np.transpose(cropped,(0,1,3,2))
    return center_of_mass(cropped[0])

   
def rotate3DAboutAPoint(data,angle_deg,axes,point):
    angle = np.deg2rad(angle_deg)
    i,j = axes
    
    #identity matrix
    A = np.eye(3)
    
    c,s = np.cos(angle),np.sin(angle)
    
    A[i,i], A[j,j] = c,c
    A[i,j], A[j,i] = -s,s
    
    offset = point - A @ point
    
    return affine_transform(data, A, offset=offset, order=3)
    
   
     
def rotatePhasesInSaggitalPlane(pid,n_samples):
    COM = getCentreOfMass(pid)
    data = croppedPhases[f"{pid}_vis1"]
    step = 360/n_samples
    for angle in range(0,360,step):
        rotate3DAboutAPoint(data,angle,(1,2),COM)
    


cm = getCentreOfMass("ISPY1_1004")

test = croppedPhases["ISPY1_1004_vis1"][2]

data4d = [rotate3DAboutAPoint(test,i,(1,2),cm) for i in range(1,15+1)]
viz3D_with_slider(data4d)
viz3D_with_slider(np.transpose(cropPhases(np.array([nib.load(f"spy1_mask/ISPY1_1004_spy1_vis1_mask.nii.gz").get_fdata()]),"ISPY1_1004",metadata,margin=6),(0,1,3,2)))


'''
test = croppedPhases[croppedPhases.files[0]][...,0]

point = np.array(test.shape)//2


data4d = [rotate3DAboutAPoint(test,i,(1,2),(0,0,0)) for i in range(1,180+1)]


viz3D_with_slider(data4d)




for key in croppedPhases:
    pid, vis = key.split("_vis")
    if len(viz_and_acq_dict[pid]["vis"][vis]["acq"])==6:
        print(pid,metadata.loc[metadata["pid"]==pid].iloc[0].pCR)

    
    '''
    



