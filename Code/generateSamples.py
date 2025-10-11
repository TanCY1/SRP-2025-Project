import numpy as np
import os
from scipy.ndimage import rotate
from itertools import combinations
#from viz import viz3D, viz3D_with_slider

def loadPatient(pid,dataset_path):
    file = np.load(os.path.join(dataset_path,f"ISPY2-{str(pid)}.npz"))
    T0 = file["T0"]
    T3 = file["T3"]
    return T0,T3

def rotateStackedPhasesInSaggitalPlane(phases,n_samples:int):
    #COM = getCentreOfMass(pid)
    angleUnit = 360/n_samples
    rotatedStackedPhases = []
    for step in range(n_samples):
        angle = angleUnit*step
        stackedPhasesAfterRotation = []
        for phase in phases:
            rotatedPhase = rotate(phase,angle,(1,2),reshape=False)
            stackedPhasesAfterRotation.append(rotatedPhase)
        #print(f"rotated {angle}")
        stackedPhasesAfterRotation = np.stack(stackedPhasesAfterRotation,axis=0)
        rotatedStackedPhases.append(stackedPhasesAfterRotation)
    rotatedStackedPhases = np.stack(rotatedStackedPhases,axis=0)
    
    #shape is (angles,t,x,y,z)
    return rotatedStackedPhases

def cropCentre(data,target_size):
    data_shape = data.shape
    slices = []
    
    for i, t in zip(data_shape, target_size):
        start = (i - t) // 2
        end = start + t
        slices.append(slice(start, end))
    
    return data[tuple(slices)]

def resize_to_6_phases(data):
    num_phases = data.shape[0]
    if num_phases==6:
        return data
    if num_phases < 6:
        raise ValueError("Cannot select 6 phases: data has fewer than 6 phases")
    volume_mean = data.mean(axis=(1,2,3))


    candidate_indices = list(combinations(range(num_phases), 6))
    best_indices = None
    lowest_error = float('inf')

    for indices in candidate_indices:
        # interpolate selected phases to full length
        interp_curve = np.interp(np.arange(num_phases), indices, volume_mean[list(indices)])
        error = np.mean((volume_mean - interp_curve)**2)  # MSE
        #print(indices,error)
        if error < lowest_error:
            lowest_error = error
            best_indices = indices
    #print(best_indices)
    return data[np.array(best_indices)]

def augment(data:np.ndarray,num_samples):
    if num_samples==1:
        return [data]
    if num_samples==2:
        return [data,np.rot90(data,k=2,axes=(1,2))]    
    else: return rotateStackedPhasesInSaggitalPlane(data,num_samples)


def generateSamples(pid,num_samples,dataset_path):
    T0,T3 = loadPatient(pid,dataset_path)
    T0 = resize_to_6_phases(T0)
    T3 = resize_to_6_phases(T3)
    T0s = augment(T0,num_samples)
    T3s = augment(T3,num_samples)
    T0s = [[cropCentre(phase,(16,128,128)) for phase in augmented_sample] for augmented_sample in T0s]
    T3s = [[cropCentre(phase,(16,128,128)) for phase in augmented_sample] for augmented_sample in T3s]
    return T0s,T3s
    
