import numpy as np

def stack_3phases(pre, early, late):
    """
    Stack 3 normalized DCE-MRI phases into a single volume with 3 channels.

    Inputs:
    - pre, early, late: np.ndarrays of shape (D, H, W)

    Returns:
    - stacked: np.ndarray of shape (3, D, H, W)
    """
    assert pre.shape == early.shape == late.shape, "Phases must have same shape"
    
    stacked = np.stack([pre, early, late], axis=0)  # shape: (3, D, H, W)
    return stacked
