from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
def upscale(data):
    print(data)
    return zoom(data,zoom=(2,2,2),order=3)

def sharpen_3d(img, alpha=1.5, sigma=1):
    blurred = gaussian_filter(img, sigma=sigma)
    return img + alpha * (img - blurred)