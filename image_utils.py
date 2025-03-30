import cv2
import urllib.request
from skimage import data
import numpy as np

def download_lena():
    """Download Lena image from Wikipedia."""
    lena_url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    lena_path = "lena.png"
    urllib.request.urlretrieve(lena_url, lena_path)
    lena = cv2.imread(lena_path)
    return cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)

def get_camera_image():
    """Get grayscale camera image from skimage."""
    return data.camera()

def entropy(image):
    """Calculate the entropy of an image."""
    hist, _ = np.histogram(image.flatten(), bins=256, density=True) #autoadjust bins
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))
