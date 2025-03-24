import cv2
import numpy as np

def GKernel(a):
    """Generate a 5x5 kernel based on parameter a."""
    w = np.array([0.25 - a / 2, 0.25, a, 0.25, 0.25 - a / 2])
    kernel = np.outer(w, w)
    return kernel / np.sum(kernel)

def GREDUCE(I, h):
    """Reduce the image using kernel h."""
    I_blurred = cv2.filter2D(I, -1, h)
    return I_blurred[::2, ::2]

def GPyramid(I, a, depth):
    """Construct Gaussian Pyramid."""
    G = [I]
    h = GKernel(a)
    for _ in range(depth - 1):
        G.append(GREDUCE(G[-1], h))
    return G

def GEXPAND(I, h):
    """Expand the image using kernel h."""
    I_expanded = np.zeros((I.shape[0] * 2, I.shape[1] * 2), dtype=I.dtype)
    I_expanded[::2, ::2] = I
    return cv2.filter2D(I_expanded, -1, h * 4)

def LPyramid(I, a, depth):
    """Construct Laplacian Pyramid."""
    G = GPyramid(I, a, depth)
    L = []
    h = GKernel(a)
    for i in range(depth - 1):
        expanded = GEXPAND(G[i + 1], h)
        L.append(cv2.subtract(G[i], expanded))
    L.append(G[-1])
    return L

def L_Pyramid_Decode(L, a):
    """Decode Laplacian Pyramid."""
    h = GKernel(a)
    I_out = L[-1]
    for i in range(len(L) - 2, -1, -1):
        I_out = cv2.add(GEXPAND(I_out, h), L[i])
    return I_out

def L_Quantization(L, bin_size):
    """Quantize the Laplacian Pyramid."""
    return [np.round(layer / bin_size) * bin_size for layer in L]
