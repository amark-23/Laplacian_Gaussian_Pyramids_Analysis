import matplotlib.pyplot as plt
import cv2
import numpy as np
from pyramid import LPyramid, L_Pyramid_Decode, L_Quantization
from image_utils import download_lena, get_camera_image, entropy

def process_image(img, a, depth):
    """Apply Laplacian Pyramid encoding and decoding."""
    if len(img.shape) == 2:  # Grayscale
        L = LPyramid(img, a, depth)
        decoded = L_Pyramid_Decode(L, a)
    else:  # Color image
        channels = cv2.split(img)
        L_channels = [LPyramid(ch, a, depth) for ch in channels]
        decoded_channels = [L_Pyramid_Decode(L, a) for L in L_channels]
        decoded = cv2.merge(decoded_channels).astype(np.uint8)
    return decoded

def test_different_a(image, is_color=True):
    """
    Test decoding with different values of 'a' for a single image.
    
    Args:
        image: Input image (color or grayscale)
        is_color: Set to True if the image is RGB, False if grayscale
    """
    a_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    for a in a_values:
        decoded = process_image(image, a, depth=5)
        plt.figure(figsize=(6, 5))
        if is_color:
            plt.imshow(decoded)
        else:
            plt.imshow(decoded, cmap='gray')
        plt.title(f"Decoded Image (a={a})")
        plt.axis("off")
        plt.show()

def test_different_depths(image, a):
    """Test pyramid decoding at different depths."""
    depths = [3, 4, 5, 6]
    fig, axes = plt.subplots(1, len(depths), figsize=(16, 4))
    for i, depth in enumerate(depths):
        decoded = process_image(image, a, depth)
        axes[i].imshow(decoded if len(decoded.shape) == 3 else decoded, cmap='gray')
        axes[i].set_title(f"Depth {depth}")
        axes[i].axis("off")
    plt.suptitle("Effect of Pyramid Depth")
    plt.show()

def compare_entropy_quantization(image, a, depth, bin_sizes):
    """Apply quantization, reconstruct image, and compute entropy."""
    original_entropy = entropy(image)
    print(f"Original Entropy: {original_entropy:.4f}")

    if len(image.shape) == 2:  # Grayscale
        L = LPyramid(image, a, depth)
        for bin_size in bin_sizes:
            Lq = L_Quantization(L, bin_size)
            decoded = L_Pyramid_Decode(Lq, a)
            e = entropy(decoded)
            print(f"Bin Size: {bin_size} -> Entropy: {e:.4f}")
            plt.imshow(decoded, cmap='gray')
            plt.title(f'Quantized (bin={bin_size}) - Entropy={e:.2f}')
            plt.axis('off')
            plt.show()
    else:
        # Color: process per channel
        channels = cv2.split(image)
        for bin_size in bin_sizes:
            L_channels = [LPyramid(ch, a, depth) for ch in channels]
            Lq_channels = [L_Quantization(L, bin_size) for L in L_channels]
            decoded_channels = [L_Pyramid_Decode(Lq, a) for Lq in Lq_channels]
            decoded = cv2.merge(decoded_channels).astype(np.uint8)
            e = entropy(decoded)
            print(f"Bin Size: {bin_size} -> Entropy: {e:.4f}")
            plt.imshow(decoded)
            plt.title(f'Quantized (bin={bin_size}) - Entropy={e:.2f}')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    lena = download_lena()
    camera = get_camera_image()

    print("\n Testing with different α values:")
    test_different_a(lena,is_color=True)
    test_different_a(camera, is_color=False)


    print("\n Testing different pyramid depths on Lena:")
    test_different_depths(lena, a=0.4)

    print("\n📉 Quantization & Entropy on grayscale Camera:")
    compare_entropy_quantization(camera, a=0.4, depth=5, bin_sizes=[5, 10, 20, 40])
