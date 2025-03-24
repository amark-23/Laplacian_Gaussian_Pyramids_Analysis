##  Laplacian & Gaussian Pyramids for Image Processing

This project implements **Gaussian and Laplacian Pyramids** for image decomposition and reconstruction using Python and OpenCV. It allows analysis, quantization, and entropy measurement on grayscale and color images, and helps visualize the effect of pyramid depth and quantization levels.

---

##  Theory



### 🔹 Sampling & Multi-Resolution Representations

In digital image processing, controlling spatial resolution is essential for tasks like compression, transmission, multiscale analysis, and object recognition.

- **Downsampling** reduces image size by discarding information, usually after applying a **low-pass filter** to avoid aliasing.
- **Upsampling** increases resolution by interpolating missing pixel values.

These operations form the basis of **image pyramids**, which store multiple levels of detail in a compact hierarchical structure.

---

### 🔹 Gaussian Kernel 

We use a parametric 5x5 kernel based on a smoothing factor $a$:

$$
w = \left[0.25 - \frac{a}{2},\ 0.25,\ a,\ 0.25,\ 0.25 - \frac{a}{2} \right]
$$

The 2D kernel is then formed via the outer product:

$$
h = w \otimes w
$$

Where:

- $a$ controls the shape and spread of the kernel (similar to $\sigma$ in a traditional Gaussian)
- The resulting kernel is **normalized** so that:

$$
\sum h = 1
$$

This method is computationally efficient and allows for flexible smoothing in both Gaussian and Laplacian pyramids.


### 🔹 Gaussian Pyramid

A **Gaussian Pyramid** is a series of increasingly smaller and smoother versions of an image.

Each level $G_i$ is computed by:

$$
G_{i+1} = \text{Reduce}(G_i, h) = (G_i * h) \downarrow 2
$$

Where:
- $h$ is a Gaussian-like kernel (e.g., based on a parameter $a$)
- $*$ denotes convolution
- $\downarrow 2$ means subsampling by a factor of 2

The result is a progressively blurred and downsampled image that retains low-frequency content.

---

### 🔹 Laplacian Pyramid

The **Laplacian Pyramid** stores the difference between Gaussian levels, capturing edge and detail information (i.e., band-pass data).

Each level $L_i$ is calculated as:

$$
L_i = G_i - \text{Expand}(G_{i+1}, h)
$$

Where:
- `Expand` upsamples $G_{i+1}$ to match the size of $G_i$, typically using interpolation followed by convolution with $h$

The pyramid can be **perfectly reconstructed**:

$$
G_i = L_i + \text{Expand}(G_{i+1}, h)
$$

This makes Laplacian Pyramids useful for compression, texture synthesis, and image blending.

---

### 🔹 Quantization & Entropy

**Quantization** reduces precision by mapping pixel values to discrete bins, commonly used in compression.

Given a bin size $b$, quantization is defined as:

$$
L'_i = \text{round}\left(\frac{L_i}{b}\right) \cdot b
$$

To measure the effect of quantization, we compute the **Shannon entropy** of the image:

$$
H = -\sum p(x) \log_2 p(x)
$$

Where:
- $p(x)$ is the probability (histogram) of intensity $x$

Entropy quantifies the amount of information:
- **High entropy** → more detail, noise, or randomness
- **Low entropy** → more compressible but possibly degraded

---

### 🔹 Role of Parameters: $a$ and Pyramid Depth

- The **parameter $a$** defines the shape of the 1D kernel used to build the Gaussian kernel. It controls the **amount of smoothing** applied:
  - Lower $a$ values yield less smoothing
  - Higher $a$ values result in more blur
  - Common range: $0.3 \leq a \leq 0.7$

  The kernel $w$ used in convolution is generated as:
  $$
  w = \left[0.25 - \frac{a}{2},\ 0.25,\ a,\ 0.25,\ 0.25 - \frac{a}{2} \right]
  $$

- **Pyramid depth** defines how many times the image is reduced and differences are stored. Greater depth:
  - Captures more levels of detail
  - Increases computation
  - Can lead to excessive blurring or resolution loss if too deep

Choosing the right combination of $a$ and depth is crucial for balancing detail preservation and compression.


---

## How to Run

Make sure Python is installed, install packages and run the project:

```bash
pip install -r requirements.txt
python main.py
```

It will:
- Download test images (Lena & Camera)
  ![image](https://github.com/user-attachments/assets/94a3cdc8-07b9-49c8-beee-f5912c32d0e6)

- Run Laplacian pyramid decomposition & reconstruction
- Display visual outputs for different values of `a`, pyramid `depth`, and quantization `bin size`
  ![image](https://github.com/user-attachments/assets/38e6b9ed-0217-46ae-94a5-c30d9339aa3a)
  ![image](https://github.com/user-attachments/assets/400cf7f1-3eb5-400d-ba1e-f9f6c5dcb712)
  ![image](https://github.com/user-attachments/assets/1376cf90-564d-43ba-8e56-0c1d918ee582)
  ![image](https://github.com/user-attachments/assets/5e0ce591-84f2-4ed6-8349-558639985c3e)

- Print entropy values to console

---

## Using Custom Images

To run the algorithm on your own image:

1. Add your image file (e.g. `my_image.jpg`) to the project folder
2. Modify `main.py` like this:

```python
import cv2
image = cv2.imread("my_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # For color
# or
image = cv2.imread("my_image.jpg", cv2.IMREAD_GRAYSCALE)  # For grayscale

test_different_a(image, is_color=True)  # or False if grayscale

compare_entropy_quantization(image, a=0.4, depth=5, bin_sizes=[10, 20, 30])
```

---


## Project Structure

```bash
.
├── main.py              # Entry point to run the pipeline
├── pyramid.py           # Pyramid construction & decoding logic
├── image_utils.py       # Image loading, entropy calculation, etc.
├── requirements.txt     # All needed Python packages
```

