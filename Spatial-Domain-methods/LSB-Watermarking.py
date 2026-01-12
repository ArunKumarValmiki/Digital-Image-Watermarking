# Least Significant Watermarking(LSB)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load cover image (grayscale, uint8: 0–255)
a = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
m, n = a.shape

# --- Step 1: Bit-plane decomposition ---
b = np.zeros((m, n, 8), dtype=np.uint8)

for k in range(8):
    b[:, :, k] = (a >> k) & 1   # extract k-th bit-plane

# Show all 8 bit-planes
for i in range(8):
    plt.figure()
    plt.title(f"Bit-plane {i+1}")
    plt.imshow(b[:, :, i]*255, cmap='gray')
    plt.axis("off")

# --- Step 2: Load watermark image (binary) ---
w = cv2.imread("rice.jpeg", cv2.IMREAD_GRAYSCALE)
w = cv2.resize(w, (n, m))   # resize watermark to fit cover image
_, w = cv2.threshold(w, 127, 1, cv2.THRESH_BINARY)  # binarize (0/1)

plt.figure()
plt.title("Watermark")
plt.imshow(w*255, cmap='gray')
plt.axis("off")

# --- Step 3: Embed watermark into LSB (bit-plane 0), can embed in any plane at the cost of imperceptibility ---
b[:, :, 0] = w

# Reconstruct watermarked image from bit-planes
wi = np.zeros((m, n), dtype=np.uint8)
for k in range(8):
    wi += (b[:, :, k] << k)

plt.figure()
plt.title("Watermarked Image")
plt.imshow(wi, cmap='gray')
plt.axis("off")


# --- Step 4: Extract watermark from watermarked image ---
b_extracted = np.zeros((m, n, 8), dtype=np.uint8)
for k in range(8):
    b_extracted[:, :, k] = (wi >> k) & 1

plt.figure()
plt.title("Extracted Watermark")
plt.imshow(b_extracted[:, :, 0]*255, cmap='gray')
plt.axis("off")

plt.show()
