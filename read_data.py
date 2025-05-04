import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import cv2

# Load files
depth_img = np.load("data/one-box.depth.npdata.npy")
color_img = np.load("data/one-box.color.npdata.npy")
cam_extr = np.load("data/extrinsics.npy")
cam_intr = np.load("data/intrinsics.npy")

plt.imshow(depth_img, cmap="gray")
matplotlib.image.imsave("data/depth_img.png", depth_img)

plt.imshow(color_img,cmap="gray")
matplotlib.image.imsave("data/color_img.png", color_img)

# Inspect data
print("Color image shape:", color_img.shape, "dtype:", color_img.dtype)
print("Depth image shape:", depth_img.shape, "dtype:", depth_img.dtype)
print("Color min/max:", np.min(color_img), np.max(color_img))
print("Depth min/max:", np.min(depth_img), np.max(depth_img))

print("Extrincsic data shape:", cam_extr.shape, "dtype:", cam_extr.dtype)
print(f"Camera Extrinsics:\n{cam_extr}")

print("Intrinsic data shape:", cam_intr.shape, "dtype:", cam_intr.dtype)
print(f"Camera Intrinsics:\n{cam_intr}")