import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import cv2

# Load files
depth_img = np.load("Robotics Perception Take-Home Task - Additional Files/one-box.depth.npdata.npy")
color_img = np.load("Robotics Perception Take-Home Task - Additional Files/one-box.color.npdata.npy")
cam_extr = np.load("Robotics Perception Take-Home Task - Additional Files/extrinsics.npy")
cam_intr = np.load("Robotics Perception Take-Home Task - Additional Files/intrinsics.npy")

# Inspect data
print("Color image shape:", color_img.shape, "dtype:", color_img.dtype)
print("Depth image shape:", depth_img.shape, "dtype:", depth_img.dtype)
print("Color min/max:", np.min(color_img), np.max(color_img))
print("Depth min/max:", np.min(depth_img), np.max(depth_img))

print("Extrincsic data shape:", cam_extr.shape, "dtype:", cam_extr.dtype)
print(cam_extr)

print("Intrinsic data shape:", cam_intr.shape, "dtype:", cam_intr.dtype)
print(cam_intr)
