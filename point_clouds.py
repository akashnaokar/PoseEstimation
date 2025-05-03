import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

color_raw = np.load("data/one-box.color.npdata.npy")
depth_raw = np.load("data/one-box.depth.npdata.npy")

color_normalized = cv2.normalize(color_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
color_img = np.stack([color_normalized] * 3, axis=-1)

depth_o3d = o3d.geometry.Image(depth_raw.astype(np.float32))
color_o3d = o3d.geometry.Image(color_img)

image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d)

print(image)

plt.subplot(1,2,1)
plt.title("Color Image noramlized")
plt.imshow(image.color)
plt.subplot(1,2,2)
plt.title("Depth Image (float) in meters")
plt.imshow(image.depth)
plt.show()