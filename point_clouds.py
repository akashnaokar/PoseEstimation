import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import math

color_raw = np.load("data/one-box.color.npdata.npy")
depth_raw = np.load("data/one-box.depth.npdata.npy")

width, height = depth_raw.shape[1], depth_raw.shape[0]
print(f"Width: {width}\tHeight: {height}")

color_normalized = cv2.normalize(color_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
color_img = np.stack([color_normalized] * 3, axis=-1)

print("Depth min/max:", np.min(depth_raw), np.max(depth_raw))
depth_o3d = o3d.geometry.Image(depth_raw.astype(np.float32))
color_o3d = o3d.geometry.Image(color_img)


image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=math.ceil(np.max(depth_raw)),
            convert_rgb_to_intensity=False)

print(image)



plt.subplot(1,2,1)
plt.title("Color Image noramlized")
plt.imshow(image.color)
plt.subplot(1,2,2)
plt.title("Depth Image (float) in meters")
plt.imshow(image.depth)
plt.show()

cam_extr = np.load("data/extrinsics.npy")
cam_intr = np.load("data/intrinsics.npy")

fx, fy , cx, cy = cam_intr[0,0], cam_intr[1,1], cam_intr[0,2], cam_intr[1,2]
print(f"fx: {fx}\t fy:{fy}\t cx:{cx}\t cy:{cy}")

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    image,
    intrinsic
)

points = np.asarray(pcd.points)
print("Point cloud XYZ min:", points.min(axis=0))
print("Point cloud XYZ max:", points.max(axis=0))

pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd], zoom=0.5, window_name="Stage")

o3d.io.write_point_cloud("data/stage.pcd", pcd)