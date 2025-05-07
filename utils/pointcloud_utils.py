import cv2
import math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def generate_pointcloud_from_npy(color_path: str, depth_path: str,
                                 intrinsics_path: str, extrinsics_path: str,
                                 save_path: str = None, visualize: bool = False) -> o3d.geometry.PointCloud:
    """
    Generate a point cloud from .npy color and depth images using Open3D.

    Args:
        color_path (str): Path to the color image saved as a NumPy array (.npy).
        depth_path (str): Path to the depth image saved as a NumPy array (.npy).
        intrinsics_path (str): Path to the camera intrinsics (.npy) - 3x3 matrix.
        extrinsics_path (str): Path to the camera extrinsics (.npy) - 4x4 matrix.
        save_path (str, optional): File path to save the resulting point cloud (.pcd or .ply). Default: None.
        visualize (bool, optional): Whether to visualize the point cloud and RGBD images. Default: False.

    Returns:
        o3d.geometry.PointCloud: The resulting point cloud object.
    """
    # Load data
    color_raw = np.load(color_path)
    depth_raw = np.load(depth_path)
    cam_intr = np.load(intrinsics_path)
    cam_extr = np.load(extrinsics_path)

    height, width = depth_raw.shape
    print(f"Width: {width}\tHeight: {height}")

    # Normalize and convert color to 8-bit 3-channel
    color_normalized = cv2.normalize(color_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_img = np.stack([color_normalized] * 3, axis=-1)

    print("Depth min/max:", np.min(depth_raw), np.max(depth_raw))
    depth_o3d = o3d.geometry.Image(depth_raw.astype(np.float32))
    color_o3d = o3d.geometry.Image(color_img)

    # Create RGBD image
    image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=math.ceil(np.max(depth_raw)),
        convert_rgb_to_intensity=False
    )

    # Optionally show the RGBD images
    if visualize:
        plt.subplot(1, 2, 1)
        plt.title("Color Image Normalized")
        plt.imshow(image.color)
        plt.subplot(1, 2, 2)
        plt.title("Depth Image (float) in meters")
        plt.imshow(image.depth)
        plt.show()

    # Get intrinsics from matrix
    fx, fy, cx, cy = cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image, intrinsic)

    # Flip point cloud for visualization (Open3D coordinate system)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    print("Point cloud XYZ min:", np.min(np.asarray(pcd.points), axis=0))
    print("Point cloud XYZ max:", np.max(np.asarray(pcd.points), axis=0))

    # Optionally visualize
    if visualize:
        o3d.visualization.draw_geometries([pcd], zoom=0.5, window_name="Stage")

    # Optionally save
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved point cloud to {save_path}")

    return pcd
