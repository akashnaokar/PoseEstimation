import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def segment_table_and_objects(scene_path: str,
                               table_save_path: str = None,
                               objects_save_path: str = None,
                               box_save_path: str = None,
                               visualize: bool = False,
                               target_cluster_idx: int = 0,
                               plane_distance_threshold: float = 0.18,
                               dbscan_eps: float = 0.03,
                               dbscan_min_points: int = 30) -> o3d.geometry.PointCloud:
    """
    Segments a scene point cloud into a table and clustered objects using RANSAC and DBSCAN.

    Args:
        scene_path (str): Path to the input scene point cloud (.pcd).
        table_save_path (str, optional): Where to save the segmented table point cloud. Default: None.
        objects_save_path (str, optional): Where to save the clustered object points. Default: None.
        box_save_path (str, optional): Where to save a specific target object (cluster) point cloud. Default: None.
        visualize (bool, optional): Whether to visualize each stage of segmentation. Default: False.
        target_cluster_idx (int, optional): Index of the object cluster to extract. Default: 0.
        plane_distance_threshold (float, optional): RANSAC plane segmentation threshold. Default: 0.018.
        dbscan_eps (float, optional): DBSCAN epsilon for clustering. Default: 0.03.
        dbscan_min_points (int, optional): DBSCAN minimum points per cluster. Default: 30.

    Returns:
        o3d.geometry.PointCloud: The extracted target object (cluster) point cloud.
    """
    # Load the scene point cloud
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    # Plane segmentation (table detection)
    plane_model, inliers = scene_pcd.segment_plane(
        distance_threshold=plane_distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )

    table = scene_pcd.select_by_index(inliers)
    objects = scene_pcd.select_by_index(inliers, invert=True)

    if table_save_path:
        o3d.io.write_point_cloud(table_save_path, table)
        print(f"Saved table point cloud to {table_save_path}")
    if visualize:
        o3d.visualization.draw_geometries([table], window_name="Segmented Table")

    # Object clustering using DBSCAN
    labels = np.array(objects.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points))
    max_label = labels.max()
    print(f"Detected {max_label + 1} object clusters")

    # Color clusters for visualization
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label >= 0 else 1))
    colors[labels < 0] = 0  # Noise points
    objects.colors = o3d.utility.Vector3dVector(colors[:, :3])

    if objects_save_path:
        o3d.io.write_point_cloud(objects_save_path, objects)
        print(f"Saved object clusters to {objects_save_path}")
    if visualize:
        o3d.visualization.draw_geometries([objects], window_name="Segmented Object Clusters")

    # Extract specific target cluster (e.g., a box)
    target_cluster_indices = np.where(labels == target_cluster_idx)[0]
    box_pcd = objects.select_by_index(target_cluster_indices)

    if box_save_path:
        o3d.io.write_point_cloud(box_save_path, box_pcd)
        print(f"Saved extracted box model to {box_save_path}")
    if visualize:
        o3d.visualization.draw_geometries([box_pcd], window_name="Extracted Object Cluster")

    return box_pcd
