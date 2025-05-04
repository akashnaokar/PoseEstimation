import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

scene_pcd = o3d.io.read_point_cloud("data/stage.pcd")

scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))


plane_model, inliers = scene_pcd.segment_plane(distance_threshold=0.2,  # try 0.005–0.02 based on noise
                                         ransac_n=3,
                                         num_iterations=1000)

table = scene_pcd.select_by_index(inliers)
o3d.visualization.draw_geometries([table], window_name="Table")
objects = scene_pcd.select_by_index(inliers, invert=True)

labels = np.array(objects.cluster_dbscan(eps=0.03, min_points=30))  # Adjust eps based on box size
max_label = labels.max()
print(f"Detected {max_label + 1} object clusters")

colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label >= 0 else 1))
colors[labels < 0] = 0  # For noise/unclustered points
objects.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([objects], window_name="Objects after Filtering")

target_cluster_idx = 0  # Change based on what you observe visually
box_pcd = objects.select_by_index(np.where(labels == target_cluster_idx)[0])

# Visualize the box alone
o3d.visualization.draw_geometries([box_pcd], window_name="Box Top Surface")

o3d.io.write_point_cloud("data/box_model_extracted.pcd", box_pcd)