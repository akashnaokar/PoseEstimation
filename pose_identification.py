import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import copy

# Load point clouds
scene_pcd = o3d.io.read_point_cloud("data/stage.pcd")
box_pcd = o3d.io.read_point_cloud("data/box_model_extracted.pcd")

# Estimate normals for the scene
scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

# Step 1: Compute PCA-based Oriented Bounding Box (OBB) for the box
points = np.asarray(box_pcd.points)
mean = np.mean(points, axis=0)
cov = np.cov((points - mean).T)
eigenvalues, eigenvectors = np.linalg.eig(cov)
sort_idx = eigenvalues.argsort()[::-1]
eigenvectors = eigenvectors[:, sort_idx]

if np.linalg.det(eigenvectors) < 0:
    eigenvectors[:, 2] = -eigenvectors[:, 2]

projected = np.dot(points - mean, eigenvectors)
min_bound = np.min(projected, axis=0)
max_bound = np.max(projected, axis=0)
extent = max_bound - min_bound

box_obb = o3d.geometry.OrientedBoundingBox()
box_obb.center = mean
box_obb.R = eigenvectors
box_obb.extent = extent
box_obb.color = (0, 0, 1)

# Step 2: Compute the box bottom center
gravity_alignment = np.abs(np.dot(eigenvectors.T, np.array([0, 0, 1])))
gravity_axis = np.argmax(gravity_alignment)
bottom_offset = eigenvectors[:, gravity_axis] * (extent[gravity_axis] / 2)
if bottom_offset[2] > 0:  # ensure downward
    bottom_offset = -bottom_offset
box_bottom_center = box_obb.center + bottom_offset

# Step 3: Segment multiple planes in the scene
remaining_pcd = scene_pcd.voxel_down_sample(voxel_size=0.005)
remaining_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

planes = []
plane_models = []
max_planes = 5
for _ in range(max_planes):
    plane_model, inliers = remaining_pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=500
    )
    inlier_cloud = remaining_pcd.select_by_index(inliers)
    print(len(inlier_cloud.points))
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    o3d.visualization.draw_geometries([inlier_cloud], window_name="Planes")
    remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    if len(inlier_cloud.points) < 15000:
        continue
    planes.append(inlier_cloud)
    plane_models.append(plane_model)

print(f"Detected {len(planes)} significant planes.")

# Step 4: Choose the best plane (closest to box bottom center)
min_dist = float('inf')
best_plane_idx = -1
for i, model in enumerate(plane_models):
    a, b, c, d = model
    normal = np.array([a, b, c])
    dist = abs(np.dot(normal, box_bottom_center) + d) / np.linalg.norm(normal)
    if dist < min_dist:
        min_dist = dist
        best_plane_idx = i

assert best_plane_idx >= 0, "No valid plane found."

best_plane = planes[best_plane_idx]
[a, b, c, d] = plane_models[best_plane_idx]
plane_normal = np.array([a, b, c])
if plane_normal[2] < 0:
    plane_normal = -plane_normal
    d = -d

plane_normal = plane_normal / np.linalg.norm(plane_normal)
print(f"Selected plane #{best_plane_idx} with normal {plane_normal} and distance {min_dist:.4f} m")

# Step 5: Distance from box bottom to table
distance_to_plane = abs(np.dot(plane_normal, box_bottom_center) + d)
print(f"Distance from box bottom to table: {distance_to_plane:.4f} m")

# Visualize box bottom point and projection
box_bottom_point = o3d.geometry.PointCloud()
box_bottom_point.points = o3d.utility.Vector3dVector([box_bottom_center])
box_bottom_point.paint_uniform_color([1, 0, 0])

table_point = box_bottom_center - distance_to_plane * plane_normal
line_points = [box_bottom_center, table_point]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

# Step 6: Adjust height and recompute corrected box extent
visible_box_height = extent[gravity_axis]
full_box_height = visible_box_height + distance_to_plane
corrected_extent = np.copy(extent)
corrected_extent[gravity_axis] = full_box_height

print(f"Corrected box dimensions (WxHxD): {corrected_extent[0]*100:.2f} x {corrected_extent[1]*100:.2f} x {corrected_extent[2]*100:.2f} cm")

# Step 7: Create corrected box and transform it
corrected_box = o3d.geometry.TriangleMesh.create_box(
    width=corrected_extent[0],
    height=corrected_extent[1],
    depth=corrected_extent[2]
)
corrected_box.compute_vertex_normals()
corrected_box.paint_uniform_color([0.75, 1, 0.75])
corrected_box.translate(-corrected_extent / 2)

corrected_box_transform = np.eye(4)
corrected_box_transform[:3, :3] = box_obb.R
corrected_box_center = table_point + (corrected_extent[gravity_axis] / 2) * plane_normal
corrected_box_transform[:3, 3] = corrected_box_center
corrected_box.transform(corrected_box_transform)

# Step 8: Save and visualize
np.save("data/box_dimensions_corrected.npy", corrected_extent)
np.save("data/box_pose_corrected.npy", corrected_box_transform)

print("Saved corrected box dimensions and pose.")

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
best_plane.paint_uniform_color([0.8, 0.8, 0.8])  # Visualize selected plane

# Visualize
o3d.visualization.draw_geometries([
    box_pcd, 
    best_plane,
    box_obb,
    box_bottom_point,
    line_set,
    coordinate_frame
], window_name="Box to Table Distance")

o3d.visualization.draw_geometries([
    scene_pcd,
    corrected_box,
    coordinate_frame
], window_name="Corrected Box in Scene")
