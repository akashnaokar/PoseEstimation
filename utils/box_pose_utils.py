import numpy as np
import open3d as o3d

def compute_corrected_box_pose_and_dimensions(
    scene_pcd_path: str,
    box_model_path: str,
    table_pcd_path: str,
    save_dir: str = "data"
):
    """
    Computes the corrected pose and dimensions of a box placed on a table using 3D point cloud data.
    
    Args:
        scene_pcd_path (str): Path to the full scene point cloud (.pcd file).
        box_model_path (str): Path to the extracted box point cloud (.pcd file).
        table_pcd_path (str): Path to the table point cloud (.pcd file).
        save_dir (str): Directory where the corrected pose and dimensions will be saved.

    Returns:
        corrected_extent (np.ndarray): Corrected 3D size of the box (width, height, depth) in meters.
        corrected_box_transform (np.ndarray): 4x4 homogeneous transform matrix representing the box pose.
    """
    # Load point clouds
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)
    box_pcd_full = o3d.io.read_point_cloud(box_model_path)
    table_pcd = o3d.io.read_point_cloud(table_pcd_path)
    box_pcd = box_pcd_full.voxel_down_sample(voxel_size=0.01)

    # Estimate normals
    scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    box_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

    # PCA on box
    points = np.asarray(box_pcd.points)
    mean = np.mean(points, axis=0)
    cov = np.cov((points - mean).T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    projected = np.dot(points - mean, eigenvectors)
    min_bound = np.min(projected, axis=0)
    max_bound = np.max(projected, axis=0)
    extent = max_bound - min_bound

    # Refine orientation using normals
    normals = np.asarray(box_pcd.normals)
    abs_normals = np.abs(normals)
    x_dom, y_dom, z_dom = (abs_normals[:, i] > 0.9 for i in range(3))

    refined_eigenvectors = eigenvectors.copy()
    axes_refined = 0
    for i, dominant in enumerate([x_dom, y_dom, z_dom]):
        if np.sum(dominant) > 100:
            avg_normal = np.mean(normals[dominant], axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            refined_eigenvectors[:, i] = avg_normal * np.sign(np.dot(avg_normal, eigenvectors[:, i]))
            axes_refined += 1

    if axes_refined >= 2:
        refined_eigenvectors[:, 1] = np.cross(refined_eigenvectors[:, 2], refined_eigenvectors[:, 0])
        refined_eigenvectors[:, 1] /= np.linalg.norm(refined_eigenvectors[:, 1])
        refined_eigenvectors[:, 2] = np.cross(refined_eigenvectors[:, 0], refined_eigenvectors[:, 1])
        refined_eigenvectors[:, 2] /= np.linalg.norm(refined_eigenvectors[:, 2])
        eigenvectors = refined_eigenvectors

    box_obb = o3d.geometry.OrientedBoundingBox(center=mean, R=eigenvectors, extent=extent)

    # Get bottom center
    gravity_alignment = np.abs(np.dot(eigenvectors.T, np.array([0, 0, 1])))
    gravity_axis = np.argmax(gravity_alignment)
    bottom_offset = eigenvectors[:, gravity_axis] * (extent[gravity_axis] / 2)
    if bottom_offset[2] > 0:
        bottom_offset = -bottom_offset
    box_bottom_center = box_obb.center + bottom_offset

    # Segment planes
    remaining_pcd = scene_pcd.voxel_down_sample(voxel_size=0.005)
    remaining_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    planes, plane_models = [], []
    for _ in range(5):
        plane_model, inliers = remaining_pcd.segment_plane(0.01, 3, 500)
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        if len(inlier_cloud.points) < 15000:
            continue
        planes.append(inlier_cloud)
        plane_models.append(plane_model)

    # Find closest plane to box bottom
    min_dist = float('inf')
    best_plane_idx = -1
    for i, model in enumerate(plane_models):
        a, b, c, d = model
        dist = abs(np.dot([a, b, c], box_bottom_center) + d) / np.linalg.norm([a, b, c])
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

    print(f"Corrected box dimensions (WxHxD): {corrected_extent[0]*1000:.2f} x {corrected_extent[1]*1000:.2f} x {corrected_extent[2]*1000:.2f} mm")

    # Step 7: Create corrected box and transform it
    corrected_box = o3d.geometry.TriangleMesh.create_box(
        width=corrected_extent[0],
        height=corrected_extent[1],
        depth=corrected_extent[2]
    )
    corrected_box.compute_vertex_normals()
    corrected_box.paint_uniform_color([0.25, 0.25, 0.25])
    corrected_box.translate(-corrected_extent / 2)

    corrected_box_transform = np.eye(4)
    corrected_box_transform[:3, :3] = box_obb.R
    corrected_box_center = table_point + (corrected_extent[gravity_axis] / 2) * plane_normal
    corrected_box_transform[:3, 3] = corrected_box_center
    corrected_box.transform(corrected_box_transform)

    print("Box pose in camera frame:")
    print(corrected_box_transform)

    # Save results
    np.save(f"{save_dir}/box_dimensions_corrected.npy", corrected_extent)
    np.save(f"{save_dir}/box_pose_corrected.npy", corrected_box_transform)

    print("Saved corrected box dimensions and pose.")
    print(f"Corrected box dimensions (m): {corrected_extent}")
    print(f"Box pose:\n{corrected_box_transform}")
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    best_plane.paint_uniform_color([0.8, 0.8, 0.8])  # Visualize selected plane 

    # Visualize
    table_pcd.paint_uniform_color([1, 0.5, 0])
    o3d.visualization.draw_geometries([
        box_pcd, 
        best_plane,
        box_obb,
        box_bottom_point,
        line_set,
        coordinate_frame
    ], window_name="Box to Table Distance")

    o3d.visualization.draw_geometries([
        table_pcd,
        corrected_box,
        coordinate_frame
    ], window_name="Corrected Box in Scene")

    return corrected_extent, corrected_box_transform
