from utils.pointcloud_utils import generate_pointcloud_from_npy
from utils.segmentation_utils import segment_table_and_objects
from utils.box_pose_utils import compute_corrected_box_pose_and_dimensions

def main():
    # Step 1: Generate point cloud from RGB-D + calibration data
    print("[1] Generating point cloud from NPY files...")
    pcd = generate_pointcloud_from_npy(
        color_path="data/one-box.color.npdata.npy",
        depth_path="data/one-box.depth.npdata.npy",
        intrinsics_path="data/intrinsics.npy",
        extrinsics_path="data/extrinsics.npy",
        save_path="data/stage.pcd",
        visualize=True
    )

    # Step 2: Segment the table and objects
    print("[2] Segmenting table and objects from the scene...")
    box = segment_table_and_objects(
        scene_path="data/stage.pcd",
        table_save_path="data/table.pcd",
        objects_save_path="data/objects.pcd",
        box_save_path="data/box_model_extracted.pcd",
        visualize=True,
        target_cluster_idx=0,  # Change this index if the box isn't cluster 0
        plane_distance_threshold=0.18,
        dbscan_eps=0.03,
        dbscan_min_points=30
    )

    # Step 3: Compute box pose and dimensions
    print("[3] Computing box pose and dimensions...")
    dims, pose = compute_corrected_box_pose_and_dimensions(
        scene_pcd_path="data/stage.pcd",
        box_model_path="data/box_model_extracted.pcd",
        table_pcd_path="data/table.pcd"
    )

    print("Final box dimensions (meters):", dims)
    print("Final box pose:\n", pose)

if __name__ == "__main__":
    main()
