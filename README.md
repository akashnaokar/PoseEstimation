# 3D Box Pose and Dimension Estimation

A Python-based system for accurate estimation of box dimensions and pose from RGB-D camera data. This repository implements a pipeline that processes depth and color images to generate point clouds, segment objects, and calculate precise measurements of box-shaped objects on a table surface.

## Overview

This system solves the problem of automatically measuring box dimensions and determining their orientation in 3D space using RGB-D camera data. The pipeline consists of:

1. Point cloud generation from RGB-D data
2. Table and object segmentation
3. Box extraction and pose correction
4. Accurate dimension estimation

## Directory Structure

```
.
├── data/                           # Input and output data directory
│   ├── intrinsics.npy              # Camera intrinsic parameters (3x3 matrix)
│   ├── extrinsics.npy              # Camera extrinsic parameters (4x4 matrix)
│   ├── one-box.color.npdata.npy    # Color image as NumPy array
│   └── one-box.depth.npdata.npy    # Depth image as NumPy array
│
├── utils/                          # Utility modules
│   ├── box_pose_utils.py           # Box pose and dimension computation
│   ├── pointcloud_utils.py         # Point cloud generation from RGB-D data
│   └── segmentation_utils.py       # Point cloud segmentation utilities
│
├── main.py                         # Main execution script
└── README.md                       # This documentation file
```

## Requirements

### Dependencies

- NumPy
- Open3D
- OpenCV (cv2)
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

### Input Data Format

- **Color images**: RGB images stored as NumPy arrays (.npy)
- **Depth images**: Depth maps stored as NumPy arrays (.npy)
- **Camera intrinsics**: 3×3 camera intrinsic parameter matrix as NumPy array (.npy)
- **Camera extrinsics**: 4×4 camera extrinsic parameter matrix as NumPy array (.npy)

## Usage

### Step 1: Prepare Data

Place your input data in the `data/` directory:
- RGB image: `data/one-box.color.npdata.npy`
- Depth image: `data/one-box.depth.npdata.npy`
- Camera intrinsics: `data/intrinsics.npy`
- Camera extrinsics: `data/extrinsics.npy`

### Step 2: Run the Pipeline

Execute the main script:
```bash
python main.py
```

This will:
1. Generate a point cloud from the RGB-D data
2. Segment the table and objects in the scene
3. Extract the box object
4. Compute and correct the box pose and dimensions

### Step 3: View Results

The pipeline generates several output files in the `data/` directory:
- `stage.pcd`: Full scene point cloud
- `table.pcd`: Segmented table point cloud
- `objects.pcd`: Segmented object clusters
- `box_model_extracted.pcd`: Extracted box point cloud
- `box_dimensions_corrected.npy`: Corrected box dimensions (width, height, depth) in meters
- `box_pose_corrected.npy`: 4×4 transformation matrix representing box pose

Additionally, several visualization windows will appear during execution if `visualize=True` is set.

## Configuration

Key parameters can be adjusted in `main.py`:

```python
# Point cloud generation
pcd = generate_pointcloud_from_npy(
    color_path="data/one-box.color.npdata.npy",
    depth_path="data/one-box.depth.npdata.npy",
    intrinsics_path="data/intrinsics.npy",
    extrinsics_path="data/extrinsics.npy",
    save_path="data/stage.pcd",
    visualize=True  # Set to False to disable visualization
)

# Table and object segmentation
box = segment_table_and_objects(
    scene_path="data/stage.pcd",
    table_save_path="data/table.pcd",
    objects_save_path="data/objects.pcd",
    box_save_path="data/box_model_extracted.pcd",
    visualize=True,  # Set to False to disable visualization
    target_cluster_idx=0,  # Change this index if the box isn't cluster 0
    plane_distance_threshold=0.18,  # RANSAC threshold for plane detection
    dbscan_eps=0.03,  # DBSCAN epsilon for clustering
    dbscan_min_points=30  # Minimum points per cluster
)
```
<!--
## Function Documentation

### Point Cloud Generation

```python
generate_pointcloud_from_npy(color_path, depth_path, intrinsics_path, extrinsics_path, save_path=None, visualize=False)
```

Generates a point cloud from RGB-D data using Open3D.

### Table and Object Segmentation

```python
segment_table_and_objects(scene_path, table_save_path=None, objects_save_path=None, box_save_path=None, visualize=False, target_cluster_idx=0, plane_distance_threshold=0.18, dbscan_eps=0.03, dbscan_min_points=30)
```

Segments a scene point cloud into a table and clustered objects using RANSAC and DBSCAN.

### Box Pose and Dimension Computation

```python
compute_corrected_box_pose_and_dimensions(scene_pcd_path, box_model_path, table_pcd_path, save_dir="data")
```

Computes the corrected pose and dimensions of a box placed on a table using 3D point cloud data.
-->
## Troubleshooting

1. **No objects detected**: Try adjusting `plane_distance_threshold`, `dbscan_eps`, and `dbscan_min_points`.
2. **Wrong object selected**: Change `target_cluster_idx` to select a different cluster.
3. **Inaccurate dimensions**: Check that the table is properly segmented and visible in the point cloud.

## Custom Data

To use your own data:
1. Convert your RGB-D images to NumPy arrays
2. Ensure camera calibration parameters are stored as NumPy arrays
3. Update file paths in `main.py`
4. Adjust segmentation parameters if needed

## Limitations

- Works best with box-shaped objects on flat surfaces
- Requires good quality RGB-D data with minimal noise
- Box should be clearly visible and not occluded

## Example Output

When the pipeline completes successfully, it will print information like:

```
Final box dimensions (meters): [0.152 0.076 0.198]
Final box pose:
[[ 0.989  0.012  0.147  0.231]
 [-0.013  0.999  0.002  0.032]
 [-0.147  0.000  0.989  0.114]
 [ 0.000  0.000  0.000  1.000]]
```

This indicates a box with dimensions 15.2 × 7.6 × 19.8 cm positioned at coordinates (0.231, 0.032, 0.114) in the camera frame.