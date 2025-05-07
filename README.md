# PoseEstimation
Estimate 3d pose using single depth and color image
# 3D Box Pose Estimation from RGB-D Data

This repository contains code for a 3D object pose estimation pipeline that processes RGB-D data to identify and locate box-shaped objects in a scene. The implementation uses point cloud processing techniques to segment planes and cluster objects.

## Project Overview

This project was developed as a solution to a robotics perception task that requires estimating the 3D pose of a box-shaped object from depth data. The pipeline includes:

- RGB-D data loading and preprocessing
- Point cloud generation from depth data
- Planar segmentation using RANSAC
- Object clustering using DBSCAN
- Pose estimation of identified objects

## Repository Structure

```
├── main.py                 # Main script integrating the complete pipeline
├── data/                   # Directory for input data files
│   ├── one-box.color.npdata.npy    # Color image data
│   ├── one-box.depth.npdata.npy    # Depth image data
│   ├── extrinsics.npy              # Camera extrinsic parameters
│   └── intrinsics.npy              # Camera intrinsic parameters
├── results/                # Directory for output results (created by script)
│   ├── rgbd_visualization.png      # RGB-D data visualization
│   ├── scene.pcd                   # Full scene point cloud
│   ├── table_plane.pcd             # Segmented table plane
│   ├── clustered_objects.pcd       # Clustered objects
│   └── cluster_*.pcd               # Individual object clusters
└── README.md               # This file
```

## Dependencies

This project requires the following Python libraries:

```
numpy
opencv-python (cv2)
open3d
matplotlib
```

You can install the dependencies using:

```bash
pip install numpy opencv-python open3d matplotlib
```

## Usage

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/3d-box-pose-estimation.git
   cd 3d-box-pose-estimation
   ```

2. Create a data directory and add the required input files:
   ```bash
   mkdir -p data
   # Copy the input files to the data directory
   # - one-box.color.npdata.npy
   # - one-box.depth.npdata.npy
   # - extrinsics.npy
   # - intrinsics.npy
   ```

### Running the Pipeline

Execute the main script to run the complete pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the RGB-D data
2. Generate a 3D point cloud
3. Segment the table plane
4. Cluster and identify objects
5. Save results to the `results` directory
6. Display visualizations of each step

### Customization

You can adjust key parameters in the `main.py` file:

- `distance_threshold`: Threshold for planar segmentation (default: 0.2)
- `eps`: Maximum distance between points in a cluster for DBSCAN (default: 0.03)
- `min_points`: Minimum number of points to form a cluster (default: 30)

## Pipeline Details

### 1. Data Loading

The pipeline begins by loading RGB-D data and camera parameters, providing basic information about the input data dimensions and ranges.

### 2. Point Cloud Generation

RGB-D data is transformed into a 3D point cloud using the camera intrinsic parameters, applying appropriate coordinate transformations for visualization.

### 3. Planar Segmentation

RANSAC algorithm is used to identify the dominant plane (table surface) in the scene, separating it from potential objects.

### 4. Object Clustering

DBSCAN clustering is applied to non-planar points to identify distinct objects, with visualization of the clustering results.

## Results

The pipeline generates several files in the `results` directory:

- Point cloud files (.pcd) for the scene, table plane, and clustered objects
- Visualization images (.png) showing RGB-D data and processing results

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses the [Open3D](http://www.open3d.org/) library for point cloud processing and visualization.