import numpy as np 
from PIL import Image
import matplotlib.image
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from typing import List, Tuple
import colorsys
import os 
import argparse
import tifffile

############## PLAN ##################
'''

0th version: just retrieve the pointclouds using semantic segmentation and pass it to Andrea's code

1st version: load images and do the clustering using DBSCAN on a pointcloud.
            - check if the clusters converge with semantic segmentation
            
2nd version - use segmentation in clustering, but how? It's just a label provided by another model

Pass segmented pointcloud to Andreas code to fit the cylinder

ALSO: collect everything in one repo to have infer + pointcloud creation + Andreas model in end to end manner

'''


def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    img = Image.open(file_path)
    img_array = np.array(img)
   # Display the image
    plt.imshow(img_array)
    plt.axis('off')  # Turn off axis numbers
    plt.show()
    return img_array

def apply_threshold_to_semantics(semantic_image, threshold=0.3):
    """
    Apply a threshold to a semantic segmentation probability map.
    
    Parameters:
    semantic_image (numpy.ndarray): x by y array where each value represents
                                    the probability of a pixel belonging to a particular class.
    threshold (float): The threshold value (default is 0.5).
    
    Returns:
    numpy.ndarray: Binary mask where 1 indicates the pixel belongs to the class,
                   and 0 indicates it doesn't.
    """
    # Ensure the input is a numpy array
    semantic_image = np.array(semantic_image)
    
    # Normalize the array to [0, 1] range
    min_val = np.min(semantic_image)
    max_val = np.max(semantic_image)
    
    if min_val != max_val:
        normalized_image = (semantic_image - min_val) / (max_val - min_val)
    else:
        # Handle the case where all values are the same
        normalized_image = np.zeros_like(semantic_image)
    
    # Apply threshold to the normalized image
    binary_mask = (normalized_image >= threshold).astype(int)
    
    return binary_mask


def binary_mask_to_rgb(binary_mask, color_positive=(255, 0, 0), color_negative=(0, 0, 255)):
    """
    Convert a binary mask to an RGB image.
    
    Parameters:
    binary_mask (numpy.ndarray): 2D array where 1 indicates positive class and 0 indicates negative class.
    color_positive (tuple): RGB color for positive class (default is red: (255, 0, 0)).
    color_negative (tuple): RGB color for negative class (default is blue: (0, 0, 255)).
    
    Returns:
    numpy.ndarray: RGB image representation of the binary mask.
    """
    # Ensure the input is a numpy array
    binary_mask = np.array(binary_mask)
    
    # Create an empty RGB image
    rgb_image = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    
    # Set colors for positive and negative classes
    rgb_image[binary_mask == 1] = color_positive
    rgb_image[binary_mask == 0] = color_negative
    
    return rgb_image

    
def create_pointcloud_from_rgbd(rgb_image, depth_image, semantic_image, depth_unit='meters', depth_trunc=5.0, colors = False):
    """
    Create a pointcloud from RGB, depth, and semantic images.
    
    Parameters:
    - depth_unit: 'meters' or 'millimeters'
    - depth_trunc: Maximum depth value in meters
    """
    height, width = rgb_image.shape[:2]
    
    # Ensure depth_image is in float32 format
    depth_image = depth_image.astype(np.float32)
    
    # Set depth_scale based on the unit
    if depth_unit == 'millimeters':
        depth_scale = 1000.0  # 1 meter = 1000 millimeters
    elif depth_unit == 'meters':
        depth_scale = 1.0
    else:
        raise ValueError("depth_unit must be 'meters' or 'millimeters'")

    # Debug: Print depth image statistics
    print(f"Depth image shape: {depth_image.shape}")
    print(f"Depth range: {depth_image.min() * depth_scale:.3f} to {depth_image.max() * depth_scale:.3f} meters")
    
    # Create Open3D images
    rgb_o3d = o3d.geometry.Image(rgb_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

    # Create pinhole camera intrinsic, scene in the middel of the picture
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx=500, fy=500, cx=width/2, cy=height/2)

    # Create pointcloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    # Apply voxel downsampling to reduce noise and point density
    #pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    # Remove statistical outliers
    #pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    semantics_masked = apply_threshold_to_semantics(semantic_image, threshold=0.4)
    
    #transfered mask to rgb for open3d visualisation
    semantics_masked_rgb = binary_mask_to_rgb(semantics_masked)
    
    if colors == True: 
        # Extract RGB colors from the image
        colors = rgb_image[:, :, :3].reshape(-1, 3) / 255.0  # Normalize to [0, 1]
        #print(colors.shape)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Extract colors from the semantic image
        semantics_color =  semantics_masked_rgb[:, :, :3].reshape(-1, 3) / 255.0 # Normalize to [0, 1]
        #print(semantics_color.shape)
        pcd.colors = o3d.utility.Vector3dVector(semantics_color)

    return pcd, semantics_masked

def filter_save_visualize_pointcloud(pcd, mask, center=True, min_points=10):
    """
    Filter a point cloud based on a mask, optionally center it around zero, and perform diagnostics.

    Parameters:
    pcd (open3d.geometry.PointCloud): The input point cloud.
    mask (numpy.ndarray): Boolean mask for filtering points (True for points to keep).
    center (bool): Whether to center the filtered point cloud around zero.
    min_points (int): Minimum number of points required to proceed.

    Returns:
    open3d.geometry.PointCloud: The filtered and optionally centered point cloud.
    dict: Diagnostic information.
    """
    
    # Ensure mask is the correct shape
    if mask.ndim == 2:
        mask = mask.flatten()
    elif mask.ndim != 1:
        raise ValueError("Mask must be either 1D or 2D.")

    if mask.shape[0] != len(pcd.points):
        raise ValueError(f"Mask shape {mask.shape} does not match number of points in the point cloud ({len(pcd.points)}).")

    # Convert mask to boolean
    mask_bool = mask.astype(bool)

    # Create a new point cloud with only the masked points
    masked_pcd = pcd.select_by_index(np.where(mask_bool)[0])
    
    # Check if we have enough points
    if len(masked_pcd.points) < min_points:
        raise ValueError(f"Not enough points after masking. Got {len(masked_pcd.points)}, need at least {min_points}.")

    # Initialize diagnostics
    diagnostics = {
        "original_points": len(pcd.points),
        "masked_points": len(masked_pcd.points),
        "mask_ratio": len(masked_pcd.points) / len(pcd.points),
    }

    if center:
        # Calculate the centroid of the masked point cloud
        centroid = np.mean(np.asarray(masked_pcd.points), axis=0)
        
        # Center the points by subtracting the centroid
        centered_points = np.asarray(masked_pcd.points) - centroid
        
        # Create a new point cloud with the centered points
        centered_pcd = o3d.geometry.PointCloud()
        centered_pcd.points = o3d.utility.Vector3dVector(centered_points)
        
        # Copy colors if they exist
        if masked_pcd.has_colors():
            centered_pcd.colors = masked_pcd.colors
        
        # Add centering diagnostics
        diagnostics["centroid"] = centroid.tolist()
        diagnostics["centered_std"] = np.std(centered_points, axis=0).tolist()
        
        result_pcd = centered_pcd
    else:
        result_pcd = masked_pcd

    # Compute the covariance matrix of the points
    covariance = np.cov(np.asarray(result_pcd.points).T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    
    # Sort eigenvalues and corresponding eigenvectors
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Add eigenvalue diagnostics
    diagnostics["eigenvalues"] = eigenvalues.tolist()
    diagnostics["planarity"] = 1 - eigenvalues[2] / eigenvalues[1]  # Should be close to 1 for planar data

    return result_pcd, diagnostics


def estimate_eps(points: np.ndarray, n_neighbors: int = 2) -> float:
    """
    Estimate a good eps value for DBSCAN using the knee method.

    Args:
    points (np.ndarray): The input points.
    n_neighbors (int): Number of neighbors to consider.

    Returns:
    float: Estimated eps value.
    """
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(points)
    distances, indices = nbrs.kneighbors(points)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]  # Get the distance to the nearest neighbor
    
    kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
    eps = distances[kneedle.knee]
    
    return eps

def generate_distinct_colors(n):
    """Generate n distinct colors"""
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    return [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]

def add_noise(points: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Add small random noise to the points.
    
    Args:
    points (np.ndarray): Array of shape (n, 3) containing n 3D points.
    noise_level (float): Standard deviation of the Gaussian noise to be added.
    
    Returns:
    np.ndarray: Array of shape (n, 3) with added noise.
    """
    noise = np.random.normal(0, noise_level, points.shape)
    return points + noise

def normalize_o3d_point_cloud(pcd):
    # Convert to numpy array
    points = np.asarray(pcd.points)

    # Center the point cloud
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Scale the point cloud
    max_distance = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
    normalized_points = centered_points / max_distance

    # Create a new Open3D point cloud with normalized points
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)

    # If the original point cloud has colors, normalize them too
    if pcd.has_colors():
        normalized_pcd.colors = pcd.colors

    # If the original point cloud has normals, keep them (they don't need normalization)
    if pcd.has_normals():
        normalized_pcd.normals = pcd.normals

    return normalized_pcd, centroid, max_distance


def adaptive_clustering(pcd: o3d.geometry.PointCloud, 
                        method: str = 'optics', 
                        min_samples: int = 10, 
                        max_eps: float = np.inf) -> List[o3d.geometry.PointCloud]:
    """
    Perform adaptive clustering on a point cloud using either OPTICS or DBSCAN with estimated eps.

    Args:
    pcd (open3d.geometry.PointCloud): The input point cloud.
    method (str): Clustering method, either 'optics' or 'dbscan'.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    max_eps (float): Maximum eps value for OPTICS (only used if method is 'optics').

    Returns:
    List[open3d.geometry.PointCloud]: A list of sub-point clouds.
    """
    points = np.asarray(pcd.points)

    if method == 'optics':
        clustering = OPTICS(min_samples=min_samples, max_eps=max_eps)
        labels = clustering.fit_predict(points)
    elif method == 'dbscan':
        eps = estimate_eps(points)
        print(f"Estimated eps: {eps}")
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points)
    else:
        raise ValueError("Method must be either 'optics' or 'dbscan'")

    # Generate distinct colors for each cluster
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise if present
    distinct_colors = generate_distinct_colors(n_clusters)
    color_map = {label: color for label, color in zip([l for l in unique_labels if l != -1], distinct_colors)}
    color_map[-1] = (0.5, 0.5, 0.5)  # Gray color for noise points

    # Split into sub-point clouds
    sub_point_clouds = []
    for label in unique_labels:
        if label != -1:  # Ignore noise points
            cluster_points = points[labels == label]
            
            # Add noise to the cluster points to make them non-colinear preserving the structure of pointcloud
            #noisy_cluster_points = add_noise(cluster_points, 0.0005)
            
            sub_pcd = o3d.geometry.PointCloud()
            sub_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            
            # Assign color to the cluster
            cluster_color = np.array([color_map[label]] * len(cluster_points))
            sub_pcd.colors = o3d.utility.Vector3dVector(cluster_color)
            
            sub_point_clouds.append(sub_pcd)

    return sub_point_clouds

def analyze_point_cloud_geometry(pcd: o3d.geometry.PointCloud) -> dict:
    """
    Analyze the geometry of a point cloud using PCA.
    
    Args:
    pcd (open3d.geometry.PointCloud): The input point cloud.
    
    Returns:
    dict: A dictionary containing geometric properties of the point cloud.
    """
    points = np.asarray(pcd.points)
    pca = PCA(n_components=3)
    pca.fit(points)
    
    eigenvalues = pca.explained_variance_
    total_variance = np.sum(eigenvalues)
    
    # Calculate various geometric properties
    linearity = (eigenvalues[0] - eigenvalues[1]) / total_variance
    planarity = (eigenvalues[1] - eigenvalues[2]) / total_variance
    scattering = eigenvalues[2] / total_variance
    
    return {
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "num_points": len(points)
    }

def classify_point_cloud(geometry: dict, 
                         planarity_threshold: float = 0.6, 
                         linearity_threshold: float = 0.6,
                         min_points: int = 10) -> str:
    """
    Classify a point cloud based on its geometric properties.
    
    Args:
    geometry (dict): Dictionary of geometric properties.
    planarity_threshold (float): Threshold for planarity.
    linearity_threshold (float): Threshold for linearity.
    min_points (int): Minimum number of points to consider.
    
    Returns:
    str: Classification of the point cloud.
    """
    if geometry['num_points'] < min_points:
        return "too_small"
    elif geometry['planarity'] > planarity_threshold:
        return "planar"
    elif geometry['linearity'] > linearity_threshold:
        return "linear"
    else:
        return "complex"

def filter_point_clouds(sub_point_clouds: List[o3d.geometry.PointCloud], 
                        planarity_threshold: float = 0.6,
                        linearity_threshold: float = 0.6,
                        min_points: int = 10) -> Tuple[List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud]]:
    """
    Filter and classify point clouds based on their geometry.
    
    Args:
    sub_point_clouds (List[open3d.geometry.PointCloud]): List of sub-point clouds.
    planarity_threshold (float): Threshold for determining planarity.
    linearity_threshold (float): Threshold for determining linearity.
    min_points (int): Minimum number of points to consider a sub-point cloud.
    
    Returns:
    Tuple[List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud], List[o3d.geometry.PointCloud]]: 
        Complex, planar, linear, and too small point clouds.
    """
    complex_clouds = []
    planar_clouds = []
    linear_clouds = []
    too_small_clouds = []
    
    for sub_pcd in sub_point_clouds:
        geometry = analyze_point_cloud_geometry(sub_pcd)
        classification = classify_point_cloud(geometry, planarity_threshold, linearity_threshold, min_points)
        
        if classification == "complex":
            complex_clouds.append(sub_pcd)
        elif classification == "planar":
            planar_clouds.append(sub_pcd)
        elif classification == "linear":
            linear_clouds.append(sub_pcd)
        else:  # too_small
            too_small_clouds.append(sub_pcd)
    
    return complex_clouds, planar_clouds, linear_clouds, too_small_clouds



def save_point_clouds(point_clouds: List[o3d.geometry.PointCloud], 
                      output_dir: str, 
                      base_name: str = "pointcloud", 
                      format: str = "ply") -> None:
    """
    Save a list of point clouds to individual files.

    Args:
    point_clouds (List[o3d.geometry.PointCloud]): List of point clouds to save.
    output_dir (str): Directory to save the point clouds.
    base_name (str): Base name for the point cloud files.
    format (str): File format to save the point clouds. Options: "ply", "pcd", "xyz".

    Returns:
    None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Supported formats and their corresponding save functions
    save_functions = {
        "ply": o3d.io.write_point_cloud,
        "pcd": o3d.io.write_point_cloud,
        "xyz": o3d.io.write_point_cloud
    }

    if format.lower() not in save_functions:
        raise ValueError(f"Unsupported format: {format}. Supported formats are: {', '.join(save_functions.keys())}")

    save_func = save_functions[format.lower()]

    for i, pcd in enumerate(point_clouds):
        filename = f"{base_name}_{i:04d}.{format.lower()}"
        filepath = os.path.join(output_dir, filename)
        success = save_func(filepath, pcd)
        if success:
            print(f"Saved point cloud to {filepath}")
        else:
            print(f"Failed to save point cloud to {filepath}")

def plot_info(rgb, depth, semantics, mask): 
    
        # Create a figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot RGB image
    ax1.imshow(rgb)
    ax1.set_title('RGB Image')
    ax1.axis('off')

    # Plot Depth image
    ax2.imshow(depth, cmap='jet')
    ax2.set_title('Depth Estimation')
    ax2.axis('off')

    # Plot Semantic image
    ax3.imshow(semantics, cmap='jet')
    ax3.set_title('Semantic Segmentation')
    ax3.axis('off')
    
    # Plot Masekd Semantic image
    ax4.imshow(mask, cmap='jet')
    ax4.set_title('Masked Semantics')
    ax4.axis('off')

    # Adjust the layout and display the plot
    plt.tight_layout()
    output_path = 'combined_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Plot saved to {output_path}")

def convert_depth_image(depth):
    
    rgba_image = depth
    
    # Convert the image to 32-bit float
    rgba_image = rgba_image.astype(np.float32)
    
    # Normalize each channel to [0, 1] range
    rgba_image /= 255.0
    
    # Combine channels
    # This assumes each channel contributes equally to the final depth
    # You may need to adjust the weights if some channels are more significant
    depth_image = (rgba_image[:,:,0] + 
                   rgba_image[:,:,1] * 256 + 
                   rgba_image[:,:,2] * 256**2 + 
                   rgba_image[:,:,3] * 256**3)
    
    # Normalize the combined depth to [0, 1] range
    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    
    # Convert back to 8-bit for saving as an image
    depth_image = (depth_image * 255).astype(np.uint8)
    
    print(f"Converted depth image shape: {depth_image.shape}")
    
    # Optionally, save the depth image
    Image.fromarray(depth_image).save("depth_image_claude.png")
    
    return depth_image

def load_and_visualize_images(rgb_path, depth_path, semantic_path):
    # Load RGB image
    rgb_image = np.load(rgb_path)
    print(f"Loaded RGB image with shape: {rgb_image.shape}")

    # Load depth image
    depth_image = tifffile.imread(depth_path)
    print(f"Loaded depth image with shape: {depth_image.shape}")

    # Load semantic image
    semantic_image = np.load(semantic_path)
    print(f"Loaded semantic image with shape: {semantic_image.shape}")

    # Visualize the images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # RGB image
    ax1.imshow(rgb_image)
    ax1.set_title('RGB Image')
    ax1.axis('off')

    # Depth image
    depth_display = ax2.imshow(depth_image, cmap='viridis')
    ax2.set_title('Depth Image')
    ax2.axis('off')
    #plt.colorbar(depth_display, ax=ax2, fraction=0.046, pad=0.04)

    # Semantic image
    semantic_display = ax3.imshow(semantic_image, cmap='jet')
    ax3.set_title('Semantic Image (TV Monitor)')
    ax3.axis('off')
    #plt.colorbar(semantic_display, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return rgb_image, depth_image, semantic_image

def main():
    
    #parser=argparse.ArgumentParser()
    #parser.add_argument("--input", help="Orginal input image")
    #parser.add_argument("--input_pre_rgb", help="Input preprocessed RGB image")
    #parser.add_argument("--input_depth", help="Input RGB image")
    #parser.add_argument("--output_path", help="Output folder")
    #parser.add_argument("--output_ply_dir", help = "Output folder of .ply files")

    #args=parser.parse_args()
    
    # Load images
    rgb_path_viz = "data_check/rgb_0_visual.png"
    depth_path_viz = "data_check/depth_0_visual.png"  
    semantic_path_viz = "data_check/semantic_0_tvmonitor_visual.png"
    
    rgb_path = "data_check/rgb_0.npy"
    depth_path = "data_check/depth_0.tiff"
    semantic_path = "data_check/semantic_0_tvmonitor.npy"
    
    output_path = "./"
    
    rgb_image, depth_image, semantic_image = load_and_visualize_images(rgb_path, depth_path, semantic_path)
    
    # Create pointcloud
    pcd, mask = create_pointcloud_from_rgbd(rgb_image, depth_image, semantic_image, colors = True)
    
    # Optional: Estimate normals for better visualization
    #pcd.estimate_normals()
    o3d.io.write_point_cloud("rgb_pointcloud.ply", pcd)
    
    # Visualize pointcloud
    o3d.visualization.draw_geometries([pcd])
    
    masked_pcd, diagnostics = filter_save_visualize_pointcloud(pcd, mask, center=False)

    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    # Check planarity
    if diagnostics["planarity"] > 0.99:
        print("Warning: The point cloud is very close to planar. This may cause issues with plane fitting.")

    # Check if points are too few or too collinear
    if min(diagnostics["eigenvalues"]) / max(diagnostics["eigenvalues"]) < 1e-6:
        print("Warning: The points may be too collinear for reliable plane fitting.")
    
    # Save the masked point cloud
    o3d.io.write_point_cloud(output_path, masked_pcd)
    print(f"Filtered point cloud saved to {output_path}")

    # Visualize pointcloud
    o3d.visualization.draw_geometries([masked_pcd], window_name = "Masked pointcloud")
    
    normalized_pcd, centroid, scale = normalize_o3d_point_cloud(masked_pcd)
    #print(np.asarray(normalized_pcd.points))
    
    # To denormalize :
    # original_points = (np.asarray(normalized_pcd.points) * scale) + centroid
    # original_pcd = o3d.geometry.PointCloud()
    # original_pcd.points = o3d.utility.Vector3dVector(original_points)
    
    #o3d.visualization.draw_geometries([normalized_pcd], window_name = "Normalised pointcloud")
        
    # Preprocess the point cloud using adaptive clustering to generate smaller pointcloudss
    sub_point_clouds = adaptive_clustering(normalized_pcd, method='optics', min_samples=150, max_eps=0.5)
    
    print(f"Number of sub-point clouds: {len(sub_point_clouds)}")
    
    #sub_point_clouds is already a list of o3d geometries
    o3d.visualization.draw_geometries(sub_point_clouds, window_name = "Clusters")
    
    # Filter and classify point clouds
    complex_clouds, planar_clouds, linear_clouds, too_small_clouds = filter_point_clouds(sub_point_clouds)
    
    print(f"Number of complex sub-point clouds: {len(complex_clouds)}")
    print(f"Number of planar sub-point clouds: {len(planar_clouds)}")
    print(f"Number of linear sub-point clouds: {len(linear_clouds)}")
    print(f"Number of too small sub-point clouds: {len(too_small_clouds)}")
    
    o3d.visualization.draw_geometries(complex_clouds, window_name = "complex clusters")
    o3d.visualization.draw_geometries(linear_clouds, window_name = "linear clusters")
    
    #output_directory = str(args.output_ply_dir)
    output_directory = "./"
    # If you have classified point clouds, you might want to save them separately:
    save_point_clouds(complex_clouds, os.path.join(output_directory, "pointclouds"), base_name="complex", format="ply")
    save_point_clouds(planar_clouds, os.path.join(output_directory, "planar"), base_name="planar", format="ply")
    save_point_clouds(linear_clouds, os.path.join(output_directory, "pointclouds"), base_name="linear", format="ply")

    plot_info(rgb_image, depth_image, semantic_image, mask)
    
if __name__ == '__main__':
    main()