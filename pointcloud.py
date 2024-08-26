import numpy as np 
from PIL import Image
import matplotlib.image
import matplotlib.pyplot as plt
import open3d as o3d

############## PLAN ##################
'''

0th version: just retrieve the pointclouds using semantic segmentation and pass it to Andrea's code

1st version: load images and do the clustering using DBSCAN on a pointcloud.
            - check if the clusters converge with semantic segmentation
            
2nd version - use segmentation in clustering, but how? It's just a label provided by another model

Pass segmented pointcloud to Andreas code to fit the cylinder

ALSO: collect everything in one repo to have infer + pointcloud creation + Andreas model in end to end manner

'''

import numpy as np
import open3d as o3d
from PIL import Image

def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    img = Image.open(file_path)
    img_array = np.array(img)
   # Display the image
    #plt.imshow(img_array)
    #plt.axis('off')  # Turn off axis numbers
    #plt.show()
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
    
    #plt.imshow(binary_mask)
    #plt.axis('off')  # Turn off axis numbers
    #plt.show()
    
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

    

def create_pointcloud_from_rgbd(rgb_image, depth_image, semantic_image, depth_scale=1000.0, depth_trunc=5.0, colors = False):
    """Create a pointcloud from RGB and depth images."""
    height, width = rgb_image.shape[:2]
    
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
    
    
    semantics_masked = apply_threshold_to_semantics(semantic_image,threshold=0.3)
    
    semantics_masked_rgb = binary_mask_to_rgb(semantics_masked)
    
    if colors == True: 
        # Extract RGB colors from the image
        colors = rgb_image[:, :, :3].reshape(-1, 3) / 255.0  # Normalize to [0, 1]
        #print(colors.shape)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Extract colors from the semantic image
        semantics_color =  semantics_masked_rgb[:, :, :3].reshape(-1, 3)
        #print(semantics_color.shape)
        pcd.colors = o3d.utility.Vector3dVector(semantics_color)

    return pcd

def main():
    # Load images
    rgb_path = "data/preprocessed_image.png"
    depth_path = "data/depth.png"  
    semantic_path = "data/semantic_tvmonitor.png"
    
    rgb_image = load_image(rgb_path)
    depth_image = load_image(depth_path)
    semantic_image = load_image(semantic_path)
    
    print(semantic_image.shape)

    # Ensure depth image is grayscale
    if len(depth_image.shape) == 3:
        depth_image = np.mean(depth_image, axis=2).astype(np.uint8)
        
    if len(semantic_image.shape) == 3:
        semantic_image = np.mean(semantic_image, axis=2).astype(np.uint8)
        
    #plt.imshow(semantic_image)
    #plt.axis('off')  # Turn off axis numbers
    #plt.show()
    
    # Create pointcloud
    pcd = create_pointcloud_from_rgbd(rgb_image, depth_image, semantic_image, colors = False)
    
    # Optional: Estimate normals for better visualization
    #pcd.estimate_normals()
    o3d.io.write_point_cloud("rgb_pointcloud.ply", pcd)
    
    # Visualize pointcloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()