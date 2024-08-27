import open3d as o3d
import numpy as np
from skspatial.objects import Plane, Points
import cv2
import sys
from sklearn.decomposition import PCA
from circle_fit import taubinSVD

from cylinder_fitting import create_cylinder_point_cloud, create_cylinder_point_cloud_with_seed, T_Linkage, transform_circle_to_3d, transform_cylinder_to_3d

def load_point_cloud(file_path):
    """
    Load a point cloud from a .ply file with error checking and debugging.
    
    Args:
    file_path (str): Path to the .ply file.
    
    Returns:
    o3d.geometry.PointCloud: Loaded point cloud.
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError(f"Loaded point cloud from {file_path} has no points")
        print(f"Successfully loaded {len(pcd.points)} points from {file_path}")
        return pcd
    except Exception as e:
        print(f"Error loading point cloud: {str(e)}")
        return None

def prepare_point_cloud(pcd):
    """
    Prepare the point cloud for processing.
    
    Args:
    pcd (o3d.geometry.PointCloud): Input point cloud.
    
    Returns:
    o3d.geometry.PointCloud: Prepared point cloud.
    """
    if pcd is None:
        return None
    
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    print(f"Point cloud has {len(pcd.points)} points and {len(pcd.normals)} normals")
    
    if not pcd.has_colors():
        # Ensure the point cloud is visible by setting a non-black color
        pcd.paint_uniform_color([0.0, 0.0, 0.0])  # black color
        print("Black color added to the loaded Point cloud")

    return pcd

def visualize_point_clouds(*point_clouds, window_name="Point Clouds Visualization"):
    """
    Visualize multiple point clouds with error checking and default Open3D background.
    
    Args:
    *point_clouds: Variable number of o3d.geometry.PointCloud objects.
    window_name (str): Name of the visualization window.
    """
    if not point_clouds:
        print("Error: No point clouds provided for visualization")
        return
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # Add all point clouds to the visualizer
    for i, pcd in enumerate(point_clouds):
        if not isinstance(pcd, o3d.geometry.PointCloud):
            print(f"Warning: Object {i} is not a PointCloud. Skipping.")
            continue
        if not pcd.has_points():
            print(f"Warning: PointCloud {i} is empty. Skipping.")
            continue
        vis.add_geometry(pcd)
        print(f"Added PointCloud {i} with {len(pcd.points)} points")
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Increase point size for better visibility
    #render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark blue-grey to match default
    
    # Set a default viewpoint
    ctr = vis.get_view_control()
    ctr.set_front([-0.2, -0.2, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()

def main(file_path):
    # Load point cloud from .ply file
    pcd = load_point_cloud(file_path)
    if pcd is None:
        print("Failed to load point cloud. Exiting.")
        return

    pcd = prepare_point_cloud(pcd)
    if pcd is None:
        print("Failed to prepare point cloud. Exiting.")
        return

    o3d.visualization.draw_geometries([pcd], window_name = "Loaded Point Cloud")
    #visualize_point_clouds(pcd)
    
    '''
    #cylinder for test
    
    cyl = create_cylinder_point_cloud(radius=2.0, height=8.0, num_points=1000)
    cyl = create_cylinder_point_cloud_with_seed(radius=2.0, height=8.0, num_points=1000, seed = 12345)
    pcd = cyl
    
    pcd.paint_uniform_color([0, 0, 0])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    o3d.visualization.draw([pcd], title = "Generated Cylinder")
    '''
    
    # save the points and normal into np array
    pcyl = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    print('normals', normals.shape)

    # to represents a cylinder we need an orientation (3D vector, a plane such that it's normal is in the same direction of the axis of the cylinder),
    #  radius (scalar), center in the 3D space (3D vector) [ or in the local frame of the cylinder]
    # embed the normals (unitary vector) into a unitary Sprhere, called Gaussian Sphere. 
    # Cylinder follow a peculirty: all the normals on the curvature of the cylinder (not the basis) form a great circle (circonferenza di raggio massimo sulla sfera).
    # Then this great circle has to lay on a plane passing through the origin. Let's search this plane!

    iput_gauss_sphere = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normals.astype('float64'))).paint_uniform_color([1,0,0])

    plane = T_Linkage(normals)

    plane = plane[:3]

    vet_to_plane = np.zeros(3)

    # The plane founded represent the orientation of the cylinder in the space. We can now project all points on this plane

    n = plane
    n = n / np.linalg.norm(n)
    proj = []
    for p in pcyl:
        dis = p.dot(n)
        proj.append(p - dis * n)
    proj_points = np.array(proj) 


    u = np.array([1, 0, -plane[0] / plane[2]])
    u = u / np.linalg.norm(u)
    v = np.cross(plane, u)

    # change the base of the representation: from the 3D, we want to trasform points into the 2D plane founded above.
    # This to ensure an isometry tha can lead as to compute the circle fitting in a 2D setting, leaving the dimension unchanged
    # then create a orthonormal base [u, v, n], with n the norml of the plane 

    base = np.array([u,v,plane]).T

    boh = proj_points @ base

    p_projected = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(boh.astype('float64'))).paint_uniform_color([0,0,1])

    print('proj', proj_points.shape)

    p_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proj_points.astype('float64'))).paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([pcd, p_projected, p_points], window_name = "Projected Circles")

    o3d.visualization.draw_geometries([p_projected], window_name= "Circle")

    proj_points = boh

    a, b, c = plane

    # find the circle

    best_cir = T_Linkage(proj_points, model='circle')

    print('radius', best_cir[2])

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points_2d = np.array([best_cir[0] + best_cir[2] * np.cos(theta), best_cir[1] + best_cir[2] * np.sin(theta)]).T
    print('circle_points_2d', circle_points_2d.shape)
    print(np.zeros(circle_points_2d.shape[0]).shape)
    circle_points_2d = np.hstack([circle_points_2d, np.zeros(circle_points_2d.shape[0]).reshape((-1,1))])
    print('circle_points_2d', circle_points_2d.shape)
    circle_points_2d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(circle_points_2d.astype('float64'))).paint_uniform_color([1,0,0])

    pcir,center = transform_circle_to_3d(best_cir[:2], best_cir[2], vet_to_plane, plane)
    print('center', center)
    pcir = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcir.astype('float64'))).paint_uniform_color([1,0,0])

    o3d.visualization.draw_geometries([pcd, pcir], window_name = "Circles")

    inl_cyl = []
    i = 0
    for p in pcyl:
        dist = np.linalg.norm(np.cross(plane, (p-center)))/np.linalg.norm(plane)
        if np.abs(dist) < best_cir[2] + 0.1 and dist > best_cir[2] - 0.1:
            inl_cyl.append(i)
        i += 1

    print('inl_cyl', len(inl_cyl))
    estim = pcyl[inl_cyl]

    p_proj = np.dot(estim, plane)

    z_min = np.min(p_proj)
    z_max = np.max(p_proj)
    print('zmin', z_min)

    obtained_cyl = transform_cylinder_to_3d(best_cir[:2], best_cir[2], vet_to_plane, plane, z_min, z_max)
    obtained_cyl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obtained_cyl.astype('float64'))).paint_uniform_color([1,0,0])
    print(f'obtained cylinder: center: {center}, radius {best_cir[2]}, normal of plane: {plane}')
    o3d.visualization.draw_geometries([pcd, obtained_cyl], window_name = "Obtained Cylinder")    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_ply_file>")
        sys.exit(1)
    main(sys.argv[1])