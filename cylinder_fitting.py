import open3d as o3d
import numpy as np
from skspatial.objects import Plane, Points
import cv2
import sys
from sklearn.decomposition import PCA
from circle_fit import taubinSVD



# compute the preference function (how two points are similar wrt the distance between models) 
def PrefFunc(dist_points_model, tau): # dist_points_model vector of distance from model for each points in the point cloud

    ret = np.zeros_like(dist_points_model)

    for i in range(dist_points_model.shape[0]):

        if dist_points_model[i] <  tau :  
            ret[i] = np.exp(-dist_points_model[i]/(tau))

        else:
            ret[i] = 0
    
    return ret

# debug function to visualize retrieved plane
def vis_plane(plane, center):
    # plane is th enormal vector of the plane
    d = np.dot(plane, center)
    # Generate random (x, y) within a range
    x_vals = np.random.uniform(-1, 1, 1000)
    y_vals = np.random.uniform(-1, 1, 1000)

    # Calculate corresponding z values
    z_vals = (d - plane[0] * x_vals - plane[1] * y_vals) / plane[2]

    # Create the matrix of points
    points_matrix = np.column_stack((x_vals, y_vals, z_vals))

    return points_matrix


def Tdistance(p, q):    # Tdist = 0, as if p and q are the same point
                        # Tdist = 1, p orthogonal to q
    pin = np.inner(p, q)
    pnorm = np.linalg.norm(p)**2
    qnorm = np.linalg.norm(q)**2

    return 1 - (pin / (pnorm + qnorm - pin))

# merge two clusters
def Tmerge(PFp, PFq):
    newPF = np.zeros_like(PFp)
    for i in range(PFp.shape[0]):
        newPF[i] = np.min([PFp[i], PFq[i]])
    return newPF


# take out clusters of outlier 
def cluster_identification(clusters, num_points):

    real_clusters = []
    best = 0
    best_i = None

    for i in clusters:
        if len(i) > 0.35 * num_points:
            real_clusters.append(i)
        if len(i) > best:
            best_i = i
    if len(real_clusters) == 0:
        real_clusters = [best_i]

    # TODO: reorder the clusters by consensus
    return real_clusters


# compute the clusters to be merged. If not, T-linkage has finish
def checkOrthogality(PF, Tanidistance, TaniIndex):
    # compute similarity bettween models with tanimoto distance
    if Tanidistance is None: # if no Tanidistance compute until now, compute it
        Tanidistance = []
        TaniIndex = []
        for i in range(PF.shape[1]):
            for j in range(i + 1, PF.shape[1]):
                tdist = Tdistance(PF[:,i], PF[:,j])
                Tanidistance.append(tdist)
                TaniIndex.append((i, j))
        Tanidistance = np.array(Tanidistance)
    # otherwise the tanidistance is being update
    if np.allclose(Tanidistance, np.ones_like(Tanidistance)):
        return False, 0, 0, 0
    else:
        minimum = np.argmin(Tanidistance) # indice del megavettore di dist, da capire il giusto index
        return True, TaniIndex[minimum] , Tanidistance, TaniIndex
            

def update_PF(prefFunc, tobemerged, trace_points, Tanidistance, TaniIndex): # tobemerged tuple of index

    # compute the merged PF
    newPF = Tmerge(prefFunc[:, tobemerged[0]], prefFunc[:, tobemerged[1]]) 

    # put the newPF instead of the first old cluster

    prefFunc[:,tobemerged[0]] = newPF


    prefFunc[:, tobemerged[1]] = np.zeros_like(prefFunc[:,tobemerged[1]])

    for x in trace_points[tobemerged[1]]:
        trace_points[tobemerged[0]].append(x)

    i = 0
    elim = []
    update = []
    for fr, to in TaniIndex:
        if fr == tobemerged[1] or to == tobemerged[1]:
            elim.append(i)
        elif fr == tobemerged[0] or to == tobemerged[0]:
            update.append(i)
        i += 1
    for up in update:
        tup = TaniIndex[up]
        tdist = Tdistance(prefFunc[:,tup[0]], prefFunc[:,tup[1]])
        Tanidistance[up] = tdist
    
    Tanidistance = np.delete(Tanidistance, elim)
    for e in reversed(elim):
        TaniIndex.pop(e)



    return prefFunc, trace_points,  Tanidistance, TaniIndex

def casual_color():
    return np.random.random(3)

def checkCollinear(points, epsilon=1e-9):
    # Ensure we have at least 3 points
    if points.shape[0] < 3:
        return True  # Less than 3 points are always collinear

    # Normalize the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    max_distance = np.max(np.sqrt(np.sum(centered_points**2, axis=1)))
    
    # Avoid division by zero
    if max_distance < 1e-20:
        return True  # All points are essentially the same

    normalized_points = centered_points / max_distance

    # Check collinearity on normalized points
    x, y, z = normalized_points.T
    v1 = y - x
    v2 = z - x
    cross = np.cross(v1, v2)
   # magnitude = np.linalg.norm(cross)
    
    is_collinear = np.all(cross < epsilon) 
    return is_collinear

# main body of T-linkage, a multi model procedure proposed by prof. Luca Magri
def T_Linkage(pointcloud, model = 'plane'):
    iput_gauss_sphere = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud.astype('float64'))).paint_uniform_color([1,1,0])

    trace_points = [[x] for x in range(pointcloud.shape[0])]
    # TODO: keep trace of inliers of models found
    tau = 0.05
    # search for H models and compute distances
    founded_models = []
    max_attempts = 2000 # Maximum number of attempts to find non-collinear points
    for i in range(50):
        # TODO: choose random points, 2 (+ origin) for plane, 3 for circle
        if model == 'plane':
            attempts = 0 
            while attempts < max_attempts:
                # Choose two random points
                choose = np.random.randint(0, pointcloud.shape[0], 2)
                pc = pointcloud[choose]
                
                # Add the origin as the third point
                pc = np.vstack((pc, np.zeros(3).reshape((1,3))))
                
                # Check if the points are NOT collinear
                if not checkCollinear(pc):
                    #print("Found non-collinear points.")
                    break
                
                #print("Points are collinear. Selecting new points.")
                attempts += 1
                
            if attempts == max_attempts:
                print(f"Failed to find non-collinear points for plane {i + 1} after {max_attempts} attempts.")
                continue
             
            # At this point, pc should contains two points from pointcloud and the origin, non-collinear
            flag, m, dist = fit_plane(pc, pointcloud)
            if flag == False: 
                continue
                
            dist = np.array(dist)
            #print(dist)
            
        elif model == 'circle':
            choose = np.random.randint(0, pointcloud.shape[0], 10)
            pc = pointcloud[choose]
            m, dist = fit_circle(pc, pointcloud)
            
            
        founded_models.append((m, dist))
    
    # for each model, compute the Preference Function
    preferenceFunction = np.zeros((len(founded_models), pointcloud.shape[0]))
    i = 0
    for m, d in founded_models:
        pfm = PrefFunc(d, tau)
        preferenceFunction[i,:] = pfm
        i += 1

    to_elim = []
    iput_gauss_sphere = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud.astype('float64'))).paint_uniform_color([0,0,0])
    for i in range(preferenceFunction.shape[1]):
        if np.allclose(preferenceFunction[:,i], np.zeros_like(preferenceFunction[:,i])):
            to_elim.append(i)
    print('to be eliminated, no support', len(to_elim))
    elim = pointcloud[to_elim]
    elim = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(elim.astype('float64'))).paint_uniform_color([0,0,1])

    preferenceFunction = np.delete(preferenceFunction, to_elim, axis=1)
    trace_points = np.delete(trace_points, to_elim, axis=0).tolist()

    print('preferenceFunction', preferenceFunction.shape)
    print('trace_points', len(trace_points))

    i = 0
    # now we can search for clusters to be merged via Tanimoto Distance
    repeat, clus_to_be_merged, Tdis, Tidx = checkOrthogality(preferenceFunction, None, None)
    merged = []
    while(repeat):
        #print('iter', i)
        preferenceFunction, trace_points, Tdis, Tidx = update_PF(preferenceFunction, clus_to_be_merged, trace_points, Tdis, Tidx)
        repeat, clus_to_be_merged, Tdis, _ = checkOrthogality(preferenceFunction, Tdis, Tidx) # clus_to_be_merged is a tuple of index to be merged
        try:
            merged.append(clus_to_be_merged[1])
        except:
            continue
        i += 1
    # now we have good clusters, TODO: add oulier detection
    preferenceFunction = np.delete(preferenceFunction, merged, axis=1)
    for md in merged:
        trace_points[md] = []

    print('clusters found by T-linkage' , preferenceFunction.shape)

    #o3d.visualization.draw(all)
    real_clus = cluster_identification(trace_points, pointcloud.shape[0])
    print('total_cluster_founds', len(real_clus))

    for r_c in real_clus:
        refit = pointcloud[r_c]

        if model == 'plane':
            refit = np.vstack((refit, np.zeros(3).reshape((1,3))))
            flag, best_model, res = fit_plane(refit, pointcloud)
        elif model == 'circle':
            best_model, res = fit_circle(refit, pointcloud)
        


    return best_model
    
    # TODO: refit models on optimal clusters found 


# create points of a circle for visualization purpose
def create_circle_points(radius, x0, y0,  n=72):
    sp = np.linspace(0, 2.0 * np.pi, num=250)
    nx = sp.shape[0]
    u = np.tile(sp, nx)
    v = np.repeat(sp, nx)
    x = x0 + np.cos(u) * radius
    y = y0 + np.sin(u) * radius
    return x, y

# NEVER USED
def project_points_onto_plane(points, normal_vector):
    centroid = np.mean(points, axis=0)

    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the normal vector
    projected_points = []
    
    for point in points:
        vector_from_centroid = point - centroid
        distance_to_plane = np.dot(vector_from_centroid, normal_vector)
        projection = point - distance_to_plane * normal_vector
        projected_points.append(projection)
    
    return np.array(projected_points)

# get back from 2d circle found to the 3d
def transform_circle_to_3d(center_2d, radius, centroid, normal_vector):
    # Generate a set of points on the circle in 2D
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points_2d = np.array([center_2d[0] + radius * np.cos(theta), center_2d[1] + radius * np.sin(theta)]).T
    
    # Create a basis for the plane
    u = np.array([1, 0, -normal_vector[0] / normal_vector[2]])
    u = u / np.linalg.norm(u)
    v = np.cross(normal_vector, u)
    
    # Transform the circle points back to 3D
    circle_points_3d = []
    for point_2d in circle_points_2d:
        point_3d = centroid + point_2d[0] * u + point_2d[1] * v
        circle_points_3d.append(point_3d)
    
    new_center = centroid + center_2d[0] * u + center_2d[1] * v 
    circle_points_3d.append(new_center)
    
    
    return np.array(circle_points_3d), new_center

def transform_cylinder_to_3d(center_2d, radius, centroid, normal_vector, zmin, zmax):
    # Generate a set of points on the circle in 2D
    theta = np.linspace(0, 2 * np.pi, 100)
    zz = np.linspace(zmin, zmax, 100)
    circle_points_2d = np.array([center_2d[0] + radius * np.cos(theta), center_2d[1] + radius * np.sin(theta), zz]).T
    
    # Create a basis for the plane
    u = np.array([1, 0, -normal_vector[0] / normal_vector[2]])
    u = u / np.linalg.norm(u)
    v = np.cross(normal_vector, u)
    # Transform the circle points back to 3D
    circle_points_3d = []
    for point_2d in circle_points_2d:
        point_3d = centroid + point_2d[0] * u + point_2d[1] * v #+ point_2d[2] * normal_vector#+ np.array([0,0,1])
        #circle_points_3d.append(point_3d)
        for z in zz:
            p3d = point_3d + z * normal_vector
            circle_points_3d.append(p3d)
    
    
    return np.array(circle_points_3d)


def fit_circle(points, data):

    xc, yc, r, sigma = taubinSVD(points[:,:2])


    dists = np.abs(np.linalg.norm(data[:,:2] - [xc, yc], axis=1) - r) 


    return [xc, yc, r], dists # , cons, inl


def fit_plane(toBeFitted, data, threshold=0.01):
    """
    Fit a plane to the given points and calculate inliers from the data.
    
    Args:
    toBeFitted (np.ndarray): Points to fit the plane to.
    data (np.ndarray): Data points to check for inliers.
    threshold (float): Distance threshold for inliers. Default is 0.01.
    
    Returns:
    Tuple[bool, np.ndarray | None, List[float]]: 
        - Boolean indicating success or failure
        - Normal vector of the plane (or None if fitting failed)
        - List of distances from each data point to the plane
    """
    points = Points(toBeFitted)
    n = None
    dists = []
    
    try:
        plane = Plane.best_fit(points)
        n = plane.normal
        n = n / np.linalg.norm(n)
        
        for d in data:
            dist = np.abs(np.dot(n, d)) / np.linalg.norm(n)
            dists.append(dist)
        
        #inliers = [d for d in data if np.abs(np.dot(n, d)) / np.linalg.norm(n) < threshold]
        #inlier_count = len(inliers)
        
        #print(f"Successfully fitted plane. Inlier count: {inlier_count}")
        return True, n, dists
    
    except ValueError as e:
        print(f"Error processing point cloud: {str(e)}")
        return False, None, []
        



def create_cylinder_point_cloud(radius=2.0, height=8.0, num_points=1000):
    """Create a point cloud of a cylinder."""
    RR = cv2.Rodrigues(np.array([np.pi, 0, np.pi]))[0]
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=200, split=4)
    cyl = cylinder.sample_points_uniformly(number_of_points=num_points).rotate(RR, center=np.array([0,0,0])).translate([10,5,-4])
    cyl.paint_uniform_color([0, 0, 0])
    cyl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    return cyl

def create_cylinder_point_cloud_with_seed(radius=2.0, height=8.0, num_points=1000, seed=42):
    """
    Create a point cloud of a cylinder with reproducible uniform sampling.
    
    Args:
    radius (float): Radius of the cylinder.
    height (float): Height of the cylinder.
    num_points (int): Number of points to sample.
    seed (int): Random seed for reproducibility.
    
    Returns:
    o3d.geometry.PointCloud: Point cloud of the cylinder.
    """
    np.random.seed(seed)
    
    # Generate uniform points on the cylinder surface
    theta = np.random.uniform(0, 2*np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Combine coordinates
    points = np.column_stack((x, y, z))
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Apply rotation and translation
    RR = cv2.Rodrigues(np.array([np.pi, 0, np.pi]))[0]
    pcd.rotate(RR, center=np.array([0,0,0]))
    pcd.translate([10, 5, -4])
    
    # Paint and estimate normals
    pcd.paint_uniform_color([0, 0, 0])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    
    return pcd

def project_points_to_plane(points, plane_normal):
    """Project points onto a plane."""
    proj = []
    for p in points:
        dis = p.dot(plane_normal)
        proj.append(p - dis * plane_normal)
    return np.array(proj)

def find_cylinder_inliers(points, plane_normal, center, radius, tolerance=0.1):
    """Find inliers of the cylinder model."""
    inl_cyl = []
    for i, p in enumerate(points):
        dist = np.linalg.norm(np.cross(plane_normal, (p-center)))/np.linalg.norm(plane_normal)
        if np.abs(dist - radius) < tolerance:
            inl_cyl.append(i)
    return inl_cyl

def get_cylinder_height(points, plane_normal):
    """Determine the height of the cylinder."""
    p_proj = np.dot(points, plane_normal)
    return np.min(p_proj), np.max(p_proj)




