import numpy as np
from . import nettracer
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import KDTree
from scipy import ndimage
import concurrent.futures
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Union, Tuple, List, Optional
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import functools

# Related to morphological border searching:

def get_reslice_indices(slice_obj, dilate_xy, dilate_z, array_shape):
    """Convert slice object to padded indices accounting for dilation and boundaries"""
    if slice_obj is None:
        return None, None, None
        
    z_slice, y_slice, x_slice = slice_obj
    
    # Extract min/max from slices
    z_min, z_max = z_slice.start, z_slice.stop - 1
    y_min, y_max = y_slice.start, y_slice.stop - 1
    x_min, x_max = x_slice.start, x_slice.stop - 1

    # Add dilation padding
    y_max = y_max + ((dilate_xy-1)/2) + 1
    y_min = y_min - ((dilate_xy-1)/2) - 1
    x_max = x_max + ((dilate_xy-1)/2) + 1
    x_min = x_min - ((dilate_xy-1)/2) - 1
    z_max = z_max + ((dilate_z-1)/2) + 1
    z_min = z_min - ((dilate_z-1)/2) - 1

    # Boundary checks
    y_max = min(y_max, array_shape[1] - 1)
    x_max = min(x_max, array_shape[2] - 1)
    z_max = min(z_max, array_shape[0] - 1)
    y_min = max(y_min, 0)
    x_min = max(x_min, 0)
    z_min = max(z_min, 0)

    y_vals = [y_min, y_max]
    x_vals = [x_min, x_max]
    z_vals = [z_min, z_max]

    return z_vals, y_vals, x_vals

def reslice_3d_array(args):
    """Internal method used for the secondary algorithm to reslice subarrays around nodes."""

    input_array, z_range, y_range, x_range = args
    z_start, z_end = z_range
    z_start, z_end = int(z_start), int(z_end)
    y_start, y_end = y_range
    y_start, y_end = int(y_start), int(y_end)
    x_start, x_end = x_range
    x_start, x_end = int(x_start), int(x_end)
    
    # Reslice the array
    resliced_array = input_array[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
    
    return resliced_array



def _get_node_node_dict(label_array, label, dilate_xy, dilate_z, fastdil = False, xy_scale = 1, z_scale = 1, search = 0):
    """Internal method used for the secondary algorithm to find which nodes interact 
    with which other nodes based on proximity."""
    
    # Create a boolean mask where elements with the specified label are True
    binary_array = label_array == label
    binary_array = nettracer.dilate(binary_array, search, xy_scale, z_scale, fast_dil = fastdil, dilate_xy = dilate_xy, dilate_z = dilate_z) #Dilate the label to see where the dilated label overlaps
    label_array = label_array * binary_array  # Filter the labels by the node in question
    label_array = label_array.flatten()  # Convert 3d array to 1d array
    label_array = nettracer.remove_zeros(label_array)  # Remove zeros
    label_array = label_array[label_array != label]
    label_array = set(label_array)  # Remove duplicates
    label_array = list(label_array)  # Back to list
    return label_array

def process_label(args):
    """Modified to use pre-computed bounding boxes instead of argwhere"""
    nodes, label, dilate_xy, dilate_z, array_shape, bounding_boxes = args
    print(f"Processing node {label}")
    
    # Get the pre-computed bounding box for this label
    slice_obj = bounding_boxes[int(label)-1]  # -1 because label numbers start at 1
    if slice_obj is None:
        return None, None
        
    z_vals, y_vals, x_vals = get_reslice_indices(slice_obj, dilate_xy, dilate_z, array_shape)
    if z_vals is None:
        return None, None
        
    sub_nodes = reslice_3d_array((nodes, z_vals, y_vals, x_vals))
    return label, sub_nodes


def create_node_dictionary(nodes, num_nodes, dilate_xy, dilate_z, targets=None, fastdil = False, xy_scale = 1, z_scale = 1, search = 0):
    """pre-compute all bounding boxes using find_objects"""
    node_dict = {}
    array_shape = nodes.shape

    
    # Get all bounding boxes at once
    bounding_boxes = ndimage.find_objects(nodes)
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Create args list with bounding_boxes included
        args_list = [(nodes, i, dilate_xy, dilate_z, array_shape, bounding_boxes) 
                    for i in range(1, int(num_nodes) + 1)]

        if targets is not None:
            args_list = [tup for tup in args_list if tup[1] in targets]

        results = executor.map(process_label, args_list)

        # Process results in parallel
        for label, sub_nodes in results:
            executor.submit(create_dict_entry, node_dict, label, sub_nodes, dilate_xy, dilate_z, fastdil = fastdil, xy_scale = xy_scale, z_scale = z_scale, search = search)

    return node_dict

def create_dict_entry(node_dict, label, sub_nodes, dilate_xy, dilate_z, fastdil = False, xy_scale = 1, z_scale = 1, search = 0):
    """Internal method used for the secondary algorithm to pass around args in parallel."""

    if label is None:
        pass
    else:
        node_dict[label] = _get_node_node_dict(sub_nodes, label, dilate_xy, dilate_z, fastdil = fastdil, xy_scale = xy_scale, z_scale = z_scale, search = search)

def find_shared_value_pairs(input_dict):
    """Internal method used for the secondary algorithm to look through discrete 
    node-node connections in the various node dictionaries"""
    # List comprehension approach
    return [[key, value, 0] for key, values in input_dict.items() for value in values]



#Related to kdtree centroid searching:

def populate_array(centroids, clip=False, shape = None):
    """
    Create a 3D array from centroid coordinates.
    
    Args:
        centroids: Dictionary where keys are object IDs and values are [z,y,x] coordinates
        clip: Boolean, if True, transpose all centroids so minimum values become 0
    
    Returns:
        If clip=False: 3D numpy array where values are object IDs at their centroid locations
        If clip=True: Tuple of (3D numpy array, dictionary with clipped centroids)
    """
    # Input validation
    if not centroids:
        raise ValueError("Centroids dictionary is empty")
    
    # Convert to numpy array and get bounds
    coords = np.array(list(centroids.values()))
    # Round coordinates to nearest integer
    coords = np.round(coords).astype(int)
    if shape is None:
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
    else:
        min_coords = [0, 0, 0]
        max_coords = shape
    
    # Check for negative coordinates only if not clipping
    #if not clip and np.any(min_coords < 0):
        #raise ValueError("Negative coordinates found in centroids")
    
    # Apply clipping if requested
    clipped_centroids = {}
    if clip:
        # Transpose all coordinates so minimum becomes 0
        coords = coords - min_coords
        max_coords = max_coords - min_coords
        min_coords = np.zeros_like(min_coords)
        
        # Create dictionary with clipped centroids
        for i, obj_id in enumerate(centroids.keys()):
            clipped_centroids[obj_id] = coords[i].tolist()
    
    if shape is None:
        # Create array
        array = np.zeros((max_coords[0] + 1, 
                         max_coords[1] + 1, 
                         max_coords[2] + 1), dtype=int)
    else:
        array = np.zeros((max_coords[0], 
                         max_coords[1], 
                         max_coords[2]), dtype=int)
    
    # Populate array with (possibly clipped) rounded coordinates
    for i, (obj_id, coord) in enumerate(centroids.items()):
        if clip:
            z, y, x = coords[i]  # Use pre-computed clipped coordinates
        else:
            z, y, x = np.round([coord[0], coord[1], coord[2]]).astype(int)
        try:
            array[z, y, x] = obj_id
        except:
            pass
        
    if clip:
        return array, clipped_centroids
    else:
        return array

def _process_chunk_centroids(args):
    """Process a chunk of neighbor indices for centroids mode"""
    chunk_data, idx_to_node, query_indices, tree, points, max_neighbors = args
    output = []
    
    for i, neighbors in chunk_data:
        query_idx = query_indices[i]
        query_value = idx_to_node[query_idx]
        query_point = points[query_idx]
        
        # Filter out self-reference
        filtered_neighbors = [n for n in neighbors if n != query_idx]
        
        # If max_neighbors is specified and we have more neighbors than allowed
        if max_neighbors is not None and len(filtered_neighbors) > max_neighbors:
            # Use KDTree to get distances efficiently - query for more than we need
            # to ensure we get the exact closest ones
            k = min(len(filtered_neighbors), max_neighbors + 1)  # +1 in case query point is included
            distances, indices = tree.query(query_point, k=k)
            
            # Filter out self and limit to max_neighbors
            selected_neighbors = []
            for dist, idx in zip(distances, indices):
                if idx != query_idx and idx in filtered_neighbors:
                    selected_neighbors.append(idx)
                    if len(selected_neighbors) >= max_neighbors:
                        break
            
            filtered_neighbors = selected_neighbors
        
        # Add all selected neighbors to output
        for neighbor_idx in filtered_neighbors:
            neighbor_value = idx_to_node[neighbor_idx]
            output.append([query_value, neighbor_value, 0])
    
    return output

def _process_chunk_array(args):
    """Process a chunk of neighbor indices for array mode"""
    chunk_data, array, point_tuples, query_indices, tree, points, max_neighbors = args
    output = []
    
    for i, neighbors in chunk_data:
        query_idx = query_indices[i]
        query_value = array[point_tuples[query_idx]]
        query_point = points[query_idx]
        
        # Filter out self-reference
        filtered_neighbors = [n for n in neighbors if n != query_idx]
        
        # If max_neighbors is specified and we have more neighbors than allowed
        if max_neighbors is not None and len(filtered_neighbors) > max_neighbors:
            # Use KDTree to get distances efficiently - query for more than we need
            # to ensure we get the exact closest ones
            k = min(len(filtered_neighbors), max_neighbors + 1)  # +1 in case query point is included
            distances, indices = tree.query(query_point, k=k)
            
            # Filter out self and limit to max_neighbors
            selected_neighbors = []
            for dist, idx in zip(distances, indices):
                if idx != query_idx and idx in filtered_neighbors:
                    selected_neighbors.append(idx)
                    if len(selected_neighbors) >= max_neighbors:
                        break
            
            filtered_neighbors = selected_neighbors
        
        # Add all selected neighbors to output
        for neighbor_idx in filtered_neighbors:
            neighbor_value = array[point_tuples[neighbor_idx]]
            output.append([query_value, neighbor_value, 0])
    
    return output

def find_neighbors_kdtree(radius, centroids=None, array=None, targets=None, n_jobs=None, chunk_size=None, max_neighbors=None):
    """
    Find neighbors using KDTree with optional parallelization.
    
    Parameters:
    -----------
    radius : float
        Search radius for finding neighbors
    centroids : dict or list, optional
        Dictionary mapping node IDs to coordinates or list of points
    array : numpy.ndarray, optional
        Array to search for nonzero points
    targets : list, optional
        Specific targets to query for neighbors
    n_jobs : int, optional
        Number of parallel jobs. If None, uses cpu_count(). Set to 1 to disable parallelization.
    chunk_size : int, optional
        Size of chunks for parallel processing. If None, auto-calculated based on data size.
    max_neighbors : int, optional
        Maximum number of nearest neighbors to return per query point within the radius.
        If None, returns all neighbors within radius (original behavior).
    """
    
    # Get coordinates of nonzero points
    if centroids:
        # If centroids is a dictionary mapping node IDs to coordinates
        if isinstance(centroids, dict):
            # Extract the node IDs and points
            node_ids = list(centroids.keys())
            points_list = list(centroids.values())
            points = np.array(points_list, dtype=np.int32)
        else:
            # If centroids is just a list of points
            points = np.array(centroids, dtype=np.int32)
            node_ids = list(range(1, len(points) + 1))  # Default sequential IDs
        
        # Create direct index-to-node mapping instead of sparse array
        idx_to_node = {i: node_ids[i] for i in range(len(points))}
        
    elif array is not None:
        points = np.transpose(np.nonzero(array))
        node_ids = None  # Not used in array-based mode
        # Pre-convert points to tuples once to avoid repeated conversions
        point_tuples = [tuple(point) for point in points]
    else:
        return []
    
    print("Building KDTree...")
    # Create KD-tree from all nonzero points
    tree = KDTree(points)
    
    if targets is None:
        # Original behavior: find neighbors for all points
        query_points = np.array(points)
        query_indices = list(range(len(points)))
    else:
        # Convert targets to set for O(1) lookup
        targets_set = set(targets)
        
        # Find coordinates of target values
        target_points = []
        target_indices = []
        
        if array is not None:
            # Standard array-based filtering
            for idx, point_tuple in enumerate(point_tuples):
                if array[point_tuple] in targets_set:
                    target_points.append(points[idx])
                    target_indices.append(idx)
        else:
            # Filter based on node IDs directly
            for idx, node_id in enumerate(node_ids):
                if node_id in targets_set:
                    target_points.append(points[idx])
                    target_indices.append(idx)
        
        # Convert to numpy array for querying
        query_points = np.array(target_points)
        query_indices = target_indices
        
        # Handle case where no target values were found
        if len(query_points) == 0:
            return []
    
    print("Querying KDTree...")

    # Query for all points within radius of each query point
    neighbor_indices = tree.query_ball_point(query_points, radius)
    
    print("Sorting Through Output...")

    # Determine parallelization parameters
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Skip parallelization for small datasets or when n_jobs=1
    if n_jobs == 1 or len(neighbor_indices) < 100:
    #if True:
        # Sequential processing (original logic with max_neighbors support)
        output = []
        for i, neighbors in enumerate(neighbor_indices):
            query_idx = query_indices[i]
            query_point = points[query_idx]
            
            # Filter out self-reference
            filtered_neighbors = [n for n in neighbors if n != query_idx]
            
            # If max_neighbors is specified and we have more neighbors than allowed
            if max_neighbors is not None and len(filtered_neighbors) > max_neighbors:
                # Use KDTree to get distances efficiently - query for more than we need
                # to ensure we get the exact closest ones
                k = min(len(filtered_neighbors), max_neighbors + 1)  # +1 in case query point is included
                distances, indices = tree.query(query_point, k=k)
                
                # Filter out self and limit to max_neighbors
                selected_neighbors = []
                for dist, idx in zip(distances, indices):
                    if idx != query_idx and idx in filtered_neighbors:
                        selected_neighbors.append(idx)
                        if len(selected_neighbors) >= max_neighbors:
                            break
                
                filtered_neighbors = selected_neighbors
            
            # Process the selected neighbors
            if centroids:
                query_value = idx_to_node[query_idx]
                for neighbor_idx in filtered_neighbors:
                    neighbor_value = idx_to_node[neighbor_idx]
                    output.append([query_value, neighbor_value, 0])
            else:
                query_value = array[point_tuples[query_idx]]
                for neighbor_idx in filtered_neighbors:
                    neighbor_value = array[point_tuples[neighbor_idx]]
                    output.append([query_value, neighbor_value, 0])
        return output
    
    # Parallel processing
    if chunk_size is None:
        # Auto-calculate chunk size: aim for ~4x more chunks than processes
        chunk_size = max(1, len(neighbor_indices) // (n_jobs * 4))
    
    # Create chunks of (index, neighbors) pairs
    chunks = []
    for i in range(0, len(neighbor_indices), chunk_size):
        chunk = [(j, neighbor_indices[j]) for j in range(i, min(i + chunk_size, len(neighbor_indices)))]
        chunks.append(chunk)
    
    # Process chunks in parallel
    with Pool(processes=n_jobs) as pool:
        if centroids:
            # Prepare arguments for centroids mode
            chunk_args = [(chunk, idx_to_node, query_indices, tree, points, max_neighbors) for chunk in chunks]
            chunk_results = pool.map(_process_chunk_centroids, chunk_args)
        else:
            # Prepare arguments for array mode
            chunk_args = [(chunk, array, point_tuples, query_indices, tree, points, max_neighbors) for chunk in chunks]
            chunk_results = pool.map(_process_chunk_array, chunk_args)
    
    # Flatten results
    output = []
    for chunk_result in chunk_results:
        output.extend(chunk_result)

    print("Organizing Network...")

    
    return output

def extract_pairwise_connections(connections):
    output = []

    for i, sublist in enumerate(connections):
        list_index_value = i + 1  # Element corresponding to the sublist's index
        for number in sublist:
            if number != list_index_value:  # Exclude self-pairing
                output.append([list_index_value, number, 0])
                print(f'sublist: {sublist}, adding: {[list_index_value, number, 0]}')

    return output


def average_nearest_neighbor_distances(point_centroids, root_set, compare_set, xy_scale=1.0, z_scale=1.0, num=1, do_borders=False):
    """
    Calculate the average distance between each point in root_set and its nearest neighbor in compare_set.
    
    Args:
        point_centroids (dict): Dictionary mapping point IDs to [Z, Y, X] coordinates (when do_borders=False)
                               OR dictionary mapping labels to border coordinates (when do_borders=True)
        root_set (set or dict): Set of point IDs (when do_borders=False) 
                               OR dict {label: border_coords} (when do_borders=True)
        compare_set (set or numpy.ndarray): Set of point IDs (when do_borders=False)
                                           OR 1D array of border coordinates (when do_borders=True)
        xy_scale (float): Scaling factor for X and Y coordinates
        z_scale (float): Scaling factor for Z coordinate
        num (int): Number of nearest neighbors (ignored when do_borders=True, always uses 1)
        do_borders (bool): If True, perform border-to-border distance calculation
    
    Returns:
        tuple: (average_distance, distances_dict)
    """
    
    if do_borders:
        # Border comparison mode
        if not isinstance(compare_set, np.ndarray):
            raise ValueError("When do_borders=True, compare_set must be a numpy array of coordinates")
        
        # Vectorized scaling for compare coordinates
        compare_coords_scaled = compare_set.astype(float)
        compare_coords_scaled[:, 0] *= z_scale  # Z coordinates
        compare_coords_scaled[:, 1:] *= xy_scale  # Y and X coordinates
        
        distances = {}
        
        for label, border_coords in root_set.items():
            if len(border_coords) == 0:
                continue
                
            # Vectorized scaling for border coordinates
            border_coords_scaled = border_coords.astype(float)
            border_coords_scaled[:, 0] *= z_scale  # Z coordinates
            border_coords_scaled[:, 1:] *= xy_scale  # Y and X coordinates
            
            # Remove overlapping coordinates to avoid distance = 0
            # Create a set of tuples for fast membership testing
            border_coords_set = set(map(tuple, border_coords_scaled))
            
            # Filter out overlapping coordinates from compare set
            non_overlapping_mask = np.array([
                tuple(coord) not in border_coords_set 
                for coord in compare_coords_scaled
            ])
            
            if not np.any(non_overlapping_mask):
                # All compare coordinates overlap - skip this object or set to NaN
                distances[label] = np.nan
                continue
            
            filtered_compare_coords = compare_coords_scaled[non_overlapping_mask]
            
            # Build KDTree with filtered coordinates
            tree = KDTree(filtered_compare_coords)
            
            # Vectorized nearest neighbor search for all border points at once
            distances_to_all, _ = tree.query(border_coords_scaled, k=1)
            
            # Find minimum distance for this object
            distances[label] = np.min(distances_to_all)
        
        # Calculate average excluding NaN values
        valid_distances = [d for d in distances.values() if not np.isnan(d)]
        avg = np.mean(valid_distances) if valid_distances else np.nan
        return avg, distances
    
    else:
        # Original centroid comparison mode (unchanged)
        # Extract coordinates for compare_set
        compare_coords = np.array([point_centroids[point_id] for point_id in compare_set])
        
        # Vectorized scaling for compare coordinates
        compare_coords_scaled = compare_coords.astype(float)
        compare_coords_scaled[:, 0] *= z_scale  # Z coordinates
        compare_coords_scaled[:, 1:] *= xy_scale  # Y and X coordinates
        
        # Build KDTree for efficient nearest neighbor search
        tree = KDTree(compare_coords_scaled)
        
        distances = {}
        same_sets = root_set == compare_set
        
        # Extract and scale root coordinates all at once
        root_coords = np.array([point_centroids[root_id] for root_id in root_set])
        root_coords_scaled = root_coords.astype(float)
        root_coords_scaled[:, 0] *= z_scale  # Z coordinates
        root_coords_scaled[:, 1:] *= xy_scale  # Y and X coordinates
        
        # Vectorized nearest neighbor search for all root points
        if same_sets:
            distances_to_all, indices = tree.query(root_coords_scaled, k=(num + 1))
            # Remove self-matches (first column) and average the rest
            if num == 1:
                distances_array = distances_to_all[:, 1]  # Just take second nearest
            else:
                distances_array = np.mean(distances_to_all[:, 1:], axis=1)
        else:
            distances_to_all, _ = tree.query(root_coords_scaled, k=num)
            if num == 1:
                distances_array = distances_to_all.flatten()
            else:
                distances_array = np.mean(distances_to_all, axis=1)
        
        # Map back to root_ids
        for i, root_id in enumerate(root_set):
            distances[root_id] = distances_array[i]
        
        avg = np.mean(distances_array) if len(distances_array) > 0 else 0.0
        return avg, distances


#voronois:
def create_voronoi_3d_kdtree(centroids: Dict[Union[int, str], Union[Tuple[int, int, int], List[int]]], 
                            shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Create a 3D Voronoi diagram using scipy's KDTree for faster computation.
    
    Args:
        centroids: Dictionary with labels as keys and (z,y,x) coordinates as values
        shape: Optional tuple of (Z,Y,X) dimensions. If None, calculated from centroids
    
    Returns:
        3D numpy array where each cell contains the label of the closest centroid as uint32
    """
    
    # Convert string labels to integers if necessary
    if any(isinstance(k, str) for k in centroids.keys()):
        label_map = {label: idx for idx, label in enumerate(centroids.keys())}
        centroids = {label_map[k]: v for k, v in centroids.items()}
    
    # Convert centroids to array and keep track of labels
    labels = np.array(list(centroids.keys()), dtype=np.uint32)
    centroid_points = np.array([centroids[label] for label in labels])
    
    # Calculate shape if not provided
    if shape is None:
        max_coords = centroid_points.max(axis=0)
        shape = tuple(max_coord + 1 for max_coord in max_coords)
    
    # Create KD-tree
    tree = KDTree(centroid_points)
    
    # Create coordinate arrays
    coords = np.array(np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )).reshape(3, -1).T
    
    # Find nearest centroid for each point
    _, indices = tree.query(coords)
    
    # Convert indices to labels and ensure uint32 dtype
    label_array = labels[indices].astype(np.uint32)
    
    # Reshape to final shape
    return label_array.reshape(shape)



#Ripley cluster analysis:

def convert_centroids_to_array(centroids_list, xy_scale = 1, z_scale = 1):
    """
    Convert a dictionary of centroids to a numpy array suitable for Ripley's K calculation.
    
    Parameters:
    centroids_list: List of centroid coordinate arrays
    
    Returns:
    numpy array of shape (n, d) where n is number of points and d is dimensionality
    """
    # Determine how many centroids we have
    n_points = len(centroids_list)

    # Get dimensionality from the first centroid
    dim = len(list(centroids_list)[0])
    
    # Create empty array
    points_array = np.zeros((n_points, dim))
    
    # Fill array with coordinates
    for i, coords in enumerate(centroids_list):
        points_array[i] = coords

    points_array[:, 1:] = points_array[:, 1:] * xy_scale #account for scaling

    points_array[:, 0] = points_array[:, 0] * z_scale #account for scaling

    return points_array

def generate_r_values(points_array, step_size, bounds = None, dim = 2, max_proportion=0.5, max_r = None):
    """
    Generate an array of r values based on point distribution and step size.
    
    Parameters:
    points_array: numpy array of shape (n, d) with point coordinates
    step_size: user-defined step size for r values
    max_proportion: maximum proportion of the study area extent to use (default 0.5)
                   This prevents analyzing at distances where edge effects dominate
    
    Returns:
    numpy array of r values
    """

    if bounds is None:
        if dim == 2:
            min_coords = np.array([0,0])
        else:
            min_coords = np.array([0,0,0])
        max_coords = np.max(points_array, axis=0)
        max_coords = np.flip(max_coords)
    else:
        min_coords, max_coords = bounds

    
    # Calculate the longest dimension
    try:
        dimensions = max_coords - min_coords
    except: # Presume dimension mismatch
        min_coords = np.array([0,0,0])
        dimensions = max_coords - min_coords

    if 1 in dimensions:
        dimensions = np.delete(dimensions, 0) #Presuming 2D data 

    min_dimension = np.min(dimensions) #Biased for smaller dimension now for safety
    
    # Calculate maximum r value (typically half the shortest side for 2D,
    # or scaled by max_proportion for general use)
    if max_r is None:
        max_r = min_dimension * max_proportion
        if max_proportion < 1:
            print(f"Omitting search radii beyond {max_r}")
    else:
        print(f"Omitting search radii beyond {max_r} (to keep analysis within the mask)")

    
    # Generate r values from 0 to max_r with step_size increments
    num_steps = int(max_r / step_size)
    r_values = np.linspace(step_size, max_r, num_steps)

    if r_values[0] == 0:
        r_values = np.delete(r_values, 0)
    
    return r_values

def convert_augmented_array_to_points(augmented_array):
    """
    Convert an array where first column is 1 and remaining columns are coordinates.
    
    Parameters:
    augmented_array: 2D array where first column is 1 and rest are coordinates
    
    Returns:
    numpy array with just the coordinate columns
    """
    # Extract just the coordinate columns (all except first column)
    return augmented_array[:, 1:]

def optimized_ripleys_k(reference_points, subset_points, r_values, bounds=None, dim = 2, is_subset = False, volume = None, n_subset = None):
    """
    Optimized computation of Ripley's K function using KD-Tree with simplified but effective edge correction.
    
    Parameters:
    reference_points: numpy array of shape (n, d) containing coordinates (d=2 or d=3)
    subset_points: numpy array of shape (m, d) containing coordinates
    r_values: numpy array of distances at which to compute K
    bounds: tuple of (min_coords, max_coords) defining the study area boundaries
    edge_correction: Boolean indicating whether to apply edge correction
    
    Returns:
    K_values: numpy array of K values corresponding to r_values
    """
    n_ref = len(reference_points)
    if n_subset is None:
        n_subset = len(subset_points)

    # Determine bounds if not provided
    if bounds is None:
        min_coords = np.min(reference_points, axis=0)
        max_coords = np.max(reference_points, axis=0)
        bounds = (min_coords, max_coords)
    
    # Calculate volume of study area
    min_bounds, max_bounds = bounds
    sides = max_bounds - min_bounds

    if volume is None:
        if dim == 2:
            volume = sides[0] * sides[1]
        else:
            volume = np.prod(sides)
    
    # Point intensity (points per unit volume)
    intensity = n_ref / volume
    
    # Build KD-Tree for efficient nearest neighbor search
    tree = KDTree(reference_points)
    
    # Initialize K values
    K_values = np.zeros(len(r_values))
    
    # For each r value, compute cumulative counts
    for i, r in enumerate(r_values):
        total_count = 0

        # Query the tree for all points within radius r of each subset point
        for j, point in enumerate(subset_points):
            # Find all reference points within radius r
            indices = tree.query_ball_point(point, r)
            count = len(indices)
                    
            total_count += count

        # Subtract self-counts if points appear in both sets
        if is_subset or np.array_equal(reference_points, subset_points):
            total_count -= n_ref  # Subtract all self-counts

        # Normalize
        K_values[i] = total_count / (n_subset * intensity)
    
    return K_values

def ripleys_h_function_3d(k_values, r_values):
    """
    Convert K values to H values for 3D point patterns with edge correction.
    
    Parameters:
    k_values: numpy array of K function values
    r_values: numpy array of distances at which K was computed
    edge_weights: optional array of edge correction weights
    
    Returns:
    h_values: numpy array of H function values
    """
    h_values = np.cbrt(k_values / (4/3 * np.pi)) - r_values
    
    return h_values

def ripleys_h_function_2d(k_values, r_values):
    """
    Convert K values to H values for 2D point patterns with edge correction.
    
    Parameters:
    k_values: numpy array of K function values
    r_values: numpy array of distances at which K was computed
    edge_weights: optional array of edge correction weights
    
    Returns:
    h_values: numpy array of H function values
    """
    h_values = np.sqrt(k_values / np.pi) - r_values
    
    return h_values

def compute_ripleys_h(k_values, r_values, dimension=2):
    """
    Compute Ripley's H function (normalized K) with edge correction.
    
    Parameters:
    k_values: numpy array of K function values
    r_values: numpy array of distances at which K was computed
    edge_weights: optional array of edge correction weights
    dimension: dimensionality of the point pattern (2 for 2D, 3 for 3D)
    
    Returns:
    h_values: numpy array of H function values
    """
    if dimension == 2:
        return ripleys_h_function_2d(k_values, r_values)
    elif dimension == 3:
        return ripleys_h_function_3d(k_values, r_values)
    else:
        raise ValueError("Dimension must be 2 or 3")

def plot_ripley_functions(r_values, k_values, h_values, dimension=2, rootiden = None, compiden = None, figsize=(12, 5)):
    """
    Plot Ripley's K and H functions with theoretical Poisson distribution references
    adjusted for edge effects.
    
    Parameters:
    r_values: numpy array of distances at which K and H were computed
    k_values: numpy array of K function values
    h_values: numpy array of H function values (normalized K)
    edge_weights: optional array of edge correction weights
    dimension: dimensionality of the point pattern (2 for 2D, 3 for 3D)
    figsize: tuple specifying figure size (width, height)
    """

    #plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Theoretical values for complete spatial randomness (CSR)
    if dimension == 2:
        theo_k = np.pi * r_values**2  # πr² for 2D
    elif dimension == 3:
        theo_k = (4/3) * np.pi * r_values**3  # (4/3)πr³ for 3D
    else:
        raise ValueError("Dimension must be 2 or 3")
    
    # Theoretical H values are always 0 for CSR
    theo_h = np.zeros_like(r_values)
    
    # Plot K function
    ax1.plot(r_values, k_values, 'b-', label='Observed K(r)')
    ax1.plot(r_values, theo_k, 'r--', label='Theoretical K(r) for CSR')
    ax1.set_xlabel('Distance (r)')
    ax1.set_ylabel('L(r)')
    if rootiden is None or compiden is None:
        ax1.set_title("Ripley's K Function")
    else:
        ax1.set_title(f"Ripley's K Function for {compiden} Clustering Around {rootiden}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot H function
    ax2.plot(r_values, h_values, 'b-', label='Observed H(r)')
    ax2.plot(r_values, theo_h, 'r--', label='Theoretical H(r) for CSR')
    ax2.set_xlabel('Distance (r)')
    ax2.set_ylabel('L(r) Normalized')
    if rootiden is None or compiden is None:
        ax2.set_title("Ripley's H Function")
    else:
        ax2.set_title(f"Ripley's H Function for {compiden} Clustering Around {rootiden}")
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    #plt.clf()




def partition_objects_into_cells(object_centroids, cell_size):
    """
    Partition objects into 3D grid cells based on their centroids.
    
    Args:
        object_centroids (dict): Dictionary with object labels as keys and [z,y,x] coordinates as values
        cell_size (tuple or int): Size of each cell. If int, creates cubic cells. If tuple, (z_size, y_size, x_size)
    
    Returns:
        dict: Dictionary with cell numbers as keys and lists of object labels as values
    """
    
    if not object_centroids:
        return {}
    
    # Handle cell_size input
    if isinstance(cell_size, (int, float)):
        cell_size = (cell_size, cell_size, cell_size)
    elif len(cell_size) == 1:
        cell_size = (cell_size[0], cell_size[0], cell_size[0])
    
    # Extract centroids and find bounds
    centroids = np.array(list(object_centroids.values()))
    labels = list(object_centroids.keys())
    
    # Find the bounding box of all centroids
    min_coords = np.min(centroids, axis=0)  # [min_z, min_y, min_x]
    max_coords = np.max(centroids, axis=0)  # [max_z, max_y, max_x]
    
    # Calculate number of cells in each dimension
    dimensions = max_coords - min_coords
    num_cells = np.ceil(dimensions / np.array(cell_size)).astype(int)
    
    # Initialize result dictionary
    cell_assignments = defaultdict(list)
    
    # Assign each object to a cell
    for i, (label, centroid) in enumerate(object_centroids.items()):
        # Calculate which cell this centroid belongs to
        relative_pos = np.array(centroid) - min_coords
        cell_indices = np.floor(relative_pos / np.array(cell_size)).astype(int)
        
        # Ensure indices don't exceed bounds (handles edge cases)
        cell_indices = np.minimum(cell_indices, num_cells - 1)
        cell_indices = np.maximum(cell_indices, 0)
        
        # Convert 3D cell indices to a single cell number
        cell_number = (cell_indices[0] * num_cells[1] * num_cells[2] + 
                      cell_indices[1] * num_cells[2] + 
                      cell_indices[2])
        
        cell_assignments[int(cell_number)].append(int(label))
    
    # Convert defaultdict to regular dict and sort keys
    return dict(sorted(cell_assignments.items()))