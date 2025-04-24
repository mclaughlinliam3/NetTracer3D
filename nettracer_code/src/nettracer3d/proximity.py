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
    slice_obj = bounding_boxes[label-1]  # -1 because label numbers start at 1
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
                    for i in range(1, num_nodes + 1)]

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

def populate_array(centroids):
    """
    Create a 3D array from centroid coordinates.
    
    Args:
        centroids: Dictionary where keys are object IDs and values are [z,y,x] coordinates
    
    Returns:
        3D numpy array where values are object IDs at their centroid locations
    """
    # Input validation
    if not centroids:
        raise ValueError("Centroids dictionary is empty")
    
    # Convert to numpy array and get bounds
    coords = np.array(list(centroids.values()))
    # Round coordinates to nearest integer
    coords = np.round(coords).astype(int)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Check for negative coordinates
    if np.any(min_coords < 0):
        raise ValueError("Negative coordinates found in centroids")
    
    # Create array
    array = np.zeros((max_coords[0] + 1, 
                     max_coords[1] + 1, 
                     max_coords[2] + 1), dtype=int)
    
    # Populate array with rounded coordinates
    for obj_id, coord in centroids.items():
        z, y, x = np.round([coord[0], coord[1], coord[2]]).astype(int)
        array[z, y, x] = obj_id
        
    return array

def find_neighbors_kdtree(radius, centroids=None, array=None, targets=None):
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
        
        # Create a temporary array only if we need it for lookups
        if array is None:
            max_coords = np.max(points, axis=0) + 1
            array = np.zeros(tuple(max_coords), dtype=np.int32)
            for i, point in enumerate(points):
                array[tuple(point)] = node_ids[i]  # Use the actual node ID
    elif array is not None:
        points = np.transpose(np.nonzero(array))
        node_ids = None  # Not used in array-based mode
    else:
        return []
    
    # Create KD-tree from all nonzero points
    tree = KDTree(points)
    
    if targets is None:
        # Original behavior: find neighbors for all points
        query_points = points
        query_indices = range(len(points))
    else:
        # Find coordinates of target values
        target_points = []
        target_indices = []
        
        if array is not None:
            # Standard array-based filtering
            for idx, point in enumerate(points):
                point_tuple = tuple(point)
                if array[point_tuple] in targets:
                    target_points.append(point)
                    target_indices.append(idx)
        else:
            # Filter based on node IDs directly
            for idx, node_id in enumerate(node_ids):
                if node_id in targets:
                    target_points.append(points[idx])
                    target_indices.append(idx)
        
        # Convert to numpy array for querying
        query_points = np.array(target_points)
        query_indices = target_indices
        
        # Handle case where no target values were found
        if len(query_points) == 0:
            return []
    
    # Query for all points within radius of each query point
    neighbor_indices = tree.query_ball_point(query_points, radius)
    
    # Initialize output list
    output = []
    
    # Generate pairs
    for i, neighbors in enumerate(neighbor_indices):
        query_idx = query_indices[i]
        for neighbor_idx in neighbors:
            # Skip self-pairing
            if neighbor_idx != query_idx:
                # Use node IDs from the dictionary if available
                if array is not None:
                    query_value = array[tuple(points[query_idx])]
                    neighbor_value = array[tuple(points[neighbor_idx])]
                else:
                    query_value = node_ids[query_idx]
                    neighbor_value = node_ids[neighbor_idx]
                output.append([query_value, neighbor_value, 0])
    
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

def generate_r_values(points_array, step_size, bounds = None, dim = 2, max_proportion=0.5):
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
    dimensions = max_coords - min_coords
    max_dimension = np.max(dimensions)
    
    # Calculate maximum r value (typically half the shortest side for 2D,
    # or scaled by max_proportion for general use)
    max_r = max_dimension * max_proportion
    
    # Generate r values from 0 to max_r with step_size increments
    num_steps = int(max_r / step_size)
    r_values = np.linspace(step_size, max_r, num_steps)

    if r_values[0] == 0:
        np.delete(r_values, 0)
    
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

def optimized_ripleys_k(reference_points, subset_points, r_values, bounds=None, edge_correction=True, dim = 2, is_subset = False):
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
    n_subset = len(subset_points)
    
    # Determine bounds if not provided
    if bounds is None:
        min_coords = np.min(reference_points, axis=0)
        max_coords = np.max(reference_points, axis=0)
        bounds = (min_coords, max_coords)
    
    # Calculate volume of study area
    min_bounds, max_bounds = bounds
    sides = max_bounds - min_bounds
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
            
            # Apply edge correction if needed
            if edge_correction:
                # Calculate edge correction weight
                weight = 1.0
                
                if dim == 2:
                    # For 2D - check all four boundaries
                    x, y = point
                    
                    # Distances to all boundaries
                    x_min_dist = x - min_bounds[0]
                    x_max_dist = max_bounds[0] - x
                    y_min_dist = y - min_bounds[1]
                    y_max_dist = max_bounds[1] - y
                    
                    proportion_in = 1.0
                    # Apply correction for each boundary if needed
                    if x_min_dist < r:
                        proportion_in -= 0.5 * (1 - x_min_dist/r)
                    if x_max_dist < r:
                        proportion_in -= 0.5 * (1 - x_max_dist/r)
                    if y_min_dist < r:
                        proportion_in -= 0.5 * (1 - y_min_dist/r)
                    if y_max_dist < r:
                        proportion_in -= 0.5 * (1 - y_max_dist/r)
                    
                    # Corner correction
                    if ((x_min_dist < r and y_min_dist < r) or 
                        (x_min_dist < r and y_max_dist < r) or
                        (x_max_dist < r and y_min_dist < r) or
                        (x_max_dist < r and y_max_dist < r)):
                        proportion_in += 0.1  # Add a small boost for corners
                    
                elif dim == 3:
                    # For 3D - check all six boundaries
                    x, y, z = point
                    
                    # Distances to all boundaries
                    x_min_dist = x - min_bounds[0]
                    x_max_dist = max_bounds[0] - x
                    y_min_dist = y - min_bounds[1]
                    y_max_dist = max_bounds[1] - y
                    z_min_dist = z - min_bounds[2]
                    z_max_dist = max_bounds[2] - z
                    
                    proportion_in = 1.0
                    # Apply correction for each boundary if needed
                    if x_min_dist < r:
                        proportion_in -= 0.25 * (1 - x_min_dist/r)
                    if x_max_dist < r:
                        proportion_in -= 0.25 * (1 - x_max_dist/r)
                    if y_min_dist < r:
                        proportion_in -= 0.25 * (1 - y_min_dist/r)
                    if y_max_dist < r:
                        proportion_in -= 0.25 * (1 - y_max_dist/r)
                    if z_min_dist < r:
                        proportion_in -= 0.25 * (1 - z_min_dist/r)
                    if z_max_dist < r:
                        proportion_in -= 0.25 * (1 - z_max_dist/r)
                    
                    # Corner correction for 3D (if point is near a corner)
                    num_close_edges = (
                        (x_min_dist < r) + (x_max_dist < r) +
                        (y_min_dist < r) + (y_max_dist < r) +
                        (z_min_dist < r) + (z_max_dist < r)
                    )
                    if num_close_edges >= 2:
                        proportion_in += 0.05 * num_close_edges  # Stronger boost for more edges
                
                # Ensure proportion_in stays within reasonable bounds
                proportion_in = max(0.1, min(1.0, proportion_in))
                weight = 1.0 / proportion_in
                
                count *= weight
            
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

def plot_ripley_functions(r_values, k_values, h_values, dimension=2, figsize=(12, 5)):
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
    ax1.set_title("Ripley's K Function")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot H function
    ax2.plot(r_values, h_values, 'b-', label='Observed H(r)')
    ax2.plot(r_values, theo_h, 'r--', label='Theoretical H(r) for CSR')
    ax2.set_xlabel('Distance (r)')
    ax2.set_ylabel('L(r) Normalized')
    ax2.set_title("Ripley's H Function")
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    #plt.clf()