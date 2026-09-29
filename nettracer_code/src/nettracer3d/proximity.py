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
from . import smart_dilate as sdl
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm
import re

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
    #print(f"Processing node {label}")
    
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

def find_neighbors_kdtree(radius, centroids=None, array=None, targets=None, max_neighbors=None):
    """
    Find neighbors using KDTree.
    
    Parameters:
    -----------
    radius : float or None
        Search radius for finding neighbors. If None and max_neighbors is set,
        returns k nearest neighbors at any distance.
    centroids : dict or list, optional
        Dictionary mapping node IDs to coordinates or list of points
    array : numpy.ndarray, optional
        Array to search for nonzero points
    targets : list, optional
        Specific targets to query for neighbors
    max_neighbors : int, optional
        Maximum number of nearest neighbors to return per query point.
        If radius is also set, returns up to max_neighbors within radius.
        If radius is None, returns exactly max_neighbors at any distance.
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
        
    # Determine query strategy based on parameters
    if radius is None and max_neighbors is not None:
        # Case 1: k-nearest neighbors at any distance
        # Query for k+1 to include self, which we'll filter out
        k = max_neighbors + 1
        distances, neighbor_indices = tree.query(query_points, k=k)
        
    elif max_neighbors is not None:
        # Case 2: k-nearest neighbors within radius
        # Query for enough neighbors to ensure we get max_neighbors within radius
        # Use a larger k to be safe, then filter by radius
        k = min(len(points), max_neighbors * 3 + 1)  # Heuristic: query more than needed
        distances, neighbor_indices = tree.query(query_points, k=k)
        
    else:
        # Case 3: all neighbors within radius (original behavior)
        neighbor_indices = tree.query_ball_point(query_points, radius)
        distances = None  # Not needed for this case
        
    # Process results efficiently
    query_indices = np.asarray(query_indices)
    
    # ------------------------------------------------------------------
    # STEP 1: build flat (row_query_idx, neighbor_idx) pairs, filtered,
    # using whole-matrix numpy ops instead of per-row work.
    # ------------------------------------------------------------------
    if distances is not None:
        neigh = np.asarray(neighbor_indices)
        dist = np.asarray(distances)
    
        if neigh.ndim == 1:
            # Original semantics: every query row reuses the same 1-D result
            neigh = np.broadcast_to(neigh, (len(query_indices), neigh.shape[0]))
            dist = np.broadcast_to(dist, (len(query_indices), dist.shape[0]))
    
        # mask: not self, and within radius if set
        mask = neigh != query_indices[:, None]
        if radius is not None:
            mask &= dist <= radius
    
        # max_neighbors: keep only the first m surviving entries per row.
        # Rows are pre-sorted by distance, so a cumulative count along the
        # row reproduces the original early-break exactly.
        if max_neighbors is not None:
            mask &= np.cumsum(mask, axis=1) <= max_neighbors
    
        rows, cols = np.nonzero(mask)          # row-major -> preserves order
        q_flat = query_indices[rows]
        n_flat = neigh[rows, cols]
    else:
        # Radius mode: ragged list of neighbor lists
        lengths = np.fromiter((len(x) for x in neighbor_indices),
                              dtype=np.int64, count=len(neighbor_indices))
        if lengths.sum() == 0:
            return []
        n_flat = np.concatenate([np.asarray(x, dtype=np.int64)
                                 for x in neighbor_indices])
        q_flat = np.repeat(query_indices, lengths)
        keep = n_flat != q_flat
        q_flat = q_flat[keep]
        n_flat = n_flat[keep]
    
    if len(n_flat) == 0:
        return []
    
    # ------------------------------------------------------------------
    # STEP 2: map indices -> values with array lookups (no per-row dict
    # hits or tuple indexing inside a Python loop).
    # ------------------------------------------------------------------
    if centroids:
        # Build a dense lookup table from the dict once per call.
        hi = int(max(q_flat.max(), n_flat.max())) + 1
        lut = np.zeros(hi, dtype=np.int64)
        present = np.zeros(hi, dtype=bool)
        for k, v in idx_to_node.items():
            if 0 <= k < hi:
                lut[k] = v
                present[k] = True
    
        # Original behavior: missing *neighbor* keys are silently skipped
        # (the try/except), but a missing *query* key raised KeyError
        # because that lookup was outside the try block.
        if not present[q_flat].all():
            bad = int(q_flat[~present[q_flat]][0])
            raise KeyError(bad)
        keep = present[n_flat]
        q_vals = lut[q_flat[keep]]
        n_vals = lut[n_flat[keep]]
    else:
        # One pass to turn point_tuples -> value array, then pure fancy indexing
        pt = np.asarray(point_tuples)          # (P, ndim)
        vals = array[tuple(pt.T)]              # value for every point index
        q_vals = vals[q_flat]
        n_vals = vals[n_flat]
    
    # ------------------------------------------------------------------
    # STEP 3: assemble [[q, n, 0], ...] in one shot
    # ------------------------------------------------------------------
    out = np.empty((len(q_vals), 3), dtype=np.int64)
    out[:, 0] = q_vals
    out[:, 1] = n_vals
    out[:, 2] = 0

    return out.tolist()
    
    
    #return output

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
    (docstring unchanged)
    """

    if do_borders:
        # Border comparison mode
        if not isinstance(compare_set, np.ndarray):
            raise ValueError("When do_borders=True, compare_set must be a numpy array of coordinates")

        compare_coords_scaled = compare_set.astype(float)
        compare_coords_scaled[:, 0] *= z_scale
        compare_coords_scaled[:, 1:] *= xy_scale

        distances = {}

        for label, border_coords in root_set.items():
            if len(border_coords) == 0:
                continue

            border_coords_scaled = border_coords.astype(float)
            border_coords_scaled[:, 0] *= z_scale
            border_coords_scaled[:, 1:] *= xy_scale

            border_coords_set = set(map(tuple, border_coords_scaled))

            non_overlapping_mask = np.array([
                tuple(coord) not in border_coords_set
                for coord in compare_coords_scaled
            ])

            if not np.any(non_overlapping_mask):
                distances[label] = np.nan
                continue

            filtered_compare_coords = compare_coords_scaled[non_overlapping_mask]

            tree = KDTree(filtered_compare_coords)
            distances_to_all, _ = tree.query(border_coords_scaled, k=1)
            distances[label] = np.min(distances_to_all)

        valid_distances = [d for d in distances.values() if not np.isnan(d)]
        avg = np.mean(valid_distances) if valid_distances else np.nan
        return avg, distances

    else:
        # Centroid comparison mode
        compare_coords = np.array([point_centroids[point_id] for point_id in compare_set])
        compare_coords_scaled = compare_coords.astype(float)
        compare_coords_scaled[:, 0] *= z_scale   # Z
        compare_coords_scaled[:, 1:] *= xy_scale  # Y and X

        tree = KDTree(compare_coords_scaled)

        root_coords = np.array([point_centroids[root_id] for root_id in root_set])
        root_coords_scaled = root_coords.astype(float)
        root_coords_scaled[:, 0] *= z_scale
        root_coords_scaled[:, 1:] *= xy_scale

        epsilon = 1e-10

        # Query num+1 neighbors so a self-match (distance ~0) can be dropped when a
        # root point also lives in compare_set. Never request more neighbors than
        # there are points in the tree -- this avoids scipy padding with inf and,
        # critically, avoids k==1 (which makes query() return a 1-D array).
        k_query = min(num + 1, len(tree.data))
        distances_to_all, _ = tree.query(root_coords_scaled, k=k_query)

        # scipy squeezes the trailing axis when k == 1 (e.g. compare_set has a single
        # point). Restore a consistent 2-D shape so the column logic below works.
        if distances_to_all.ndim == 1:
            distances_to_all = distances_to_all[:, np.newaxis]

        # Per-row self-match removal: push ~0 distances to the back, then take the
        # `num` nearest remaining neighbors. A row whose only neighbor is itself
        # (no valid comparison point) becomes NaN.
        masked = np.where(distances_to_all < epsilon, np.inf, distances_to_all)
        masked.sort(axis=1)

        selected = masked[:, :num]
        valid_mask = ~np.isinf(selected)
        valid_counts = valid_mask.sum(axis=1)
        sums = np.where(valid_mask, selected, 0.0).sum(axis=1)
        distances_array = np.where(
            valid_counts > 0,
            sums / np.maximum(valid_counts, 1),
            np.nan,
        )

        # Map back to root_ids
        distances = {}
        for i, root_id in enumerate(root_set):
            distances[root_id] = distances_array[i]

        valid = distances_array[~np.isnan(distances_array)]
        avg = np.mean(valid) if len(valid) > 0 else np.nan
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

def convert_centroids_to_array(centroids_list, xy_scale=1, z_scale=1):
    """
    Convert an iterable of centroids to an (n, d) array in physical units.
 
    3D centroids are taken as (z, y, x) and 2D centroids as (y, x); the
    z_scale is only ever applied to a leading z column that actually exists.
    """
    centroids_list = list(centroids_list)
    if not centroids_list:
        return np.zeros((0, 3))
 
    points_array = np.asarray(centroids_list, dtype=float)
    if points_array.ndim != 2:
        raise ValueError(
            f"expected a sequence of coordinate tuples, got shape {points_array.shape}")
 
    if points_array.shape[1] == 3:
        points_array[:, 0] *= z_scale        # z
        points_array[:, 1:] *= xy_scale      # y, x
    elif points_array.shape[1] == 2:
        points_array *= xy_scale             # y, x — no z to scale
    else:
        raise ValueError(
            f"centroids must be 2D or 3D, got {points_array.shape[1]} components")
 
    return points_array
 
 
def convert_augmented_array_to_points(augmented_array):
    """Drop the leading constant column from a (1, y, x) style array."""
    return augmented_array[:, 1:]
 
 
# ---------------------------------------------------------------------
# radius grid
# ---------------------------------------------------------------------
 
 
def generate_r_values(points_array, step_size, bounds=None, dim=2,
                      max_proportion=0.5, max_r=None,
                      xy_scale=1.0, z_scale=1.0):
    """
    Build the radius grid.
 
    `bounds` is expected in (x, y, z) order and in voxel units, matching what
    get_ripley assembles. `points_array` is in scaled units, so the bounds
    are scaled here before max_r is derived from them — otherwise the cap is
    wrong by a factor of the pixel size.
 
    `max_r`, when supplied, is assumed to already be in scaled units (the
    `min_legal` distance-transform value from a mask is).
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")
 
    if bounds is None:
        # points_array is (n, d) with d matching dim; extents straight off it.
        extents = np.max(points_array, axis=0) - np.min(points_array, axis=0)
        extents = np.flip(extents)                      # -> (x, y[, z])
    else:
        min_coords, max_coords = bounds
        min_coords = np.asarray(min_coords, dtype=float)
        max_coords = np.asarray(max_coords, dtype=float)
        if min_coords.shape != max_coords.shape:
            # Legacy callers sometimes pass a 3-vector min against a 2-vector
            # max; fall back to an origin of the right length.
            min_coords = np.zeros_like(max_coords)
        extents = max_coords - min_coords
 
        # bounds are voxel counts, points are scaled — put them in the
        # same units before deriving a radius from them.
        scale = np.array([xy_scale, xy_scale, z_scale][:extents.size])
        extents = extents * scale
 
    # Keep only the axes that actually exist. A single-slice stack has a
    # z extent of 1 in the *last* position under the (x, y, z) convention.
    extents = np.asarray(extents, dtype=float)[:2 if dim == 2 else 3]
    extents = extents[extents > 0]
    if extents.size == 0:
        raise ValueError("degenerate bounds: every extent is zero")
 
    if max_r is None:
        max_r = float(np.min(extents)) * max_proportion
        if max_proportion < 1:
            print(f"Omitting search radii beyond {max_r}")
    else:
        max_r = float(max_r)
        print(f"Omitting search radii beyond {max_r} "
              "(to keep analysis within the mask)")
 
    if max_r < step_size:
        raise ValueError(
            f"step_size ({step_size}) exceeds the largest usable radius "
            f"({max_r:.4g}); nothing to compute")
 
    # arange, not linspace: the grid must actually step by step_size.
    r_values = np.arange(step_size, max_r + step_size * 1e-9, step_size)
    return r_values
 
 
# ---------------------------------------------------------------------
# the K function
# ---------------------------------------------------------------------
 
 
def _count_coincident_pairs(reference_points, subset_points, tol=1e-9):
    """
    Number of (reference, subset) pairs sitting at the same location.
 
    These are the self-pairs: a point that appears in both sets contributes
    a distance of zero, which lands in every cumulative bin. Counting them
    directly means the caller never has to declare whether the sets overlap,
    and it stays correct when only *some* of the points are shared — which
    is exactly the case after the interior filter, and after edge-correction
    mirroring duplicates points into both sets.
    """
    if len(reference_points) == 0 or len(subset_points) == 0:
        return 0
    return int(KDTree(reference_points).count_neighbors(
        KDTree(subset_points), tol))
 
 
def optimized_ripleys_k(reference_points, subset_points, r_values, bounds=None,
                        dim=2, is_subset=False, volume=None, n_subset=None,
                        n_ref=None, reference_tree=None, subset_tree=None):
    """
    Ripley's K (or cross-K) for a reference and a subset population.
 
        K(r) = |W| * (pairs closer than r) / (n_ref * n_subset)
 
    reference_points : the roots, in scaled units
    subset_points    : the targets, in scaled units
    volume           : |W|; derived from bounds when omitted
    n_ref, n_subset  : true population sizes. Supply these when the point
                       arrays have been augmented by edge-correction
                       mirroring, so the normalisation uses real counts
                       rather than inflated ones.
    is_subset        : accepted for backward compatibility and ignored.
                       Overlap is now measured, not declared.
    """
    reference_points = np.asarray(reference_points, dtype=float)
    subset_points = np.asarray(subset_points, dtype=float)
 
    if reference_points.size == 0 or subset_points.size == 0:
        raise ValueError("Ripley's K needs at least one root and one target")
 
    if n_ref is None:
        n_ref = len(reference_points)
    if n_subset is None:
        n_subset = len(subset_points)
    if n_ref <= 0 or n_subset <= 0:
        raise ValueError("n_ref and n_subset must be positive")
 
    if volume is None:
        if bounds is None:
            min_coords = np.min(reference_points, axis=0)
            max_coords = np.max(reference_points, axis=0)
        else:
            min_coords, max_coords = bounds
        sides = np.asarray(max_coords, dtype=float) - np.asarray(min_coords, dtype=float)
        volume = float(sides[0] * sides[1] if dim == 2 else np.prod(sides))
    if volume <= 0:
        raise ValueError(f"study volume must be positive, got {volume}")
 
    r_values = np.asarray(r_values, dtype=float)
 
    if reference_tree is None:
        reference_tree = KDTree(reference_points)
    if subset_tree is None:
        subset_tree = KDTree(subset_points)

    # One vectorised sweep instead of a tree query per (target, radius).
    pair_counts = reference_tree.count_neighbors(
        subset_tree, r_values, cumulative=True).astype(float)

    # Points present in both sets pair with themselves at distance zero and
    # so appear in every bin. Remove exactly as many as are really there.
    self_pairs = int(reference_tree.count_neighbors(subset_tree, 1e-9))
    pair_counts -= self_pairs
 
    if np.any(pair_counts < 0):
        raise RuntimeError(
            "negative pair count after self-pair removal — this should be "
            "impossible; check that reference and subset arrays are in the "
            "same units")
 
    k_values = volume * pair_counts / (float(n_ref) * float(n_subset))
 
    if np.any(k_values < 0):
        raise RuntimeError("K(r) came out negative, which is not possible")
 
    return k_values
 
 
# ---------------------------------------------------------------------
# the H function
# ---------------------------------------------------------------------
 
 
def _check_k(k_values):
    k_values = np.asarray(k_values, dtype=float)
    if np.any(~np.isfinite(k_values)):
        raise ValueError("K contains NaN or inf")
    if np.any(k_values < 0):
        raise ValueError(
            "K contains negative values, so L(r) is undefined. K is a pair "
            "count divided by an intensity and cannot be negative — the fault "
            "is upstream in optimized_ripleys_k, not here.")
    return k_values
 
 
def ripleys_h_function_3d(k_values, r_values):
    """Besag's L for 3D, centred: (3K / 4pi)^(1/3) - r."""
    return np.cbrt(_check_k(k_values) / (4 / 3 * np.pi)) - np.asarray(r_values, float)
 
 
def ripleys_h_function_2d(k_values, r_values):
    """Besag's L for 2D, centred: sqrt(K / pi) - r."""
    return np.sqrt(_check_k(k_values) / np.pi) - np.asarray(r_values, float)
 
 
def compute_ripleys_h(k_values, r_values, dimension=2):
    """Ripley's H = L(r) - r. Zero under CSR, positive when clustered."""
    if dimension == 2:
        return ripleys_h_function_2d(k_values, r_values)
    if dimension == 3:
        return ripleys_h_function_3d(k_values, r_values)
    raise ValueError("Dimension must be 2 or 3")
 
 
# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------
 
 
def plot_ripley_functions(r_values, k_values, h_values, dimension=2,
                          rootiden=None, compiden=None, figsize=(12, 5)):
    """Plot K and H against their complete-spatial-randomness references."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
 
    if dimension == 2:
        theo_k = np.pi * np.asarray(r_values) ** 2
    elif dimension == 3:
        theo_k = (4 / 3) * np.pi * np.asarray(r_values) ** 3
    else:
        raise ValueError("Dimension must be 2 or 3")
 
    ax1.plot(r_values, k_values, 'b-', label='Observed K(r)')
    ax1.plot(r_values, theo_k, 'r--', label='Theoretical K(r) for CSR')
    ax1.set_xlabel('Distance (r)')
    ax1.set_ylabel('K(r)')
    ax1.set_title("Ripley's K Function" if rootiden is None or compiden is None
                  else f"Ripley's K Function for {compiden} Clustering Around {rootiden}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    ax2.plot(r_values, h_values, 'b-', label='Observed H(r)')
    ax2.plot(r_values, np.zeros_like(r_values), 'r--',
             label='Theoretical H(r) for CSR')
    ax2.set_xlabel('Distance (r)')
    ax2.set_ylabel('H(r) = L(r) - r')
    ax2.set_title("Ripley's H Function" if rootiden is None or compiden is None
                  else f"Ripley's H Function for {compiden} Clustering Around {rootiden}")
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.show()
    #plt.clf()

class BatchRipley:
    """
    Ripley's K and H for every ordered identity pair, sharing one radius grid.

    Every pair is computed against the same r values, the same study volume
    and the same interior root criterion, so the columns of an output file are
    directly comparable and the expensive geometry is paid for once.

    Ordered, not unordered: roots are restricted to the mask interior while
    targets are not, so (A as root, B as target) is a different statistic from
    (B as root, A as target). Both are computed.

    Parameters mirror `get_ripley`, minus root and targ. All-node combinations
    are deliberately excluded — batch mode iterates identities only.
    """

    def __init__(self, network, distance=5.0, edgecorrect=False, bounds=None,
                 ignore_dims=True, proportion=0.5, mode=0, safe=False,
                 factor=0.25):

        if network.node_identities is None:
            raise ValueError(
                "Batch mode needs node identities; there is nothing to pair up.")
        if not network.node_centroids:
            raise ValueError("Batch mode needs node centroids.")

        self.network = network
        self.distance = distance
        self.edgecorrect = edgecorrect
        self.ignore_dims = ignore_dims
        self.mode = mode
        self.safe = safe
        self.factor = factor

        # ---- geometry, once ------------------------------------------------
        self.min_coords, self.max_coords = network._ripley_bounds(bounds)
        self.bounds = (self.min_coords, self.max_coords)
        sides = self.max_coords - self.min_coords

        self.dim = network._ripley_dim()
        if self.dim == 2:
            self.volume = sides[0] * sides[1] * network.xy_scale ** 2
        else:
            self.volume = np.prod(sides) * network.z_scale * network.xy_scale ** 2

        self.max_r = None
        # `safe` overrides the proportion, exactly as get_ripley does.
        self.proportion = factor if safe else proportion

        # ---- node tables, once ---------------------------------------------
        self.node_ids = np.array([int(node) for node in network.node_centroids])
        self.centroids = np.array(
            [network.node_centroids[node] for node in network.node_centroids],
            dtype=float)                                     # unscaled (z, y, x)

        # Scaled points. get_ripley drops the degenerate z column for 2D data
        # after edge correction, so keep both forms: the 3-column version goes
        # to the r-value generator and the edge corrector, the trimmed one to
        # the trees and the K function.
        self._points_full = convert_centroids_to_array(
            self.centroids, xy_scale=network.xy_scale, z_scale=network.z_scale)
        self._points = (self._points_full[:, 1:] if self.dim == 2
                        else self._points_full)

        row_of = {node: i for i, node in enumerate(self.node_ids)}

        # ---- identity -> row indices, once ---------------------------------
        rows_by_identity = {}
        for node, identity_list in network.node_identities.items():
            row = row_of.get(int(node))
            if row is None:                 # identity without a centroid
                continue
            for identity in identity_list:
                rows_by_identity.setdefault(identity, []).append(row)

        self.identities = sorted(rows_by_identity)
        self._rows = {identity: np.array(sorted(rows), dtype=int)
                      for identity, rows in rows_by_identity.items()}

        # ---- interior root criterion, once ---------------------------------
        # This is the part that would otherwise redo a whole-image distance
        # transform for all n^2 pairs.
        interior = np.ones(len(self._points), dtype=bool)
        if ignore_dims:
            if mode == 0:
                interior = self._interior_by_image_bounds(factor)
            else:
                interior_ids, self.volume, self.max_r = \
                    network._ripley_interior_ids(mode, self.dim, factor, safe)
                interior = np.zeros(len(self._points), dtype=bool)
                for node in interior_ids:
                    row = row_of.get(int(node))
                    if row is not None:
                        interior[row] = True
        self._interior = interior

        # ---- the shared radius grid, once ----------------------------------
        # Built from every node rather than one pair's points, so all columns
        # of an output file line up. With bounds supplied (the normal case)
        # the point cloud is not consulted at all.
        self.r_values = generate_r_values(
            self._points_full, distance, bounds=self.bounds, dim=self.dim,
            max_proportion=self.proportion, max_r=self.max_r,
            xy_scale=network.xy_scale, z_scale=network.z_scale)

        self._root_cache = {}
        self._targ_cache = {}
        self.failures = {}

    # -----------------------------------------------------------------
    # interior selection for mode 0
    # -----------------------------------------------------------------

    def _interior_by_image_bounds(self, factor):
        """
        Vectorised form of `_ripley_roots_inside_image` over every node at once.

        Bounds are (x, y, z) ordered and centroids are (z, y, x), hence the
        column reversal.
        """
        span = self.max_coords - self.min_coords
        xyz = self.centroids[:, ::-1]                  # -> (x, y, z)
        n_axes = 3 if self.dim == 3 else 2

        keep = np.ones(len(xyz), dtype=bool)
        for axis in range(n_axes):
            margin = span[axis] * factor
            keep &= (xyz[:, axis] - self.min_coords[axis] > margin)
            keep &= (self.max_coords[axis] - xyz[:, axis] > margin)
        return keep

    # -----------------------------------------------------------------
    # cached per-identity point sets and trees
    # -----------------------------------------------------------------

    def root_set(self, identity):
        """(points, tree, n) for an identity acting as root: interior only."""
        if identity not in self._root_cache:
            rows = self._rows[identity][self._interior[self._rows[identity]]]
            points = self._points[rows]
            tree = KDTree(points) if len(points) else None
            self._root_cache[identity] = (points, tree, rows)
        return self._root_cache[identity]

    def target_set(self, identity):
        """(points, tree, rows) for an identity acting as target: all of them."""
        if identity not in self._targ_cache:
            rows = self._rows[identity]
            points = self._points[rows]
            tree = KDTree(points) if len(points) else None
            self._targ_cache[identity] = (points, tree, rows)
        return self._targ_cache[identity]

    def root_count(self, identity):
        return len(self.root_set(identity)[0])

    # -----------------------------------------------------------------
    # one pair
    # -----------------------------------------------------------------

    def pair(self, root, targ):
        """
        K and H for one ordered identity pair.

        Returns (k_values, h_values). Both are NaN-filled and the reason is
        recorded in `self.failures` when the pair cannot be computed — an
        empty population, or an error from the estimator. One bad pair must
        not abandon the other n^2 - 1.
        """
        blank = np.full(len(self.r_values), np.nan)

        root_points, root_tree, root_rows = self.root_set(root)
        targ_points, targ_tree, targ_rows = self.target_set(targ)

        if len(root_points) == 0:
            self.failures[(root, targ)] = "no interior root points"
            return blank, blank.copy()
        if len(targ_points) == 0:
            self.failures[(root, targ)] = "no target points"
            return blank, blank.copy()

        n_ref, n_subset = len(root_points), len(targ_points)

        try:
            if self.edgecorrect:
                # Mirroring produces new arrays, so the cached trees are no
                # use here and K has to build its own.
                mirrored_roots, mirrored_targs = apply_edge_correction_to_ripley(
                    self._points_full[root_rows], self._points_full[targ_rows],
                    self.proportion, self.bounds, self.dim,
                    node_centroids=self.network.node_centroids)
                if self.dim == 2:
                    mirrored_roots = convert_augmented_array_to_points(
                        mirrored_roots)
                    mirrored_targs = convert_augmented_array_to_points(
                        mirrored_targs)
                k_values = optimized_ripleys_k(
                    mirrored_roots, mirrored_targs, self.r_values,
                    bounds=self.bounds, dim=self.dim, volume=self.volume,
                    n_subset=n_subset, n_ref=n_ref)
            else:
                k_values = optimized_ripleys_k(
                    root_points, targ_points, self.r_values,
                    bounds=self.bounds, dim=self.dim, volume=self.volume,
                    n_subset=n_subset, n_ref=n_ref,
                    reference_tree=root_tree, subset_tree=targ_tree)

            h_values = compute_ripleys_h(
                k_values, self.r_values, self.dim)
        except Exception as error:
            self.failures[(root, targ)] = f"{type(error).__name__}: {error}"
            return blank, blank.copy()

        return k_values, h_values

    # -----------------------------------------------------------------
    # the batch
    # -----------------------------------------------------------------

    def run(self, output_dir, progress=None):
        """
        Compute every ordered identity pair and write the results out.

        One CSV per root identity per statistic: `Ripley K/<root>.csv` holds a
        radius column plus one column per target identity, including the root
        against itself. n identities give 2n files rather than 2n^2.

        A matching PNG goes beside each CSV. Figures are rendered through the
        Agg canvas directly rather than pyplot, so nothing is drawn into the
        application's own figure manager and no windows appear.

        `progress` is an optional callable (done, total, label) -> bool.
        Returning False cancels; `run` then returns None having written only
        the identities it finished.
        """
        output_dir = Path(output_dir)
        k_dir = output_dir / "Ripley K"
        h_dir = output_dir / "Ripley H"
        k_dir.mkdir(parents=True, exist_ok=True)
        h_dir.mkdir(parents=True, exist_ok=True)

        self._write_parameters(output_dir)

        total = len(self.identities) ** 2
        done = 0
        written = 0
        used_names = {}

        for root in self.identities:
            k_columns = {}
            h_columns = {}

            for targ in self.identities:
                k_values, h_values = self.pair(root, targ)
                k_columns[targ] = k_values
                h_columns[targ] = h_values
                done += 1

                if progress is not None and not progress(
                        done, total, f"{root} vs {targ}"):
                    return None

            stem = self._filename(root, used_names)
            radius = {"Radius (scaled)": self.r_values}

            pd.DataFrame({**radius, **k_columns}).to_csv(
                k_dir / f"{stem}.csv", index=False)
            pd.DataFrame({**radius, **h_columns}).to_csv(
                h_dir / f"{stem}.csv", index=False)

            self._save_figure(k_dir / f"{stem}.png", root, k_columns,
                              statistic="K")
            self._save_figure(h_dir / f"{stem}.png", root, h_columns,
                              statistic="H")
            written += 1

        return written

    # -----------------------------------------------------------------
    # output helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _filename(identity, used_names):
        """A filesystem-safe stem, disambiguated if sanitising collides."""
        stem = re.sub(r"[^\w\-. ]", "_", str(identity)).strip() or "identity"
        stem = stem[:120]
        if stem in used_names:
            used_names[stem] += 1
            stem = f"{stem}_{used_names[stem]}"
        else:
            used_names[stem] = 0
        return stem

    def _write_parameters(self, output_dir):
        """Record the settings, so a folder of results stays interpretable."""
        lines = [
            "Batch Ripley parameters",
            "=" * 40,
            f"identities        : {len(self.identities)}",
            f"pairs             : {len(self.identities) ** 2}",
            f"dimension         : {self.dim}D",
            f"study volume      : {self.volume}",
            f"bucket distance   : {self.distance}",
            f"radius grid       : {self.r_values[0]} .. {self.r_values[-1]} "
            f"({len(self.r_values)} values)",
            f"proportion        : {self.proportion}",
            f"factor            : {self.factor}",
            f"exclude border    : {self.ignore_dims}",
            f"boundary mode     : {self.mode}",
            f"radius cap (safe) : {self.safe} ({self.max_r})",
            f"edge correction   : {self.edgecorrect}",
            f"xy scale          : {self.network.xy_scale}",
            f"z scale           : {self.network.z_scale}",
            "",
            "interior root counts per identity:",
        ]
        for identity in self.identities:
            lines.append(f"  {identity}: {self.root_count(identity)} of "
                         f"{len(self._rows[identity])}")

        if self.failures:
            lines += ["", "pairs that could not be computed:"]
            lines += [f"  {root} vs {targ}: {why}"
                      for (root, targ), why in sorted(self.failures.items())]

        (output_dir / "batch_parameters.txt").write_text("\n".join(lines))

    def _save_figure(self, path, root, columns, statistic):
        """One figure per root identity, every target overlaid."""
        figure = Figure(figsize=(9, 5.5))
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)

        plotted = 0
        palette = np.linspace(0, 1, max(len(columns), 2))

        for index, (targ, values) in enumerate(columns.items()):
            if not np.any(np.isfinite(values)):
                continue
            axes.plot(self.r_values, values, lw=1.2,
                      color=cm.viridis(palette[index]), label=str(targ))
            plotted += 1

        if statistic == "K":
            theoretical = (np.pi * self.r_values ** 2 if self.dim == 2
                           else (4 / 3) * np.pi * self.r_values ** 3)
            axes.plot(self.r_values, theoretical, 'k--', lw=1.4,
                      label="CSR")
            axes.set_ylabel("K(r)")
        else:
            axes.axhline(0, color='k', ls='--', lw=1.4)
            axes.set_ylabel("H(r) = L(r) - r")

        axes.set_xlabel("Distance r (scaled)")
        axes.set_title(f"Ripley's {statistic}: targets clustering around {root}")
        axes.grid(True, alpha=0.3)

        # A legend for seventy curves is worse than none. Nothing labelled at
        # all means every column was blank, so skip it rather than warn.
        if plotted <= 15 and (plotted > 0 or statistic == "K"):
            axes.legend(fontsize=8, ncol=2, frameon=False)
        else:
            axes.text(0.99, 0.02, f"{plotted} target identities",
                      transform=axes.transAxes, ha="right", fontsize=8,
                      alpha=0.7)

        figure.tight_layout()
        figure.savefig(path, dpi=150)




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



# To use with the merge node identities manual calculation: 

#Numba implementation:

from numba import jit, prange


def convert_bboxes_to_array(bounding_boxes, array_shape):
    """Convert scipy bounding boxes to Numba-compatible array"""
    num_labels = len(bounding_boxes)
    bboxes_array = np.full((num_labels, 6), -1, dtype=np.int32)
    
    for i, bbox in enumerate(bounding_boxes):
        if bbox is None:
            continue
        
        z_slice, y_slice, x_slice = bbox
        
        z_min, z_max = z_slice.start, z_slice.stop - 1
        y_min, y_max = y_slice.start, y_slice.stop - 1
        x_min, x_max = x_slice.start, x_slice.stop - 1
        
        # Boundary checks
        z_max = min(z_max, array_shape[0] - 1)
        y_max = min(y_max, array_shape[1] - 1)
        x_max = min(x_max, array_shape[2] - 1)
        z_min = max(z_min, 0)
        y_min = max(y_min, 0)
        x_min = max(x_min, 0)
        
        bboxes_array[i] = [z_min, z_max, y_min, y_max, x_min, x_max]
    
    return bboxes_array

@jit(nopython=True, parallel=True)
def compute_all_means_numba(nodes, edges, select_nodes, bboxes_array):
    """
    Single Numba function with automatic parallelization.
    Only computes means for the node IDs in select_nodes.
    """
    n = len(select_nodes)
    results = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        label = select_nodes[i]  # Already the 1-based node ID
        idx = label - 1          # 0-based index into bboxes_array

        # Get bounding box
        z_min, z_max, y_min, y_max, x_min, x_max = bboxes_array[idx]

        # Skip if invalid bounding box
        if z_min < 0:
            results[i] = 0.0
            continue

        # Extract subcell
        sub_nodes = nodes[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        sub_edges = edges[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

        # Compute mean for this label
        sum_val = 0.0
        count = 0

        nodes_flat = sub_nodes.ravel()
        edges_flat = sub_edges.ravel()

        for j in range(len(nodes_flat)):
            if nodes_flat[j] == label and edges_flat[j] > 0:
                sum_val += edges_flat[j]
                count += 1

        results[i] = sum_val / count if count > 0 else 0.0

    return results


def create_node_dictionary_id_numba(nodes, edges, num_nodes, bounding_boxes, select_nodes):
    """
    Pure Numba version with byte order handling for TIFF compatibility.
    """
    # Convert to native byte order if needed
    if nodes.dtype.byteorder == '>' or nodes.dtype.byteorder == '<':
        nodes = np.ascontiguousarray(nodes, dtype=nodes.dtype.newbyteorder('='))
    if edges.dtype.byteorder == '>' or edges.dtype.byteorder == '<':
        edges = np.ascontiguousarray(edges, dtype=edges.dtype.newbyteorder('='))

    # Ensure select_nodes is a numpy array for Numba
    select_nodes_arr = np.asarray(select_nodes, dtype=np.int64)

    # Single Numba call — only processes select_nodes
    results = compute_all_means_numba(nodes, edges, select_nodes_arr, bounding_boxes)

    # Convert to dictionary keyed by the actual node IDs
    node_dict = {select_nodes_arr[i]: results[i] for i in range(len(select_nodes_arr))}

    return node_dict


def create_node_dictionary_id(nodes, edges, num_nodes, bounding_boxes, select_nodes=None):
    if select_nodes is None:
        select_nodes = list(range(1, num_nodes + 1))
    try:
        from numba import jit
        return create_node_dictionary_id_numba(nodes, edges, num_nodes, bounding_boxes, select_nodes)
    except:
        import traceback
        print(traceback.format_exc())
        print("Attempting without numba...")
        return create_node_dictionary_id_python(nodes, edges, num_nodes, bounding_boxes, select_nodes)


#Non numba:

def get_reslice_indices_for_id(slice_obj, array_shape):
    """Convert slice object to padded indices accounting for dilation and boundaries"""
    if slice_obj is None:
        return None, None, None
        
    z_slice, y_slice, x_slice = slice_obj
    
    # Extract min/max from slices
    z_min, z_max = z_slice.start, z_slice.stop - 1
    y_min, y_max = y_slice.start, y_slice.stop - 1
    x_min, x_max = x_slice.start, x_slice.stop - 1

    # Boundary checks
    y_max = min(y_max, array_shape[1] - 1)
    x_max = min(x_max, array_shape[2] - 1)
    z_max = min(z_max, array_shape[0] - 1)
    y_min = max(y_min, 0)
    x_min = max(x_min, 0)
    z_min = max(z_min, 0)

    return [z_min, z_max], [y_min, y_max], [x_min, x_max]


def _get_node_edge_dict_id(label_array, edge_array, label):
    """Internal method used for the secondary algorithm to find which nodes interact with which edges."""
    
    # Create compound condition: label matches AND edge value > 0
    valid_mask = (label_array == label) & (edge_array > 0)
    valid_edges = edge_array[valid_mask]
    
    if len(valid_edges) > 0:
        edge_val = np.mean(valid_edges)
    else:
        edge_val = 0  
    
    return edge_val
    
def process_label_id(args):
    """Modified to use pre-computed bounding boxes instead of argwhere"""
    nodes, edges, label, array_shape, bounding_boxes = args
    
    # Get the pre-computed bounding box for this label
    slice_obj = bounding_boxes[int(label)-1]  # -1 because label numbers start at 1
    if slice_obj is None:
        return None, None, None
        
    z_vals, y_vals, x_vals = get_reslice_indices_for_id(slice_obj, array_shape)
    if z_vals is None:
        return None, None, None
        
    sub_nodes = reslice_3d_array((nodes, z_vals, y_vals, x_vals))
    sub_edges = reslice_3d_array((edges, z_vals, y_vals, x_vals))
    return label, sub_nodes, sub_edges


def create_node_dictionary_id_python(nodes, edges, num_nodes, bounding_boxes):
    """Modified to pre-compute all bounding boxes using find_objects"""
    node_dict = {}
    array_shape = nodes.shape
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Create args list with bounding_boxes included
        args_list = [(nodes, edges, i, array_shape, bounding_boxes) 
                    for i in range(1, int(num_nodes) + 1)]

        # Execute parallel tasks to process labels
        results = executor.map(process_label_id, args_list)

        # Process results in parallel
        for label, sub_nodes, sub_edges in results:
            executor.submit(create_dict_entry_id, node_dict, label, sub_nodes, sub_edges)

    return node_dict

def create_dict_entry_id(node_dict, label, sub_nodes, sub_edges):
    """Internal method used for the secondary algorithm to pass around args in parallel."""

    if label is None:
        pass
    else:
        node_dict[label] = _get_node_edge_dict_id(sub_nodes, sub_edges, label)


# For the continuous structure labeler:

def get_reslice_space(slice_obj, array_shape):
    z_slice, y_slice, x_slice = slice_obj
    
    # Extract min/max from slices
    z_min, z_max = z_slice.start, z_slice.stop - 1
    y_min, y_max = y_slice.start, y_slice.stop - 1
    x_min, x_max = x_slice.start, x_slice.stop - 1
    # Add padding
    y_max = y_max + 1
    y_min = y_min - 1
    x_max = x_max + 1
    x_min = x_min - 1
    z_max = z_max + 1
    z_min = z_min - 1
    # Boundary checks
    y_max = min(y_max, array_shape[1] - 1)
    x_max = min(x_max, array_shape[2] - 1)
    z_max = min(z_max, array_shape[0] - 1)
    y_min = max(y_min, 0)
    x_min = max(x_min, 0)
    z_min = max(z_min, 0)
    return [z_min, z_max], [y_min, y_max], [x_min, x_max]

def reslice_array(args):
    """Internal method used for the secondary algorithm to reslice subarrays."""
    input_array, z_range, y_range, x_range = args
    z_start, z_end = z_range
    z_start, z_end = int(z_start), int(z_end)
    y_start, y_end = y_range
    y_start, y_end = int(y_start), int(y_end)
    x_start, x_end = x_range
    x_start, x_end = int(x_start), int(x_end)
    
    # Reslice the array
    resliced_array = input_array[z_start:z_end + 1, y_start:y_end + 1, x_start:x_end + 1]
    
    return resliced_array

def _reassign_label_by_continuous_proximity(sub_to_assign, sub_labels, label):
    """Internal method used for the secondary algorithm to find pixel involvement of component to be labeled based on nearby labels."""
    
    # Create a boolean mask where elements with the specified label are True
    sub_to_assign = sub_to_assign == label
    sub_to_assign = nettracer.dilate_3D_old(sub_to_assign, 3, 3, 3) #Dilate the label by 1 to see where the dilated label overlaps
    sub_to_assign = sub_to_assign != 0
    sub_labels = sub_labels * sub_to_assign # Isolate only adjacent label
    sub_to_assign = sdl.smart_label_single(sub_to_assign, sub_labels) # Assign labeling schema from 'sub_to_assign' to 'sub_labels'
    return sub_to_assign

def process_and_write_voxels(args):
    """Optimized version using vectorized operations"""
    to_assign, labels, label, array_shape, bounding_boxes, result_array = args
    print(f"Processing node {label}")

    # Get the pre-computed bounding box for this label
    slice_obj = bounding_boxes[label-1]  # -1 because label numbers start at 1
    
    z_vals, y_vals, x_vals = get_reslice_space(slice_obj, array_shape)
    z_start, z_end = z_vals
    y_start, y_end = y_vals
    x_start, x_end = x_vals
    
    # Extract subarrays
    sub_to_assign = reslice_array((to_assign, z_vals, y_vals, x_vals))
    sub_labels = reslice_array((labels, z_vals, y_vals, x_vals))
    
    # Create mask for this label BEFORE relabeling (critical!)
    label_mask = (sub_to_assign == label)
    
    # Get local coordinates of voxels belonging to this label
    local_coords = np.where(label_mask)
    
    # Do the relabeling on the subarray
    # Note: relabeled may contain multiple different label values now
    relabeled = _reassign_label_by_continuous_proximity(sub_to_assign, sub_labels, label)

    # Apply offsets (vectorized)
    global_coords = (
        local_coords[0] + z_start,
        local_coords[1] + y_start,
        local_coords[2] + x_start
    )
    
    # Single vectorized write operation
    # This copies all new label values (potentially multiple different labels)
    # for voxels that originally belonged to 'label'
    result_array[global_coords] = relabeled[local_coords]

def create_label_map(to_assign, labels, num_labels, array_shape):
    """Modified to pre-compute all bounding boxes and write voxels in parallel"""
    
    # Get all bounding boxes at once
    bounding_boxes = ndimage.find_objects(to_assign)
    
    # Clone to_assign for modifications (original used for reading subarrays)
    result_array = to_assign.copy()
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Create args list with bounding_boxes and result_array included
        args_list = [(to_assign, labels, i, array_shape, bounding_boxes, result_array) 
                    for i in range(1, num_labels + 1)]
        
        # Execute parallel tasks - each writes directly to result_array
        futures = [executor.submit(process_and_write_voxels, args) for args in args_list]
        
        # Wait for all to complete
        for future in futures:
            future.result()
    
    return result_array

def label_continuous(to_assign, labels):
    array_shape = to_assign.shape
    num_labels = np.max(to_assign)
    result = create_label_map(to_assign, labels, num_labels, array_shape)
    return result


# Some methods for the nearest neighbors batch method:

import ast
import numpy as np
from scipy.spatial import KDTree

EPSILON = 1e-10


# ---------------------------------------------------------------- identities

def natural_key(value):
    """Sort key so 'id2' precedes 'id10' and mixed types don't explode."""
    s = str(value)
    out = []
    num = ''
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            if num:
                out.append((1, int(num), ''))
                num = ''
            out.append((0, 0, ch.lower()))
    if num:
        out.append((1, int(num), ''))
    return out


def parse_identity_spec(spec):
    """
    Normalize an identity selector into (key, exact).

    exact=False -> membership test: key in node_identities[node]
    exact=True  -> whole-identity test: str(list(iden)) == key

    Accepts plain strings ('Tumor'), lists/tuples (['Tumor', 'CD8']),
    and legacy stringified lists ("['Tumor']") for backwards compatibility.
    A single-element selector is treated as membership, which matches the
    historical behavior of the old "['x']" format.
    """
    if spec is None:
        return None, False

    if isinstance(spec, (list, tuple, set)):
        items = list(spec)
        if len(items) == 0:
            return None, False
        if len(items) == 1:
            return items[0], False
        return str(list(items)), True

    if isinstance(spec, str):
        stripped = spec.strip()
        if stripped[:1] in ('[', '('):
            try:
                return parse_identity_spec(ast.literal_eval(stripped))
            except Exception:
                return spec, False
        return spec, False

    return spec, False


def identity_label(spec):
    """Human-readable label for a spec (no stringified-list noise)."""
    key, exact = parse_identity_spec(spec)
    if not exact:
        return str(key)
    try:
        return " + ".join(str(m) for m in ast.literal_eval(key))
    except Exception:
        return str(key)


def enumerate_identity_specs(node_identities, include_multi=False):
    """
    Build the list of identity selectors for a batch run.

    include_multi=False -> every individual identity string (multi-identity
                           nodes contribute to each of their identities)
    include_multi=True  -> every unique identity combination, matched exactly
                           (single-element combos still behave as membership,
                           matching the old dialog's semantics)
    """
    if include_multi:
        combos = {tuple(iden) for iden in node_identities.values()}
        ordered = sorted(combos, key=lambda c: (len(c), [natural_key(x) for x in c]))
        return [list(c) for c in ordered]

    singles = set()
    for iden in node_identities.values():
        singles.update(iden)
    return sorted(singles, key=natural_key)


def resolve_identity_sets(node_identities, specs, restrict_to=None):
    """
    Map identity selectors -> node id arrays in a single pass over the nodes.

    Returns (identity_sets, labels) where identity_sets is {label: np.array(ids)}
    and labels preserves the order of `specs`.
    """
    valid = set(restrict_to) if restrict_to is not None else None

    resolved = []
    used_labels = set()
    for spec in specs:
        key, exact = parse_identity_spec(spec)
        label = identity_label(spec)
        while label in used_labels:          # defensive; labels should be unique
            label += "'"
        used_labels.add(label)
        resolved.append({'label': label, 'key': key, 'exact': exact, 'ids': []})

    for node, iden in node_identities.items():
        if valid is not None and node not in valid:
            continue
        iden_str = None
        for entry in resolved:
            if entry['exact']:
                if iden_str is None:
                    iden_str = str(list(iden))
                if iden_str == entry['key']:
                    entry['ids'].append(node)
            elif entry['key'] in iden:
                entry['ids'].append(node)

    identity_sets = {e['label']: np.asarray(e['ids']) for e in resolved}
    labels = [e['label'] for e in resolved]
    return identity_sets, labels


# ---------------------------------------------------------------- geometry

def scale_coords(coords, xy_scale=1.0, z_scale=1.0):
    """(N, 3) -> z on axis 0; (N, 2) -> xy on both axes."""
    out = np.asarray(coords, dtype=float).copy()
    if out.shape[1] >= 3:
        out[:, 0] *= z_scale
        out[:, 1:] *= xy_scale
    else:
        out *= xy_scale
    return out


def build_coordinate_index(point_centroids, identity_sets, labels=None):
    """
    Pack every node referenced by identity_sets into one coordinate array.

    Returns (all_ids, coords, rows_by_label) where rows_by_label holds integer
    row indices into `coords`. A node belonging to several identities appears
    once in `coords` and in several row arrays.
    """
    labels = list(identity_sets) if labels is None else labels

    all_ids, row_of = [], {}
    for label in labels:
        for nid in identity_sets[label]:
            if nid in row_of or nid not in point_centroids:
                continue
            row_of[nid] = len(all_ids)
            all_ids.append(nid)

    if not all_ids:
        return [], np.empty((0, 3)), {label: np.empty(0, dtype=int) for label in labels}

    dim = len(point_centroids[all_ids[0]])
    coords = np.empty((len(all_ids), dim), dtype=float)
    for i, nid in enumerate(all_ids):
        coords[i] = point_centroids[nid]

    rows_by_label = {
        label: np.array([row_of[n] for n in identity_sets[label] if n in row_of], dtype=int)
        for label in labels
    }
    return all_ids, coords, rows_by_label


# ---------------------------------------------------------------- helpers
 
def _mean_of_nearest(dists_sorted, k_eff):
    """
    Mean of the k_eff smallest distances per row, ignoring +inf entries
    (self-matches / coincident points). Rows with no finite entry -> NaN.
    """
    sel = dists_sorted[:, :k_eff]
    finite = np.isfinite(sel)
    counts = finite.sum(axis=1)
    sums = np.where(finite, sel, 0.0).sum(axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
 
 
# ---------------------------------------------------------------- batch core
 
def _matrix_from_coords(coords_scaled, rows_by_label, labels, num,
                        query_rows_by_label=None, return_node_table=False):
    """
    One tree per target identity, one vectorized query per tree, sliced per root.
 
    Replicates average_nearest_neighbor_distances() exactly for every pair:
    self-matches (and any coincident point) are pushed to +inf, the remaining
    `num_eff` nearest are averaged per root, and roots with no valid neighbor
    drop out as NaN.
 
    If return_node_table is True, also returns (q_rows, node_table) where
    node_table has shape (len(q_rows), len(labels)): entry [i, j] is the mean
    nearest-neighbor distance from node q_rows[i] to identity labels[j], or NaN
    when that node has no valid neighbor in that identity.
 
    Self-exclusion in the node table is decided by *node membership*: if the
    node itself belongs to labels[j], one neighbor slot is dropped
    (num_eff = min(num, n_t - 1)), matching the diagonal convention.
    """
    matrix = {r: {t: None for t in labels} for r in labels}
    if query_rows_by_label is None:
        query_rows_by_label = rows_by_label
 
    stack = [query_rows_by_label[l] for l in labels if len(query_rows_by_label[l])]
    if not stack:
        empty = (np.empty(0, dtype=int), np.empty((0, len(labels))))
        return (matrix, *empty) if return_node_table else matrix
 
    q_rows = np.unique(np.concatenate(stack))
    q_coords = coords_scaled[q_rows]
    q_index = {l: np.searchsorted(q_rows, query_rows_by_label[l]) for l in labels}
 
    node_table = np.full((len(q_rows), len(labels)), np.nan) if return_node_table else None
 
    for col, targ in enumerate(labels):
        t_rows = rows_by_label[targ]
        n_t = len(t_rows)
        if n_t == 0:
            continue
 
        tree = KDTree(coords_scaled[t_rows])
        # Cap k at the tree size so scipy never pads with inf.
        k = int(min(num + 1, n_t))
        dists, _ = tree.query(q_coords, k=k)
        dists = np.asarray(dists, dtype=float)
        if dists.ndim == 1:                       # scipy squeezes when k == 1
            dists = dists[:, np.newaxis]
 
        dists = np.where(dists < EPSILON, np.inf, dists)
        dists = np.sort(dists, axis=1)
 
        # ---- per-node column (no extra tree work, just two more slices)
        if return_node_table:
            member = np.isin(q_rows, t_rows)
            for mask, num_eff in ((member, min(num, n_t - 1)),
                                  (~member, min(num, n_t))):
                if num_eff <= 0 or not mask.any():
                    continue
                node_table[mask, col] = _mean_of_nearest(dists[mask], num_eff)
 
        # ---- unchanged aggregate matrix
        for root in labels:
            idx = q_index[root]
            if len(idx) == 0:
                continue
            num_eff = min(num, n_t - 1) if root == targ else min(num, n_t)
            if num_eff <= 0:
                continue
 
            per_root = _mean_of_nearest(dists[idx], num_eff)
            valid = per_root[~np.isnan(per_root)]
            if valid.size:
                matrix[root][targ] = float(np.mean(valid))
 
    return (matrix, q_rows, node_table) if return_node_table else matrix
 
 
def batch_average_nearest_neighbor_distances(point_centroids, identity_sets, labels=None,
                                             xy_scale=1.0, z_scale=1.0, num=1,
                                             return_node_table=True):
    """
    Observed nearest-neighbor matrix for every (root, target) identity pair.
 
    Returns (matrix, (all_ids, rows_by_label)) as before. With
    return_node_table=True, returns
    (matrix, (all_ids, rows_by_label), (node_ids, node_table)) where node_ids
    is a list of node ids (rows) and node_table is a float array of shape
    (len(node_ids), len(labels)) whose columns follow `labels`.
    """
    labels = list(identity_sets) if labels is None else labels
    all_ids, coords, rows_by_label = build_coordinate_index(point_centroids, identity_sets, labels)
 
    if not all_ids:
        matrix = {r: {t: None for t in labels} for r in labels}
        if return_node_table:
            return matrix, (all_ids, rows_by_label), ([], np.empty((0, len(labels))))
        return matrix, (all_ids, rows_by_label)
 
    coords_scaled = scale_coords(coords, xy_scale, z_scale)
 
    if return_node_table:
        matrix, q_rows, node_table = _matrix_from_coords(
            coords_scaled, rows_by_label, labels, num, return_node_table=True)
        node_ids = [all_ids[r] for r in q_rows]
        return matrix, (all_ids, rows_by_label), (node_ids, node_table)
 
    matrix = _matrix_from_coords(coords_scaled, rows_by_label, labels, num)
    return matrix, (all_ids, rows_by_label)
 
 
# ---------------------------------------------------------------- convenience
 
def node_table_to_dataframe(node_ids, node_table, labels, identity_sets=None):
    """
    Optional pandas view. If identity_sets is given, adds one boolean column per
    identity recording membership, so you can group rows by root identity.
    """
    import pandas as pd
 
    df = pd.DataFrame(node_table, index=pd.Index(node_ids, name="node_id"),
                      columns=list(labels))
    if identity_sets is not None:
        for label in labels:
            members = set(identity_sets[label])
            df[f"in_{label}"] = [n in members for n in node_ids]
    return df

# ---------------------------------------------------------------- simulation

def _positions_to_coords(flat_idx, shape, valid_positions):
    if valid_positions is not None:
        return np.stack([axis[flat_idx] for axis in valid_positions], axis=1).astype(float)
    return np.stack(np.unravel_index(flat_idx, shape), axis=1).astype(float)


def _subsample_rows(rows_by_label, labels, num_query, rng):
    if not num_query:
        return rows_by_label
    out = {}
    for label in labels:
        rows = rows_by_label[label]
        if len(rows) > num_query:
            out[label] = np.sort(rng.choice(rows, size=num_query, replace=False))
        else:
            out[label] = rows
    return out


def simulate_batch_nearest_neighbor_distances(rows_by_label, labels, n_points, shape,
                                              xy_scale=1.0, z_scale=1.0, num=1,
                                              theoretical='Random', mask=None,
                                              replicates=1, num_query=None, seed=None):
    """
    Null matrix from ONE point placement per replicate.

    All n_points objects are placed once (randomly, or on a uniform lattice),
    identity membership is permuted across those positions, and the full
    combination matrix is read off that single arrangement. `replicates`
    therefore costs `replicates` placements total, not one per pair.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)

    if mask is not None:
        valid_positions = np.where(mask)
        total = len(valid_positions[0])
    else:
        valid_positions = None
        total = int(np.prod(shape))

    if total == 0:
        raise ValueError("No valid positions available for the simulated distribution")

    replace = n_points > total
    if replace:
        print(f"Warning: {n_points} objects but only {total} available positions; "
              f"sampling simulated positions with replacement.")

    accum = {r: {t: [] for t in labels} for r in labels}

    for _ in range(max(1, int(replicates))):
        if theoretical == 'Uniform':
            flat = np.linspace(0, total - 1, n_points).astype(int)
        else:
            flat = rng.choice(total, size=n_points, replace=replace)

        coords = _positions_to_coords(flat, shape, valid_positions)
        # Scramble which object lands on which position so identity groups are
        # spatially interleaved (matters for the uniform lattice).
        coords = coords[rng.permutation(len(coords))]
        coords_scaled = scale_coords(coords, xy_scale, z_scale)

        query_rows = _subsample_rows(rows_by_label, labels, num_query, rng)
        rep = _matrix_from_coords(coords_scaled, rows_by_label, labels, num,
                                  query_rows_by_label=query_rows)

        for r in labels:
            for t in labels:
                if rep[r][t] is not None:
                    accum[r][t].append(rep[r][t])

        if theoretical == 'Uniform' and replicates > 1 and len(labels) == 1:
            break  # nothing left to vary

    matrix = {}
    for r in labels:
        matrix[r] = {t: (float(np.mean(accum[r][t])) if accum[r][t] else None) for t in labels}
    return matrix

def simulate_batch_label_scramble(coords_scaled, rows_by_label, labels, num=1,
                                  replicates=1, num_query=None, seed=None):
    """
    Null matrix from permuting identity labels across FIXED observed positions.

    Positions never move — only which position carries which identity changes.
    Spatial architecture (clustering, density gradients, tissue boundary) is
    preserved exactly; the null asks whether identities are exchangeable across
    the observed arrangement.

    One permutation per replicate feeds the entire combination matrix, matching
    the cost structure of simulate_batch_nearest_neighbor_distances().

    Multi-identity nodes move as a unit: a single index permutation is applied
    to every label's row list, so an index appearing under several labels lands
    at the same new position in all of them.

    Parameters
    ----------
    coords_scaled : ndarray, shape (n_points, ndim)
        Observed centroids, already scaled. Row order must match the indices
        used in rows_by_label.
    rows_by_label : dict
        {label: array of row indices into coords_scaled}
    """
    rng = np.random.default_rng(seed)
    n_points = len(coords_scaled)

    if n_points == 0:
        raise ValueError("No positions available for the label scramble")

    rows_by_label = {l: np.asarray(rows_by_label.get(l, []), dtype=int) for l in labels}

    accum = {r: {t: [] for t in labels} for r in labels}

    for _ in range(max(1, int(replicates))):
        perm = rng.permutation(n_points)
        scrambled = {l: perm[rows] for l, rows in rows_by_label.items()}

        query_rows = _subsample_rows(scrambled, labels, num_query, rng)
        rep = _matrix_from_coords(coords_scaled, scrambled, labels, num,
                                  query_rows_by_label=query_rows)

        for r in labels:
            for t in labels:
                if rep[r][t] is not None:
                    accum[r][t].append(rep[r][t])

    matrix = {}
    for r in labels:
        matrix[r] = {t: (float(np.mean(accum[r][t])) if accum[r][t] else None) for t in labels}
    return matrix
    
# ---------------------------------------------------------------- formatting

def matrix_to_rows(matrix, labels):
    """{root: {targ: v}} -> {root: [v aligned to labels]} for the table/heatmap calls."""
    return {r: [matrix[r][t] for t in labels] for r in labels}


def ratio_matrix(numer, denom, labels):
    """Simulated / observed, cellwise, tolerant of Nones and zeros."""
    out = {}
    for r in labels:
        row = {}
        for t in labels:
            a, b = numer[r][t], denom[r][t]
            ok = (a is not None and b is not None and np.isfinite(a)
                  and np.isfinite(b) and b != 0)
            row[t] = float(a / b) if ok else None
        out[r] = row
    return out