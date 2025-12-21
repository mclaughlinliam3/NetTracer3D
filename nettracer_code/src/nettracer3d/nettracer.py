import pandas as pd
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
import cv2
import ast
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import zoom
import multiprocessing as mp
import os
import copy
import statistics as stats
try:
    import napari
except:
    pass
import networkx as nx
from scipy.signal import find_peaks
try:
    import cupy as cp
except:
    pass
from . import node_draw
from . import network_draw
from skimage import morphology as mpg
from . import smart_dilate
from . import modularity
from . import simple_network
from . import community_extractor
from . import network_analysis
from . import morphology
from . import proximity
from skimage.segmentation import watershed as water



#These next several methods relate to searching with 3D objects by dilating each one in a subarray around their neighborhood although I don't explicitly use this anywhere... can call them deprecated although I may want to use them later again so I have them still written out here.


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

    return [z_min, z_max], [y_min, y_max], [x_min, x_max]

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



def _get_node_edge_dict(label_array, edge_array, label, dilate_xy, dilate_z):
    """Internal method used for the secondary algorithm to find which nodes interact with which edges."""
    
    import tifffile
    # Create a boolean mask where elements with the specified label are True
    label_array = label_array == label
    label_array = dilate_3D(label_array, dilate_xy, dilate_xy, dilate_z) #Dilate the label to see where the dilated label overlaps
    edge_array = edge_array * label_array  # Filter the edges by the label in question
    edge_array = edge_array.flatten()  # Convert 3d array to 1d array
    edge_array = remove_zeros(edge_array)  # Remove zeros
    edge_array = set(edge_array)  # Remove duplicates
    edge_array = list(edge_array)  # Back to list

    return edge_array

def process_label(args):
    """Modified to use pre-computed bounding boxes instead of argwhere"""
    nodes, edges, label, dilate_xy, dilate_z, array_shape, bounding_boxes = args
    #print(f"Processing node {label}")
    
    # Get the pre-computed bounding box for this label
    slice_obj = bounding_boxes[int(label)-1]  # -1 because label numbers start at 1
    if slice_obj is None:
        return None, None, None
        
    z_vals, y_vals, x_vals = get_reslice_indices(slice_obj, dilate_xy, dilate_z, array_shape)
    if z_vals is None:
        return None, None, None
        
    sub_nodes = reslice_3d_array((nodes, z_vals, y_vals, x_vals))
    sub_edges = reslice_3d_array((edges, z_vals, y_vals, x_vals))
    return label, sub_nodes, sub_edges


def create_node_dictionary(nodes, edges, num_nodes, dilate_xy, dilate_z):
    """Modified to pre-compute all bounding boxes using find_objects"""
    print("Calculating network...")
    node_dict = {}
    array_shape = nodes.shape
    
    # Get all bounding boxes at once
    bounding_boxes = ndimage.find_objects(nodes)
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Create args list with bounding_boxes included
        args_list = [(nodes, edges, i, dilate_xy, dilate_z, array_shape, bounding_boxes) 
                    for i in range(1, int(num_nodes) + 1)]

        # Execute parallel tasks to process labels
        results = executor.map(process_label, args_list)

        # Process results in parallel
        for label, sub_nodes, sub_edges in results:
            executor.submit(create_dict_entry, node_dict, label, sub_nodes, sub_edges, 
                          dilate_xy, dilate_z)

    return node_dict

def create_dict_entry(node_dict, label, sub_nodes, sub_edges, dilate_xy, dilate_z):
    """Internal method used for the secondary algorithm to pass around args in parallel."""

    if label is None:
        pass
    else:
        node_dict[label] = _get_node_edge_dict(sub_nodes, sub_edges, label, dilate_xy, dilate_z)

def find_shared_value_pairs(input_dict):
    """Internal method used for the secondary algorithm to look through discrete node-node connections in the various node dictionaries"""

    master_list = []
    compare_dict = input_dict.copy()

    # Iterate through each key in the dictionary
    for key1, values1 in input_dict.items():
        # Iterate through each other key in the dictionary
        for key2, values2 in compare_dict.items():
            # Avoid comparing the same key to itself
            if key1 != key2:
                # Find the intersection of values between the two keys
                shared_values = set(values1) & set(values2)
                # If there are shared values, create pairs and add to master list
                if shared_values:
                    for value in shared_values:
                        master_list.append([key1, key2, value])
        del compare_dict[key1]

    return master_list



#Below are helper methods that are used for the main algorithm (calculate_all)

def array_trim(edge_array, node_array):
    """Internal method used by the primary algorithm to efficiently and massively reduce extraneous search regions for edge-node intersections"""
    edge_list = edge_array.flatten() #Turn arrays into lists
    node_list = node_array.flatten()

    edge_bools = edge_list != 0 #establish where edges/nodes exist by converting to a boolean list
    node_bools = node_list != 0

    overlaps = edge_bools * node_bools #Establish boolean list where edges and nodes intersect.

    edge_overlaps = overlaps * edge_list #Set all vals in the edges/nodes to 0 where intersections are not occurring
    node_overlaps = overlaps * node_list

    edge_overlaps = remove_zeros(edge_overlaps) #Remove all values where intersections are not present, so we don't have to iterate through them later
    node_overlaps = remove_zeros(node_overlaps)

    return edge_overlaps, node_overlaps

def establish_connections_parallel(edge_labels, num_edge, node_labels):
    """Internal method used by the primary algorithm to look at dilated edges array and nodes array. Iterates through edges. 
    Each edge will see what nodes it overlaps. It will put these in a list."""
    print("Processing edge connections...")
    
    all_connections = []

    def process_edge(label):

        if label not in edge_labels:
            return None

        edge_connections = []

        # Get the indices corresponding to the current edge label
        indices = np.argwhere(edge_labels == label).flatten()

        for index in indices:

            edge_connections.append(node_labels[index])

        my_connections = list(set(edge_connections))


        edge_connections = [my_connections, label]


        #Edges only interacting with one node are not used:
        if len(my_connections) > 1:

            return edge_connections
        else:
            return None

    #These lines makes CPU run for loop iterations simultaneously, speeding up the program:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_edge, range(1, num_edge + 1)))

    all_connections = [result for result in results if result is not None]

    return all_connections


def extract_pairwise_connections(connections):
    """Parallelized method to break lists of edge interactions into trios."""
    def chunk_data_pairs(data, num_chunks):
        """Helper function to divide data into roughly equal chunks."""
        chunk_size = len(data) // num_chunks
        remainder = len(data) % num_chunks
        chunks = []
        start = 0
        for i in range(num_chunks):
            extra = 1 if i < remainder else 0  # Distribute remainder across the first few chunks
            end = start + chunk_size + extra
            chunks.append(data[start:end])
            start = end
        return chunks

    def process_sublist_pairs(connections):
        """Helper function to process each sublist and generate unique pairs."""
        pairwise_connections = []
        for connection in connections:
            nodes = connection[0]  # Get the list of nodes
            edge_ID = connection[1]  # Get the edge ID
            pairs_within_sublist = [(nodes[i], nodes[j], edge_ID) for i in range(len(nodes))
                                  for j in range(i + 1, len(nodes))]
            pairwise_connections.extend(set(map(tuple, pairs_within_sublist)))
        pairwise_connections = [list(pair) for pair in pairwise_connections]
        return pairwise_connections

    pairwise_connections = []
    num_cpus = mp.cpu_count()  # Get the number of CPUs available
    
    # Chunk the data
    connection_chunks = chunk_data_pairs(connections, num_cpus)
    
    # Use ThreadPoolExecutor to parallelize the processing of the chunks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        # Submit the chunks for processing in parallel
        futures = [executor.submit(process_sublist_pairs, chunk) for chunk in connection_chunks]
        # Retrieve the results as they are completed
        for future in concurrent.futures.as_completed(futures):
            pairwise_connections.extend(future.result())
    
    return pairwise_connections


#Saving outputs
def create_and_save_dataframe(pairwise_connections, excel_filename=None):
    """Internal method used to convert lists of discrete connections into an excel output"""
    
    # Create DataFrame directly from the connections with 3 columns
    df = pd.DataFrame(pairwise_connections, columns=['Node A', 'Node B', 'Edge C'])
    
    if excel_filename is not None:
        # Remove file extension if present to use as base path
        base_path = excel_filename.rsplit('.', 1)[0]
        
        # First try to save as CSV
        try:
            csv_path = f"{base_path}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Network file saved to {csv_path}")
            return
        except Exception as e:
            print(f"Could not save as CSV: {str(e)}")
            
            # If CSV fails, try to save as Excel
            try:
                xlsx_path = f"{base_path}.xlsx"
                df.to_excel(xlsx_path, index=False)
                print(f"Network file saved to {xlsx_path}")
            except Exception as e:
                print(f"Unable to write network file to disk... please make sure that {base_path}.xlsx is being saved to a valid directory and try again")
    else:
        return df

def create_and_save_dataframe_old(pairwise_connections, excel_filename = None):
    """Internal method used to convert lists of discrete connections into an excel output"""
    # Determine the length of the input list
    length = len(pairwise_connections)
    
    # Initialize counters for column assignment
    col_start = 0
    
    # Initialize master list to store sublists
    master_list = []
    
    # Split the input list into sublists of maximum length 1 million
    while col_start < length:
        # Determine the end index for the current sublist
        col_end = min(col_start + 1000000, length)
        
        # Append the current sublist to the master list
        master_list.append(pairwise_connections[col_start:col_end])
        
        # Update column indices for the next sublist
        col_start = col_end
    
    # Create an empty DataFrame
    df = pd.DataFrame()
    
    # Assign trios to columns in the DataFrame
    for i, sublist in enumerate(master_list):
        # Determine column names for the current sublist
        column_names = ['Node {}A'.format(i+1), 'Node {}B'.format(i+1), 'Edge {}C'.format(i+1)]
        
        # Create a DataFrame from the current sublist
        temp_df = pd.DataFrame(sublist, columns=column_names)
        
        # Concatenate the DataFrame with the master DataFrame
        df = pd.concat([df, temp_df], axis=1)

    if excel_filename is not None:
        # Remove file extension if present to use as base path
        base_path = excel_filename.rsplit('.', 1)[0]
        
        # First try to save as CSV
        try:
            csv_path = f"{base_path}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Network file saved to {csv_path}")
            return
        except Exception as e:
            print(f"Could not save as CSV: {str(e)}")
            
            # If CSV fails, try to save as Excel
            try:
                xlsx_path = f"{base_path}.xlsx"
                df.to_excel(xlsx_path, index=False)
                print(f"Network file saved to {xlsx_path}")
            except Exception as e:
                print(f"Unable to write network file to disk... please make sure that {base_path}.xlsx is being saved to a valid directory and try again")

    else:
        return df




#General supporting methods below:

def invert_dict(d):
    inverted = {}
    for key, value in d.items():
        inverted.setdefault(value, []).append(key)
    return inverted

def revert_dict(d):
    inverted = {}
    for key, value_list in d.items():
        for value in value_list:
            inverted[value] = key
    return inverted

def invert_dict_special(d):

    d = invert_dict(d)

    new_dict = copy.deepcopy(d)

    for key, vals in d.items():

        try:
            idens = ast.literal_eval(key)
            for iden in idens:
                try:
                    new_dict[iden].extend(vals)
                except:
                    new_dict[iden] = vals
            del new_dict[key]
        except:
            pass
    return new_dict


def invert_array(array):
    """Internal method used to flip node array indices. 0 becomes 255 and vice versa."""
    inverted_array = np.where(array == 0, 255, 0).astype(np.uint8)
    return inverted_array

def invert_boolean(array):
    """Internal method to flip a boolean array"""
    inverted_array = np.where(array == False, True, False).astype(np.uint8)
    return inverted_array

def establish_edges(nodes, edge):
    """Internal  method used to black out where edges interact with nodes"""
    invert_nodes = invert_array(nodes)
    edges = edge * invert_nodes
    return edges

def establish_inner_edges(nodes, edge):
    """Internal method to find inner edges that may exist betwixt dilated nodes."""
    inner_edges = edge * nodes
    return inner_edges


def upsample_with_padding(data, factor=None, original_shape=None):
    """
    Upsample a 3D or 4D array with optional different scaling factors per dimension.
    
    Parameters:
    -----------
    data : ndarray
        Input 3D array or 4D array (where 4th dimension is RGB) to be upsampled
    factor : float or tuple, optional
        Upsampling factor. If float, same factor is applied to all dimensions.
        If tuple, should contain three values for z, y, x dimensions respectively.
        If None, factor is calculated from original_shape.
    original_shape : tuple, optional
        Target shape for the output array. Used to calculate factors if factor is None.
        
    Returns:
    --------
    ndarray
        Upsampled and padded array matching the original shape
    """
    if original_shape is None:
        raise ValueError("original_shape must be provided")
        
    # Handle 4D color arrays
    is_color = len(data.shape) == 4 and (data.shape[-1] == 3 or data.shape[-1] == 4)
    if is_color:
        # Split into separate color channels
        channels = [data[..., i] for i in range(3)]
        upsampled_channels = []
        
        for channel in channels:
            # Upsample each channel separately
            upsampled_channel = _upsample_3d_array(channel, factor, original_shape)
            upsampled_channels.append(upsampled_channel)
            
        # Stack the channels back together
        return np.stack(upsampled_channels, axis=-1)
    else:
        # Handle regular 3D array
        return _upsample_3d_array(data, factor, original_shape)

def _upsample_3d_array(data, factor, original_shape):
    """Helper function to handle the upsampling of a single 3D array"""
    original_shape = np.array(original_shape)
    current_shape = np.array(data.shape)
    
    # Calculate factors if not provided
    if factor is None:
        # Compute the ratio between original and current shape for each dimension
        factors = [os / cs for os, cs in zip(original_shape, current_shape)]
        # If all factors are the same, use a single number for efficiency
        if len(set(factors)) == 1:
            factor = factors[0]
        else:
            factor = tuple(factors)
    elif isinstance(factor, (int, float)):
        factor = factor  # Keep it as a single number
        
    # Upsample the input array
    binary_array = zoom(data, factor, order=0)
    upsampled_shape = np.array(binary_array.shape)
    
    # Calculate the positive differences in dimensions
    difference_dims = original_shape - upsampled_shape
    
    # Calculate the padding amounts for each dimension
    padding_dims = np.maximum(difference_dims, 0)
    padding_before = padding_dims // 2
    padding_after = padding_dims - padding_before
    
    # Pad the binary array along each dimension
    padded_array = np.pad(binary_array, 
                         [(padding_before[0], padding_after[0]),
                          (padding_before[1], padding_after[1]),
                          (padding_before[2], padding_after[2])],
                         mode='constant',
                         constant_values=0)
    
    # Calculate the subtraction amounts for each dimension
    sub_dims = np.maximum(-difference_dims, 0)
    sub_before = sub_dims // 2
    sub_after = sub_dims - sub_before
    
    # Remove excess dimensions sequentially
    # Remove planes from the beginning and end
    if sub_dims[0] == 0:
        trimmed_planes = padded_array
    else:
        trimmed_planes = padded_array[sub_before[0]:-sub_after[0], :, :]
    
    # Remove rows from the beginning and end
    if sub_dims[1] == 0:
        trimmed_rows = trimmed_planes
    else:
        trimmed_rows = trimmed_planes[:, sub_before[1]:-sub_after[1], :]
    
    # Remove    columns from the beginning and end
    if sub_dims[2] == 0:
        trimmed_array = trimmed_rows
    else:
        trimmed_array = trimmed_rows[:, :, sub_before[2]:-sub_after[2]]
    
    return trimmed_array


def remove_branches_new(skeleton, length):
    """Used to compensate for overly-branched skeletons resulting from the scipy 3d skeletonization algorithm"""
    def find_coordinate_difference(arr):
        try:
            arr[1,1,1] = 0
            # Find the indices of non-zero elements
            indices = np.array(np.nonzero(arr)).T
            
            # Calculate the difference
            diff = np.array([1,1,1]) - indices[0]
            
            return diff
        except:
            return None
    
    skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0) #Add black planes over the 3d space to avoid index errors
    image_copy = np.copy(skeleton)
    
    # Find all endpoints ONCE at the beginning
    nonzero_coords = np.transpose(np.nonzero(image_copy))
    endpoints = []
    nubs = []
    
    for x, y, z in nonzero_coords:
        mini = image_copy[x-1:x+2, y-1:y+2, z-1:z+2]
        nearby_sum = np.sum(mini)
        threshold = 2 * image_copy[x, y, z]
        
        if nearby_sum <= threshold:
            endpoints.append((x, y, z))
    
    x, y, z = endpoints[0]
    original_val = image_copy[x, y, z]

    # Process each endpoint individually for nub assessment
    for start_x, start_y, start_z in endpoints:
            
        # Trace the branch from this endpoint, removing points as we go
        branch_coords = []
        current_coord = (start_x, start_y, start_z)
        nub_reached = False
        
        for step in range(length):
            x, y, z = current_coord
            
            # Store original value and coordinates
            branch_coords.append((x, y, z))
            
            # Remove this point temporarily
            image_copy[x, y, z] = 0
            
            # If we've reached the maximum length without hitting a nub, break
            if step == length - 1:
                break
            
            # Find next coordinate in the branch
            mini = image_copy[x-1:x+2, y-1:y+2, z-1:z+2]
            dif = find_coordinate_difference(mini.copy())
            if dif is None:
                break
                
            next_coord = (x - dif[0], y - dif[1], z - dif[2])
            
            # Check if next coordinate is valid and exists
            nx, ny, nz = next_coord
            
            # Check if next point is a nub (has more neighbors than expected)
            next_mini = image_copy[nx-1:nx+2, ny-1:ny+2, nz-1:nz+2]
            next_nearby_sum = np.sum(next_mini)
            next_threshold = 2 * image_copy[nx, ny, nz]
            
            if next_nearby_sum > next_threshold:
                nub_reached = True
                nubs.append(next_coord)
                nubs.append(current_coord) # Note, if we don't add the current coord here (and restore it below), the behavior of this method can be changed to trim branches beneath previous branches, which could be neat but its somewhat unpredictable so I opted out of it.
                image_copy[x, y, z] = original_val
                #image_copy[nx, ny, nz] = 0
                break
                
            current_coord = next_coord
        
        # If no nub was reached, restore all the points we removed
        if not nub_reached:
            for i, (bx, by, bz) in enumerate(branch_coords):
                image_copy[bx, by, bz] = original_val
        # If nub was reached, points stay removed (branch is eliminated)

    for item in nubs: #The nubs are endpoints of length = 1. They appear a bit different in the array so we just note when one is created and remove them all at the end in a batch.
        image_copy[item[0], item[1], item[2]] = 0 # Removing the nub itself leaves a hole in the skeleton but for branchpoint detection that doesn't matter, which is why it behaves this way. To fill the hole, one option is to dilate once then erode/skeletonize again, but we want to avoid making anything that looks like local branching so I didn't bother.

    # Remove padding and return
    image_copy = (image_copy[1:-1, 1:-1, 1:-1]).astype(np.uint8)
    return image_copy

import numpy as np
from collections import deque, defaultdict

    
def remove_branches(skeleton, length):
    """Used to compensate for overly-branched skeletons resulting from the scipy 3d skeletonization algorithm"""

    def find_coordinate_difference(arr):
        try:
            arr[1,1,1] = 0
            # Find the indices of non-zero elements
            indices = np.array(np.nonzero(arr)).T
            
            # Calculate the difference
            diff = np.array([1,1,1]) - indices[0]
            
            return diff
        except:
            return


    skeleton = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0) #Add black planes over the 3d space to avoid index errors

    # Find all nonzero voxel coordinates
    nonzero_coords = np.transpose(np.nonzero(skeleton))
    x, y, z = nonzero_coords[0]
    threshold = 2 * skeleton[x, y, z]
    nubs = []
    

    for b in range(length):

        new_coords = []

        # Create a copy of the image to modify
        image_copy = np.copy(skeleton)


        # Iterate through each nonzero voxel
        for x, y, z in nonzero_coords: #We are looking for endpoints, which designate a branch terminus, that will be removed and move onto the next endpoint equal for iterations equal to user length param

            # Count nearby pixels including diagonals
            mini = skeleton[x-1:x+2, y-1:y+2, z-1:z+2]
            nearby_sum = np.sum(mini)
            
            # If sum is one, remove this endpoint
            if nearby_sum <= threshold:

                try:

                    dif = find_coordinate_difference(mini)
                    new_coord = [x - dif[0], y - dif[1], z - dif[2]]
                    new_coords.append(new_coord)
                except:
                    pass
                    
                nonzero_coords = new_coords

                image_copy[x, y, z] = 0
            elif b > 0:
                nub = [x, y, z]
                nubs.append(nub)

        if b == length - 1:
            for item in nubs: #The nubs are endpoints of length = 1. They appear a bit different in the array so we just note when one is created and remove them all at the end in a batch.
                #x, y, z = item[0], item[1], item[2]
                image_copy[item[0], item[1], item[2]] = 0
                #image_copy[x-1:x+2, y-1:y+2, z-1:z+2] = 0



        skeleton = image_copy

    image_copy = (image_copy[1:-1, 1:-1, 1:-1]).astype(np.uint8)

    return image_copy



def estimate_object_radii(labeled_array, gpu=False, n_jobs=None, xy_scale = 1, z_scale = 1):
    """
    Estimate the radii of labeled objects in a 3D numpy array.
    Dispatches to appropriate implementation based on parameters.
    
    Parameters:
    -----------
    labeled_array : numpy.ndarray
        3D array where each object has a unique integer label (0 is background)
    gpu : bool
        Whether to use GPU acceleration via CuPy (if available)
    n_jobs : int or None
        Number of parallel jobs for CPU version. If None, uses all available cores.
    
    Returns:
    --------
    dict: Dictionary mapping object labels to estimated radii
    dict: (optional) Dictionary of shape statistics for each label
    """
    # Check if GPU is requested but not available
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpx
        HAS_CUPY = True
    except ImportError:
        HAS_CUPY = False

    if gpu and not HAS_CUPY:
        print("Warning: GPU acceleration requested but CuPy not available. Falling back to CPU.")
        gpu = False
    
    if gpu:
        return morphology.estimate_object_radii_gpu(labeled_array, xy_scale = xy_scale, z_scale = z_scale)
    else:
        return morphology.estimate_object_radii_cpu(labeled_array, n_jobs, xy_scale = xy_scale, z_scale = z_scale)

def get_surface_areas(labeled, xy_scale=1, z_scale=1):
    labels = np.unique(labeled)
    labels = labels[labels > 0]
    max_label = int(np.max(labeled))
    
    surface_areas = np.zeros(max_label + 1, dtype=np.float64)
    
    for axis in range(3):
        if axis == 2:
            face_area = xy_scale * xy_scale
        else:
            face_area = xy_scale * z_scale
        
        for direction in [-1, 1]:
            # Pad with zeros only on the axis we're checking
            pad_width = [(1, 1) if i == axis else (0, 0) for i in range(3)]
            padded = np.pad(labeled, pad_width, mode='constant', constant_values=0)
            
            # Roll the padded array
            shifted = np.roll(padded, direction, axis=axis)
            
            # Extract the center region (original size) from shifted
            slices = [slice(1, -1) if i == axis else slice(None) for i in range(3)]
            shifted_cropped = shifted[tuple(slices)]
            
            # Find exposed faces
            exposed_faces = (labeled != shifted_cropped) & (labeled > 0)
            
            face_counts = np.bincount(labeled[exposed_faces], 
                                     minlength=max_label + 1)
            surface_areas += face_counts * face_area
    
    result = {int(label): float(surface_areas[label]) for label in labels}
    return result

def break_and_label_skeleton(skeleton, peaks = 1, branch_removal = 0, comp_dil = 0, max_vol = 0, directory = None, return_skele = False, nodes = None, compute = True, unify = False, xy_scale = 1, z_scale = 1):
    """Internal method to break open a skeleton at its branchpoints and label the remaining components, for an 8bit binary array"""

    if type(skeleton) == str:
        broken_skele = skeleton
        skeleton = tifffile.imread(skeleton)
    else:
        broken_skele = None

    if nodes is None:

        verts = label_vertices(skeleton, peaks = peaks, branch_removal = branch_removal, comp_dil = comp_dil, max_vol = max_vol, return_skele = return_skele, compute = compute)

    else:
        verts = nodes

    verts = invert_array(verts)

    """
    if compute: # We are interested in the endpoints if we are doing the optional computation later
        endpoints = []
        image_copy = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0)
        nonzero_coords = np.transpose(np.nonzero(image_copy))
        for x, y, z in nonzero_coords:
            mini = image_copy[x-1:x+2, y-1:y+2, z-1:z+2]
            nearby_sum = np.sum(mini)
            threshold = 2 * image_copy[x, y, z]
            
            if nearby_sum <= threshold:
                endpoints.append((x, y, z))
    """

    image_copy = skeleton * verts

 
    # Label the modified image to assign new labels for each branch
    #labeled_image, num_labels = measure.label(image_copy, connectivity=2, return_num=True)
    labeled_image, num_labels = label_objects(image_copy)

    if type(broken_skele) == str:
        if directory is None:
            filename = f'broken_skeleton_with_labels.tif'
        else:
            filename = f'{directory}/broken_skeleton_with_labels.tif'

        tifffile.imwrite(filename, labeled_image, photometric='minisblack')
        print(f"Broken skeleton saved to {filename}")

    if not unify:
        verts = None
    else:
        verts = invert_array(verts)

    if compute:

        return labeled_image, verts, skeleton, None

    return labeled_image, verts, None, None

def compute_optional_branchstats(verts, labeled_array, endpoints, xy_scale = 1, z_scale = 1):

    #Lengths:
    # Get all non-background coordinates and their labels in one pass
    z, y, x = np.where(labeled_array != 0)
    labels = labeled_array[z, y, x]

    # Sort by label
    sort_idx = np.argsort(labels)
    labels_sorted = labels[sort_idx]
    z_sorted = z[sort_idx]
    y_sorted = y[sort_idx]
    x_sorted = x[sort_idx]

    # Find where each label starts
    unique_labels, split_idx = np.unique(labels_sorted, return_index=True)
    split_idx = split_idx[1:]  # Remove first index for np.split

    # Split into groups
    z_split = np.split(z_sorted, split_idx)
    y_split = np.split(y_sorted, split_idx)
    x_split = np.split(x_sorted, split_idx)

    # Build dict
    coords_dict = {label: np.column_stack([z, y, x]) 
                   for label, z, y, x in zip(unique_labels, z_split, y_split, x_split)}

    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist, squareform
    len_dict = {}
    tortuosity_dict = {}
    angle_dict = {}
    for label, coords in coords_dict.items():
        len_dict[label] = morphology.calculate_skeleton_lengths(labeled_array.shape, xy_scale=xy_scale, z_scale=z_scale, skeleton_coords=coords)
        
        # Find neighbors for all points at once
        nbrs = NearestNeighbors(radius=1.74, algorithm='kd_tree').fit(coords)
        neighbor_counts = nbrs.radius_neighbors(coords, return_distance=False)
        neighbor_counts = np.array([len(n) - 1 for n in neighbor_counts])  # -1 to exclude self
        
        # Endpoints have exactly 1 neighbor
        endpoints = coords[neighbor_counts == 1]
        
        if len(endpoints) > 1:
            # Scale endpoints
            scaled_endpoints = endpoints.copy().astype(float)
            scaled_endpoints[:, 0] *= z_scale  # z dimension
            scaled_endpoints[:, 1] *= xy_scale  # y dimension
            scaled_endpoints[:, 2] *= xy_scale  # x dimension
            
            # calculate distances on scaled coordinates
            distances = pdist(scaled_endpoints, metric='euclidean')
            max_distance = distances.max()
            
            tortuosity_dict[label] = len_dict[label]/max_distance

        for branch, length in len_dict.items():
            if length == 0: # This can happen for branches that are 1 pixel which shouldn't have '0' length technically, so we just set them to the length of a pixel
                len_dict[branch] = xy_scale
                tortuosity_dict[branch] = 1

    """
    verts = invert_array(verts)
    for x, y, z in endpoints:
        try:
            verts[z,y,x] = 1
        except IndexError:
            print(x, y, z)

    temp_network = Network_3D(nodes = verts, edges = labeled_array, xy_scale = xy_scale, z_scale = z_scale)
    temp_network.calculate_all(temp_network.nodes, temp_network.edges, xy_scale = temp_network.xy_scale, z_scale = temp_network.z_scale, search = None, diledge = None, inners = False, remove_trunk = 0, ignore_search_region = True, other_nodes = None, label_nodes = True, directory = None, GPU = False, fast_dil = False, skeletonize = False, GPU_downsample = None)
    temp_network.calculate_node_centroids()
    from itertools import combinations
    for node in temp_network.network.nodes:
        neighbors = list(temp_network.network.neighbors(node))
        
        # Skip if fewer than 2 neighbors (endpoints or isolated nodes)
        if len(neighbors) < 2:
            continue
        
        # Get all unique pairs of neighbors
        neighbor_pairs = combinations(neighbors, 2)
        
        angles = []
        for neighbor1, neighbor2 in neighbor_pairs:
            # Get coordinates from centroids
            point_a = temp_network.node_centroids[neighbor1]
            point_b = temp_network.node_centroids[node]  # vertex
            point_c = temp_network.node_centroids[neighbor2]
            
            # Calculate angle
            angle_result = calculate_3d_angle(point_a, point_b, point_c, xy_scale = xy_scale, z_scale = z_scale)
            angles.append(angle_result)
        
        angle_dict[node] = angles
    """

    return len_dict, tortuosity_dict, angle_dict

def calculate_3d_angle(point_a, point_b, point_c, xy_scale = 1, z_scale = 1):
    """Calculate 3D angle at vertex B between points A-B-C."""
    z1, y1, x1 = point_a
    z2, y2, x2 = point_b  # vertex
    z3, y3, x3 = point_c
    
    # Apply scaling
    scaled_a = np.array([x1 * xy_scale, y1 * xy_scale, z1 * z_scale])
    scaled_b = np.array([x2 * xy_scale, y2 * xy_scale, z2 * z_scale])
    scaled_c = np.array([x3 * xy_scale, y3 * xy_scale, z3 * z_scale])
    
    # Create vectors from vertex B
    vec_ba = scaled_a - scaled_b
    vec_bc = scaled_c - scaled_b
    
    # Calculate angle using dot product
    dot_product = np.dot(vec_ba, vec_bc)
    magnitude_ba = np.linalg.norm(vec_ba)
    magnitude_bc = np.linalg.norm(vec_bc)
    
    # Avoid division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return {'angle_degrees': 0}
    
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def threshold(arr, proportion, custom_rad = None):

    """Internal method to apply a proportional threshold on an image"""

    def find_closest_index(target: float, num_list: list[float]) -> int:
       return min(range(len(num_list)), key=lambda i: abs(num_list[i] - target))


    if custom_rad is not None:

        threshold_value = custom_rad

    else:
        # Step 1: Flatten the array
        flattened = arr.flatten()

        # Step 2: Filter out the zero values
        non_zero_values = list(set(flattened[flattened > 0]))

        # Step 3: Sort the remaining values
        sorted_values = np.sort(non_zero_values)

        threshold_index = int(len(sorted_values) * proportion)
        threshold_value = sorted_values[threshold_index]
        print(f"Thresholding as if smallest_radius was assigned {threshold_value}")


    mask = arr > threshold_value

    arr = arr * mask

    return arr

def generate_3d_bounding_box(shape, foreground_value=1, background_value=0):
    """
    Generate a 3D bounding box array with edges connecting the corners.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the array in format (Z, Y, X)
    foreground_value : int or float, default=1
        Value to use for the bounding box edges and corners
    background_value : int or float, default=0
        Value to use for the background
    
    Returns:
    --------
    numpy.ndarray
        3D array with bounding box edges
    """
    if len(shape) > 3:
        shape = (shape[0], shape[1], shape[2])

    z_size, y_size, x_size = shape
    
    # Create empty array filled with background value
    box_array = np.full(shape, background_value, dtype=np.float64)
    
    # Define the 8 corners of the 3D box
    corners = [
        (0, 0, 0),           # corner 0
        (0, 0, x_size-1),    # corner 1
        (0, y_size-1, 0),    # corner 2
        (0, y_size-1, x_size-1),  # corner 3
        (z_size-1, 0, 0),    # corner 4
        (z_size-1, 0, x_size-1),  # corner 5
        (z_size-1, y_size-1, 0),  # corner 6
        (z_size-1, y_size-1, x_size-1)  # corner 7
    ]
    
    # Set corner values
    for corner in corners:
        box_array[corner] = foreground_value
    
    # Define edges connecting adjacent corners
    # Each edge connects two corners that differ by only one coordinate
    edges = [
        # Bottom face edges (z=0)
        (0, 1), (1, 3), (3, 2), (2, 0),
        # Top face edges (z=max)
        (4, 5), (5, 7), (7, 6), (6, 4),
        # Vertical edges connecting bottom to top
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Draw edges using linspace
    for start_idx, end_idx in edges:
        start_corner = corners[start_idx]
        end_corner = corners[end_idx]
        
        # Calculate the maximum distance along any axis to determine number of points
        max_distance = max(
            abs(end_corner[0] - start_corner[0]),
            abs(end_corner[1] - start_corner[1]),
            abs(end_corner[2] - start_corner[2])
        )
        num_points = max_distance + 1
        
        # Generate points along the edge using linspace
        z_points = np.linspace(start_corner[0], end_corner[0], num_points, dtype=int)
        y_points = np.linspace(start_corner[1], end_corner[1], num_points, dtype=int)
        x_points = np.linspace(start_corner[2], end_corner[2], num_points, dtype=int)
        
        # Set foreground values along the edge
        for z, y, x in zip(z_points, y_points, x_points):
            box_array[int(z), int(y), int(x)] = foreground_value
    
    return box_array

def show_3d(arrays_3d=None, arrays_4d=None, down_factor=None, order=0, xy_scale=1, z_scale=1, colors=['red', 'green', 'white', 'cyan', 'yellow'], box = False):
    """
    Show 3d (or 2d) displays of array data using napari.
    Params: arrays - A list of 3d or 2d numpy arrays to display
    down_factor (int) - Optional downsampling factor to speed up display
    """
    import os
    # Force PyQt6 usage to avoid binding warning
    os.environ['QT_API'] = 'pyqt6'
    
    import napari
    from qtpy.QtWidgets import QApplication

    if down_factor is not None:
        # Downsample arrays if specified
        arrays_3d = [downsample(array, down_factor, order=order) for array in arrays_3d] if arrays_3d is not None else None
        arrays_4d = [downsample(array, down_factor, order=order) for array in arrays_4d] if arrays_4d is not None else None
        scale = [z_scale * down_factor, xy_scale * down_factor, xy_scale * down_factor]
    else:
        scale = [z_scale, xy_scale, xy_scale]

    
    viewer = napari.Viewer(ndisplay=3)
    
    # Add 3D arrays if provided
    if arrays_3d is not None:
        for arr, color in zip(arrays_3d, colors):
            shape = arr.shape
            viewer.add_image(
                arr,
                scale=scale,
                colormap=color,
                rendering='mip',
                blending='additive',
                opacity=0.5,
                name=f'Channel_{color}'
            )
        
    if arrays_4d is not None:
        for i, arr in enumerate(arrays_4d):
            # Check if the last dimension is 3 (RGB) or 4 (RGBA)
            if arr.shape[-1] not in [3, 4]:
                print(f"Warning: Array {i} doesn't appear to be RGB/RGBA. Skipping.")
                continue
                
            if arr.shape[3] == 4:
                arr = arr[:, :, :, :3]  # Remove alpha
            
            shape = arr.shape

            # Add each color channel separately
            colors = ['red', 'green', 'blue']
            for c in range(3):
                viewer.add_image(
                    arr[:,:,:,c],  # Take just one color channel
                    scale=scale,
                    colormap=colors[c],  # Use corresponding color
                    rendering='mip',
                    blending='additive',
                    opacity=0.5,
                    name=f'Channel_{colors[c]}_{i}'
                )

    if box:
        viewer.add_image(
            generate_3d_bounding_box(shape),
            scale=scale,
            colormap='white',
            rendering='mip',
            blending='additive',
            opacity=0.5,
            name=f'Bounding Box'
        )



    napari.run()

def z_project(array3d, method='max'):
    """
    Project a 3D numpy array along the Z axis to create a 2D array.
    
    Parameters:
        array3d (numpy.ndarray): 3D input array with shape (Z, Y, X)
        method (str): Projection method - 'max', 'mean', 'min', 'sum', or 'std'
    
    Returns:
        numpy.ndarray: 2D projected array with shape (Y, X)
    """
    if not isinstance(array3d, np.ndarray) or array3d.ndim != 3:
        raise ValueError("Input must be a 3D numpy array")
        
    if method == 'max':
        return np.max(array3d, axis=0)
    elif method == 'mean':
        return np.mean(array3d, axis=0)
    elif method == 'min':
        return np.min(array3d, axis=0)
    elif method == 'sum':
        return np.sum(array3d, axis=0)
    elif method == 'std':
        return np.std(array3d, axis=0)
    else:
        raise ValueError("Method must be one of: 'max', 'mean', 'min', 'sum', 'std'")

def fill_holes_3d(array, head_on = False, fill_borders = True):
    def process_slice(slice_2d, border_threshold=0.08, fill_borders = True):
        """
        Process a 2D slice, considering components that touch less than border_threshold
        of any border length as potential holes.
        
        Args:
            slice_2d: 2D binary array
            border_threshold: proportion of border that must be touched to be considered background
        """
        from scipy.ndimage import binary_fill_holes
        
        slice_2d = slice_2d.astype(np.uint8)

        # Apply scipy's binary_fill_holes to the result
        slice_2d = binary_fill_holes(slice_2d)
        
        return slice_2d
        
    print("Filling Holes...")
    
    array = binarize(array)
    #inv_array = invert_array(array)
    
    # Create arrays for all three planes
    array_xy = np.zeros_like(array, dtype=np.uint8)
    array_xz = np.zeros_like(array, dtype=np.uint8)
    array_yz = np.zeros_like(array, dtype=np.uint8)
    
    # Process XY plane
    for z in range(array.shape[0]):
        array_xy[z] = process_slice(array[z], fill_borders = fill_borders)
        
    if (array.shape[0] > 3) and not head_on: #only use these dimensions for sufficiently large zstacks
        
        # Process XZ plane    
        for y in range(array.shape[1]):
            slice_xz = array[:, y, :]
            array_xz[:, y, :] = process_slice(slice_xz, fill_borders = fill_borders)
            
        # Process YZ plane
        for x in range(array.shape[2]):
            slice_yz = array[:, :, x]
            array_yz[:, :, x] = process_slice(slice_yz, fill_borders = fill_borders)
        
        # Combine results from all three planes
        filled = (array_xy | array_xz | array_yz) * 255
        return array + filled
    else:
        # Apply scipy's binary_fill_holes to each XY slice
        from scipy.ndimage import binary_fill_holes
        for z in range(array_xy.shape[0]):
            array_xy[z] = binary_fill_holes(array_xy[z])
        return array_xy * 255

def fill_holes_3d_old(array, head_on = False, fill_borders = True):

    def process_slice(slice_2d, border_threshold=0.08, fill_borders = True):
        """
        Process a 2D slice, considering components that touch less than border_threshold
        of any border length as potential holes.
        
        Args:
            slice_2d: 2D binary array
            border_threshold: proportion of border that must be touched to be considered background
        """
        slice_2d = slice_2d.astype(np.uint8)
        labels, num_features = ndimage.label(slice_2d)

        if not fill_borders:
            border_threshold = 0 #Testing
        
        if num_features == 0:
            return np.zeros_like(slice_2d)
        
        # Get dimensions for threshold calculations
        height, width = slice_2d.shape
        
        # Dictionary to store border intersection lengths for each label
        border_proportions = {}
        
        for label in range(1, num_features + 1):
            mask = labels == label
            
            # Calculate proportion of each border this component touches
            top_prop = np.sum(mask[0, :]) / width
            bottom_prop = np.sum(mask[-1, :]) / width
            left_prop = np.sum(mask[:, 0]) / height
            right_prop = np.sum(mask[:, -1]) / height
            
            # If it touches any border significantly, consider it background
            border_proportions[label] = max(top_prop, bottom_prop, left_prop, right_prop)
        
        # Create mask of components that either don't touch borders
        # or touch less than the threshold proportion

        background_labels = {label for label, prop in border_proportions.items() 
                            if prop > border_threshold}

        
        holes_mask = ~np.isin(labels, list(background_labels))
        
        return holes_mask

    print("Filling Holes...")
    
    array = binarize(array)
    inv_array = invert_array(array)

    
    # Create arrays for all three planes
    array_xy = np.zeros_like(inv_array, dtype=np.uint8)
    array_xz = np.zeros_like(inv_array, dtype=np.uint8)
    array_yz = np.zeros_like(inv_array, dtype=np.uint8)


    # Process XY plane
    for z in range(inv_array.shape[0]):
        array_xy[z] = process_slice(inv_array[z], fill_borders = fill_borders)

    if (array.shape[0] > 3) and not head_on: #only use these dimensions for sufficiently large zstacks
        
        # Process XZ plane    
        for y in range(inv_array.shape[1]):
            slice_xz = inv_array[:, y, :]
            array_xz[:, y, :] = process_slice(slice_xz, fill_borders = fill_borders)
            
        # Process YZ plane
        for x in range(inv_array.shape[2]):
            slice_yz = inv_array[:, :, x]
            array_yz[:, :, x] = process_slice(slice_yz, fill_borders = fill_borders)
        
        # Combine results from all three planes
        filled = (array_xy | array_xz | array_yz) * 255
        return array + filled
    else:
        return array_xy * 255





def resize(array, factor, order = 0):
    """Simply resizes an array by a factor"""

    if len(array.shape) == 4:  # presumably this is a color image
        processed_arrays = []
        for i in range(array.shape[3]):  # iterate through the color dimension
            color_array = array[:, :, :, i]  # get 3D array for each color channel
            processed_color = zoom(color_array, (factor), order = order)

            processed_arrays.append(processed_color)
        
        # Stack them back together along the 4th dimension
        result = np.stack(processed_arrays, axis=3)
        return result

    array = zoom(array, (factor), order = order)

    return array



def _rescale(array, original_shape, xy_scale, z_scale):
    """Internal method to help 3D visualization"""
    if xy_scale != 1 or z_scale != 1: #Handle seperate voxel scalings by resizing array dimensions
        if z_scale > xy_scale:
            array = zoom(array, (xy_scale/z_scale, 1, 1), order = 3)
        elif xy_scale > z_scale:
            array = zoom(array, (1, z_scale/xy_scale, z_scale/xy_scale))
    return array


def remove_trunk(edges, num_iterations=1):
    """
    Removes the largest connected objects from a 3D binary array in-place.
    
    Parameters:
    -----------
    edges : ndarray
        3D binary array containing objects to process.
        Will be modified in-place.
    num_iterations : int, optional
        Number of largest objects to remove, default is 1.
        
    Returns:
    --------
    ndarray
        Reference to the modified input array.
    """
    # Label connected components
    labeled_array, num_features = measure.label(edges, background=0, return_num=True)
    
    # If there are fewer objects than requested iterations, adjust
    iterations = min(num_iterations, num_features)
    
    if iterations == 0 or num_features == 0:
        return edges
    
    # Count occurrences of each label
    label_counts = np.bincount(labeled_array.ravel())
    
    # Skip background (label 0)
    label_counts = label_counts[1:]
    
    # Find indices of largest objects (argsort returns ascending order, so we reverse it)
    largest_indices = np.argsort(label_counts)[::-1][:iterations]
    
    # Convert back to actual labels (add 1 because we skipped background)
    largest_labels = largest_indices + 1
    
    # Modify the input array in-place
    for label in largest_labels:
        edges[labeled_array == label] = 0
    
    return edges

def get_all_label_coords(labeled_array, background=0):
    """
    Get coordinates for all labels using single pass method.
    
    Parameters:
    -----------
    labeled_array : numpy.ndarray
        Labeled array with integer labels
    background : int, optional
        Background label to exclude (default: 0)
    
    Returns:
    --------
    dict : {label: coordinates_array}
        Dictionary mapping each label to its coordinate array
    """
    coords_dict = {}
    
    # Get all non-background coordinates at once
    all_coords = np.argwhere(labeled_array != background)
    
    if len(all_coords) == 0:
        return coords_dict
    
    # Get the label values at those coordinates
    labels_at_coords = labeled_array[tuple(all_coords.T)]
    
    # Group by label
    unique_labels = np.unique(labels_at_coords)
    for label in unique_labels:
        mask = labels_at_coords == label
        coords_dict[label] = all_coords[mask]
    
    return coords_dict

def approx_boundaries(array, iden_set = None, node_identities = None, keep_labels = False):

    """Hollows out an array, can do it for only a set number of identities. Returns coords as dict if labeled or as 1d numpy array if binary is desired"""

    if node_identities is not None:

        nodes = []

        for node in node_identities:

            if node_identities[node] in iden_set: #Filter out only idens we need
                nodes.append(node)

        mask = np.isin(array, nodes)

        if keep_labels:

            array = array * mask
        else:
            array = mask
        del mask

    from skimage.segmentation import find_boundaries

    borders = find_boundaries(array, mode='thick')
    array = array * borders
    del borders
    if not keep_labels:
        return np.argwhere(array != 0)
    else:
        return get_all_label_coords(array)



def hash_inners(search_region, inner_edges, GPU = False):
    """Internal method used to help sort out inner edge connections. The inner edges of the array will not differentiate between what nodes they contact if those nodes themselves directly touch each other.
    This method allows these elements to be efficiently seperated from each other"""

    from skimage.segmentation import find_boundaries

    borders = find_boundaries(search_region, mode='thick')

    inner_edges = inner_edges * borders #And as a result, we can mask out only 'inner edges' that themselves exist within borders

    inner_edges = dilate_3D_old(inner_edges, 3, 3, 3) #Not sure if dilating is necessary. Want to ensure that the inner edge pieces still overlap with the proper nodes after the masking.

    return inner_edges

def hash_inners_old(search_region, inner_edges, GPU = True):
    """Internal method used to help sort out inner edge connections. The inner edges of the array will not differentiate between what nodes they contact if those nodes themselves directly touch each other.
    This method allows these elements to be efficiently seperated from each other. Originally this was implemented using the gaussian blur because i didn't yet realize skimage could do the same more efficiently."""

    print("Performing gaussian blur to hash inner edges.")

    blurred_search = smart_dilate.gaussian(search_region, GPU = GPU) 

    borders = binarize((blurred_search - search_region)) #By subtracting the original image from the guassian blurred version, we set all non-border regions to 0

    del blurred_search

    inner_edges = inner_edges * borders #And as a result, we can mask out only 'inner edges' that themselves exist within borders

    inner_edges = dilate_3D_old(inner_edges, 3, 3, 3) #Not sure if dilating is necessary. Want to ensure that the inner edge pieces still overlap with the proper nodes after the masking.

    return inner_edges


def dilate_2D(array, search, scaling = 1):

    inv = array < 1

    inv = smart_dilate.compute_distance_transform_distance(inv)

    inv = inv * scaling

    inv = inv <= search

    return inv


def dilate_3D_dt(array, search_distance, xy_scaling=1.0, z_scaling=1.0, fast_dil = False):
    """
    Dilate a 3D array using distance transform method. Dt dilation produces perfect results but only works in euclidean geometry and lags in big arrays.
    
    Parameters:
    array -- Input 3D binary array
    search_distance -- Distance within which to dilate
    xy_scaling -- Scaling factor for x and y dimensions (default: 1.0)
    z_scaling -- Scaling factor for z dimension (default: 1.0)
    
    Returns:
    Dilated 3D array
    """

    if array.shape[0] == 1:

        return dilate_2D(array, search_distance, scaling = xy_scaling) #Use the 2d method in psueod-3d cases


    # Invert the array (find background)
    inv = array < 1

    del array

    """
    # Determine which dimension needs resampling
    if (z_scaling > xy_scaling):
        # Z dimension needs to be stretched
        zoom_factor = [z_scaling/xy_scaling, 1, 1]  # Scale factor for [z, y, x]
        rev_factor = [xy_scaling/z_scaling, 1, 1] 
        cardinal = xy_scaling
    elif (xy_scaling > z_scaling):
        # XY dimensions need to be stretched
        zoom_factor = [1, xy_scaling/z_scaling, xy_scaling/z_scaling]  # Scale factor for [z, y, x]
        rev_factor = [1, z_scaling/xy_scaling, z_scaling/xy_scaling]  # Scale factor for [z, y, x]
        cardinal = z_scaling
    else:
        # Already uniform scaling, no need to resample
        zoom_factor = None
        rev_factor = None
        cardinal = xy_scaling

    # Resample the mask if needed
    if zoom_factor:
        inv = ndimage.zoom(inv, zoom_factor, order=0)  # Use order=0 for binary masks
    """

    # Compute distance transform (Euclidean)
    inv = smart_dilate.compute_distance_transform_distance(inv, sampling = [z_scaling, xy_scaling, xy_scaling], fast_dil = fast_dil)

    #inv = inv * cardinal
    
    # Threshold the distance transform to get dilated result
    inv = inv <= search_distance

    #if rev_factor:
        #inv = ndimage.zoom(inv, rev_factor, order=0)  # Use order=0 for binary masks
    
    return inv.astype(np.uint8)

def erode_2D(array, search, scaling=1, preserve_labels = False):
    """
    Erode a 2D array using distance transform method.
    
    Parameters:
    array -- Input 2D binary array
    search -- Distance within which to erode
    scaling -- Scaling factor (default: 1)
    
    Returns:
    Eroded 2D array
    """
    # For erosion, we work directly with the foreground
    # No need to invert the array

    if preserve_labels:
        from skimage.segmentation import find_boundaries
        borders = find_boundaries(array, mode='thick')
        mask = array * invert_array(borders)
        mask = smart_dilate.compute_distance_transform_distance(mask)
        mask = mask * scaling
        mask = mask >= search
        array = mask * array
    else:
        # Compute distance transform on the foreground
        dt = smart_dilate.compute_distance_transform_distance(array)
        
        # Apply scaling
        dt = dt * scaling
        
        # Threshold to keep only points that are at least 'search' distance from the boundary
        array = dt > search
    
    return array

def erode_3D_dt(array, search_distance, xy_scaling=1.0, z_scaling=1.0, fast_dil = False, preserve_labels = False):
    """
    Erode a 3D array using distance transform method. DT erosion produces perfect results 
    with Euclidean geometry, but may be slower for large arrays.
    
    Parameters:
    array -- Input 3D binary array
    search_distance -- Distance within which to erode
    xy_scaling -- Scaling factor for x and y dimensions (default: 1.0)
    z_scaling -- Scaling factor for z dimension (default: 1.0)
    GPU -- Whether to use GPU acceleration if available (default: False)
    
    Returns:
    Eroded 3D array
    """
    
    if array.shape[0] == 1:
        # Handle 2D case
        return erode_2D(array, search_distance, scaling=xy_scaling, preserve_labels = True)
    

    if preserve_labels:


        from skimage.segmentation import find_boundaries

        borders = find_boundaries(array, mode='thick')
        mask = array * invert_array(borders)
        mask = smart_dilate.compute_distance_transform_distance(mask, sampling = [z_scaling, xy_scaling, xy_scaling], fast_dil = fast_dil)
        mask = mask >= search_distance
        array = mask * array
    else:
        array = smart_dilate.compute_distance_transform_distance(array, sampling = [z_scaling, xy_scaling, xy_scaling], fast_dil = fast_dil)
        # Threshold the distance transform to get eroded result
        # For erosion, we keep only the points that are at least search_distance from the boundary
        array = array > search_distance
    
    return array.astype(np.uint8)


def dilate_3D(tiff_array, dilated_x, dilated_y, dilated_z):
    """Internal method to dilate an array in 3D. Dilation this way is much faster than using a distance transform although the latter is more accurate.
    Arguments are an array,  and the desired pixel dilation amounts in X, Y, Z. Uses psuedo-3D kernels (imagine a 3D + sign rather than a cube) to approximate 3D neighborhoods but will miss diagonally located things with larger kernels, if those are needed use the distance transform version.
    """

    if dilated_x == 3 and dilated_y == 3  and dilated_z == 3:

        return dilate_3D_old(tiff_array, dilated_x, dilated_y, dilated_z)

    if tiff_array.shape[0] == 1:
        return dilate_2D(tiff_array, ((dilated_x - 1) / 2))

    def create_circular_kernel(diameter):
        """Create a 2D circular kernel with a given radius.

        Parameters:
        radius (int or float): The radius of the circle.

        Returns:
        numpy.ndarray: A 2D numpy array representing the circular kernel.
        """
        # Determine the size of the kernel
        radius = diameter/2
        size = radius  # Diameter of the circle
        size = int(np.ceil(size))  # Ensure size is an integer
        
        # Create a grid of (x, y) coordinates
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        
        # Calculate the distance from the center (0,0)
        distance = np.sqrt(x**2 + y**2)
        
        # Create the circular kernel: points within the radius are 1, others are 0
        kernel = distance <= radius
        
        # Convert the boolean array to integer (0 and 1)
        return kernel.astype(np.uint8)

    def create_ellipsoidal_kernel(long_axis, short_axis):
        """Create a 2D ellipsoidal kernel with specified axis lengths and orientation.

        Parameters:
        long_axis (int or float): The length of the long axis.
        short_axis (int or float): The length of the short axis.

        Returns:
        numpy.ndarray: A 2D numpy array representing the ellipsoidal kernel.
        """
        semi_major, semi_minor = long_axis / 2, short_axis / 2

        # Determine the size of the kernel

        size_y = int(np.ceil(semi_minor))
        size_x = int(np.ceil(semi_major))
        
        # Create a grid of (x, y) coordinates centered at (0,0)
        y, x = np.ogrid[-semi_minor:semi_minor+1, -semi_major:semi_major+1]
        
        # Ellipsoid equation: (x/a)^2 + (y/b)^2 <= 1
        ellipse = (x**2 / semi_major**2) + (y**2 / semi_minor**2) <= 1
        
        return ellipse.astype(np.uint8)


    # Function to process each slice
    def process_slice(z):
        tiff_slice = tiff_array[z].astype(np.uint8)
        dilated_slice = cv2.dilate(tiff_slice, kernel, iterations=1)
        return z, dilated_slice

    def process_slice_other(y):
        tiff_slice = tiff_array[:, y, :].astype(np.uint8)
        dilated_slice = cv2.dilate(tiff_slice, kernel, iterations=1)
        return y, dilated_slice

    """
    def process_slice_third(x):
        tiff_slice = tiff_array[:, :, x].astype(np.uint8)
        dilated_slice = cv2.dilate(tiff_slice, kernel, iterations=1)
        return x, dilated_slice
    """

    # Create empty arrays to store the dilated results for the XY and XZ planes
    dilated_xy = np.zeros_like(tiff_array, dtype=np.uint8)
    dilated_xz = np.zeros_like(tiff_array, dtype=np.uint8)
    #dilated_yz = np.zeros_like(tiff_array, dtype=np.uint8)

    kernel_x = int(dilated_x)
    kernel = create_circular_kernel(kernel_x)

    num_cores = mp.cpu_count()

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_slice, z): z for z in range(tiff_array.shape[0])}

        for future in as_completed(futures):
            z, dilated_slice = future.result()
            dilated_xy[z] = dilated_slice

    kernel_x = int(dilated_x)
    kernel_z = int(dilated_z)

    if kernel_x == kernel_z:
        kernel = create_circular_kernel(kernel_z)
    else:
        kernel = create_ellipsoidal_kernel(kernel_x, kernel_z)

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_slice_other, y): y for y in range(tiff_array.shape[1])}
        
        for future in as_completed(futures):
            y, dilated_slice = future.result()
            dilated_xz[:, y, :] = dilated_slice

    """
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_slice_other, x): x for x in range(tiff_array.shape[2])}
        
        for future in as_completed(futures):
            x, dilated_slice = future.result()
            dilated_yz[:, :, x] = dilated_slice
    """


    # Overlay the results
    final_result = (dilated_xy | dilated_xz)

    return final_result

def dilate_3D_old(tiff_array, dilated_x=3, dilated_y=3, dilated_z=3):
    """
    Dilate a 3D array using scipy.ndimage.binary_dilation with a 3x3x3 cubic kernel.
    
    Arguments:
    tiff_array -- Input 3D binary array
    dilated_x -- Fixed at 3 for X dimension
    dilated_y -- Fixed at 3 for Y dimension
    dilated_z -- Fixed at 3 for Z dimension
    
    Returns:
    Dilated 3D array
    """
    
    # Create a simple 3x3x3 cubic kernel (all ones)
    kernel = np.ones((3, 3, 3), dtype=bool)
    
    # Perform binary dilation
    dilated_array = ndimage.binary_dilation(tiff_array.astype(bool), structure=kernel)
    
    return dilated_array.astype(np.uint8)


def dilation_length_to_pixels(xy_scaling, z_scaling, micronx, micronz):
    """Internal method to find XY and Z dilation parameters based on voxel micron scaling"""
    dilate_xy = 2 * int(round(micronx/xy_scaling))

    # Ensure the dilation param is odd to have a center pixel
    dilate_xy += 1 if dilate_xy % 2 == 0 else 0

    dilate_z = 2 * int(round(micronz/z_scaling))

    # Ensure the dilation param is odd to have a center pixel
    dilate_z += 1 if dilate_z % 2 == 0 else 0

    return dilate_xy, dilate_z

def label_objects(nodes, dtype=int):
    """Internal method to labels objects with cubic 3D labelling scheme"""
    if len(nodes.shape) == 3:
        structure_3d = np.ones((3, 3, 3), dtype=int)

    elif len(nodes.shape) == 2:
        structure_3d = np.ones((3, 3), dtype = int)
    nodes, num_nodes = ndimage.label(nodes, structure = structure_3d)

    # Choose a suitable data type based on the number of labels
    if num_nodes < 256:
        dtype = np.uint8
    elif num_nodes < 65536:
        dtype = np.uint16
    else:
        dtype = np.uint32

    # Convert the labeled array to the chosen data type
    nodes = nodes.astype(dtype)

    return nodes, num_nodes


def remove_zeros(input_list):
    """Internal method to remove zeroes from an array"""
    # Use boolean indexing to filter out zeros
    result_array = input_list[input_list != 0] #note - presumes your list is an np array

    return result_array


def overlay_arrays_simple(edge_labels_1, edge_labels_2):
    """
    Superimpose edge_labels_2 on top of edge_labels_1 without any offset.
    Where edge_labels_2 > 0, use those values directly.
    """
    mask = edge_labels_1 > 0
    return np.where(mask, edge_labels_1, edge_labels_2)

def combine_edges(edge_labels_1, edge_labels_2):
    """
    let NumPy handle promotion automatically
    """
    # Early exit if no combination needed
    mask = (edge_labels_1 == 0) & (edge_labels_2 > 0)
    if not np.any(mask):
        return edge_labels_1.copy()
    
    max_val = np.max(edge_labels_1)
    
    # Let NumPy handle dtype promotion automatically
    # This will promote to the smallest type that can handle the operation
    offset_labels = edge_labels_2 + max_val
    
    return np.where(mask, offset_labels, edge_labels_1)

def directory_info(directory = None):
    """Internal method to get the files in a directory, optionally the current directory if nothing passed"""
    
    if directory is None:
        items = os.listdir()
    else:
        # Get the list of all items in the directory
        items = os.listdir(directory)
    
    return items


# Ripley's K Helpers:

def mirror_points_for_edge_correction(points_array, bounds, max_r, dim=3):
    """
    Mirror points near boundaries to handle edge effects in Ripley's K analysis.
    Works with actual coordinate positions, not spatial grid placement.
    
    Parameters:
    points_array: numpy array of shape (n, 3) with [z, y, x] coordinates (already scaled)
    bounds: tuple of (min_coords, max_coords) where each is array - can be 2D or 3D
    max_r: maximum search radius (determines mirroring distance)
    dim: dimension (2 or 3) - affects which coordinates are used
    
    Returns:
    numpy array with original points plus mirrored points
    """
    min_coords, max_coords = bounds
    
    # Ensure bounds are numpy arrays and handle dimension mismatch
    min_coords = np.array(min_coords)
    max_coords = np.array(max_coords)
    
    # Handle case where bounds might be 2D but points are 3D
    if len(min_coords) == 2 and points_array.shape[1] == 3:
        # Extend 2D bounds to 3D by adding z=0 dimension at the front
        min_coords = np.array([0, min_coords[0], min_coords[1]])  # [0, min_x, min_y] -> [min_z, min_y, min_x]
        max_coords = np.array([0, max_coords[0], max_coords[1]])  # [0, max_x, max_y] -> [max_z, max_y, max_x]
    elif len(min_coords) == 3 and points_array.shape[1] == 3:
        # Already 3D, but ensure it's in [z,y,x] format (your bounds are [x,y,z] and get flipped)
        pass  # Should already be handled by the flip in your bounds calculation
    
    # Start with original points
    all_points = points_array.copy()
    
    if dim == 2:
        # For 2D: work with y, x coordinates (indices 1, 2), z should be 0
        active_dims = [1, 2]  # y, x
        # 8 potential mirror regions for 2D (excluding center)
        mirror_combinations = [
            [0, -1], [0, 1],   # left, right (y direction)
            [-1, 0], [1, 0],   # bottom, top (x direction)
            [-1, -1], [-1, 1], # corners
            [1, -1], [1, 1]
        ]
    else:
        # For 3D: work with z, y, x coordinates (indices 0, 1, 2)
        active_dims = [0, 1, 2]  # z, y, x
        # 26 potential mirror regions for 3D (3^3 - 1, excluding center)
        mirror_combinations = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if not (dz == 0 and dy == 0 and dx == 0):  # exclude center
                        mirror_combinations.append([dz, dy, dx])
    
    # Process each potential mirror region
    for mirror_dir in mirror_combinations:
        # Find points that need this specific mirroring
        needs_mirror = np.ones(len(points_array), dtype=bool)
        
        # Check each active dimension
        for i, dim_idx in enumerate(active_dims):
            direction = mirror_dir[i] if dim == 3 else mirror_dir[i]
            
            # Safety check: make sure we have bounds for this dimension
            if dim_idx >= len(min_coords) or dim_idx >= len(max_coords):
                needs_mirror = np.zeros(len(points_array), dtype=bool)  # Skip this mirror if bounds insufficient
                break
            
            if direction == -1:  # Points near minimum boundary
                # Distance from point to min boundary < max_r
                needs_mirror &= (points_array[:, dim_idx] - min_coords[dim_idx]) < max_r
            elif direction == 1:  # Points near maximum boundary  
                # Distance from point to max boundary < max_r
                needs_mirror &= (max_coords[dim_idx] - points_array[:, dim_idx]) < max_r
            # direction == 0 means no constraint for this dimension
        
        # Create mirrored points if any qualify
        if np.any(needs_mirror):
            mirrored_points = points_array[needs_mirror].copy()
            
            # Apply mirroring transformation for each active dimension
            for i, dim_idx in enumerate(active_dims):
                direction = mirror_dir[i] if dim == 3 else mirror_dir[i]
                
                # Safety check again
                if dim_idx >= len(min_coords) or dim_idx >= len(max_coords):
                    continue
                
                if direction == -1:  # Mirror across minimum boundary
                    # Reflection formula: new_coord = 2 * boundary - old_coord
                    mirrored_points[:, dim_idx] = 2 * min_coords[dim_idx] - mirrored_points[:, dim_idx]
                elif direction == 1:  # Mirror across maximum boundary
                    # Reflection formula: new_coord = 2 * boundary - old_coord
                    mirrored_points[:, dim_idx] = 2 * max_coords[dim_idx] - mirrored_points[:, dim_idx]
            
            # Add mirrored points to collection
            all_points = np.vstack([all_points, mirrored_points])
    
    return all_points
    
def get_max_r_from_proportion(bounds, proportion):
    """
    Calculate max_r based on bounds and proportion, matching your generate_r_values logic.
    
    Parameters:
    bounds: tuple of (min_coords, max_coords)
    proportion: maximum proportion of study area extent
    
    Returns:
    max_r value
    """
    min_coords, max_coords = bounds
    min_coords = np.array(min_coords)
    max_coords = np.array(max_coords)
    
    # Calculate dimensions
    dimensions = max_coords - min_coords
    
    # Remove placeholder dimensions (where dimension = 1, typically for 2D z-dimension)
    # But ensure we don't end up with an empty array
    filtered_dimensions = dimensions[dimensions != 1]
    if len(filtered_dimensions) == 0:
        # If all dimensions were 1 (shouldn't happen), use original dimensions
        filtered_dimensions = dimensions
    
    # Use minimum dimension for safety (matches your existing logic)
    min_dimension = np.min(filtered_dimensions)
    max_r = min_dimension * proportion
    
    return max_r

def apply_edge_correction_to_ripley(roots, targs, proportion, bounds, dim, node_centroids=None):
    """
    Apply edge correction through mirroring to target points.
    
    This should be called AFTER convert_centroids_to_array but BEFORE 
    convert_augmented_array_to_points (for 2D case).
    
    Parameters:
    roots: array of root points (search centers) - already scaled
    targs: array of target points (points being searched for) - already scaled  
    proportion: the proportion parameter from your workflow
    bounds: boundary tuple (min_coords, max_coords) or None
    dim: dimension (2 or 3)
    node_centroids: dict of node centroids (needed if bounds is None)
    
    Returns:
    tuple: (roots, mirrored_targs) where mirrored_targs includes edge corrections
    """
    # Handle bounds calculation if not provided (matching your existing logic)
    if bounds is None:
        if node_centroids is None:
            # Fallback: calculate from the points we have
            all_points = np.vstack([roots, targs])
        else:
            # Use your existing method
            import proximity  # Assuming this is available
            big_array = proximity.convert_centroids_to_array(list(node_centroids.values()))
            all_points = big_array
        
        min_coords = np.array([0, 0, 0])
        max_coords = [np.max(all_points[:, 0]), np.max(all_points[:, 1]), np.max(all_points[:, 2])]
        max_coords = np.flip(max_coords)  # Convert [x,y,z] to [z,y,x] format
        bounds = (min_coords, max_coords)
        
        if 'big_array' in locals():
            del big_array
    
    # Calculate max_r using your existing logic
    max_r = get_max_r_from_proportion(bounds, proportion)
    
    # Mirror target points for edge correction
    mirrored_targs = mirror_points_for_edge_correction(targs, bounds, max_r, dim)
    
    print(f"Original target points: {len(targs)}, After mirroring: {len(mirrored_targs)}")
    print(f"Added {len(mirrored_targs) - len(targs)} mirrored points for edge correction")
    print(f"Using max_r = {max_r} for mirroring threshold")
    print(f"Bounds used: min={bounds[0]}, max={bounds[1]}")
    
    return roots, mirrored_targs


#CLASSLESS FUNCTIONS THAT MAY BE USEFUL TO USERS TO RUN DIRECTLY THAT SUPPORT ANALYSIS IN SOME WAY. NOTE THESE METHODS SOMETIMES ARE USED INTERNALLY AS WELL:

def downsample(data, factor, directory=None, order=0):
    """
    Can be used to downsample an image by some arbitrary factor. Downsampled output will be saved to the active directory if none is specified.
    
    :param data: (Mandatory, string or ndarray) - If string, a path to a tif file to downsample. Note that the ndarray alternative is for internal use mainly and will not save its output.
    :param factor: (Mandatory, int) - A factor by which to downsample the image.
    :param directory: (Optional - Val = None, string) - A filepath to save outputs.
    :param order: (Optional - Val = 0, int) - The order of interpolation for scipy.ndimage.zoom
    :returns: a downsampled ndarray.
    """
    # Load the data if it's a file path
    if isinstance(data, str):
        data2 = data
        data = tifffile.imread(data)
    else:
        data2 = None

    if len(data.shape) == 4:  # presumably this is a color image
        processed_arrays = []
        for i in range(data.shape[3]):  # iterate through the color dimension
            color_array = data[:, :, :, i]  # get 3D array for each color channel
            processed_color = downsample(color_array, factor, directory = None, order = order) #right now this is only for internal use - color array downsampling that is
            processed_arrays.append(processed_color)
        
        # Stack them back together along the 4th dimension
        result = np.stack(processed_arrays, axis=3)
        return result
    
    # Check if Z dimension is too small relative to downsample factor
    if data.ndim == 3 and data.shape[0] < factor * 4:
        print(f"Warning: Z dimension ({data.shape[0]}) is less than 4x the downsample factor ({factor}). "
              f"Skipping Z-axis downsampling to preserve resolution.")
        zoom_factors = (1, 1/factor, 1/factor)
    else:
        zoom_factors = 1/factor


    # Apply downsampling
    data = zoom(data, zoom_factors, order=order)
    
    # Save if input was a file path
    if isinstance(data2, str):
        if directory is None:
            filename = "downsampled.tif"
        else:
            filename = f"{directory}/downsampled.tif"
        tifffile.imwrite(filename, data)
    
    return data


def otsu_binarize(image_array, non_bool = False):

    """Automated binarize method for seperating the foreground"""

    from skimage.filters import threshold_otsu

    threshold = threshold_otsu(image_array)
    binary_mask = image_array > threshold

    if non_bool:
        binary_mask = binary_mask * 255
        
    return binary_mask

def binarize(arrayimage, directory = None):
    """
    Can be used to binarize an image. Binary output will be saved to the active directory if none is specified.
    :param arrayimage: (Mandatory, string or ndarray) - If string, a path to a tif file to binarize. Output will be 8bit with 0 representing background and 255 representing signal. Note that the ndarray alternative is for internal use mainly and will not save its output, and will also contain vals of 0 and 1.
    :param directory: (Optional - Val = None, string) - A filepath to save outputs.
    :returns: a binary ndarray.
    """
    if type(arrayimage) == str:
        print("Binarizing...")
        image = arrayimage
        arrayimage = tifffile.imread(arrayimage)

    arrayimage = arrayimage != 0

    arrayimage = arrayimage * 255

    if type(arrayimage) == str:
        arrayimage = arrayimage * 255
        if directory is None:
            tifffile.imwrite(f"binary.tif", arrayimage)
        else:
            tifffile.imwrite(f"{directory}/binary.tif", arrayimage)


    return arrayimage.astype(np.uint8)

def dilate(arrayimage, amount, xy_scale = 1, z_scale = 1, directory = None, fast_dil = False, recursive = False, dilate_xy = None, dilate_z = None):
    """
    Can be used to dilate a binary image in 3D. Dilated output will be saved to the active directory if none is specified. Note that dilation is done with single-instance kernels and not iterations, and therefore
    objects will lose their shape somewhat and become cube-ish if the 'amount' param is ever significantly larger than the objects in quesiton.
    :param arrayimage: (Mandatory, string or ndarray) - If string, a path to a tif file to dilate. Note that the ndarray alternative is for internal use mainly and will not save its output.
    :param amount: (Mandatory, int) - The amount to dilate the array. Note that if xy_scale and z_scale params are not passed, this will correspond one-to-one with voxels. Otherwise, it will correspond with what voxels represent (ie microns).
    :param xy_scale: (Optional; Val = 1, float) - The scaling of pixels.
    :param z_scale: (Optional - Val = 1; float) - The depth of voxels.
    :param directory: (Optional - Val = None, string) - A filepath to save outputs.
    :param fast_dil: (Optional - Val = False, boolean) - A boolean that when True will utilize faster cube dilation but when false will use slower spheroid dilation.
    :returns: a dilated ndarray.
    """

    if type(arrayimage) == str:
        print("Dilating...")
        image = arrayimage
        arrayimage = tifffile.imread(arrayimage).astype(np.uint8)
    else:
        image = None

    if not dilate_xy:
        dilate_xy, dilate_z = dilation_length_to_pixels(xy_scale, z_scale, amount, amount)

    if len(np.unique(arrayimage)) > 2: #binarize
        arrayimage = binarize(arrayimage)

    if fast_dil:
        arrayimage = (dilate_3D(arrayimage, dilate_xy, dilate_xy, dilate_z))
    else:
        arrayimage = (dilate_3D_dt(arrayimage, amount, xy_scale, z_scale))


    if type(image) == str:
        if directory is None:
            filename = f'dilated.tif'
        else:
            filename = f'{directory}/dilated.tif'

        tifffile.imwrite(filename, arrayimage)
        print(f"Dilated array saved to {filename}")

    return arrayimage

def erode(arrayimage, amount, xy_scale = 1, z_scale = 1, mode = 0, preserve_labels = False):
    if not preserve_labels and len(np.unique(arrayimage)) > 2: #binarize
        arrayimage = binarize(arrayimage)

    if mode == 0 or mode == 2:
        fast_dil = True
    else:
        fast_dil = False

    arrayimage = erode_3D_dt(arrayimage, amount, xy_scaling=xy_scale, z_scaling=z_scale, fast_dil = fast_dil, preserve_labels = preserve_labels)

    if np.max(arrayimage) == 1:
        arrayimage = arrayimage * 255

    return arrayimage

def iden_set(idens):

    idens = set(idens)
    real_iden_set = []
    for iden in idens:
        try:
            options = ast.literal_eval(iden)
            for opt in options:
                real_iden_set.append(opt)
        except:
            real_iden_set.append(iden)

    return set(real_iden_set)



def skeletonize(arrayimage, directory = None):
    """
    Can be used to 3D skeletonize a binary image. Skeletonized output will be saved to the active directory if none is specified. Note this works better on already thin filaments and may make mistakes on larger trunkish objects.
    :param arrayimage: (Mandatory, string or ndarray) - If string, a path to a tif file to skeletonize. Note that the ndarray alternative is for internal use mainly and will not save its output.
    :param directory: (Optional - Val = None, string) - A filepath to save outputs.
    :returns: a skeletonized ndarray.
    """
    print("Skeletonizing...")


    if type(arrayimage) == str:
        image = arrayimage
        arrayimage = tifffile.imread(arrayimage).astype(np.uint8)
    else:
        image = None

    arrayimage = (mpg.skeletonize(arrayimage))

    if type(image) == str:
        if directory is None:
            filename = f'skeletonized.tif'
        else:
            filename = f'{directory}/skeletonized.tif'

        tifffile.imwrite(filename, arrayimage)
        print(f"Skeletonized array saved to {filename}")

    return arrayimage

def label_branches(array, peaks = 0, branch_removal = 0, comp_dil = 0, max_vol = 0, down_factor = None, directory = None, nodes = None, bonus_array = None, GPU = True, arrayshape = None, compute = False, unify = False, union_val = 10, mode = 0, xy_scale = 1, z_scale = 1):
    """
    Can be used to label branches a binary image. Labelled output will be saved to the active directory if none is specified. Note this works better on already thin filaments and may over-divide larger trunkish objects.
    :param array: (Mandatory, string or ndarray) - If string, a path to a tif file to label. Note that the ndarray alternative is for internal use mainly and will not save its output.
    :param branch_removal: (Optional, Val = None; int) - An optional into to specify what size of pixels to remove branches. Use this if the skeleton is branchy and you want to remove the branches from the larger filaments.
    :param comp_dil: (Optional, Val = 0; int) - An optional value to merge nearby vertices. This algorithm may be prone to leaving a few, disconnected vertices next to each other that otherwise represent the same branch point but will confound the network a bit. These can be combined into a single object by dilation. Note this dilation will be applied post downsample, so take that into account when assigning a value, as the value will not take resampling into account and will just apply as is on a downsample.
    :param max_vol: (Optional, Val = 0, int) - An optional value of the largest volume of an object to keep in the vertices output. Will only filter if > 0.
    :param down_factor: (Optional, Val = None; int) - An optional factor to downsample internally to speed up computation. Note that this method will try to use the GPU if one is available, which may
    default to some internal downsampling.
    :param directory: (Optional - Val = None; string) - A filepath to save outputs.
    :returns: an ndarray with labelled branches.
    """
    if type(array) == str:
        stringbool = True
        array = tifffile.imread(array)
    else:
        stringbool = False

    if down_factor is not None and nodes is None:
        array = downsample(array, down_factor)
        arrayshape = array.shape
    else:
        arrayshape = arrayshape

    if nodes is None:

        array = array > 0

        other_array = skeletonize(array)

        other_array, verts, skele, endpoints = break_and_label_skeleton(other_array, peaks = peaks, branch_removal = branch_removal, comp_dil = comp_dil, max_vol = max_vol, nodes = nodes, compute = compute, unify = unify, xy_scale = xy_scale, z_scale = z_scale)

    else:
        if down_factor is not None:
            bonus_array = downsample(bonus_array, down_factor)
        array, verts, skele, endpoints = break_and_label_skeleton(array, peaks = peaks, branch_removal = branch_removal, comp_dil = comp_dil, max_vol = max_vol, nodes = nodes, compute = compute, unify = unify, xy_scale = xy_scale, z_scale = z_scale)

    if unify is True and nodes is not None:
        from . import branch_stitcher
        verts = dilate_3D_old(verts, 3, 3, 3,)
        verts, _ = label_objects(verts)
        print("Merging branches...")
        array = branch_stitcher.trace(bonus_array, array, verts, score_thresh = union_val, xy_scale = xy_scale, z_scale = z_scale)
        verts = None


    if nodes is None:

        array = smart_dilate.smart_label(array, other_array, GPU = GPU, remove_template = True, mode = mode)
        #distance = smart_dilate.compute_distance_transform_distance(array)
        #array = water(-distance, other_array, mask=array) #Tried out skimage watershed as shown and found it did not label branches as well as smart_label (esp combined combined with post-processing label splitting if needed)

    else:
        if down_factor is not None:
            array = smart_dilate.smart_label(bonus_array, array, GPU = GPU, predownsample = down_factor, remove_template = True, mode = mode)
            #distance = smart_dilate.compute_distance_transform_distance(bonus_array)
            #array = water(-distance, array, mask=bonus_array)
        else:

            array = smart_dilate.smart_label(bonus_array, array, GPU = GPU, remove_template = True, mode = mode)
            #distance = smart_dilate.compute_distance_transform_distance(bonus_array)
            #array = water(-distance, array, mask=bonus_array)


    if down_factor is not None and nodes is None:
        array = upsample_with_padding(array, down_factor, arrayshape)

    if stringbool:
        if directory is not None:
            filename = f'{directory}/labelled_branches.tif'
        else:
            filename = f'labelled_branches.tif'

        tifffile.imwrite(filename, other_array)
        print(f"Labelled branches saved to {filename}")
    else:
        print("Branches labelled")

    if nodes is not None and down_factor is not None:
        array = upsample_with_padding(array, down_factor, arrayshape)


    return array, verts, skele, endpoints

def fix_branches_network(array, G, communities, fix_val = None):

    def get_degree_threshold(community_degrees):
        degrees = np.array(community_degrees, dtype=float)
        hist, bins = np.histogram(degrees, bins='auto')        
        peaks, _ = find_peaks(hist)
        if len(peaks) > 1:
            # Get bin value after first peak as threshold
            return bins[peaks[0] + 1]
        return 4  # Default fallback

    avg_degree = G.number_of_edges() * 2 / G.number_of_nodes()

    targs = []

    inverted = invert_dict(communities)

    community_degrees = {}

    for com in inverted:
        subgraph = G.subgraph(inverted[com])
        sub_degree = subgraph.number_of_edges() * 2/ subgraph.number_of_nodes()
        community_degrees[com] = sub_degree


    if fix_val is None:
        threshold = get_degree_threshold(list(community_degrees.values()))
    else:
        threshold = fix_val

    for com in community_degrees:
        if community_degrees[com] > threshold: #This method of comparison could possibly be more nuanced. 
            targs.append(com)


    return targs

def fix_branches(array, G, max_val):
    """
    Parameters:
    array: numpy array containing the labeled regions
    G: Graph representing connectivity relationships
    max_val: The target value to find neighbors for
    
    Returns:
    Modified array with fused regions
    """
    # Get all nodes
    all_nodes = set(G.nodes())
    
    # Initially safe nodes are direct neighbors of max_val
    safe_initial = set(G.neighbors(max_val))
    
    # Not-safe nodes are all other nodes except max_val
    not_safe_initial = all_nodes - safe_initial - {max_val}
    
    # Get adjacency view (much faster for repeated neighbor lookups)
    adj = G.adj
    
    # Find all neighbors of not_safe nodes in one pass
    neighbors_of_not_safe = set()
    for node in not_safe_initial:
        neighbors_of_not_safe.update(adj[node])
    
    # Remove max_val if present
    neighbors_of_not_safe.discard(max_val)
    
    # Find safe nodes that should be moved
    nodes_to_move = safe_initial & neighbors_of_not_safe
    
    # Update sets
    not_safe = not_safe_initial | nodes_to_move
    
    # The rest of the function - FIX STARTS HERE
    targs = np.array(list(not_safe))
    
    if len(targs) == 0:
        return array
        
    mask = np.isin(array, targs)
    
    labeled, num_components = label_objects(mask)
    
    # Get the current maximum label in the array to avoid collisions
    current_max = np.max(array)
    
    # Assign new unique labels to each connected component
    for component_id in range(1, num_components + 1):
        component_mask = labeled == component_id
        array[component_mask] = current_max + component_id
    
    return array






def label_vertices(array, peaks = 0, branch_removal = 0, comp_dil = 0, max_vol = 0, down_factor = 0, directory = None, return_skele = False, order = 0, fastdil = True):
    """
    Can be used to label vertices (where multiple branches connect) a binary image. Labelled output will be saved to the active directory if none is specified. Note this works better on already thin filaments and may over-divide larger trunkish objects.
    Note that this can be used in tandem with an edge segmentation to create an image containing 'pseudo-nodes', meaning we can make a network out of just a single edge file.
    :param array: (Mandatory, string or ndarray) - If string, a path to a tif file to label. Note that the ndarray alternative is for internal use mainly and will not save its output.
    :param peaks: (Optional, Val = 0; int) - An optional value on what size of peaks to keep. A peak is peak in the histogram of volumes of objects in the array. The number of peaks that will be kept start on the left (low volume). The point of this is to remove large, erroneous vertices that may result from skeletonizing large objects. 
    :param branch_removal: (Optional, Val = 0; int) - An optional into to specify what size of pixels to remove branches. Use this if the skeleton is branchy and you want to remove the branches from the larger filaments. Large objects tend to produce branches when skeletonized. Enabling this in the right situations will make the output significantly more accurate.
    :param comp_dil: (Optional, Val = 0; int) - An optional value to merge nearby vertices. This algorithm may be prone to leaving a few, disconnected vertices next to each other that otherwise represent the same branch point but will confound the network a bit. These can be combined into a single object by dilation. Note this dilation will be applied post downsample, so take that into account when assigning a value, as the value will not take resampling into account and will just apply as is on a downsample.
    :param max_vol: (Optional, Val = 0, int) - An optional value of the largest volume of an object to keep in the vertices output. Will only filter if > 0.
    :param directory: (Optional - Val = None; string) - A filepath to save outputs.
    :returns: an ndarray with labelled vertices.
    """    
    print("Breaking Skeleton...")

    if type(array) == str:
        broken_skele = array
        array = tifffile.imread(array)
    else:
        broken_skele = None

    if down_factor > 1:
        array_shape = array.shape
        array = downsample(array, down_factor, order)
        if order == 3:
            array = binarize(array)

    array = array > 0

    array = skeletonize(array)

    if return_skele:
        old_skeleton = copy.deepcopy(array) # The skeleton might get modified in label_vertices so we can make a preserved copy of it to use later

    if branch_removal > 0:
        array = remove_branches(array, branch_removal)

    array = np.pad(array, pad_width=1, mode='constant', constant_values=0)

    # Find all nonzero voxel coordinates
    nonzero_coords = np.transpose(np.nonzero(array))
    x, y, z = nonzero_coords[0]
    threshold = 3 * array[x, y, z]

    # Create a copy of the image to modify
    image_copy = np.zeros_like(array)

    # Iterate through each nonzero voxel
    for x, y, z in nonzero_coords:

        # Count nearby pixels including diagonals
        mini = array[x-1:x+2, y-1:y+2, z-1:z+2]
        nearby_sum = np.sum(mini)
        
        if nearby_sum > threshold:
            mini = mini.copy()
            mini[1, 1, 1] = 0
            _, test_num = ndimage.label(mini)
            if test_num > 2:
                image_copy[x-1:x+2, y-1:y+2, z-1:z+2] = 1

    image_copy = (image_copy[1:-1, 1:-1, 1:-1]).astype(np.uint8)


    # Label the modified image to assign new labels for each branch
    #labeled_image, num_labels = measure.label(image_copy, connectivity=2, return_num=True)

    if peaks > 0:
        image_copy = filter_size_by_peaks(image_copy, peaks)
        if comp_dil > 0:
            image_copy = dilate_3D_dt(image_copy, comp_dil, fast_dil = fastdil)

        labeled_image, num_labels = label_objects(image_copy)
    elif max_vol > 0:
        image_copy = filter_size_by_vol(image_copy, max_vol)
        if comp_dil > 0:
            image_copy = dilate_3D_dt(image_copy, comp_dil, fast_dil = fastdil)

        labeled_image, num_labels = label_objects(image_copy)
    else:
        if comp_dil > 0:
            image_copy = dilate_3D_dt(image_copy, comp_dil, fast_dil = fastdil)
        labeled_image, num_labels = label_objects(image_copy)

    #if down_factor > 0:
        #labeled_image = upsample_with_padding(labeled_image, down_factor, array_shape)

    if type(broken_skele) == str:
        if directory is None:
            filename = f'labelled_vertices.tif'
        else:
            filename = f'{directory}/labelled_vertices.tif'

        tifffile.imwrite(filename, labeled_image, photometric='minisblack')
        print(f"Broken skeleton saved to {filename}")

    if return_skele:

        return labeled_image, old_skeleton

    else:

        return labeled_image

def filter_size_by_peaks(binary_array, num_peaks_to_keep=1):

    binary_array = binary_array > 0
    # Label connected components
    labeled_array, num_features = ndimage.label(binary_array)
    
    # Calculate the volume of each object
    volumes = np.bincount(labeled_array.ravel())[1:]
    
    # Create a histogram of volumes
    hist, bin_edges = np.histogram(volumes, bins='auto')
    
    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, distance=1)
    
    if len(peaks) < num_peaks_to_keep + 1:
        print(f"Warning: Found only {len(peaks)} peaks. Keeping all objects up to the last peak.")
        num_peaks_to_keep = len(peaks) - 1
    
    if num_peaks_to_keep < 1:
        print("Warning: Invalid number of peaks to keep. Keeping all objects.")
        return binary_array

    print(f"Keeping all peaks up to {num_peaks_to_keep} of {len(peaks)} peaks")
    
    # Find the valley after the last peak we want to keep
    if num_peaks_to_keep == len(peaks):
        # If we're keeping all peaks, set the threshold to the maximum volume
        volume_threshold = volumes.max()
    else:
        valley_start = peaks[num_peaks_to_keep - 1]
        valley_end = peaks[num_peaks_to_keep]
        valley = valley_start + np.argmin(hist[valley_start:valley_end])
        volume_threshold = bin_edges[valley + 1]
    
    # Create a mask for objects larger than the threshold
    mask = np.isin(labeled_array, np.where(volumes > volume_threshold)[0] + 1)
    
    # Set larger objects to 0
    result = binary_array.copy()
    result[mask] = 0
    
    return result

def filter_size_by_vol(binary_array, volume_threshold):

    binary_array = binary_array > 0
    # Label connected components
    labeled_array, num_features = ndimage.label(binary_array)
    
    # Calculate the volume of each object
    volumes = np.bincount(labeled_array.ravel())[1:]
    
    # Create a mask for objects larger than the threshold
    mask = np.isin(labeled_array, np.where(volumes > volume_threshold)[0] + 1)
    
    # Set larger objects to 0
    result = binary_array.copy()
    result[mask] = 0
    
    return result

def gray_watershed(image, min_distance = 1, threshold_abs = None):


    from skimage.feature import peak_local_max

    if len(np.unique(image)) == 2:
        image = smart_dilate.compute_distance_transform_distance(image)


    is_pseudo_3d = image.shape[0] == 1
    if is_pseudo_3d:
        image = np.squeeze(image)  # Convert to 2D for processing

    #smoothed = ndimage.gaussian_filter(image.astype(float), sigma=2)

    peaks = peak_local_max(image, min_distance = min_distance, threshold_abs = threshold_abs)
    if len(peaks) < 256:
        dtype = np.uint8
    elif len(peaks) < 65535:
        dtype = np.uint16
    else:
        dytpe = np.uint32

    clone = np.zeros_like(image).astype(dtype)

    if not is_pseudo_3d:
        for i, peak in enumerate(peaks):
            z, y, x = peak
            clone[z,y,x] = i + 1
    else:
        for i, peak in enumerate(peaks):
            y, x = peak
            clone[y,x] = i + 1


    if is_pseudo_3d:
        image = np.expand_dims(image, axis = 0)
        clone = np.expand_dims(clone, axis = 0)


    binary_image = binarize(image)
    #image = smart_dilate.smart_label(image, clone, GPU = False)

    image = water(-image, clone, mask=binary_image)



    return image


def watershed(image, directory = None, proportion = 0.1, GPU = True, smallest_rad = None, fast_dil = False, predownsample = None, predownsample2 = None):
    """
    Can be used to 3D watershed a binary image. Watershedding attempts to use an algorithm to split touching objects into seperate labelled components. Labelled output will be saved to the active directory if none is specified.
    This watershed algo essentially uses the distance transform to decide where peaks are and then after thresholding out the non-peaks, uses the peaks as labelling kernels for a smart label. It runs semi slow without GPU accel since it requires two dts to be computed.
    :param image: (Mandatory, string or ndarray). - If string, a path to a binary .tif to watershed, or an ndarray containing the same.
    :param directory: (Optional - Val = None; string) - A filepath to save outputs.
    :param proportion: (Optional - Val = 0.1; float) - A zero to one value representing the proportion of watershed 'peaks' that are kept around for splitting objects. Essentially,
    making this value smaller makes the watershed break more things, however making it too small will result in some unusual failures where small objects all get the same label. 
    :param GPU: (Optional - Val = True; boolean). If True, GPU will be used to watershed. Please note this will result in internal downsampling most likely, and overall be very fast.
    However, this downsampling may kick small nodes out of the image. Do not use the GPU to watershed if your GPU wants to downsample beyond the size of the smallest node that you
    want to keep in the output. Set to False to use the CPU (no downsampling). Note using the GPU + downsample may take around a minute to process arrays that are a few GB while the CPU may take an hour or so.
    :param smallest_rad: (Optional - Val = None; int). The size (in voxels) of the radius of the smallest object you want to seperate with watershedding. Note that the
    'proportion' param is the affector of watershed outputs but itself may be confusing to tinker with. By inputting a smallest_rad, the algo will instead compute a custom proportion
    to use for your data.
    :returns: A watershedded, labelled ndarray.
    """ 

    if type(image) == str:
        image = tifffile.imread(image)

    image = image > 0

    original_shape = image.shape


    try:

        if GPU == True and cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy for distance transform.")

            try:

                if predownsample is None:
                    # Step 4: Find the nearest label for each voxel in the ring
                    distance = smart_dilate.compute_distance_transform_distance_GPU(image)
                else:
                    gotoexcept = 1/0

            except (cp.cuda.memory.OutOfMemoryError, ZeroDivisionError) as e:
                
                if predownsample is None:
                    down_factor = smart_dilate.catch_memory(e) #Obtain downsample amount based on memory missing
                else:
                    down_factor = (predownsample)**3

                while True:
                    downsample_needed = down_factor**(1./3.)
                    small_image = downsample(image, downsample_needed) #Apply downsample
                    try:
                        distance = smart_dilate.compute_distance_transform_distance_GPU(small_image) #Retry dt on downsample
                        print(f"Using {down_factor} downsample ({downsample_needed} in each dim - largest possible with this GPU)")
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        down_factor += 1
                old_mask = smart_dilate.binarize(image)
                image = small_image
                del small_image
        else:
            goto_except = 1/0
    except Exception as e:
        if GPU:
            print("GPU dt failed or did not detect GPU (cupy must be installed with a CUDA toolkit setup...). Computing CPU distance transform instead.")
            print(f"Error message: {str(e)}")
        distance = smart_dilate.compute_distance_transform_distance(image, fast_dil = fast_dil)


    distance = threshold(distance, proportion, custom_rad = smallest_rad)

    labels, _ = label_objects(distance)

    if len(labels.shape) ==2:
        labels = np.expand_dims(labels, axis = 0)

    #del distance


    if labels.shape[1] < original_shape[1]: #If downsample was used, upsample output
        labels = upsample_with_padding(labels, downsample_needed, original_shape)
        labels = labels * old_mask
        labels = water(-distance, labels, mask=old_mask) # Here i like skimage watershed over smart_label, mainly because skimage just kicks out too-small nodes from the image, while smart label just labels them sort of wrongly.
        #labels = smart_dilate.smart_label(old_mask, labels, GPU = GPU, predownsample = predownsample2)
    else:
        labels = water(-distance, labels, mask=image)
        #labels = smart_dilate.smart_label(image, labels, GPU = GPU, predownsample = predownsample2)

    if directory is None:
        pass
    else:
        tifffile.imwrite(f"{directory}/Watershed_output.tif", labels)
        print(f"Watershed saved to {directory}/'Watershed_output.tif'")

    return labels

def filter_by_size(array, proportion=0.1, directory = None):
    """
    Threshold out objects below a certain proportion of the total volume in a 3D binary array.
    
    :param array: (Mandatory; string or ndarray) - A file path to a 3D binary tif image array with objects or an ndarray of the same.
    :param proportion: (Optional - Val = 0.1; float): Proportion of the total volume to use as the threshold. Objects smaller tha this proportion of the total volume will be removed.
    :param directory: (Optional - Val = None; string): Optional file path to a directory to save output, otherwise active directory will be used.

    :returns: A 3D binary numpy array with small objects removed.
    """

    if type(array) == str:
        array = tifffile.imread(array)

    # Label connected components
    labeled_array, num_features = label_objects(array)

    # Calculate the volume of each object
    object_slices = ndimage.find_objects(labeled_array)
    object_volumes = np.array([np.sum(labeled_array[slc] == i + 1) for i, slc in enumerate(object_slices)])

    # Determine the threshold volume
    total_volume = np.sum(object_volumes)
    threshold_volume = total_volume * proportion
    print(f"threshold_volume is {threshold_volume}")

    # Filter out small objects
    large_objects = np.zeros_like(array, dtype=np.uint8)
    for i, vol in enumerate(object_volumes):
        print(f"Obj {i+1} vol is {vol}")
        if vol >= threshold_volume:
            large_objects[labeled_array == i + 1] = 1

    if directory is None:
        tifffile.imwrite('filtered_array.tif', large_objects)
    else:
        tifffile.imwrite(f'{directory}/filtered_array.tif', large_objects)

    return large_objects


def mask(image, mask, directory = None):
    """
    Can be used to mask one image with another. Masked output will be saved to the active directory if none is specified.
    :param image: (Mandatory, string or ndarray) - If string, a path to a tif file to get masked, or an ndarray of the same.
    :param mask: (Mandatory, string or ndarray) - If string, a path to a tif file to be a mask, or an ndarray of the same.
    :param directory: (Optional - Val = None; string) - A filepath to save outputs.
    :returns: a masked ndarray.
    """    
    if type(image) == str or type(mask) == str:
        string_bool = True
    else:
        string_bool = False

    if type(image) == str:
        image = tifffile.imread(image)
    if type(mask) == str:
        mask = tifffile.imread(mask)

    mask = mask != 0

    if len(image.shape) == 3:

        image = image * mask
    else:
        # Split into separate color channels
        channels = [image[..., i] for i in range(3)]
        masked_channels = []
        
        for image in channels:
            # Upsample each channel separately
            if len(image.shape) == 2:
                np.expand_dims(image, axis = 0)
            image = image * mask
            masked_channels.append(image)
            
        # Stack the channels back together
        image = np.stack(masked_channels, axis=-1)


    if string_bool:
        if directory is None:
            filename = tifffile.imwrite("masked_image.tif", image)
        else:
            filename = tifffile.imwrite(f"{directory}/masked_image.tif", image)
        print(f"Masked image saved to masked_image.tif")

    return image

def inverted_mask(image, mask, directory = None):
    """
    Can be used to mask one image with the inversion of another. Masked output will be saved to the active directory if none is specified.
    :param image: (Mandatory, string or ndarray) - If string, a path to a tif file to get masked, or an ndarray of the same.
    :param mask: (Mandatory, string or ndarray) - If string, a path to a tif file to be a mask, or an ndarray of the same.
    :param directory: (Optional - Val = None; string) - A filepath to save outputs.
    :returns: a masked ndarray.
    """    
    if type(image) == str or type(mask) == str:
        string_bool = True
    else:
        string_bool = False

    if type(image) == str:
        image = tifffile.imread(image)
    if type(mask) == str:
        mask = tifffile.imread(mask)

    mask = invert_array(mask)
    mask = mask != 0

    image = image * mask

    if string_bool:
        if directory is None:
            filename = tifffile.imwrite("masked_image.tif", image)
        else:
            filename = tifffile.imwrite(f"{directory}/masked_image.tif", image)
        print(f"Masked image saved to masked_image.tif")

    return image


def label(image, directory = None):
    """
    Can be used to label a binary image, where each discrete object is assigned its own grayscale value. Labelled output will be saved to the active directory if none is specified.
    :param image: (Mandatory, string or ndarray) - If string, a path to a tif file to get masked, or an ndarray of the same.
    :param directory: (Optional - Val = None; string) - A filepath to save outputs.
    :returns: a labelled ndarray.
    """    
    if type(image) == str:
        image = tifffile.imread(image)
    image, _ = label_objects(image)
    if directory is None:
        image = tifffile.imwrite('labelled_image.tif', image)
    else:
        image = tifffile.imwrite(f'{directory}/labelled_image.tif', image)

    return image

def encapsulate(parent_dir = None, name = None):
    """Used for saving outputs to a new directory called my_network"""

    import os

    if name is None:
        name = 'my_network'
    
    # Use current directory if no parent_dir specified
    if parent_dir is None:
        parent_dir = os.getcwd()
        
    # Create the full path for the new folder
    new_folder_path = os.path.join(parent_dir, name)
    
    # Create the folder if it doesn't exist
    os.makedirs(new_folder_path, exist_ok=True)
    
    return new_folder_path




#THE 3D NETWORK CLASS

class Network_3D:
    """A class to store various components of the 3D networks, to make working with them easier"""
    def __init__(self, nodes = None, network = None, xy_scale = 1, z_scale = 1, network_lists = None, edges = None, search_region = None, node_identities = None, node_centroids = None, edge_centroids = None, communities = None, network_overlay = None, id_overlay = None):
        """
        Constructor that initiates a Network_3D object. Note that xy_scale and z_scale attributes will default to 1 while all others will default to None.
        :attribute 1: (ndarray) _nodes - a 3D numpy array containing labelled objects that represent nodes in a network
        :attribute 2: (G) _network - a networkx graph object
        :attribute 3: (float) _xy_scale - a float representing the scale of each pixel in the nodes array.
        :attribute 4: (float) _z_scale - a float representing the depth of each voxel in the nodes array.
        :attribute 5: (dict) _network_lists - an internal set of lists that keep network data
        :attribute 6: _edges - a 3D numpy array containing labelled objects that represent edges in a network.
        :attribute 7: _search_region - a 3D numpy array containing labelled objects that represent nodes that have been expanded by some amount to search for connections.
        :attribute 8: _node_identities - a dictionary that relates all nodes to some string identity that details what the node actually represents
        :attribute 9: _node_centroids - a dictionary containing a [Z, Y, x] centroid for all labelled objects in the nodes attribute.
        :attribute 10: _edge_centroids - a dictionary containing a [Z, Y, x] centroid for all labelled objects in the edges attribute.
        :returns: a Network-3D classs object. 
        """
        self._nodes = nodes
        self._network = network
        self._xy_scale = xy_scale
        self._z_scale = z_scale
        self._network_lists = network_lists
        self._edges = edges
        self._search_region = search_region
        self._node_identities = node_identities
        self._node_centroids = node_centroids
        self._edge_centroids = edge_centroids
        self._communities = communities
        self._network_overlay = network_overlay
        self._id_overlay = id_overlay
        self.normalized_weights = None

    def copy(self):
        """
        Copies a Network_3D object so the new object can be freely editted independent of a previous one
        :return: a deep copy of a Network_3D object
        """
        return copy.deepcopy(self)

    #Getters/Setters:

    @property    
    def nodes(self):
        """
        A 3D labelled array for nodes
        :returns: the nodes attribute
        """
        return self._nodes

    @nodes.setter
    def nodes(self, array):
        """Sets the nodes property"""
        if array is not None and not isinstance(array, np.ndarray):
            raise ValueError("nodes must be a (preferably labelled) numpy array.")
        if array is not None and len(array.shape) == 2: #For dealing with 2D images
            #array = np.stack((array, array), axis = 0)
            array = np.expand_dims(array, axis=0)
        self._nodes = array

    @nodes.deleter
    def nodes(self):
        """Eliminates nodes property by setting it to 'None'"""
        self._nodes = None

    @property
    def network(self):
        """
        A networkx graph
        :returns: the network attribute.
        """
        return self._network

    @network.setter
    def network(self, G):
        """Sets the network property, which is intended be a networkx graph object. Additionally alters the network_lists property which is primarily an internal attribute"""
        if G is not None and not isinstance(G, nx.Graph):
            print("network attribute was not set to a networkX undirected graph, which may produce unintended results")
        if G is None:
            self._network = None 
            self._network_lists = None
            self.communities = None
            return

        self._network = G
        self.communities = None
        node_pairings = list(G.edges(data=True)) #Assembling the network lists property.
        lista = []
        listb = []
        listc = []

        try:
            #Networks default to have a weighted attribute of 1 if not otherwise weighted. Here we update the weights
            for u, v, data in node_pairings:
                weight = data.get('weight', 1)  # Default weight is 1 if not specified
                for _ in range(weight):
                    lista.append(u)
                    listb.append(v)
                    listc.append(0)
            
            self._network_lists = [lista, listb, listc]


        except:
            pass


    @network.deleter
    def network(self):
        """Removes the network property by setting it to none"""
        self._network = None

    @property
    def network_lists(self):
        """
        A list with three lists. The first two lists are paired nodes (matched by index), the third is the edge that joins them.
        :returns: the network_lists attribute.
        """
        return self._network_lists

    @network_lists.setter
    def network_lists(self, value):
        """Sets the network_lists attribute"""
        if value is not None and not isinstance(value, list):
            raise ValueError("network lists must be a list.")
        self._network_lists = value
        self._network, _ = network_analysis.weighted_network(self._network_lists)
        self.communities = None

    @network_lists.deleter
    def network_lists(self):
        """Removes the network_lists attribute by setting it to None"""

        self._network_lists = None

    @property
    def xy_scale(self):
        """
        Pixel scaling
        :returns: the xy_scale attribute.
        """
        return self._xy_scale

    @xy_scale.setter
    def xy_scale(self, value):
        """Sets the xy_scale property."""
        self._xy_scale = value

    @property
    def z_scale(self):
        """
        Voxel Depth
        :returns: the z_scale attribute.
        """
        return self._z_scale

    @z_scale.setter
    def z_scale(self, value):
        """Sets the z_scale property"""
        self._z_scale = value

    @property
    def edges(self):
        """
        A 3D labelled array for edges.
        :returns: the edges attribute.
        """
        return self._edges

    @edges.setter
    def edges(self, array):
        """Sets the edges property"""
        if array is not None and not isinstance(array, np.ndarray):
            raise ValueError("edges must be a (preferably labelled) numpy array.")
        if array is not None and len(array.shape) == 2: #For dealing with 2D images
            #array = np.stack((array, array), axis = 0)
            array = np.expand_dims(array, axis=0)
        self._edges = array

    @edges.deleter
    def edges(self):
        """Removes the edges attribute by setting it to None"""
        self._edges = None

    @property
    def search_region(self):
        """
        A 3D labelled array for node search regions.
        :returns: the search_region attribute.
        """
        return self._search_region

    @search_region.setter
    def search_region(self, array):
        """Sets the search_region property"""
        if array is not None and not isinstance(array, np.ndarray):
            raise ValueError("search_region must be a (preferably labelled) numpy array.")
        if array is not None and len(array.shape) == 2: #For dealing with 2D images
            #array = np.stack((array, array), axis = 0)
            array = np.expand_dims(array, axis=0)
        self._search_region = array

    @search_region.deleter
    def search_region(self):
        """Removes the search_region attribute by setting it to None"""
        del self._search_region

    @property
    def node_identities(self):
        """
        A dictionary defining what object each node label refers to (for nodes that index multiple distinct biological objects).
        :returns: the node_identities attribute.
        """
        return self._node_identities

    @node_identities.setter
    def node_identities(self, value):
        """Sets the node_identities attribute"""
        if value is not None and not isinstance(value, dict):
            raise ValueError("node_identities must be a dictionary.")
        self._node_identities = value

    @property
    def node_centroids(self):
        """
        A dictionary of centroids for each node.
        :returns: the node_centroids attribute
        """
        return self._node_centroids

    @node_centroids.setter
    def node_centroids(self, value):
        """Sets the node_centroids property"""
        if value is not None and not isinstance(value, dict):
            raise ValueError("centroids must be a dictionary.")
        self._node_centroids = value

    @property
    def edge_centroids(self):
        """
        A dictionary of centroids for each edge.
        :returns: The _edge_centroids attribute.
        """
        return self._edge_centroids

    @edge_centroids.setter
    def edge_centroids(self, value):
        """Sets the edge_centroids property"""
        if value is not None and not isinstance(value, dict):
            raise ValueError("centroids must be a dictionary.")
        self._edge_centroids = value

    @property
    def communities(self):
        """
        A dictionary of community each node.
        :returns: The _communities attribute.
        """
        return self._communities

    @communities.setter
    def communities(self, value):
        """Sets the communities property"""
        if value is not None and not isinstance(value, dict):
            raise ValueError("communities must be a dictionary.")
        self._communities = value

    @property    
    def network_overlay(self):
        """
        A 3D network overlay
        :returns: the network overlay
        """
        return self._network_overlay

    @network_overlay.setter
    def network_overlay(self, array):
        """Sets the nodes property"""
        if array is not None and not isinstance(array, np.ndarray):
            raise ValueError("network overlay must be a (preferably labelled) numpy array.")
        if array is not None and len(array.shape) == 2: #For dealing with 2D images
            #array = np.stack((array, array), axis = 0)
            array = np.expand_dims(array, axis=0)
 
        self._network_overlay = array

    @property    
    def id_overlay(self):
        """
        A 3D id overlay
        :returns: the id overlay
        """
        return self._id_overlay

    @id_overlay.setter
    def id_overlay(self, array):
        """Sets the nodes property"""
        if array is not None and not isinstance(array, np.ndarray):
            raise ValueError("id overlay must be a (preferably labelled) numpy array.")
        if array is not None and len(array.shape) == 2: #For dealing with 2D images
            #array = np.stack((array, array), axis = 0)
            array = np.expand_dims(array, axis=0)
 
        self._id_overlay = array



    #Saving components of the 3D_network to hard mem:

    def save_nodes(self, directory = None, filename = None):
        """
        Can be called on a Network_3D object to save the nodes property to hard mem as a tif. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an indended directory to save the nodes to.
        """
        if self._nodes is not None:
            imagej_metadata = {
                'spacing': self.z_scale,
                'slices': self._nodes.shape[0],
                'channels': 1,
                'axes': 'ZYX'
            }
            resolution_value = 1.0 / self.xy_scale if self.xy_scale != 0 else 1

        if filename is None:
            filename = "labelled_nodes.tif"
        elif not filename.endswith(('.tif', '.tiff')):
            filename += '.tif'

        if self._nodes is not None:
            if directory is None:
                try:
                    if len(self._nodes.shape) == 3:
                        try:
                            tifffile.imwrite(f"{filename}", self._nodes, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                        except:
                            try:
                                tifffile.imwrite(f"{filename}", self._nodes)
                            except:
                                self._nodes = binarize(self._nodes)
                                tifffile.imwrite(f"{filename}", self._nodes, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                    else:
                        tifffile.imwrite(f"{filename}", self._nodes)
                    print(f"Nodes saved to {filename}")
                except Exception as e:
                    print("Could not save nodes")
            if directory is not None:
                try:
                    if len(self._nodes.shape) == 3:
                        try:
                            tifffile.imwrite(f"{directory}/{filename}", self._nodes, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                        except:
                            try:
                                tifffile.imwrite(f"{directory}/{filename}", self._nodes)
                            except:
                                self._nodes = binarize(self._nodes)
                                tifffile.imwrite(f"{directory}/{filename}", self._nodes, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                    else:
                        tifffile.imwrite(f"{directory}/{filename}", self._nodes)
                    print(f"Nodes saved to {directory}/{filename}")
                except Exception as e:
                    print(f"Could not save nodes to {directory}")
        if self._nodes is None:
            print("Node attribute is empty, did not save...")

    def save_edges(self, directory = None, filename = None):
        """
        Can be called on a Network_3D object to save the edges property to hard mem as a tif. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an indended directory to save the edges to.
        """

        if self._edges is not None:
            imagej_metadata = {
                'spacing': self.z_scale,
                'slices': self._edges.shape[0],
                'channels': 1,
                'axes': 'ZYX'
            }

            resolution_value = 1.0 / self.xy_scale if self.xy_scale != 0 else 1

        if filename is None:
            filename = "labelled_edges.tif"
        elif not filename.endswith(('.tif', '.tiff')):
            filename += '.tif'

        if self._edges is not None:
            if directory is None:
                try:
                    tifffile.imwrite(f"{filename}", self._edges, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                except:
                    try:
                        tifffile.imwrite(f"{filename}", self._edges)
                    except:
                        self._edges = binarize(self._edges)
                        tifffile.imwrite(f"{filename}", self._edges, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                print(f"Edges saved to {filename}")

            if directory is not None:
                try:
                    tifffile.imwrite(f"{directory}/{filename}", self._edges, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                except:
                    try:
                        tifffile.imwrite(f"{directory}/{filename}", self._edges)
                    except:
                        self._edges = binarize(self._edges)
                        tifffile.imwrite(f"{directory}/{filename}", self._edges, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                print(f"Edges saved to {directory}/{filename}")

        if self._edges is None:
            print("Edges attribute is empty, did not save...")

    def save_scaling(self, directory = None):
        """
        Can be called on a Network_3D object to save the xy_scale and z_scale properties to hard mem as a .txt. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an indended directory to save the scalings to.
        """
        output_string = f"xy_scale: {self._xy_scale} \nz_scale: {self._z_scale}"

        if directory is None:
            file_name = "voxel_scalings.txt"
        else:
            file_name = f"{directory}/voxel_scalings.txt"

        with open(file_name, "w") as file:
            file.write(output_string)

        print(f"Voxel scaling has been written to {file_name}")

    def save_node_centroids(self, directory = None):
        """
        Can be called on a Network_3D object to save the node centroids properties to hard mem as a .xlsx file. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an indended directory to save the centroids to.
        """

        if self._node_centroids is not None:
            if directory is None:
                network_analysis._save_centroid_dictionary(self._node_centroids, 'node_centroids.xlsx')
                print("Centroids saved to node_centroids.xlsx")

            if directory is not None:
                network_analysis._save_centroid_dictionary(self._node_centroids, f'{directory}/node_centroids.xlsx')
                print(f"Centroids saved to {directory}/node_centroids.xlsx")

        if self._node_centroids is None:
            if directory is None:
                network_analysis._save_centroid_dictionary({}, 'node_centroids.xlsx')
                print("Centroids saved to node_centroids.xlsx")

            if directory is not None:
                network_analysis._save_centroid_dictionary({}, f'{directory}/node_centroids.xlsx')
                print(f"Centroids saved to {directory}/node_centroids.xlsx")


    def save_edge_centroids(self, directory = None):
        """
        Can be called on a Network_3D object to save the edge centroids properties to hard mem as a .xlsx file. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an indended directory to save the centroids to.
        """
        if self._edge_centroids is not None:
            if directory is None:
                network_analysis._save_centroid_dictionary(self._edge_centroids, 'edge_centroids.xlsx', index = 'Edge ID')
                print("Centroids saved to edge_centroids.xlsx")

            if directory is not None:
                network_analysis._save_centroid_dictionary(self._edge_centroids, f'{directory}/edge_centroids.xlsx', index = 'Edge ID')
                print(f"Centroids saved to {directory}/edge_centroids.xlsx")

        if self._edge_centroids is None:
            print("Edge centroids attribute is empty, did not save...")
            if directory is None:
                network_analysis._save_centroid_dictionary({}, 'edge_centroids.xlsx', index = 'Edge ID')
                print("Centroids saved to edge_centroids.xlsx")

            if directory is not None:
                network_analysis._save_centroid_dictionary({}, f'{directory}/edge_centroids.xlsx', index = 'Edge ID')
                print(f"Centroids saved to {directory}/edge_centroids.xlsx")

    def save_search_region(self, directory = None):
        """
        Can be called on a Network_3D object to save the search_region property to hard mem as a tif. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an indended directory to save the search_region to.
        """
        if self._search_region is not None:
            if directory is None:
                tifffile.imwrite("search_region.tif", self._search_region)
                print("Search region saved to search_region.tif")

            if directory is not None:
                tifffile.imwrite(f"{directory}/search_region.tif", self._search_region)
                print(f"Search region saved to {directory}/search_region.tif")

        if self._search_region is None:
            print("Search_region attribute is empty, did not save...")

    def save_network(self, directory = None):
        """
        Can be called on a Network_3D object to save the network_lists property to hard mem as a .xlsx. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an intended directory to save the network lists to.
        """
        if self._network_lists is not None:
            if directory is None:

                temp_list = network_analysis.combine_lists_to_sublists(self._network_lists)
                create_and_save_dataframe(temp_list, 'output_network.xlsx')

            if directory is not None:
                temp_list = network_analysis.combine_lists_to_sublists(self._network_lists)

                create_and_save_dataframe(temp_list, f'{directory}/output_network.xlsx')

        if self._network_lists is None:
            print("Network associated attributes are empty (must set network_lists property to save network)...")

    def save_node_identities(self, directory = None):
        """
        Can be called on a Network_3D object to save the node_identities property to hard mem as a .xlsx. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an intended directory to save the node_identities to.
        """
        if self._node_identities is not None:
            if directory is None:
                network_analysis.save_singval_dict(self._node_identities, 'NodeID', 'Identity', 'node_identities.xlsx')
                print("Node identities saved to node_identities.xlsx")

            if directory is not None:
                network_analysis.save_singval_dict(self._node_identities, 'NodeID', 'Identity', f'{directory}/node_identities.xlsx')
                print(f"Node identities saved to {directory}/node_identities.xlsx")

        if self._node_identities is None:
            if directory is None:
                network_analysis.save_singval_dict({}, 'NodeID', 'Identity', 'node_identities.xlsx')
                print("Node identities saved to node_identities.xlsx")

            if directory is not None:
                network_analysis.save_singval_dict({}, 'NodeID', 'Identity', f'{directory}/node_identities.xlsx')
                print(f"Node identities saved to {directory}/node_identities.xlsx")

    def save_communities(self, directory = None):
        """
        Can be called on a Network_3D object to save the communities property to hard mem as a .xlsx. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an intended directory to save the communities to.
        """
        if self._communities is not None:
            if directory is None:
                network_analysis.save_singval_dict(self._communities, 'NodeID', 'Community', 'node_communities.xlsx')
                print("Communities saved to node_communities.xlsx")

            if directory is not None:
                network_analysis.save_singval_dict(self._communities, 'NodeID', 'Community', f'{directory}/node_communities.xlsx')
                print(f"Communities saved to {directory}/node_communities.xlsx")

        if self._communities is None:
            if directory is None:
                network_analysis.save_singval_dict({}, 'NodeID', 'Community', 'node_communities.xlsx')
                print("Communities saved to node_communities.xlsx")

            if directory is not None:
                network_analysis.save_singval_dict({}, 'NodeID', 'Community', f'{directory}/node_communities.xlsx')
                print(f"Communities saved to {directory}/node_communities.xlsx")

    def save_network_overlay(self, directory = None, filename = None):

        if self._network_overlay is not None:
            imagej_metadata = {
                'spacing': self.z_scale,
                'slices': self._network_overlay.shape[0],
                'channels': 1,
                'axes': 'ZYX'
            }
            resolution_value = 1.0 / self.xy_scale if self.xy_scale != 0 else 1

        if filename is None:
            filename = "overlay_1.tif"
        elif not filename.endswith(('.tif', '.tiff')):
            filename += '.tif'

        if self._network_overlay is not None:
            if directory is None:
                if len(self._network_overlay.shape) == 3:
                    try:
                        tifffile.imwrite(f"{filename}", self._network_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                    except:
                        try:
                            tifffile.imwrite(f"{filename}", self._network_overlay)
                        except:
                            self._network_overlay = binarize(self._network_overlay)
                            tifffile.imwrite(f"{filename}", self._network_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                else:
                    tifffile.imwrite(f"{filename}", self._network_overlay)
                print(f"Network overlay saved to {filename}")

            if directory is not None:
                if len(self._network_overlay.shape) == 3:
                    try:
                        tifffile.imwrite(f"{directory}/{filename}", self._network_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                    except:
                        try:
                            tifffile.imwrite(f"{directory}/{filename}", self._network_overlay)
                        except:
                            self._network_overlay = binarize(self._network_overlay)
                            tifffile.imwrite(f"{directory}/{filename}", self._network_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                else:
                    tifffile.imwrite(f"{directory}/{filename}", self._network_overlay)
                print(f"Network overlay saved to {directory}/{filename}")

    def save_id_overlay(self, directory = None, filename = None):

        if self._id_overlay is not None:
            imagej_metadata = {
                'spacing': self.z_scale,
                'slices': self._id_overlay.shape[0],
                'channels': 1,
                'axes': 'ZYX'
            }
            resolution_value = 1.0 / self.xy_scale if self.xy_scale != 0 else 1

        if filename is None:
            filename = "overlay_2.tif"
        if not filename.endswith(('.tif', '.tiff')):
            filename += '.tif'

        if self._id_overlay is not None:
            if directory is None:
                if len(self._id_overlay.shape) == 3:
                    try:
                        tifffile.imwrite(f"{filename}", self._id_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                    except:
                        try:
                            tifffile.imwrite(f"{filename}", self._id_overlay)
                        except:                            
                            self._id_overlay = binarize(self._id_overlay)
                            tifffile.imwrite(f"{filename}", self._id_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                else:
                    tifffile.imwrite(f"{filename}", self._id_overlay, imagej=True)
                print(f"Network overlay saved to {filename}")

            if directory is not None:
                if len(self._id_overlay.shape) == 3:
                    try:
                        tifffile.imwrite(f"{directory}/{filename}", self._id_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                    except:
                        try:
                            tifffile.imwrite(f"{directory}/{filename}", self._id_overlay)
                        except:
                            self._id_overlay = binarize(self._id_overlay)
                            tifffile.imwrite(f"{directory}/{filename}", self._id_overlay, imagej=True, metadata=imagej_metadata, resolution=(resolution_value, resolution_value))
                else:
                    tifffile.imwrite(f"{directory}/{filename}", self._id_overlay)
                print(f"ID overlay saved to {directory}/{filename}")



    def dump(self, directory = None, parent_dir = None, name = None):
        """
        Can be called on a Network_3D object to save the all properties to hard mem. It will save to the active directory if none is specified.
        :param directory: (Optional - Val = None; String). The path to an intended directory to save the properties to.
        """

        directory = encapsulate(parent_dir = parent_dir, name = name)

        try:
            self.save_nodes(directory)
            self.save_edges(directory)
            self.save_node_centroids(directory)
            self.save_search_region(directory)
            self.save_network(directory)
            self.save_node_identities(directory)
            self.save_edge_centroids(directory)
            self.save_scaling(directory)
            self.save_communities(directory)
            self.save_network_overlay(directory)
            self.save_id_overlay(directory)

        except:
            self.save_nodes()
            self.save_edges()
            self.save_node_centroids()
            self.save_search_region()
            self.save_network()
            self.save_node_identities()
            self.save_edge_centroids()
            self.save_scaling()
            self.save_communities()
            self.save_network_overlay()
            self.save_id_overlay()


    def load_nodes(self, directory = None, file_path = None):
        """
        Can be called on a Network_3D object to load a tif into the nodes property as an ndarray. It will look for a file called 'labelled_nodes.tif' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any tiff file may be passed to load into the nodes property.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'labelled_nodes.tif' file.
        :param file_path: (Optional - Val = None; String). A path to any tif to load into the nodes property.
        """

        if file_path is not None:
            self._nodes = tifffile.imread(file_path)
            print("Succesfully loaded nodes")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'labelled_nodes.tif':
                if directory is not None:
                    self._nodes = tifffile.imread(f'{directory}/{item}')
                    print("Succesfully loaded nodes")
                    return
                else:
                    self._nodes = tifffile.imread(item)
                    print("Succesfully loaded nodes")
                    return


        print("Could not find nodes. They must be in the specified directory and named 'labelled_nodes.tif'")

    def load_edges(self, directory = None, file_path = None):

        """
        Can be called on a Network_3D object to load a tif into the edges property as an ndarray. It will look for a file called 'labelled_edges.tif' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any tiff file may be passed to load into the edges property.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'labelled_edges.tif' file.
        :param file_path: (Optional - Val = None; String). A path to any tif to load into the edges property.
        """

        if file_path is not None:
            self._edges = tifffile.imread(file_path)
            print("Succesfully loaded edges")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'labelled_edges.tif':
                if directory is not None:
                    self._edges = tifffile.imread(f'{directory}/{item}')
                    print("Succesfully loaded edges")
                    return
                else:
                    self._edges = tifffile.imread(item)
                    print("Succesfully loaded edges")
                    return

        print("Could not find edges. They must be in the specified directory and named 'labelled_edges.tif'")

    def load_scaling(self, directory = None, file_path = None):
        """
        Can be called on a Network_3D object to load a .txt into the xy_scale and z_scale properties as floats. It will look for a file called 'voxel_scalings.txt' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any txt file may be passed to load into the xy_scale/z_scale properties, however they must be formatted the same way as the 'voxel_scalings.txt' file.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'voxel_scalings.txt' file.
        :param file_path: (Optional - Val = None; String). A path to any txt to load into the xy_scale/z_scale properties.
        """
        def read_scalings(file_name):
            """Internal function for reading txt scalings"""
            # Initialize variables
            variable1 = 1
            variable2 = 1

            # Read the file and extract the variables
            with open(file_name, "r") as file:
                for line in file:
                    if "xy_scale:" in line:
                        variable1 = float(line.split(":")[1].strip())
                    elif "z_scale:" in line:
                        variable2 = float(line.split(":")[1].strip())

            return variable1, variable2

        if file_path is not None:
            self._xy_scale, self_z_scale = read_scalings(file_path)
            print(f"Succesfully loaded voxel_scalings; values overriden to xy_scale: {self.xy_scale}, z_scale: {self.z_scale}")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'voxel_scalings.txt':
                if directory is not None:
                    self._xy_scale, self._z_scale = read_scalings(f"{directory}/{item}")
                    print(f"Succesfully loaded voxel_scalings; values overriden to xy_scale: {self.xy_scale}, z_scale: {self.z_scale}")
                    return
                else:
                    self._xy_scale, self._z_scale = read_scalings(item)
                    print(f"Succesfully loaded voxel_scaling; values overriden to xy_scale: {self.xy_scale}, z_scale: {self.z_scale}s")
                    return

        print("Could not find voxel scalings. They must be in the specified directory and named 'voxel_scalings.txt'")

    def load_network(self, directory = None, file_path = None):
        """
        Can be called on a Network_3D object to load a .xlsx into the network and network_lists properties as a networx graph and a list of lists, respecitvely. It will look for a file called 'output_network.xlsx' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any .xlsx file may be passed to load into the network/network_lists properties, however they must be formatted the same way as the 'output_network.xlsx' file.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'output_network.xlsx' file.
        :param file_path: (Optional - Val = None; String). A path to any .xlsx to load into the network/network_lists properties.
        """
        if file_path is not None:
            self._network, net_weights = network_analysis.weighted_network(file_path)
            self._network_lists = network_analysis.read_excel_to_lists(file_path)
            print("Succesfully loaded network")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'output_network.xlsx' or item == 'output_network.csv':
                if directory is not None:
                    self._network, net_weights = network_analysis.weighted_network(f'{directory}/{item}')
                    self._network_lists = network_analysis.read_excel_to_lists(f'{directory}/{item}')
                    print("Succesfully loaded network")
                    return
                else:
                    self._network, net_weights = network_analysis.weighted_network(item)
                    self._network_lists = network_analysis.read_excel_to_lists(item)
                    print("Succesfully loaded network")
                    return

        print("Could not find network. It must be stored in specified directory and named 'output_network.xlsx' or 'output_network.csv'")

    def load_search_region(self, directory = None, file_path = None):

        """
        Can be called on a Network_3D object to load a tif into the search_region property as an ndarray. It will look for a file called 'search_region.tif' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any tiff file may be passed to load into the search_region property.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'search_region.tif' file.
        :param file_path: (Optional - Val = None; String). A path to any tif to load into the search_region property.
        """

        if file_path is not None:
            self._search_region = tifffile.imread(file_path)
            print("Succesfully loaded search regions")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'search_region.tif':
                if directory is not None:
                    self._search_region = tifffile.imread(f'{directory}/{item}')
                    print("Succesfully loaded search regions")
                    return
                else:
                    self._search_region = tifffile.imread(item)
                    print("Succesfully loaded search regions")
                    return

        print("Could not find search region. It must be in the specified directory and named 'search_region.tif'")

    def load_node_centroids(self, directory = None, file_path = None):
        """
        Can be called on a Network_3D object to load a .xlsx into the node_centroids property as a dictionary. It will look for a file called 'node_centroids.xlsx' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any .xlsx file may be passed to load into the node_centroids property, however they must be formatted the same way as the 'node_centroids.xlsx' file.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'node_centroids.xlsx' file.
        :param file_path: (Optional - Val = None; String). A path to any .xlsx to load into the node_centroids property.
        """

        if file_path is not None:
            self._node_centroids = network_analysis.read_centroids_to_dict(file_path)
            self._node_centroids = self.clear_null(self._node_centroids)
            print("Succesfully loaded node centroids")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'node_centroids.xlsx' or item == 'node_centroids.csv':
                if directory is not None:
                    self._node_centroids = network_analysis.read_centroids_to_dict(f'{directory}/{item}')
                    self._node_centroids = self.clear_null(self._node_centroids)
                    print("Succesfully loaded node centroids")
                    return
                else:
                    self._node_centroids = network_analysis.read_centroids_to_dict(item)
                    self._node_centroids = self.clear_null(self._node_centroids)
                    print("Succesfully loaded node centroids")
                    return

        print("Could not find node centroids. They must be in the specified directory and named 'node_centroids.xlsx'")

    def load_node_identities(self, directory = None, file_path = None):
        """
        Can be called on a Network_3D object to load a .xlsx into the node_identities property as a dictionary. It will look for a file called 'node_identities.xlsx' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any .xlsx file may be passed to load into the node_identities property, however they must be formatted the same way as the 'node_identities.xlsx' file.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'node_identities.xlsx' file.
        :param file_path: (Optional - Val = None; String). A path to any .xlsx to load into the node_identities property.
        """

        if file_path is not None:
            self._node_identities = network_analysis.read_excel_to_singval_dict(file_path)
            self._node_identities = self.clear_null(self._node_identities)
            print("Succesfully loaded node identities")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'node_identities.xlsx' or item == 'node_identities.csv':
                if directory is not None:
                    self._node_identities = network_analysis.read_excel_to_singval_dict(f'{directory}/{item}')
                    self._node_identities = self.clear_null(self._node_identities)
                    print("Succesfully loaded node identities")
                    return
                else:
                    self._node_identities = network_analysis.read_excel_to_singval_dict(item)
                    self._node_identities = self.clear_null(self._node_identities)
                    print("Succesfully loaded node identities")
                    return

        print("Could not find node identities. They must be in the specified directory and named 'node_identities.xlsx'")

    def load_communities(self, directory = None, file_path = None):
        """
        Can be called on a Network_3D object to load a .xlsx into the communities property as a dictionary. It will look for a file called 'node_communities.xlsx' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any .xlsx file may be passed to load into the node_communities property, however they must be formatted the same way as the 'node_communities.xlsx' file.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'node_identities.xlsx' file.
        :param file_path: (Optional - Val = None; String). A path to any .xlsx to load into the node_identities property.
        """

        if file_path is not None:
            self._communities = network_analysis.read_excel_to_singval_dict(file_path)
            self._communities = self.clear_null(self._communities)
            print("Succesfully loaded communities")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'node_communities.xlsx' or item == 'node_communities.csv':
                if directory is not None:
                    self._communities = network_analysis.read_excel_to_singval_dict(f'{directory}/{item}')
                    self._communities = self.clear_null(self._communities)
                    print("Succesfully loaded communities")
                    return
                else:
                    self._communities = network_analysis.read_excel_to_singval_dict(item)
                    self._communities = self.clear_null(self._communities)
                    print("Succesfully loaded communities")
                    return

        print("Could not find communities. They must be in the specified directory and named 'node_communities.xlsx'")

    def clear_null(self, some_dict):

        if some_dict == {}:
            some_dict = None
        return some_dict

    def load_edge_centroids(self, directory = None, file_path = None):
        """
        Can be called on a Network_3D object to load a .xlsx into the edge_centroids property as a dictionary. It will look for a file called 'edge_centroids.xlsx' in the specified directory,
        or the active directory if none has been selected. Alternatively, a file path to any .xlsx file may be passed to load into the edge_centroids property, however they must be formatted the same way as the 'edge_centroids.xlsx' file.
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the 'edge_centroids.xlsx' file.
        :param file_path: (Optional - Val = None; String). A path to any .xlsx to load into the edge_centroids property.
        """

        if file_path is not None:
            self._edge_centroids = network_analysis.read_centroids_to_dict(file_path)
            self._edge_centroids = self.clear_null(self._edge_centroids)
            print("Succesfully loaded edge centroids")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'edge_centroids.xlsx' or item == 'edge_centroids.csv':
                if directory is not None:
                    self._edge_centroids = network_analysis.read_centroids_to_dict(f'{directory}/{item}')
                    self._edge_centroids = self.clear_null(self._edge_centroids)
                    print("Succesfully loaded edge centroids")
                    return
                else:
                    self._edge_centroids = network_analysis.read_centroids_to_dict(item)
                    self._edge_centroids = self.clear_null(self._edge_centroids)
                    print("Succesfully loaded edge centroids")
                    return

        print("Could not find edge centroids. They must be in the specified directory and named 'edge_centroids.xlsx', or otherwise specified")


    def load_network_overlay(self, directory = None, file_path = None):


        if file_path is not None:
            self._network_overlay = tifffile.imread(file_path)
            print("Succesfully loaded network overlay")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'overlay_1.tif':
                if directory is not None:
                    self._network_overlay = tifffile.imread(f'{directory}/{item}')
                    print("Succesfully loaded network overlay")
                    return
                else:
                    self._network_overlay = tifffile.imread(item)
                    print("Succesfully loaded network overlay")
                    return


        #print("Could not find network overlay. They must be in the specified directory and named 'drawn_network.tif'")


    def load_id_overlay(self, directory = None, file_path = None):


        if file_path is not None:
            self._id_overlay = tifffile.imread(file_path)
            print("Succesfully loaded network overlay")
            return

        items = directory_info(directory)

        for item in items:
            if item == 'overlay_2.tif':
                if directory is not None:
                    self._id_overlay = tifffile.imread(f'{directory}/{item}')
                    print("Succesfully loaded id overlay")
                    return
                else:
                    self._id_overlay = tifffile.imread(item)
                    print("Succesfully loaded id overlay")
                    return


        #print("Could not find id overlay. They must be in the specified directory and named 'labelled_node_indices.tif'")


    def assemble(self, directory = None, node_path = None, edge_path = None, search_region_path = None, network_path = None, node_centroids_path = None, node_identities_path = None, edge_centroids_path = None, scaling_path = None, net_overlay_path = None, id_overlay_path = None, community_path = None ):
        """
        Can be called on a Network_3D object to load all properties simultaneously from a specified directory. It will look for files with the names specified in the property loading methods, in the active directory if none is specified.
        Alternatively, for each property a filepath to any file may be passed to look there to load. This method is intended to be used together with the dump method to easily save and load the Network_3D objects once they had been calculated. 
        :param directory: (Optional - Val = None; String). The path to an intended directory to search for the all property files.
        :param node_path: (Optional - Val = None; String). A path to any .tif to load into the nodes property.
        :param edge_path: (Optional - Val = None; String). A path to any .tif to load into the edges property.
        :param search_region_path: (Optional - Val = None; String). A path to any .tif to load into the search_region property.
        :param network_path: (Optional - Val = None; String). A path to any .xlsx file to load into the network and network_lists properties.
        :param node_centroids_path: (Optional - Val = None; String). A path to any .xlsx to load into the node_centroids property.
        :param node_identities_path: (Optional - Val = None; String). A path to any .xlsx to load into the node_identities property.
        :param edge_centroids_path: (Optional - Val = None; String). A path to any .xlsx to load into the edge_centroids property.
        :param scaling_path: (Optional - Val = None; String). A path to any .txt to load into the xy_scale and z_scale properties.
        """

        print(f"Assembling Network_3D object from files stored in directory: {directory}")
        self.load_nodes(directory, node_path)
        self.load_edges(directory, edge_path)
        #self.load_search_region(directory, search_region_path)
        self.load_network(directory, network_path)
        self.load_node_centroids(directory, node_centroids_path)
        self.load_node_identities(directory, node_identities_path)
        self.load_edge_centroids(directory, edge_centroids_path)
        self.load_scaling(directory, scaling_path)
        self.load_communities(directory, community_path)
        self.load_network_overlay(directory, net_overlay_path)
        self.load_id_overlay(directory, id_overlay_path)


    #Assembling additional Network_3D class attributes if they were not set when generating the network:

    def calculate_node_centroids(self, down_factor = None, GPU = False):

        """
        Method to obtain node centroids. Expects _nodes property to be set. Downsampling is optional to speed up the process. Centroids will be scaled to 'true' undownsampled location when downsampling is used.
        Sets the _node_centroids attribute.
        :param down_factor: (Optional - Val = None; int). Optional factor to downsample nodes during centroid calculation to increase speed.
        """

        if not hasattr(self, '_nodes') or self._nodes is None:
            print("Requires .nodes property to be set with a (preferably labelled) numpy array for node objects")
            raise AttributeError("._nodes property is not set")

        if not GPU:
            node_centroids = network_analysis._find_centroids(self._nodes, down_factor = down_factor)
        else:
            node_centroids = network_analysis._find_centroids_GPU(self._nodes, down_factor = down_factor)


        if down_factor is not None:
            for item in node_centroids:
                node_centroids[item] = node_centroids[item] * down_factor

        self._node_centroids = node_centroids

    def calculate_edge_centroids(self, down_factor = None):

        """
        Method to obtain edge centroids. Expects _edges property to be set. Downsampling is optional to speed up the process. Centroids will be scaled to 'true' undownsampled location when downsampling is used.
        Sets the _edge_centroids attribute.
        :param down_factor: (Optional - Val = None; int). Optional factor to downsample edges during centroid calculation to increase speed.
        """

        if not hasattr(self, '_edges') or self._edges is None:
            print("Requires .edges property to be set with a (preferably labelled) numpy array for node objects")
            raise AttributeError("._edges property is not set")


        edge_centroids = network_analysis._find_centroids(self._edges, down_factor = down_factor)

        if down_factor is not None:
            for item in edge_centroids:
                edge_centroids[item] = edge_centroids[item] * down_factor

        self._edge_centroids = edge_centroids

    def calculate_search_region(self, search_region_size, GPU = True, fast_dil = True, GPU_downsample = None):

        """
        Method to obtain the search region that will be used to assign connectivity between nodes. May be skipped if nodes do not want to search and only want to look for their 
        connections in their immediate overlap. Expects the nodes property to be set. Sets the search_region property.
        :param search_region_size: (Mandatory; int). Amount nodes should expand outward to search for connections. Note this value corresponds one-to-one with voxels unless voxel_scaling has been set, in which case it will correspond to whatever value the nodes array is measured in (microns, for example).
        :param GPU: (Optional - Val = True; boolean). Will use GPU if avaialble (including necessary downsampling for GPU RAM). Set to False to use CPU with no downsample.
        :param fast_dil: (Optional - Val = False, boolean) - A boolean that when True will utilize faster cube dilation but when false will use slower spheroid dilation.

        """

        dilate_xy, dilate_z = dilation_length_to_pixels(self._xy_scale, self._z_scale, search_region_size, search_region_size) #Get true dilation sizes based on voxel scaling and search region.

        if not hasattr(self, '_nodes') or self._nodes is None:
            print("Requires .nodes property to be set with a (preferably labelled) numpy array for node objects")
            raise AttributeError("._nodes property is not set")

        if search_region_size != 0:

            self._search_region = smart_dilate.smart_dilate(self._nodes, dilate_xy, dilate_z, GPU = GPU, fast_dil = fast_dil, predownsample = GPU_downsample, use_dt_dil_amount = search_region_size) #Call the smart dilate function which essentially is a fast way to enlarge nodes into a 'search region' while keeping their unique IDs.

        else:

            self._search_region = self._nodes

    def calculate_edges(self, binary_edges, diledge = None, inners = True, search = None, remove_edgetrunk = 0, GPU = True, fast_dil = False, skeletonized = False):
        """
        Method to calculate the edges that are used to directly connect nodes. May be done with or without the search region, however using search_region is recommended. 
        The search_region property must be set to use the search region, otherwise the nodes property must be set. Sets the edges property
        :param binary_edges: (Mandatory; String or ndarray). Filepath to a binary tif containing segmented edges, or a numpy array of the same. 
        :param diledge: (Optional - Val = None; int). Amount to dilate edges, to account for imaging and segmentation artifacts that have brokenup edges. Any edge masks that are within half the value of the 'diledge' param will become connected. Ideally edges should not have gaps,
        so some amount of dilation is recommended if there are any, but  not so much to create overconnectivity. This is a value that needs to be tuned by the user.
        :param inners: (Optional - Val = True; boolean). Will use inner edges if True, will not if False. Inner edges are parts of the edge mask that exist within search regions. If search regions overlap, 
        any edges that exist within the overlap will only assert connectivity if 'inners' is True.
        If True, an extra processing step is used to sort the correct connectivity amongst these search_regions. Can only be computed when search_regions property is set.
        :param search: (Optional - Val = None; int). Amount for nodes to search for connections, assuming the search_regions are not being used. Assigning a value to this param will utilize the secondary algorithm and not the search_regions.
        :param remove_edgetrunk: (Optional - Val = 0; int). Amount of times to remove the 'Trunk' from the edges. A trunk in this case is the largest (by vol) edge object remaining after nodes have broken up the edges.
        Any 'Trunks' removed will be absent for connection calculations.
        :param GPU: (Optional - Val = True; boolean). Will use GPU (if available) for the hash_inner_edges step if True, if False will use CPU. Note that the speed is comparable either way.
        :param skeletonize: (Optional - Val = False, boolean) - A boolean of whether to skeletonize the edges when using them.
        """
        if not hasattr(self, '_search_region') or self._search_region is None:
            if not hasattr(self, '_nodes') or self._nodes is None:
                print("Requires .search_region property to be set with a (preferably labelled) numpy array for node search regions, or nodes property to be set and method to be passed a 'search = 'some float'' arg")
                raise AttributeError("._search_region/_nodes property is not set")

        if type(binary_edges) == str:
            binary_edges = tifffile.imread(binary_edges)

        if skeletonized:
            binary_edges = skeletonize(binary_edges)

        if search is not None and hasattr(self, '_nodes') and self._nodes is not None and self._search_region is None:
            search_region = binarize(self._nodes)
            dilate_xy, dilate_z = dilation_length_to_pixels(self._xy_scale, self._z_scale, search, search)
            search_region = dilate_3D_dt(search_region, diledge, self._xy_scale, self._z_scale, fast_dil = fast_dil)
        else:
            search_region = binarize(self._search_region)

        outer_edges = establish_edges(search_region, binary_edges)

        if not inners:
            del binary_edges

        if remove_edgetrunk > 0:
            print(f"Snipping trunks...")
            outer_edges = remove_trunk(outer_edges, remove_edgetrunk)

        if diledge is not None:
            dilate_xy, dilate_z = dilation_length_to_pixels(self._xy_scale, self._z_scale, diledge, diledge)

            if dilate_xy <= 3 and dilate_z <= 3:
                outer_edges = dilate_3D_old(outer_edges, dilate_xy, dilate_xy, dilate_z)
            else:
                outer_edges = dilate_3D_dt(outer_edges, diledge, self._xy_scale, self._z_scale, fast_dil = fast_dil)
        else:
            outer_edges = dilate_3D_old(outer_edges)

        labelled_edges, num_edge = label_objects(outer_edges)

        if inners:
            inner_edges = hash_inners(self._search_region, binary_edges, GPU = GPU)

            del binary_edges

            inner_labels, num_edge = label_objects(inner_edges)

            del inner_edges

            labelled_edges = combine_edges(labelled_edges, inner_labels)

            num_edge = np.max(labelled_edges)

            if num_edge < 256:
                labelled_edges = labelled_edges.astype(np.uint8)
            elif num_edge < 65536:
                labelled_edges = labelled_edges.astype(np.uint16)

        self._edges = labelled_edges

    def label_nodes(self):
        """
        Method to assign a unique numerical label to all discrete objects contained in the ndarray in the nodes property.
        Expects the nodes property to be set to (presumably) a binary ndarray. Sets the nodes property.
        """
        self._nodes, num_nodes = label_objects(nodes, structure_3d)

    def combine_nodes(self, root_nodes, other_nodes, other_ID, identity_dict, root_ID = None, centroids = False, down_factor = None):

        """Internal method to merge two labelled node arrays into one"""

        print("Combining node arrays")

        mask = (root_nodes == 0) & (other_nodes > 0)
        if np.any(mask):
            max_val = np.max(root_nodes)
            other_nodes[:] = np.where(mask, other_nodes + max_val, 0)
        if centroids:
            new_dict = network_analysis._find_centroids(other_nodes, down_factor = down_factor)
            if down_factor is not None:
                for item in new_dict:
                    new_dict[item] = down_factor * new_dict[item]
            self.node_centroids.update(new_dict)

        if root_ID is not None:
            rootIDs = list(np.unique(root_nodes)) #Sets up adding these vals to the identitiy dictionary. Gets skipped if this has already been done.

            if rootIDs[0] == 0: #np unique can include 0 which we don't want.
                del rootIDs[0]

        otherIDs = list(np.unique(other_nodes)) #Sets up adding other vals to the identity dictionary.

        if otherIDs[0] == 0:
            del otherIDs[0]

        if root_ID is not None: #Adds the root vals to the dictionary if it hasn't already

            if other_ID.endswith('.tiff'):
                other_ID = other_ID[:-5]
            elif other_ID.endswith('.tif'):
                other_ID = other_ID[:-4]

            for item in rootIDs:
                identity_dict[item] = root_ID

        for item in otherIDs: #Always adds the other vals to the dictionary
            try:
                other_ID = os.path.basename(other_ID)
            except:
                pass
            if other_ID.endswith('.tiff'):
                other_ID = other_ID[:-5]
            elif other_ID.endswith('.tif'):
                other_ID = other_ID[:-4]

            identity_dict[item] = other_ID

        nodes = root_nodes + other_nodes #Combine the outer edges with the inner edges modified via the above steps

        return nodes, identity_dict

    def merge_nodes(self, addn_nodes_name, label_nodes = True, root_id = "Root_Nodes", centroids = False, down_factor = None):
        """
        Merges the self._nodes attribute with alternate labelled node images. The alternate nodes can be inputted as a string for a filepath to a tif,
        or as a directory address containing only tif images, which will merge the _nodes attribute with all tifs in the folder. The _node_identities attribute
        meanwhile will keep track of which labels in the merged array refer to which objects, letting user track multiple seperate biological objects
        in a single network. Note that an optional param, 'label_nodes' is set to 'True' by default. This will cause the program to label any intended
        additional nodes based on seperation in the image. If your nodes a prelabelled, please input the argument 'label_nodes = False'
        :param addn_nodes_name: (Mandatory; String). Path to either a tif file or a directory containing only additional node files.
        :param label_nodes: (Optional - Val = True; Boolean). Will label all discrete objects in each node file being merged if True. If False, will not label.
        """

        nodes_name = root_id

        try:
            nodes_name = os.path.splitext(os.path.basename(nodes_name))[0]
        except:
            pass
            
        identity_dict = {} #A dictionary to deliniate the node identities

        if centroids:
            self.node_centroids = network_analysis._find_centroids(self._nodes, down_factor = down_factor)
            if down_factor is not None:
                for item in self.node_centroids:
                    self.node_centroids[item] = down_factor * self.node_centroids[item]

        try: #Try presumes the input is a tif
            addn_nodes = tifffile.imread(addn_nodes_name) #If not this will fail and activate the except block

            if label_nodes is True:
                addn_nodes, num_nodes2 = label_objects(addn_nodes) # Label the node objects. Note this presumes no overlap between node masks.
                node_labels, identity_dict = self.combine_nodes(self._nodes, addn_nodes, addn_nodes_name, identity_dict, nodes_name, centroids = centroids, down_factor = down_factor) #This method stacks labelled arrays
                num_nodes = np.max(node_labels)

            else: #If nodes already labelled
                node_labels, identity_dict = self.combine_nodes(self._nodes, addn_nodes, addn_nodes_name, identity_dict, nodes_name, centroids = centroids, down_factor = down_factor)
                num_nodes = int(np.max(node_labels))

        except: #Exception presumes the input is a directory containing multiple tifs, to allow multi-node stackage.

            addn_nodes_list = directory_info(addn_nodes_name)

            for i, addn_nodes in enumerate(addn_nodes_list):
                try:
                    addn_nodes_ID = addn_nodes
                    try:
                        addn_nodes = tifffile.imread(f'{addn_nodes_name}/{addn_nodes}')
                    except:
                        continue

                    if label_nodes is True:
                        addn_nodes, num_nodes2 = label_objects(addn_nodes)  # Label the node objects. Note this presumes no overlap between node masks.
                        if i == 0:
                            node_labels, identity_dict = self.combine_nodes(self._nodes, addn_nodes, addn_nodes_ID, identity_dict, nodes_name, centroids = centroids, down_factor = down_factor)
                        else:
                            node_labels, identity_dict = self.combine_nodes(node_labels, addn_nodes, addn_nodes_ID, identity_dict, centroids = centroids, down_factor = down_factor)

                    else:
                        if i == 0:
                            node_labels, identity_dict = self.combine_nodes(self._nodes, addn_nodes, addn_nodes_ID, identity_dict, nodes_name, centroids = centroids, down_factor = down_factor)
                        else:
                            node_labels, identity_dict = self.combine_nodes(node_labels, addn_nodes, addn_nodes_ID, identity_dict, centroids = centroids, down_factor = down_factor)
                except Exception as e:
                    print("Could not open additional nodes, verify they are being inputted correctly...")

        num_nodes = int(np.max(node_labels))

        self._node_identities = identity_dict

        if num_nodes < 256:
            dtype = np.uint8
        elif num_nodes < 65536:
            dtype = np.uint16
        else:
            dtype = np.uint32

        # Convert the labeled array to the chosen data type
        node_labels = node_labels.astype(dtype)

        self._nodes = node_labels

    def calculate_network(self, search = None, ignore_search_region = False):

        """
        Method to calculate the network from the labelled nodes and edge properties, once they have been calculated. Network connections are assigned based on node overlap along
        the same edge of some particular label. Sets the network and network_lists properties.
        :param search: (Optional - Val = None; Int). Amount for nodes to search for connections if not using the search_regions to find connections.
        :param ignore_search_region: (Optional - Val = False; Boolean). If False, will use primary algorithm (with search_regions property) to find connections. If True, will use secondary algorithm (with nodes) to find connections.
        """

        if not ignore_search_region and hasattr(self, '_search_region') and self._search_region is not None and hasattr(self, '_edges') and self._edges is not None:
            num_edge_1 = np.max(self._edges)
            edge_labels, trim_node_labels = array_trim(self._edges, self._search_region)
            connections_parallel = establish_connections_parallel(edge_labels, num_edge_1, trim_node_labels)
            del edge_labels
            connections_parallel = extract_pairwise_connections(connections_parallel)
            df = create_and_save_dataframe(connections_parallel)
            self._network_lists = network_analysis.read_excel_to_lists(df)
            self._network, net_weights = network_analysis.weighted_network(df)

        if ignore_search_region and hasattr(self, '_edges') and self._edges is not None and hasattr(self, '_nodes') and self._nodes is not None:
            #dilate_xy, dilate_z = dilation_length_to_pixels(self._xy_scale, self._z_scale, search, search)
            #print(f"{dilate_xy}, {dilate_z}")
            num_nodes = np.max(self._nodes)
            #connections_parallel = create_node_dictionary(self._nodes, self._edges, num_nodes, dilate_xy, dilate_z) #Find which edges connect which nodes and put them in a dictionary.
            connections_parallel = create_node_dictionary(self._nodes, self._edges, num_nodes, 3, 3) #For now I only implement this for immediate neighbor search so we'll just use 3 and 3 here.
            connections_parallel = find_shared_value_pairs(connections_parallel) #Sort through the dictionary to find connected node pairs.
            df = create_and_save_dataframe(connections_parallel)
            self._network_lists = network_analysis.read_excel_to_lists(df)
            self._network, net_weights = network_analysis.weighted_network(df)

    def create_id_network(self, n=5):
        import ast
        import random
        
        if self.node_identities is None:
            return
        
        def invert_dict(d):
            inverted = {}
            for key, value in d.items():
                inverted.setdefault(value, []).append(key)
            return inverted
        
        # Invert to get identity -> list of nodes
        identity_to_nodes = invert_dict(self.node_identities)
        
        G = nx.Graph()
        edge_set = set()
        
        # Step 1: Connect nodes within same exact identity
        for identity, nodes in identity_to_nodes.items():
            if len(nodes) <= 1:
                continue
            
            # Each node chooses n random neighbors from its identity group
            for node in nodes:
                available = [other for other in nodes if other != node]
                num_to_choose = min(n, len(available))
                neighbors = random.sample(available, num_to_choose)
                
                for neighbor in neighbors:
                    edge = tuple(sorted([node, neighbor]))
                    edge_set.add(edge)
        
        # Step 2: For list-like identities, connect across groups with shared sub-identities
        for identity, nodes in identity_to_nodes.items():
            if identity.startswith('['):
                try:
                    sub_identities = ast.literal_eval(identity)
                    
                    # For each sub-identity in this list-like identity
                    for sub_id in sub_identities:
                        # Find all OTHER identity groups that contain this sub-identity
                        for other_identity, other_nodes in identity_to_nodes.items():
                            if other_identity == identity:
                                continue  # Skip connecting to same exact identity (already done in Step 1)
                            
                            # Check if other_identity contains sub_id
                            contains_sub_id = False
                            
                            if other_identity.startswith('['):
                                try:
                                    other_sub_ids = ast.literal_eval(other_identity)
                                    if sub_id in other_sub_ids:
                                        contains_sub_id = True
                                except (ValueError, SyntaxError):
                                    pass
                            elif other_identity == sub_id:
                                # Single identity that matches our sub-identity
                                contains_sub_id = True
                            
                            if contains_sub_id:
                                # Each node from current identity connects to n nodes from other_identity
                                for node in nodes:
                                    num_to_choose = min(n, len(other_nodes))
                                    if num_to_choose > 0:
                                        neighbors = random.sample(other_nodes, num_to_choose)
                                        
                                        for neighbor in neighbors:
                                            edge = tuple(sorted([node, neighbor]))
                                            edge_set.add(edge)
                
                except (ValueError, SyntaxError):
                    pass  # Not a valid list, treat as already handled in Step 1
        
        G.add_edges_from(edge_set)
        self.network = G





    def calculate_all(self, nodes, edges, xy_scale = 1, z_scale = 1, down_factor = None, search = None, diledge = None, inners = True, remove_trunk = 0, ignore_search_region = False, other_nodes = None, label_nodes = True, directory = None, GPU = True, fast_dil = True, skeletonize = False, GPU_downsample = None):
        """
        Method to calculate and save to mem all properties of a Network_3D object. In general, after initializing a Network_3D object, this method should be called on the node and edge masks that will be used to calculate the network.
        :param nodes: (Mandatory; String or ndarray). Filepath to segmented nodes mask or a numpy array containing the same.
        :param edges: (Mandatory; String or ndarray). Filepath to segmented edges mask or a numpy array containing the same.
        :param xy_scale: (Optional - Val = 1; Float). Pixel scaling to convert pixel sizes to some real value (such as microns).
        :param z_scale: (Optional - Val = 1; Float). Voxel depth to convert voxel depths to some real value (such as microns).
        :param down_factor: (Optional - Val = None; int). Optional factor to downsample nodes and edges during centroid calculation to increase speed. Note this only applies to centroid calculation and that the outputed centroids will correspond to the full-sized file. On-line general downsampling is not supported by this method and should be computed on masks before inputting them.
        :param search: (Optional - Val = None; int). Amount nodes should expand outward to search for connections. Note this value corresponds one-to-one with voxels unless voxel_scaling has been set, in which case it will correspond to whatever value the nodes array is measured in (microns, for example). If unset, only directly overlapping nodes and edges will find connections.
        :param diledge: (Optional - Val = None; int). Amount to dilate edges, to account for imaging and segmentation artifacts that have brokenup edges. Any edge masks that are within half the value of the 'diledge' param will become connected. Ideally edges should not have gaps,
        so some amount of dilation is recommended if there are any, but not so much to create overconnectivity. This is a value that needs to be tuned by the user.
        :param inners: (Optional - Val = True; boolean). Will use inner edges if True, will not if False. Inner edges are parts of the edge mask that exist within search regions. If search regions overlap, 
        any edges that exist within the overlap will only assert connectivity if 'inners' is True.
        If True, an extra processing step is used to sort the correct connectivity amongst these search_regions. Can only be computed when search_regions property is set.
        :param remove_trunk: (Optional - Val = 0; int). Amount of times to remove the 'Trunk' from the edges. A trunk in this case is the largest (by vol) edge object remaining after nodes have broken up the edges.
        Any 'Trunks' removed will be absent for connection calculations.
        :param ignore_search_region: (Optional - Val = False; boolean). If False, will use primary algorithm (with search_regions property) to find connections. If True, will use secondary algorithm (with nodes) to find connections.
        :param other_nodes: (Optional - Val = None; string). Path to either a tif file or a directory containing only additional node files to merge with the original nodes, assuming multiple 'types' of nodes need comparing. Node identities will be retained.
        :param label_nodes: (Optional - Val = True; boolean). If True, all discrete objects in the node param (and all those contained in the optional other_nodes param) will be assigned a label. If files a prelabelled, set this to False to avoid labelling.
        :param directory: (Optional - Val = None; string). Path to a directory to save to hard mem all Network_3D properties. If not set, these values will be saved to the active directory.
        :param GPU: (Optional - Val = True; boolean). Will use GPU if avaialble for calculating the search_region step (including necessary downsampling for GPU RAM). Set to False to use CPU with no downsample. Note this only affects the search_region step.
        :param fast_dil: (Optional - Val = False, boolean) - A boolean that when True will utilize faster psuedo3d kernel dilation but when false will use slower dt-based dilation.
        :param skeletonize: (Optional - Val = False, boolean) - A boolean of whether to skeletonize the edges when using them.
        """

        if directory is not None:
            directory = encapsulate()

        self._xy_scale = xy_scale
        self._z_scale = z_scale

        if directory is not None:
            try:
                self.save_scaling(directory)
            except:
                pass

        if search is None and ignore_search_region == False:
            search = 0

        if type(nodes) == str:
            nodes = tifffile.imread(nodes)

        self._nodes = nodes
        del nodes

        if label_nodes:
            self._nodes, num_nodes = label_objects(self._nodes)
        if other_nodes is not None:
            self.merge_nodes(other_nodes, label_nodes)

        if directory is not None:
            try:
                self.save_nodes(directory)
            except:
                pass
            try:
                self.save_node_identities(directory)
            except:
                pass

        if not ignore_search_region:
            self.calculate_search_region(search, GPU = GPU, fast_dil = fast_dil, GPU_downsample = GPU_downsample)
            #self._nodes = None # I originally put this here to micromanage RAM a little bit (it writes it to disk so I wanted to purge it from mem briefly but now idt thats necessary and I'd rather give it flexibility when lacking write permissions)
            search = None
            if directory is not None:
                try:
                    self.save_search_region(directory)
                except:
                    pass

            self.calculate_edges(edges, diledge = diledge, inners = inners, search = search, remove_edgetrunk = remove_trunk, GPU = GPU, fast_dil = fast_dil, skeletonized = skeletonize) #Will have to be moved out if the second method becomes more directly implemented
        else:
            self._edges, _ = label_objects(edges)

        del edges
        if directory is not None:
            try:
                self.save_edges(directory)
            except:
                pass

        self.calculate_network(search = search, ignore_search_region = ignore_search_region)

        if directory is not None:
            try:
                self.save_network(directory)
            except:
                pass

        if self._nodes is None:
            self.load_nodes(directory)

        self.calculate_node_centroids(down_factor)
        if directory is not None:
            try:
                self.save_node_centroids(directory)
            except:
                pass
        self.calculate_edge_centroids(down_factor)
        if directory is not None:
            try:
                self.save_edge_centroids(directory)
            except:
                pass


    def draw_network(self, directory = None, down_factor = None, GPU = False):
        """
        Method that draws the 3D network lattice for a Network_3D object, to be used as an overlay for viewing network connections. 
        Lattice will be saved as a .tif to the active directory if none is specified. Will used the node_centroids property for faster computation, unless passed the down_factor param.
        :param directory: (Optional - Val = None; string). Path to a directory to save the network lattice to.
        :param down_factor: (Optional - Val = None; int).  Factor to downsample nodes by for calculating centroids. The node_centroids property will be used if this value is not set. If there are no node_centroids, this value must be set (to 1 or higher).
        """

        if down_factor is not None:
            nodes = downsample(self._nodes, down_factor)
            centroids = self._node_centroids.copy()

            for item in centroids:
                centroids[item] = np.round(centroids[item]/down_factor)

            output = network_draw.draw_network_from_centroids(nodes, self._network_lists, centroids, twod_bool = False, directory = directory)

        else:

            if not GPU:
                output = network_draw.draw_network_from_centroids(self._nodes, self._network_lists, self._node_centroids, twod_bool = False, directory = directory)
            else:
                output = network_draw.draw_network_from_centroids_GPU(self._nodes, self._network_lists, self._node_centroids, twod_bool = False, directory = directory)

        return output        

    def draw_node_indices(self, directory = None, down_factor = None):
        """
        Method that draws the numerical IDs for nodes in a Network_3D object, to be used as an overlay for viewing node IDs. 
        IDs will be saved as a .tif to the active directory if none is specified. Will used the node_centroids property for faster computation, unless passed the down_factor param.
        :param directory: (Optional - Val = None; string). Path to a directory to save the node_indicies to.
        :param down_factor: (Optional - Val = None; int).  Factor to downsample nodes by for calculating centroids. The node_centroids property will be used if this value is not set. If there are no node_centroids, this value must be set (to 1 or higher).
        """

        num_nodes = np.max(self._nodes)

        if down_factor is not None:
            nodes = downsample(self._nodes, down_factor)
            centroids = self._node_centroids.copy()

            for item in centroids:
                centroids[item] = np.round(centroids[item]/down_factor)

            output = node_draw.draw_from_centroids(nodes, num_nodes, centroids, twod_bool = False, directory = directory)

        else:

            output = node_draw.draw_from_centroids(self._nodes, num_nodes, self._node_centroids, twod_bool = False, directory = directory)

        return output

    def draw_edge_indices(self, directory = None, down_factor = None):
        """
        Method that draws the numerical IDs for edges in a Network_3D object, to be used as an overlay for viewing edge IDs. 
        IDs will be saved as a .tif to the active directory if none is specified. Will used the edge_centroids property for faster computation, unless passed the down_factor param.
        :param directory: (Optional - Val = None; string). Path to a directory to save the edge indices to.
        :param down_factor: (Optional - Val = None; int).  Factor to downsample edges by for calculating centroids. The edge_centroids property will be used if this value is not set. If there are no edgde_centroids, this value must be set (to 1 or higher).
        """

        num_edge = np.max(self._edges)

        if down_factor is not None:
            edges = downsample(self._edges, down_factor)
            centroids = self._edge_centroids.copy()

            for item in centroids:
                centroids[item] = np.round(centroids[item]/down_factor)

            output = node_draw.draw_from_centroids(edges, num_edge, centroids, twod_bool = False, directory = directory)

        else:

            output = node_draw.draw_from_centroids(self._edges, num_edge, self._edge_centroids, twod_bool = False, directory = directory)

        return output



    #Some methods that may be useful:

    def community_partition(self, weighted = False, style = 0, dostats = True, seed = 42):
        """
        Sets the communities attribute by splitting the network into communities
        """

        self._communities, self.normalized_weights, stats = modularity.community_partition(self._network_lists, weighted = weighted, style = style, dostats = dostats, seed = seed)

        return stats

    def remove_edge_weights(self):
        """
        Remove the weights from a network. Requires _network object to be calculated. Removes duplicates from network_list and removes weights from any network object.
        Note that by default, ALL nodes that have duplicate connections through alternative edges will have a network with weights that correspond to the number of
        these connections. This will effect some networkx calculations. This method may be called on a Network_3D object to eliminate these weights, assuming only discrete connections are wanted for analysis. 
        """

        self._network_lists = network_analysis.remove_dupes(self._network_lists)

        self._network = network_analysis.open_network(self._network_lists)


    def rescale(self, array, directory = None):
        """
        Scale a downsampled overlay or extracted image object back to the size that is present in either a Network_3D's node or edge properties.
        This will allow a user to create downsampled outputs to speed up certain methods when analyzing Network_3D objects, but then scale them back to the proper size of that corresponding object.
        This will be saved to the active directory if none is specified.
        :param array: (Mandatory; string or ndarray). A path to the .tif file to be rescaled, or an numpy array of the same.
        :param directory: (Optional - Val = None; string). A path to a directory to save the rescaled output. 
        """

        if type(array) == str:
            array_name = os.path.basename(array)

        if directory is not None and type(array) == str:
            filename = f'{directory}/rescaled.tif'
        elif directory is None and type(array) == str:
            filename = f'rescaled.tif'
        elif directory is not None and type(array) != str:
            filename = f"{directory}/rescaled_array.tif"
        elif directory is None and type(array) != str:
            filename = "rescaled_array.tif"

        if type(array) == str:
            array = tifffile.imread(array)

        targ_shape = self._nodes.shape

        factor = round(targ_shape[0]/array.shape[0])

        array = upsample_with_padding(array, factor, targ_shape)

        tifffile.imsave(filename, array)
        print(f"Rescaled array saved to {filename}")

    def edge_to_node(self):
        """
        Converts all edge objects to node objects. Oftentimes, one may wonder how nodes are connected by edges in a network. Converting nodes to edges permits this visualization.
        Essentially a nodepair A-B will be reassigned as A-EdgeC and B-EdgeC.
        Alters the network and network_lists properties to absorb all edges. Edge IDs are altered to not overlap preexisting node IDs. Alters the edges property so that labels correspond
        to new edge IDs. Alters (or sets, if none exists) the node_identities property to keep track of which new nodes are 'edges'. Alters node_centroids property to now contain edge_centroids.
        """

        print("Converting all edge objects to nodes...")

        if self._nodes is not None:
            max_node = np.max(self._nodes)
        else:
            max_node = None

        df, identity_dict, max_node = network_analysis.edge_to_node(self._network_lists, self._node_identities, maxnode = max_node)

        self._network_lists = network_analysis.read_excel_to_lists(df)
        self._network, net_weights = network_analysis.weighted_network(df)
        self._node_identities = identity_dict

        print("Reassigning edge centroids to node centroids (requires both edge_centroids and node_centroids attributes to be present)")

        try:

            new_centroids = {}
            for item in self._edge_centroids:
                new_centroids[item + max_node] = self._edge_centroids[item]
            self._edge_centroids = new_centroids
            self._node_centroids = self._edge_centroids | self._node_centroids

        except Exception as e:
            print("Could not update edge/node centroids. They were likely not precomputed as object attributes. This may cause errors when drawing elements from the merged edge/node array...")

        print("Relabelling self.edge array...")

        num_edge = np.max(self._edges)

        edge_bools = self._edges > 0

        self._edges = self._edges.astype(np.uint32)

        self._edges = self._edges + max_node

        self._edges = self._edges * edge_bools

        if num_edge < 256:
            self._edges = self._edges.astype(np.uint8)
        elif num_edge < 65536:
            self._edges = self._edges.astype(np.uint16)

        node_bools = self._nodes == 0

        self._nodes = self._nodes.astype(np.uint32)
        self._edges = self._edges * node_bools
        self._nodes = self._nodes + self._edges
        num_node = np.max(self._nodes)

        if num_node < 256:
            self._nodes = self._nodes.astype(np.uint8)
        elif num_node < 65536:
            self._nodes = self._nodes.astype(np.uint16)


    def com_by_size(self):
        """Reassign communities based on size, starting with 1 for largest."""

        from collections import Counter
        
        # Convert all community values to regular ints (handles numpy scalars)
        clean_communities = {
            node: comm.item() if hasattr(comm, 'item') else comm 
            for node, comm in self.communities.items()
        }
        
        # Count community sizes and create mapping in one go
        community_sizes = Counter(clean_communities.values())
        
        # Create old->new mapping: sort by size (desc), then by community ID for ties
        old_to_new = {
            old_comm: new_comm 
            for new_comm, (old_comm, _) in enumerate(
                sorted(community_sizes.items(), key=lambda x: (-x[1], x[0])), 
                start=1
            )
        }
        
        # Apply mapping
        self.communities = {
            node: old_to_new[comm] 
            for node, comm in clean_communities.items()
        }







    def com_to_node(self, targets = None):


        def update_array(array_3d, value_dict, targets = None):
            ref_array = copy.deepcopy(array_3d)
            if targets is None:
                for key, value_list in value_dict.items():
                    for value in value_list:
                        array_3d[ref_array == value] = key
            else:
                max_val = np.max(array_3d) + 1
                for key, value_list in value_dict.items():
                    for value in value_list:
                        array_3d[ref_array == value] = max_val
                    max_val += 1

            return array_3d

        if 0 in self.communities.values():
            self.communities = {k: v + 1 for k, v in self.communities.items()} 
            if targets is not None:
                for item in targets:
                    item = item + 1

        inverted = invert_dict(self.communities)


        if targets is not None:
            new_inverted = copy.deepcopy(inverted)
            for com in inverted:
                if com not in targets:
                    del new_inverted[com]
            inverted = new_inverted


        if self._node_identities is not None:
            new_identities = {}
            for com in inverted:
                new_identities[com] = ""

        list1 = self._network_lists[0] #Get network lists to change
        list2 = self._network_lists[1]
        list3 = self._network_lists[2]
        return1 = []
        return2 = []
        return3 = []

        for i in range(len(list1)):
            list1[i] = self.communities[list1[i]] #Set node at network list spot to its community instead
            list2[i] = self.communities[list2[i]]
            if list1[i] != list2[i]: #Avoid self - self connections
                return1.append(list1[i])
                return2.append(list2[i])
                return3.append(list3[i])



        self.network_lists = [return1, return2, return3]

        if self._nodes is not None:
            self._nodes = update_array(self._nodes, inverted, targets = targets) #Set the array to match the new network

        try:

            if self._node_identities is not None:

                for key, value_list in inverted.items():
                    temp_dict = {}
                    for value in value_list:
                        if self._node_identities[value] in temp_dict:
                            temp_dict[self._node_identities[value]] += 1
                        else:
                            temp_dict[self._node_identities[value]] = 1
                    for id_type, num in temp_dict.items():
                        new_identities[key] += f'ID {id_type}:{num}, '

                self.node_identities = new_identities
        except:
            pass








    def trunk_to_node(self):
        """
        Converts the edge 'trunk' into a node. In this case, the trunk is the edge that creates the most node-node connections. There may be times when many nodes are connected by a single, expansive edge that obfuscates the rest of the edges. Converting the trunk to a node can better reveal these edges.
        Essentially a nodepair A-B that is connected via the trunk will be reassigned as A-Trunk and B-Trunk.
        Alters the network and network_lists properties to absorb the Trunk. Alters (or sets, if none exists) the node_identities property to keep track of which new nodes is a 'Trunk'.
        """

        nodesa = self._network_lists[0]
        nodesb = self._network_lists[1]
        edgesc = self._network_lists[2]
        nodea = []
        nodeb = []
        edgec = []

        from collections import Counter
        counts = Counter(edgesc)
        if 0 not in edgesc:
            trunk = stats.mode(edgesc)
        else:
            try:
                sorted_edges = counts.most_common()
                trunk = sorted_edges[1][0]
            except:
                return

        addtrunk = max(set(nodesa + nodesb)) + 1

        for i in range(len(nodesa)):
            if edgesc[i] == trunk:
                nodea.append(nodesa[i])
                nodeb.append(addtrunk)
                nodea.append(nodesb[i])
                nodeb.append(addtrunk)
                edgec.append(0)
                edgec.append(0)
            else:
                nodea.append(nodesa[i])
                nodeb.append(nodesb[i])
                edgec.append(edgesc[i])

        self.network_lists = [nodea, nodeb, edgec]

        try:
            self._node_centroids[addtrunk] = self._edge_centroids[trunk]
        except:
            pass

        if self._node_identities is None:
            self._node_identities = {}
            nodes = list(set(nodea + nodeb))
            for item in nodes:
                if item == addtrunk:
                    self._node_identities[item] = "Trunk"
                else:
                    self._node_identities[item] = "Node"
        else:
            self._node_identities[addtrunk] = "Trunk"

        if self._edges is not None and self._nodes is not None:

            node_bools = self._nodes == 0

            trunk = self._edges == trunk

            trunk = trunk * addtrunk

            trunk = trunk * node_bools

            self._nodes = self._nodes + trunk





    def prune_samenode_connections(self):
        """
        If working with a network that has multiple node identities (from merging nodes or otherwise manipulating this property),
        this method will remove from the network and network_lists properties any connections that exist between the same node identity,
        in case we want to investigate only connections between differing objects.
        """

        self._network_lists, self._node_identities = network_analysis.prune_samenode_connections(self._network_lists, self._node_identities)
        self._network, num_weights = network_analysis.weighted_network(self._network_lists)


    def isolate_internode_connections(self, ID1, ID2):
        """
        If working with a network that has at least three node identities (from merging nodes or otherwise manipulating this property),
        this method will isolate only connections between two types of nodes, as specified by the user,
        in case we want to investigate only connections between two specific node types.
        :param ID1: (Mandatory, string). The name of the first desired nodetype, as contained in the node_identities property.
        :param ID2: (Mandatory, string). The name of the second desired nodetype, as contained in the node_identities property.
        """

        self._network_lists, self._node_identities = network_analysis.isolate_internode_connections(self._network_lists, self._node_identities, ID1, ID2)
        self._network, num_weights = network_analysis.weighted_network(self._network_lists)

    def downsample(self, down_factor):
        """
        Downsamples the Network_3D object (and all its properties) by some specified factor, to make certain associated methods faster. Centroid IDs and voxel scalings are adjusted accordingly.
        :param down_factor: (Mandatory, int). The factor by which to downsample the Network_3D object.
        """
        try:
            original_shape = self._nodes.shape
        except:
            try:
                original_shape = self._edges.shape
            except:
                print("No node or edge attributes have been set.")

        try:
            self._nodes = downsample(self._nodes, down_factor)
            new_shape = self._nodes.shape
            print("Nodes downsampled...")
        except:
            print("Could not downsample nodes")
        try:
            self._edges = downsample(self._edges, down_factor)
            new_shape = self._edges.shape
            print("Edges downsampled...")
        except:
            print("Could not downsample edges")
        try:
            self._search_region = downsample(self._search_region, down_factor)
            print("Search region downsampled...")
        except:
            print("Could not downsample search region")
        try:
            centroids = self._node_centroids.copy()
            for item in self._node_centroids:
                centroids[item] = np.round((self._node_centroids[item])/down_factor)
            self._node_centroids = centroids
            print("Node centroids downsampled")
        except:
            print("Could not downsample node centroids")
        try:
            centroids = self._edge_centroids.copy()
            for item in self._edge_centroids:
                centroids[item] = np.round((self._edge_centroids[item])/down_factor)
            self._edge_centroids = centroids
            print("Edge centroids downsampled...")
        except:
            print("Could not downsample edge centroids")

        try:
            change = float(original_shape[1]/new_shape[1])
            self._xy_scale = self._xy_scale * change
            self._z_scale = self._z_scale * change
            print(f"Arrays of size {original_shape} resized to {new_shape}. Voxel scaling has been adjusted accordingly")
        except:
            print("Could not update voxel scaling")

    def upsample(self, up_factor, targ_shape):
        """
        Upsamples the Network_3D object (and all its properties) by some specified factor, usually to undo a downsample. Centroid IDs and voxel scalings are adjusted accordingly.
        Note that the upsample also asks for a target shape in the form of a tuple (Z, Y, X) (which can be obtained from numpy arrays as some_array.shape). 
        This is because simply upsampling by a factor that mirrors a downsample will not result in the exact same shape, so the target shape is also requested. Note that this method
        should only be called to undo downsamples by an equivalent factor, while inputting the original shape prior to downsampling in the targ_shape param. This method is not a general purpose rescale method
        and will give some unusual results if the up_factor does not result in an upsample whose shape is not already close to the targ_shape.
        :param up_factor: (Mandatory, int). The factor by which to upsample the Network_3D object.
        :targ_shape: (Mandatory, tuple). A (Z, Y, X) tuple of the target shape that should already be close to the shape of the upsampled array. 
        """

        try:
            original_shape = self._nodes.shape
        except:
            try:
                original_shape = self._edges.shape
            except:
                print("No node or edge attributes have been set.")

        try:
            self._nodes = upsample_with_padding(self._nodes, up_factor, targ_shape)
            print("Nodes upsampled...")
        except:
            print("Could not upsample nodes")
        try:
            self._edges = upsample_with_padding(self._edges, up_factor, targ_shape)
            print("Edges upsampled...")
        except:
            print("Could not upsample edges")
        try:
            centroids = self._node_centroids.copy()
            for item in self._node_centroids:
                centroids[item] = (self._node_centroids[item]) * up_factor
            self._node_centroids = centroids
            print("Node centroids upsampled")
        except:
            print("Could not upsample node centroids")
        try:
            centroids = self._edge_centroids.copy()
            for item in self._edge_centroids:
                centroids[item] = (self._edge_centroids[item]) * up_factor
            self._edge_centroids = centroids
            print("Edge centroids upsampled...")
        except:
            print("Could not upsample edge centroids")

        try:
            change = float(original_shape[1]/targ_shape[1])
            self._xy_scale = self._xy_scale * change
            self._z_scale = self._z_scale * change
            print(f"Arrays of size {original_shape} resized to {targ_shape}. Voxel scaling has been adjusted accordingly")
        except:
            print("Could not update voxel scaling")



    def remove_ids(self):

        new_centroids = {}

        for node in self.node_identities.keys():
            new_centroids[node] = self.node_centroids[node]

        self.node_centroids = new_centroids


    def purge_properties(self):

        """Eliminate nodes from properties that are no longer present in the nodes channel"""

        print("Trimming properties. Note this does not update the network...")

        def filter_dict_by_list(input_dict, filter_list):
            """
            Remove dictionary entries where the key is not in the filter list.
            
            Args:
                input_dict (dict): Dictionary with integer values
                filter_list (list): List of integers to keep
                
            Returns:
                dict: New dictionary with only keys that exist in filter_list
            """
            return {key: value for key, value in input_dict.items() if key in filter_list}

        nodes = np.unique(self.nodes)

        if 0 in nodes:
            np.delete(nodes, 0)

        try:
            self.node_centroids = filter_dict_by_list(self.node_centroids, nodes)
            print("Updated centroids")
        except:
            pass
        try:
            self.communities = filter_dict_by_list(self.communities, nodes)
            print("Updated communities")
        except:
            pass
        try:
            self.node_identities = filter_dict_by_list(self.node_identities, nodes)
            print("Updated identities")
        except:
            pass

    def remove_trunk_post(self):
        """
        Removes the 'edge' trunk from a network. In this case, the trunk is the edge that creates the most node-node connections. There may be times when many nodes are connected by a single, expansive edge that obfuscates the rest of the edges. Removing the trunk to a node can better reveal these edges.
        Alters the network and network_lists properties to remove the Trunk.
        """

        nodesa = self._network_lists[0]
        nodesb = self._network_lists[1]
        edgesc = self._network_lists[2]

        trunk = stats.mode(edgesc)

        for i in range(len(edgesc) - 1, -1, -1):
            if edgesc[i] == trunk:
                del edgesc[i]
                del nodesa[i]
                del nodesb[i]

        self._network_lists = [nodesa, nodesb, edgesc]
        self._network, weights = network_analysis.weighted_network(self._network_lists)



    #Methods related to visualizing the network using networkX and matplotlib

    def show_network(self, geometric = False, directory = None, show_labels = True):
        """
        Shows the network property as a simplistic graph, and some basic stats. Does not support viewing edge weights.
        :param geoemtric: (Optional - Val = False; boolean). If False, node placement in the graph will be random. If True, nodes
        will be placed in their actual spatial location (from the original node segmentation) along the XY plane, using node_centroids.
        The node size will then represent the nodes Z location, with smaller nodes representing a larger Z value. If False, nodes will be placed randomly.
        :param directory: (Optional – Val = None; string). An optional string path to a directory to save the network plot image to. If not set, nothing will be saved.
        """

        if not geometric:

            simple_network.show_simple_network(self._network_lists, directory = directory, show_labels = show_labels)

        else:
            simple_network.show_simple_network(self._network_lists, geometric = True, geo_info = [self._node_centroids, self._nodes.shape], directory = directory, show_labels = show_labels)

    def show_communities_flex(self, geometric = False, directory = None, weighted = True, partition = False, style = 0, show_labels = True):


        self._communities, self.normalized_weights = modularity.show_communities_flex(self._network, self._network_lists, self.normalized_weights, geo_info = [self._node_centroids, self._nodes.shape], geometric = geometric, directory = directory, weighted = weighted, partition = partition, style = style, show_labels = show_labels)



    def show_identity_network(self, geometric = False, directory = None, show_labels = True):
        """
        Shows the network property, and some basic stats, as a graph where nodes are labelled by colors representing the identity of the node as detailed in the node_identities property. Does not support viewing edge weights.
        :param geoemtric: (Optional - Val = False; boolean). If False, node placement in the graph will be random. If True, nodes
        will be placed in their actual spatial location (from the original node segmentation) along the XY plane, using node_centroids.
        The node size will then represent the nodes Z location, with smaller nodes representing a larger Z value. If False, nodes will be placed randomly.
        :param directory: (Optional – Val = None; string). An optional string path to a directory to save the network plot image to. If not set, nothing will be saved.
        """
        if not geometric:
            simple_network.show_identity_network(self._network_lists, self._node_identities, geometric = False, directory = directory, show_labels = show_labels)
        else:
            simple_network.show_identity_network(self._network_lists, self._node_identities, geometric = True, geo_info = [self._node_centroids, self._nodes.shape], directory = directory, show_labels = show_labels)



    def get_degrees(self, down_factor = 1, directory = None, called = False, no_img = 0, heatmap = False):
        """
        Method to obtain information on the degrees of nodes in the network, also generating overlays that relate this information to the 3D structure.
        Overlays include a grayscale image where nodes are assigned a grayscale value corresponding to their degree, and a numerical index where numbers are drawn at nodes corresponding to their degree.
        These will be saved to the active directory if none is specified. Note calculations will be done with node_centroids unless a down_factor is passed. Note that a down_factor must be passed if there are no node_centroids.
        :param down_factor: (Optional - Val = 1; int). A factor to downsample nodes by while calculating centroids, assuming no node_centroids property was set.
        :param directory: (Optional - Val = None; string). A path to a directory to save outputs.
        :returns: A dictionary of degree values for each node.
        """

        if heatmap:
            import statistics
            degrees_dict = {node: val for (node, val) in self.network.degree()}
            pred = statistics.mean(list(degrees_dict.values()))

            node_intensity = {}
            import math
            node_centroids = {}

            for node in list(self.network.nodes()):
                node_intensity[node] = math.log(self.network.degree(node)/pred)
                node_centroids[node] = self.node_centroids[node]

            from . import neighborhoods

            overlay = neighborhoods.create_node_heatmap(node_intensity, node_centroids, shape = self.nodes.shape, is_3d=True, labeled_array = self.nodes)

            return degrees_dict, overlay



        if down_factor > 1:
            centroids = self._node_centroids.copy()
            for item in self._node_centroids:
                centroids[item] = np.round((self._node_centroids[item]) / down_factor)
            nodes = downsample(self._nodes, down_factor)
            degrees, nodes = network_analysis.get_degrees(nodes, self._network, directory = directory, centroids = centroids, called = called, no_img = no_img)

        else:
            degrees, nodes = network_analysis.get_degrees(self._nodes, self._network, directory = directory, centroids = self._node_centroids, called = called, no_img = no_img)

        return degrees, nodes


    def isolate_connected_component(self, key = None, directory = None, full_edges = None, gen_images = True):
        """
        Method to isolate a connected component of a network. This can include isolating both nodes and edge images, primarily for visualization, but will also islate a .xlsx file
        to be used to analyze a connected component of a network in detail, as well as returning that networkx graph object. This method generates a number of images. By default,
        the isolated component will be presumed to be the largest one, however a key may be passed containing some node ID of any component needing to be isolated.
        :param key: (Optional - Val None; int). A node ID that is contained in the desired connected component to be isolated. If unset, the largest component will be isolated by default.
        :param directory: (Optional - Val = None; string). A path to a directory to save outputs.
        :param full_edges: (Optional - Val = False; string). If None, will not calculate 'full edges' of the connected component. Essentially, edges stored in the edges property will resemble
        how this file has been altered for connectivity calculations, but will not resemble true edges as they appeared in their original masked segmentation. To obtain edges, isolated over
        a connected component, as they appear in their segmentation, set this as a string file path to your original binary edges segmentation .tif file. Note that this requires the search_region property to be set.
        :param gen_images: (Optional - Val = True; boolean). If True, the various isolated images will be generated. However, as this costs time and memory, setting this value to False
        will cause this method to only generate the .xlsx file of the connected component and to only return the graph object, presuming the user is only interested in non-visual analytics here.
        :returns: IF NO EDGES ATTRIBUTE (will return isolated_nodes, isolated_network in that order. These components can be used to directly set a new Network_3D object
        without using load functions by setting multiple params at once, ie my_network.nodes, my_network.network = old_network.isolate_connected_component()). IF EDGES ATTRIBUTE (will
        return isolated nodes, isolated edges, and isolated network in that order). IF gen_images == False (Will return just the network).
        """

        #Removed depricated gen_images functions

        G = community_extractor._isolate_connected(self._network, key = key)
        return G


    def isolate_mothers(self, directory = None, down_factor = 1, ret_nodes = False, called = False):

        """
        Method to isolate 'mother' nodes of a network (in this case, this means nodes that exist betwixt communities), also generating overlays that relate this information to the 3D structure.
        Overlays include a grayscale image where mother nodes are assigned a grayscale value corresponding to their degree, and a numerical index where numbers are drawn at mother nodes corresponding to their degree, and a general grayscale mask with mother nodes having grayscale IDs corresponding to those stored in the nodes property.
        These will be saved to the active directory if none is specified. Note calculations must be done with node_centroids.
        :param down_factor: (Optional - Val = 1; int). A factor to downsample nodes by while drawing overlays. Note this option REQUIRES node_centroids to already be set.
        :param directory: (Optional - Val = None; string). A path to a directory to save outputs.
        :param louvain: (Optional - Val = True; boolean). If True, louvain community detection will be used. Otherwise, label propogation will be used.
        :param ret_nodes: (Optional - Val = False; boolean). If True, will return the network graph object of the 'mothers'.
        :returns: A dictionary of mother nodes and their degree values.
        """

        if ret_nodes:
            mothers = community_extractor.extract_mothers(None, self._network, self._communities, ret_nodes = True, called = called)
            return mothers
        else:

            if down_factor > 1:
                centroids = self._node_centroids.copy()
                for item in self._node_centroids:
                    centroids[item] = np.round((self._node_centroids[item]) / down_factor)
                nodes = downsample(self._nodes, down_factor)
                mothers, overlay = community_extractor.extract_mothers(nodes, self._network, self._communities, directory = directory, centroid_dic = centroids, called = called)
            else:
                mothers, overlay = community_extractor.extract_mothers(self._nodes, self._network, self._communities, centroid_dic = self._node_centroids, directory = directory, called = called)
            return mothers, overlay


    def isolate_hubs(self, proportion = 0.1, retimg = True):

        hubs = community_extractor.find_hub_nodes(self._network, proportion)

        if retimg:

            hub_img = np.isin(self._nodes, hubs) * self._nodes
        else:
            hub_img = None

        return hubs, hub_img


    def extract_communities(self, color_code = True, down_factor = None, identities = False):

        if down_factor is not None:
            original_shape = self._nodes.shape
            temp = downsample(self._nodes, down_factor)
            if color_code:
                if not identities:
                    image, output = community_extractor.assign_community_colors(self.communities, temp)
                else:
                    image, output = community_extractor.assign_community_colors(self.node_identities, temp)
            else:
                if not identities:
                    image, output = community_extractor.assign_community_grays(self.communities, temp)
                else:
                    image, output = community_extractor.assign_community_grays(self.node_identities, temp)
            image = upsample_with_padding(image, down_factor, original_shape)
        else:

            if color_code:
                if not identities:
                    image, output = community_extractor.assign_community_colors(self.communities, self._nodes)
                else:
                    image, output = community_extractor.assign_community_colors(self.node_identities, self._nodes)
            else:
                if not identities:
                    image, output = community_extractor.assign_community_grays(self.communities, self._nodes)
                else:
                    image, output = community_extractor.assign_community_grays(self.node_identities, self._nodes)


        return image, output

    def node_to_color(self, down_factor = None, mode = 0):

        if mode == 0:
            array = self._nodes
        elif mode == 1:
            array = self._edges

        items = list(np.unique(array))
        if 0 in items:
            del items[0]


        if down_factor is not None:
            original_shape = array.shape
            array = downsample(array, down_factor)

        array, output = community_extractor.assign_node_colors(items, array)

        if down_factor is not None:
            array = upsample_with_padding(array, down_factor, original_shape)

        return array, output
        


    #Methods related to analysis:

    def radial_distribution(self, radial_distance, directory = None):
        """
        Method to calculate the radial distribution of all nodes in the network. Essentially, this is a distribution of the distances between
        all connected nodes in the network, grouped into histogram buckets, which can be used to evaluate the general distances of node-node connectivity. Also displays a histogram.
        This method will save a .xlsx file of this distribution (not bucketed but instead with all vals) to the active directory if none is specified.
        :param radial_distance: (Mandatory, int). The bucket size to group nodes into for the histogram. Note this value will correspond 1-1 with voxels in the nodes array if xy_scale/z_scale have not been set, otherwise they
        will correspond with whatever true value the voxels represent (ie microns).
        :param directory: (Optional - Val = None; string): A path to a directory to save outputs.
        :returns: A list of all the distances between connected nodes in the network.
        """

        radial_dist = network_analysis.radial_analysis(self._nodes, self._network_lists, radial_distance, self._xy_scale, self._z_scale, self._node_centroids, directory = directory)

        return radial_dist

    def assign_random(self, weighted = True):

        """
        Generates a random network of equivalent edge and node count to the current Network_3D object. This may be useful, for example, in comparing aspects of the Network_3D object
        to a similar random network, to demonstrate whether the Network_3D object is a result that itself can be considered random. For example, we can find the modularity of the
        random network and compare it to the Network_3D object's modularity. Note that the random result will itself not have a consistent modularity score between instances this
        method is called, due to randomness, in which case iterating over a large number, say 100, of these random networks will give a tighter comparison point. Please note that
        since Network_3D objects are weighted for multiple connections by default, the random network will consider each additional weight as an additional edge. So a network that has
        one edge of weight one and one of weight two will cause the random network to incorperate 3 edges (that may be crunched into one weighted edge themselves). Please call remove_edge_weights()
        on the Network_3D() object prior to generating the random network if this behavior is not desired.
        :param weighted: (Optional - Val = True; boolean). By default (when True), the random network will be able to take on edge weights by assigning additional edge
        connections between the same nodes. When False, all edges will be made to be discrete. Note that if you for some reason have a supremely weighted network and want to deweight
        the random network, there is a scenario where no new connections can be found and this method will become caught in a while loop.
        :returns: an equivalent random networkx graph object
        """

        G, df = network_analysis.generate_random(self._network, self._network_lists, weighted = weighted)

        return G, df

    def degree_distribution(self, directory = None):
        """
        Method to calculate the degree distribution of all nodes in the network. Essentially, this is recomputes the distribution of degrees to show an x axis of degrees in the network,
        and a y axis of the proportion of nodes in the network that have that degree. A .xlsx file containing the degree distribution will be saved to the active directory if none is specified. 
        This method also shows a scatterplot of this result and attempts to model a power-curve over it, however I found the excel power-curve modeler to be superior so that one may be more reliable than the one included here.
        :param directory: (Optional - Val = None; string): A path to a directory to save outputs.
        :returns: A dictionary with degrees as keys and the proportion of nodes with that degree as a value.
        """

        degrees = network_analysis.degree_distribution(self._network, directory = directory)

        return degrees

    def get_network_stats(self):
        """
        Calculate comprehensive network statistics from a NetworkX graph object.
        
        Parameters:
        G (networkx.Graph): Input graph
        
        Returns:
        dict: Dictionary containing various network statistics
        """
        G = self._network
        stats = {}
        
        # Basic graph properties
        stats['num_nodes'] = G.number_of_nodes()
        stats['num_edges'] = G.number_of_edges()
        stats['density'] = nx.density(G)
        stats['is_directed'] = G.is_directed()
        stats['is_connected'] = nx.is_connected(G) if not G.is_directed() else nx.is_strongly_connected(G)

        # Component analysis
        if not G.is_directed():
            stats['num_connected_components'] = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_component_size'] = len(largest_cc)
        else:
            stats['num_strongly_connected_components'] = nx.number_strongly_connected_components(G)
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            stats['largest_strongly_connected_component_size'] = len(largest_scc)
        
        # Degree statistics
        degrees = [d for _, d in G.degree()]
        stats['avg_degree'] = sum(degrees) / len(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
        
        # Centrality measures
        # Note: These can be computationally expensive for large graphs
        try:
            stats['avg_betweenness_centrality'] = np.mean(list(nx.betweenness_centrality(G).values()))
            stats['avg_closeness_centrality'] = np.mean(list(nx.closeness_centrality(G).values()))
            stats['avg_eigenvector_centrality'] = np.mean(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
        except:
            stats['centrality_measures'] = "Failed to compute - graph might be too large or disconnected"
        
        # Clustering and transitivity
        stats['avg_clustering_coefficient'] = nx.average_clustering(G)
        stats['transitivity'] = nx.transitivity(G)
        
        # Path lengths
        if nx.is_connected(G):
            stats['diameter'] = nx.diameter(G)
            stats['avg_shortest_path_length'] = nx.average_shortest_path_length(G)
        else:
            stats['diameter'] = "Undefined - Graph is not connected"
            stats['avg_shortest_path_length'] = "Undefined - Graph is not connected"
        
        # Structural properties
        stats['is_tree'] = nx.is_tree(G)
        stats['num_triangles'] = sum(nx.triangles(G).values()) // 3
        
        # Assortativity
        try:
            stats['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            stats['degree_assortativity'] = "Failed to compute"

        try:
            nodes = np.unique(self._nodes)
            if nodes[0] == 0:
                nodes = np.delete(nodes, 0)
            stats['Unconnected nodes (left out from node image)'] = (len(nodes) - len(G.nodes()))
        except:
            stats['Unconnected nodes (left out from node image)'] = "Failed to compute"

        
        return stats


    def neighborhood_identities(self, root, directory = None, mode = 0, search = 0, fastdil = False):



        targets = []
        total_dict = {}
        neighborhood_dict = {}
        proportion_dict = {}
        G = self._network
        node_identities = self._node_identities
        for val in set(node_identities.values()):
            total_dict[val] = 0
            neighborhood_dict[val] = 0

        for node in node_identities:
            nodeid = node_identities[node]
            total_dict[nodeid] += 1
            if nodeid == root:
                targets.append(node)


        if mode == 0: #search neighbor ids within the network


            for node in G.nodes():
                try:
                    nodeid = node_identities[node]
                    neighbors = list(G.neighbors(node))
                    for subnode in neighbors:
                        subnodeid = node_identities[subnode]
                        if subnodeid == root:
                            neighborhood_dict[nodeid] += 1
                            break
                except:
                    pass

            title1 = f'Neighborhood Distribution of Nodes in Network from Node Type: {root}'
            title2 = f'Neighborhood Distribution of Nodes in Network from Node Type {root} as a Proportion (# neighbors with ID x / Total # ID x)'


        elif mode == 1: #Search neighborhoods morphologically, obtain densities
            neighborhood_dict, total_dict, densities = morphology.search_neighbor_ids(self._nodes, targets, node_identities, neighborhood_dict, total_dict, search, self._xy_scale, self._z_scale, root, fastdil = fastdil)
            title1 = f'Volumetric Neighborhood Distribution of Nodes in image that are {search} from Node Type: {root}'
            title2 = f'Density Distribution of Nodes in image that are {search} from Node Type {root} as a proportion (Vol neighors with ID x / Total vol ID x)'


        for identity in neighborhood_dict:
            proportion_dict[identity] = neighborhood_dict[identity]/total_dict[identity]

        network_analysis.create_bar_graph(neighborhood_dict, title1, "Node Identity", "Amount", directory=directory)

        network_analysis.create_bar_graph(proportion_dict, title2, "Node Identity", "Proportion", directory=directory)

        try:
            network_analysis.create_bar_graph(densities, f'Relative Density of Node Identities with {search} from Node Type {root}', "Node Identity", "Density within search region/Density within entire image", directory=directory)
        except:
            densities = None


        return neighborhood_dict, proportion_dict, title1, title2, densities



    def get_ripley(self, root = None, targ = None, distance = 1, edgecorrect = True, bounds = None, ignore_dims = False, proportion = 0.5, mode = 0, safe = False, factor = 0.25):

        is_subset = False

        if bounds is None:
            big_array = proximity.convert_centroids_to_array(list(self.node_centroids.values()))
            min_coords = np.array([0,0,0])
            max_coords = [np.max(big_array[:, 0]), np.max(big_array[:, 1]), np.max(big_array[:, 2])]
            del big_array
            max_coords = np.flip(max_coords)
            bounds = (min_coords, max_coords)
        else:
            min_coords, max_coords = bounds

        min_bounds, max_bounds = bounds
        sides = max_bounds - min_bounds
        # Set max_r to None since we've handled edge effects through mirroring


        if root is None or targ is None: #Self clustering in this case
            roots = self._node_centroids.values()
            root_ids = self.node_centroids.keys()
            targs = self._node_centroids.values()
            is_subset = True
        else:
            roots = []
            targs = []
            root_ids = []

            for node, nodeid in self.node_identities.items(): #Otherwise we need to pull out this info
                if nodeid == root:
                    roots.append(self._node_centroids[node])
                    root_ids.append(node)
                if nodeid == targ:
                    targs.append(self._node_centroids[node])

        if not is_subset:
            if np.array_equal(roots, targs):
                is_subset = True

        rooties = proximity.convert_centroids_to_array(roots, xy_scale = self.xy_scale, z_scale = self.z_scale)
        targs = proximity.convert_centroids_to_array(targs, xy_scale = self.xy_scale, z_scale = self.z_scale)
        
        try:
            if self.nodes.shape[0] == 1:
                dim = 2
            else:
                dim = 3
        except:
            dim = 2
            for centroid in self.node_centroids.values():
                if centroid[0] != 0:
                    dim = 3
                    break

        if dim == 2:
            volume = sides[0] * sides[1] * self.xy_scale**2
        else:
            volume = np.prod(sides) * self.z_scale * self.xy_scale**2

        points_array = np.vstack((rooties, targs))
        del rooties
        max_r = None
        if safe:
            proportion = factor
            

        if ignore_dims:

            new_list = []

            if mode == 0:

                try:
                    dim_list = max_coords - min_coords
                except:
                    min_coords = np.array([0,0,0])
                    bounds = (min_coords, max_coords)
                    dim_list = max_coords - min_coords

                for centroid in roots:
                    # Assuming centroid is [z, y, x] based on your indexing
                    z, y, x = centroid[0], centroid[1], centroid[2]
                    
                    # Check x-dimension
                    x_ok = (x - min_coords[0]) > dim_list[0] * factor and (max_coords[0] - x) > dim_list[0] * factor
                    # Check y-dimension  
                    y_ok = (y - min_coords[1]) > dim_list[1] * factor and (max_coords[1] - y) > dim_list[1] * factor
                    
                    if dim == 3:  # 3D case
                        # Check z-dimension
                        z_ok = (z - min_coords[2]) > dim_list[2] * factor and (max_coords[2] - z) > dim_list[2] * factor
                        if x_ok and y_ok and z_ok:
                            new_list.append(centroid)
                    else:  # 2D case
                        if x_ok and y_ok:
                            new_list.append(centroid)

            else:
                if mode == 1:
                    legal = self.edges != 0
                elif mode == 2:
                    legal = self.network_overlay != 0
                elif mode == 3:
                    legal = self.id_overlay != 0
                if self.nodes is None:
                    temp_array = proximity.populate_array(self.node_centroids, shape = legal.shape)
                else:
                    temp_array = self.nodes
                if dim == 2:
                    volume = np.count_nonzero(legal) * self.xy_scale**2
                    # Pad in x and y dimensions (assuming shape is [y, x])
                    legal = np.pad(legal, pad_width=1, mode='constant', constant_values=0)
                else:
                    volume = np.count_nonzero(legal) * self.z_scale * self.xy_scale**2
                    # Pad in x, y, and z dimensions (assuming shape is [z, y, x])
                    legal = np.pad(legal, pad_width=1, mode='constant', constant_values=0)
                
                print(f"Using {volume} for the volume measurement (Volume of provided mask as scaled by xy and z scaling)")
                
                # Compute distance transform on padded array
                legal = smart_dilate.compute_distance_transform_distance(legal, sampling = [self.z_scale, self.xy_scale, self.xy_scale])
                
                # Remove padding after distance transform
                if dim == 2:
                    legal = legal[1:-1, 1:-1]  # Remove padding from x and y dimensions
                else:
                    legal = legal[1:-1, 1:-1, 1:-1]  # Remove padding from x, y, and z dimensions
                
                max_avail = np.max(legal) # Most internal point
                min_legal = factor * max_avail # Values of stuff 25% within the tissue

                legal = legal > min_legal

                if safe:
                    max_r = min_legal


                legal = temp_array * legal

                legal = np.unique(legal)
                if 0 in legal:
                    legal = np.delete(legal, 0)
                for node in legal:
                    if node in root_ids:
                        new_list.append(self.node_centroids[node])

            roots = new_list
            print(f"Utilizing {len(roots)} root points. Note that low n values are unstable.")
            is_subset = True

        roots = proximity.convert_centroids_to_array(roots, xy_scale = self.xy_scale, z_scale = self.z_scale)

        n_subset = len(targs)

        # Apply edge correction through mirroring
        if edgecorrect:


            roots, targs = apply_edge_correction_to_ripley(
                roots, targs, proportion, bounds, dim, 
                node_centroids=self.node_centroids  # Pass this for bounds calculation if needed
            )


        if dim == 2:
            roots = proximity.convert_augmented_array_to_points(roots)
            targs = proximity.convert_augmented_array_to_points(targs)

        print(f"Using {len(roots)} root points")
        r_vals = proximity.generate_r_values(points_array, distance, bounds = bounds, dim = dim, max_proportion=proportion, max_r = max_r)

        k_vals =  proximity.optimized_ripleys_k(roots, targs, r_vals, bounds=bounds, dim = dim, is_subset = is_subset, volume = volume, n_subset = n_subset)

        h_vals = proximity.compute_ripleys_h(k_vals, r_vals, dim)

        proximity.plot_ripley_functions(r_vals, k_vals, h_vals, dim, root, targ)

        return r_vals, k_vals, h_vals




#Morphological stats or network linking:

    def volumes(self, sort = 'nodes'):

        """Calculates the volumes of either the nodes or edges"""

        if sort == 'nodes':

            return morphology.calculate_voxel_volumes(self._nodes, self._xy_scale, self._z_scale)

        elif sort == 'edges':

            return morphology.calculate_voxel_volumes(self._edges, self._xy_scale, self._z_scale)

        elif sort == 'network_overlay':

            return morphology.calculate_voxel_volumes(self._network_overlay, self._xy_scale, self._z_scale)

        elif sort == 'id_overlay':

            return morphology.calculate_voxel_volumes(self._id_overlay, self._xy_scale, self._z_scale)




    def interactions(self, search = 0, cores = 0, resize = None, save = False, skele = False, length = False, auto = True, fastdil = False):

        return morphology.quantify_edge_node(self._nodes, self._edges, search = search, xy_scale = self._xy_scale, z_scale = self._z_scale, cores = cores, resize = resize, save = save, skele = skele, length = length, auto = auto, fastdil = fastdil)



    def morph_proximity(self, search = 0, targets = None, fastdil = False):

        if type(search) == list:
            search_x, search_z = search #Suppose we just want to directly pass these params
        else:
            search_x, search_z = dilation_length_to_pixels(self._xy_scale, self._z_scale, search, search)

        num_nodes = int(np.max(self._nodes))

        my_dict = proximity.create_node_dictionary(self._nodes, num_nodes, search_x, search_z, targets = targets, fastdil = fastdil, xy_scale = self._xy_scale, z_scale = self._z_scale, search = search)

        my_dict = proximity.find_shared_value_pairs(my_dict)

        my_dict = create_and_save_dataframe(my_dict)

        self._network_lists = network_analysis.read_excel_to_lists(my_dict)

        self.remove_edge_weights()

    def centroid_array(self, clip = False, shape = None):
        """Use the centroids to populate a node array"""

        if clip:

            array, centroids = proximity.populate_array(self.node_centroids, clip = True)
            return array, centroids

        else:

            array = proximity.populate_array(self.node_centroids, shape = shape)

            return array




    def random_nodes(self, bounds = None, mask = None):

        if self.nodes is not None:
            try:
                self.nodes = np.zeros_like(self.nodes)
            except:
                pass


        if mask is not None:
            coords = np.argwhere(mask != 0)
        else:
            if bounds is not None:
                (z1, y1, x1), (z2, y2, x2) = bounds
                z1, y1, x1 = int(z1), int(y1), int(x1)
                z2, y2, x2 = int(z2), int(y2), int(x2)
                z_range = np.arange(z1, z2 + 1 )
                y_range = np.arange(y1, y2 + 1 )
                x_range = np.arange(x1, x2 + 1 )
                z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, x_range, indexing='ij')
                del z_range
                del y_range
                del x_range
                coords = np.stack([z_grid.flatten(), y_grid.flatten(), x_grid.flatten()], axis=1)
                del z_grid
                del y_grid
                del x_grid
            else:
                shape = ()
                try:
                    shape = self.nodes.shape
                except:
                    try:
                        shape = self.edges.shape
                    except:
                        try:
                            shape = self._network_overlay.shape
                        except:
                            try:
                                shape = self._id_overlay.shape
                            except:
                                pass

                ranges = [np.arange(s) for s in shape]
                
                # Create meshgrid
                mesh = np.meshgrid(*ranges, indexing='ij')
                del ranges

                # Stack and reshape
                coords = np.stack(mesh, axis=-1).reshape(-1, len(shape))
                del mesh

        if len(coords) < len(self.node_centroids):
            print(f"Warning: Only {len(coords)} positions available for {len(self.node_centroids)} labels")

        new_centroids = {}
        
        # Generate random indices without replacement
        available_count = min(len(coords), len(self.node_centroids))
        rand_indices = np.random.choice(len(coords), available_count, replace=False)
        
        # Assign random positions to labels
        for i, label in enumerate(self.node_centroids.keys()):
            if i < len(rand_indices):
                centroid = coords[rand_indices[i]]
                new_centroids[label] = centroid
                z, y, x = centroid
                try:
                    self.nodes[z, y, x] = label
                except:
                    pass
        
        # Update the centroids dictionary
        self.node_centroids = new_centroids

        return self.node_centroids, self._nodes


    def community_id_info(self):


        community_dict = invert_dict(self.communities)
        summation = 0
        id_set = iden_set(self.node_identities.values())
        output = {sort: 0 for sort in id_set}
        template = copy.deepcopy(output)

        for community in community_dict:
            counter = copy.deepcopy(template)
            nodes = community_dict[community]
            size = len(nodes)
            summation += size
            
            # Count identities in this community
            for node in nodes:
                try:
                    idens = ast.literal_eval(self.node_identities[node])
                    for iden in idens:
                        counter[iden] += 1
                except:
                    counter[self.node_identities[node]] += 1
            
            # Convert to proportions within this community and weight by size
            for sort in counter:
                if size > 0:  # Avoid division by zero
                    counter[sort] = (counter[sort]) * size  # proportion * size
                    
            # Add to running totals
            for sort, weighted_count in counter.items():
                output[sort] = output.get(sort, 0) + weighted_count

        # Normalize by total size
        dictsum = 0
        for sort in output:
            output[sort] = output[sort]/summation
            dictsum += output[sort]

        for sort in output:
            output[sort] = output[sort]/dictsum

        return output

    def centroid_umap(self):

        from . import neighborhoods

        neighborhoods.visualize_cluster_composition_umap(self.node_centroids, None, id_dictionary = self.node_identities, graph_label = "Node ID", title = 'UMAP Visualization of Node Centroids') 


    def identity_umap(self, data, mode = 0):

        try:

            neighbor_classes = {}
            import random

            umap_dict = copy.deepcopy(data)

            for item in data.keys():
                if item in self.node_identities:
                    try:
                        parse = ast.literal_eval(self.node_identities[item])
                        neighbor_classes[item] = random.choice(parse)
                    except:
                        neighbor_classes[item] = self.node_identities[item]

                else:
                    del umap_dict[item]

            #from scipy.stats import zscore

            # Z-score normalize each marker (column)
            #for key in umap_dict:
                #umap_dict[key] = zscore(umap_dict[key])

            from . import neighborhoods

            if mode == 0:
                neighborhoods.visualize_cluster_composition_umap(umap_dict, None, id_dictionary = neighbor_classes, graph_label = "Node ID", title = 'UMAP Visualization of Node Identities by Z-Score') 
            else:
                neighborhoods.visualize_cluster_composition_umap(umap_dict, None, id_dictionary = neighbor_classes, graph_label = "Node ID", title = 'UMAP Visualization of Node Identities by Z-Score', neighborhoods = self.communities, original_communities = self.communities) 

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error: {e}")

    def community_id_info_per_com(self, umap = False, label = 0, limit = 0, proportional = False, neighbors = None):

        community_dict = invert_dict(self.communities)
        summation = 0
        id_set = iden_set(self.node_identities.values())
        id_dict = {}
        for i, iden in enumerate(id_set):
            id_dict[iden] = i

        output = {}
        umap_dict = {}

        if not proportional:

            for community in community_dict:

                counter = np.zeros(len(id_set))

                nodes = community_dict[community]
                size = len(nodes)

                # Count identities in this community
                for node in nodes:
                    try:
                        idens = ast.literal_eval(self.node_identities[node])
                        for iden in idens:
                            counter[id_dict[iden]] += 1
                    except:
                        try:
                            counter[id_dict[self.node_identities[node]]] += 1 # Keep them as arrays
                        except:
                            pass

                for i in range(len(counter)): # Translate them into proportions out of 1

                    counter[i] = counter[i]/size

                output[community] = counter #Assign the finding here

                if size >= limit:
                    umap_dict[community] = counter

        else:
            idens = invert_dict_special(self.node_identities)
            iden_count = {}
            template = {}
            node_count = len(list(self.communities.keys()))

            for iden in id_set:
                template[iden] = 0

            for iden, nodes in idens.items():
                iden_count[iden] = len(nodes)

            for community in community_dict:

                iden_tracker = copy.deepcopy(template)

                nodes = community_dict[community]
                size = len(nodes)
                counter = np.zeros(len(id_set))

                for node in nodes:
                    try:
                        idents = ast.literal_eval(self.node_identities[node])
                        for iden in idents:
                            iden_tracker[iden] += 1
                    except:
                        try:
                            iden_tracker[self.node_identities[node]] += 1
                        except:
                            pass

                i = 0

                if not umap: # External calls just get the proportion for now

                    for iden, val in iden_tracker.items(): # Translate them into proportions of total number of that node of all nodes of that ID

                        counter[i] = (val/iden_count[iden])
                        i += 1

                    output[community] = counter #Assign the finding here

                    if size >= limit:
                        umap_dict[community] = counter

                else: # Internal calls for the umap get the relative proportion, demonstrating overrepresentation per community


                    for iden, val in iden_tracker.items(): # Translate them into proportions of total number of that node of all nodes of that ID

                        counter[i] = (val/iden_count[iden])/(size/node_count) # The proportion of that ID in the community vs all of that ID divided by the proportion of that community size vs all the nodes
                        i += 1

                    output[community] = counter #Assign the finding here

                    if size >= limit:
                        umap_dict[community] = counter


        if umap:
            from . import neighborhoods


            if self.communities is not None and label == 2:
                neighbor_group = {}
                for node, com in self.communities.items():
                    try:
                        neighbor_group[com] = neighbors[node]
                    except:
                        neighbor_group[com] = 0
                neighborhoods.visualize_cluster_composition_umap(umap_dict, id_set, neighborhoods = neighbor_group, original_communities = neighbors)
            elif label == 1:
                neighborhoods.visualize_cluster_composition_umap(umap_dict, id_set, label = True) 
            else:
                neighborhoods.visualize_cluster_composition_umap(umap_dict, id_set, label = False) 


            #neighborhoods.visualize_cluster_composition_umap(umap_dict, id_set, label = label) 

        return output, id_set


    def group_nodes_by_intensity(self, data, count = None):

        from . import neighborhoods

        clusters = neighborhoods.cluster_arrays(data, count, seed = 42)

        coms = {}

        for i, cluster in enumerate(clusters):
            coms[i + 1] = cluster

        self.communities = revert_dict(coms)

    def assign_neighborhoods(self, seed, count, limit = None, prev_coms = None, proportional = False, mode = 0):

        from . import neighborhoods

        if prev_coms is not None:
            self.communities = copy.deepcopy(prev_coms)

        identities, _ = self.community_id_info_per_com()

        zero_group = {}

        comus = invert_dict(self.communities)


        if limit is not None:

            for com, nodes in comus.items():

                if len(nodes) < limit:

                    del identities[com]

        try:
            if count > len(identities):
                print(f"Requested neighborhoods too large for available communities. Using {len(identities)} neighborhoods (max for these coms)")
                count = len(identities)
        except:
            pass


        if mode == 0:
            clusters = neighborhoods.cluster_arrays(identities, count, seed = seed)
        elif mode == 1:
            clusters = neighborhoods.cluster_arrays_dbscan(identities, seed = seed)

        coms = {}

        neighbors = {}
        len_dict = {}
        inc_count = 0

        for i, cluster in enumerate(clusters):

            size = len(cluster)
            inc_count += size

            len_dict[i + 1] = [size]

            for com in cluster: # For community ID per list

                coms[com] = i + 1


        copy_dict = copy.deepcopy(self.communities)

        for node, com in copy_dict.items():

            try:

                self.communities[node] = coms[com]

            except:
                del self.communities[node]
                zero_group[node] = 0

        self.com_by_size()


        if len(zero_group) > 0:
            self.communities.update(zero_group)
            len_dict[0] = [len(comus) - inc_count]


        identities, id_set = self.community_id_info_per_com()

        coms = invert_dict(self.communities)
        node_count = len(list(self.communities.keys()))

        for com, nodes in coms.items():

            len_dict[com].append(len(nodes)/node_count)

        matrixes = []

        output = neighborhoods.plot_dict_heatmap(identities, id_set, title = "Neighborhood Heatmap by Proportional Composition Per Neighborhood")

        matrixes.append(output)

        if proportional:

            identities2, id_set2 = self.community_id_info_per_com(proportional = True)
            output = neighborhoods.plot_dict_heatmap(identities2, id_set2, title = "Neighborhood Heatmap by Proportional Composition of Nodes in Neighborhood vs All Nodes in Image")
            matrixes.append(output)

            identities3 = {}
            for iden in identities2:
                identities3[iden] = identities2[iden]/len_dict[iden][1]

            output = neighborhoods.plot_dict_heatmap(identities3, id_set2, title = "Over/Underrepresentation of Node Identities per Neighborhood (val < 1 = underrepresented, val > 1 = overrepresented)", center_at_one = True)
            matrixes.append(output)

        return len_dict, matrixes, id_set



    def kd_network(self, distance = 100, targets = None, make_array = False, max_neighbors = None):

        centroids = copy.deepcopy(self._node_centroids)

        if self._xy_scale == self._z_scale:
            upsample = None
            distance = distance/self._xy_scale # Account for scaling
        else:
            upsample = [self._xy_scale, self._z_scale] # This means resolutions have to be normalized
            if self._xy_scale < self._z_scale:
                distance = distance/self._xy_scale # We always upsample to normalize
                refactor = self._z_scale/self._xy_scale
                for node, centroid in centroids.items():
                    centroids[node] = [centroid[0] * refactor, centroid[1], centroid[2]]
            elif self._z_scale < self._xy_scale:
                distance = distance/self._z_scale
                refactor = self._xy_scale/self._z_scale
                for node, centroid in centroids.items():
                    centroids[node] = [centroid[0], centroid[1] * refactor, centroid[2] * refactor]


        neighbors = proximity.find_neighbors_kdtree(distance, targets = targets, centroids = centroids, max_neighbors = max_neighbors)

        print("Creating Dataframe")

        network = create_and_save_dataframe(neighbors)

        print("Converting df to network")

        self._network_lists = network_analysis.read_excel_to_lists(network)

        #self._network is a networkx graph that stores the connections

        print("Removing Edge Weights")

        self.remove_edge_weights()

        if make_array:

            array = self.centroid_array()

            return array

    def community_cells(self, size = 32, xy_scale = 1, z_scale = 1):

        size_x = int(size * xy_scale)
        size_z = int(size * z_scale)

        if size_x == size_z:

            com_dict = proximity.partition_objects_into_cells(self.node_centroids, size_x)

        else:

            com_dict = proximity.partition_objects_into_cells(self.node_centroids, (size_z, size_x, size_x))

        self.communities = revert_dict(com_dict)

    def community_heatmap(self, num_nodes = None, is3d = True, numpy = False):

        import math

        if num_nodes == None:

            try:
                num_nodes = len(self.network.nodes())
            except:
                try:
                    num_nodes = len(self.node_centroids.keys())
                except:
                    try:
                        num_nodes = len(self.node_identities.keys())
                    except:
                        try:
                            unique = np.unique(self.nodes)
                            num_nodes = len(unique)
                            if unique[0] == 0:
                                num_nodes -= 1
                        except:
                            return

        coms = invert_dict(self.communities)

        rand_dens = num_nodes / len(coms.keys())

        heat_dict = {}

        for com, nodes in coms.items():
            heat_dict[com] = math.log(len(nodes)/rand_dens)

        try:
            shape = self.nodes.shape
        except:
            big_array = proximity.convert_centroids_to_array(list(self.node_centroids.values()))
            shape = [np.max(big_array[0, :]) + 1, np.max(big_array[1, :]) + 1, np.max(big_array[2, :]) + 1]

        from . import neighborhoods
        if not numpy:
            neighborhoods.create_community_heatmap(heat_dict, self.communities, self.node_centroids, shape = shape, is_3d=is3d)

            return heat_dict
        else:
            overlay = neighborhoods.create_community_heatmap(heat_dict, self.communities, self.node_centroids, shape = shape, is_3d=is3d, labeled_array = self.nodes)
            return heat_dict, overlay

    def get_merge_node_dictionaries(self, path, data):

        img_list = directory_info(path)
        id_dicts = []
        num_nodes = np.max(data)

        for i, img in enumerate(img_list):
            if img.endswith('.tiff') or img.endswith('.tif'):
                print(f"Processing image {img}")
                mask = tifffile.imread(f'{path}/{img}')
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis = 0)

                id_dict = proximity.create_node_dictionary_id(data, mask, num_nodes)
                id_dicts.append(id_dict)

        return id_dicts

    def merge_node_ids(self, path, data, include = True):

        if self.node_identities is None: # Prepare modular dict

            self.node_identities = {}

            nodes = list(np.unique(data))
            if 0 in nodes:
                del nodes[0]
            for node in nodes:

                self.node_identities[node] = [] # Assign to lists at first
        else:
            for node, iden in self.node_identities.items():
                try:
                    self.node_identities[node] = ast.literal_eval(iden)
                except:
                    self.node_identities[node] = [iden]



        img_list = directory_info(path)

        for i, img in enumerate(img_list):

            if img.endswith('.tiff') or img.endswith('.tif'):

                mask = tifffile.imread(f'{path}/{img}')

                if len(np.unique(mask)) != 2:

                    mask = otsu_binarize(mask)
                else:
                    mask = mask != 0

                nodes = data * mask
                nodes = np.unique(nodes)
                nodes = nodes.tolist()
                if 0 in nodes:
                    del nodes[0]

                if img.endswith('.tiff'):
                    base_name = img[:-5]
                elif img.endswith('.tif'):
                    base_name = img[:-4]
                else:
                    base_name = img

                assigned = {}


                for node in self.node_identities.keys():

                    try:

                        if int(node) in nodes:

                            self.node_identities[node].append(f'{base_name}+')

                        elif include:

                            self.node_identities[node].append(f'{base_name}-')

                    except:
                        pass

        modify_dict = copy.deepcopy(self.node_identities)

        for node, iden in self.node_identities.items():

            try:

                if len(iden) == 1:

                    modify_dict[node] = str(iden[0]) # Singleton lists become bare strings
                elif len(iden) == 0:
                    del modify_dict[node]
                else:
                    modify_dict[node] = str(iden) # We hold multi element lists as strings for compatibility

            except:
                pass

        self.node_identities = modify_dict


    def nearest_neighbors_avg(self, root, targ, xy_scale = 1, z_scale = 1, num = 1, heatmap = False, threed = True, numpy = False, quant = False, centroids = True, mask = None):

        def distribute_points_uniformly(n, shape, z_scale, xy_scale, num, is_2d=False, mask=None):
            from scipy.spatial import KDTree
            if n <= 1:
                return 0
            
            if mask is not None:
                # Handle mask-based distribution
                # Find all valid positions where mask is True
                valid_positions = np.where(mask)
                total_valid_positions = len(valid_positions[0])
                
                if total_valid_positions == 0:
                    raise ValueError("No valid positions found in mask")
                
                if n >= total_valid_positions:
                    # If we want more points than valid positions, return scaled unit distance
                    return xy_scale if is_2d else min(z_scale, xy_scale)
                
                # Create uniformly spaced indices within valid positions
                valid_indices = np.linspace(0, total_valid_positions - 1, n, dtype=int)
                
                # Convert to coordinates and apply scaling
                coords = []
                for idx in valid_indices:
                    if len(shape) == 3:
                        coord = (valid_positions[0][idx], valid_positions[1][idx], valid_positions[2][idx])
                        scaled_coord = [coord[0] * z_scale, coord[1] * xy_scale, coord[2] * xy_scale]
                    elif len(shape) == 2:
                        coord = (valid_positions[0][idx], valid_positions[1][idx])
                        scaled_coord = [coord[0] * xy_scale, coord[1] * xy_scale]
                    coords.append(scaled_coord)
                
                coords = np.array(coords)
                
                # Find a good query point (closest to center of valid region)
                if len(shape) == 3:
                    center_pos = [np.mean(valid_positions[0]) * z_scale, 
                                 np.mean(valid_positions[1]) * xy_scale,
                                 np.mean(valid_positions[2]) * xy_scale]
                else:
                    center_pos = [np.mean(valid_positions[0]) * xy_scale,
                                 np.mean(valid_positions[1]) * xy_scale]
                
                # Find point closest to center of valid region
                center_distances = np.sum((coords - center_pos)**2, axis=1)
                middle_idx = np.argmin(center_distances)
                query_point = coords[middle_idx]
                
            else:
                # Original behavior when no mask is provided
                total_positions = np.prod(shape)
                if n >= total_positions:
                    return xy_scale if is_2d else min(z_scale, xy_scale)
                
                # Create uniformly spaced indices
                indices = np.linspace(0, total_positions - 1, n, dtype=int)
                
                # Convert flat indices to coordinates
                coords = []
                for idx in indices:
                    coord = np.unravel_index(idx, shape)
                    if len(shape) == 3:
                        scaled_coord = [coord[0] * z_scale, coord[1] * xy_scale, coord[2] * xy_scale]
                    elif len(shape) == 2:
                        scaled_coord = [coord[0] * xy_scale, coord[1] * xy_scale]
                    coords.append(scaled_coord)
                
                coords = np.array(coords)
                
                # Pick a point near the middle of the array
                middle_idx = len(coords) // 2
                query_point = coords[middle_idx]
            
            # Build KDTree
            tree = KDTree(coords)
            
            # Find the num+1 nearest neighbors (including the point itself)
            distances, indices = tree.query(query_point, k=num+1)
            
            # Exclude the point itself (distance 0) and get the actual neighbors
            neighbor_distances = distances[1:num+1]
            if num == n:
                neighbor_distances[-1] = neighbor_distances[-2]
            
            avg_distance = np.mean(neighbor_distances)
            
            return avg_distance

        do_borders = not centroids

        if centroids:
            root_set = []

            compare_set = []

            if root is None:

                root_set = list(self.node_centroids.keys())
                compare_set = root_set
                title = "Nearest Neighbors Between Nodes Heatmap"

            else:

                title = f"Nearest Neighbors of ID {targ} from ID {root} Heatmap"

                for node, iden in self.node_identities.items():

                    if iden == root: # Standard behavior

                        root_set.append(node)

                    elif '[' in iden and root != "All (Excluding Targets)": # For multiple nodes
                        if root in iden:
                            root_set.append(node)

                    elif (iden == targ) or (targ == 'All Others (Excluding Self)'): # The other group

                        compare_set.append(node)

                    elif '[' in iden: # The other group, for multiple nodes
                        if targ in iden:
                            compare_set.append(node)

                    elif root == "All (Excluding Targets)": # If not assigned to the other group but the comprehensive root option is used
                        root_set.append(node)

            if root == targ:

                compare_set = root_set
                if len(compare_set) - 1 < num:

                    num = len(compare_set) - 1

                    print(f"Error: Not enough neighbor nodes for requested number of neighbors. Using max available neighbors: {num}")
                    

            if len(compare_set) < num:

                num = len(compare_set)

                print(f"Error: Not enough neighbor nodes for requested number of neighbors. Using max available neighbors: {num}")

            avg, output = proximity.average_nearest_neighbor_distances(self.node_centroids, root_set, compare_set, xy_scale=self.xy_scale, z_scale=self.z_scale, num = num, do_borders = do_borders)

        else:
            if heatmap:
                root_set = []
                compare_set = []
                if root is None and not do_borders:
                    compare_set = root_set
                    if not do_borders:
                        root_set = list(self.node_centroids.keys())
                elif self.node_identities is not None:
                    for node, iden in self.node_identities.items():

                        if iden == root:

                            root_set.append(node)

                        elif (iden == targ) or (targ == 'All Others (Excluding Self)'):

                            compare_set.append(node)

            if root is None:
                title = "Nearest Neighbors Between Nodes Heatmap"
                root_set_neigh = approx_boundaries(self.nodes, keep_labels = True)
                compare_set_neigh = approx_boundaries(self.nodes, keep_labels = False)
            else:
                title = f"Nearest Neighbors of ID {targ} from ID {root} Heatmap"

                root_set_neigh = approx_boundaries(self.nodes, [root], self.node_identities, keep_labels = True)

                if targ == 'All Others (Excluding Self)':
                    compare_set_neigh = set(self.node_identities.values())
                    compare_set_neigh.remove(root)
                    targ = compare_set_neigh
                else:
                    targ = [targ]

                compare_set_neigh = approx_boundaries(self.nodes, targ, self.node_identities, keep_labels = False)

            avg, output = proximity.average_nearest_neighbor_distances(self.node_centroids, root_set_neigh, compare_set_neigh, xy_scale=self.xy_scale, z_scale=self.z_scale, num = num, do_borders = do_borders)

        if quant:
            try:
                quant_overlay = node_draw.degree_infect(output, self._nodes, make_floats = True)
            except:
                quant_overlay = None
        else:
            quant_overlay = None

        if heatmap:


            from . import neighborhoods
            try:
                shape = self.nodes.shape
            except:
                big_array = proximity.convert_centroids_to_array(list(self.node_centroids.values()))
                shape = [np.max(big_array[0, :]) + 1, np.max(big_array[1, :]) + 1, np.max(big_array[2, :]) + 1]


            try:
                bounds = self.nodes.shape
            except:
                try:
                    bounds = self.edges.shape
                except:
                    try:
                        bounds = self.network_overlay.shape
                    except:
                        try:
                            bounds = self.id_overlay.shape
                        except:
                            big_array = proximity.convert_centroids_to_array(list(self.node_centroids.values()))
                            max_coords = [np.max(big_array[:, 0]), np.max(big_array[:, 1]), np.max(big_array[:, 2])]
                            del big_array
            volume = bounds[0] * bounds[1] * bounds[2] * self.z_scale * self.xy_scale**2
            if 1 in bounds or 0 in bounds:
                is_2d = True
            else:
                is_2d = False

            if root_set == []:
                avail_nodes = np.unique(self.nodes)
                compare_set = list(avail_nodes)
                if 0 in compare_set:
                    del compare_set[0]
                root_set = compare_set
            elif compare_set == []:
                compare_set = root_set
            pred = distribute_points_uniformly(len(compare_set), bounds, self.z_scale, self.xy_scale, num = num, is_2d = is_2d, mask = mask)

            node_intensity = {}
            import math
            node_centroids = {}

            for node in root_set:
                node_intensity[node] = math.log(pred/output[node])
                node_centroids[node] = self.node_centroids[node]

            if numpy:

                overlay = neighborhoods.create_node_heatmap(node_intensity, node_centroids, shape = shape, is_3d=threed, labeled_array = self.nodes, colorbar_label="Clustering Intensity", title = title)

                return avg, output, overlay, quant_overlay, pred

            else:
                neighborhoods.create_node_heatmap(node_intensity, node_centroids, shape = shape, is_3d=threed, labeled_array = None, colorbar_label="Clustering Intensity", title = title)

        else:
            pred = None

        return avg, output, quant_overlay, pred








if __name__ == "__main__":
    create_and_draw_network()