from . import nettracer
from . import network_analysis
import numpy as np
from scipy.ndimage import zoom
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import tifffile
from functools import partial
import concurrent.futures
from functools import partial
from scipy import ndimage
import pandas as pd
# Import CuPy conditionally for GPU support
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
import itertools
from skimage.morphology import skeletonize

try:
    import xs3d
except:
    pass



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


def _get_node_edge_dict(label_array, edge_array, label, dilate_xy, dilate_z, cores = 0, search = 0, fastdil = False, length = False, xy_scale = 1, z_scale = 1):
    """Internal method used for the secondary algorithm to find pixel involvement of nodes around an edge."""
    
    # Create a boolean mask where elements with the specified label are True
    label_array = label_array == label
    dil_array = nettracer.dilate(label_array, search, xy_scale = xy_scale, z_scale = z_scale, fast_dil = fastdil) #Dilate the label to see where the dilated label overlaps

    if cores == 0: #For getting the volume of objects. Cores presumes you want the 'core' included in the interaction.
        edge_array = edge_array * dil_array  # Filter the edges by the label in question
    elif cores == 1: #Cores being 1 presumes you do not want to 'core' included in the interaction
        label_array = dil_array - label_array
        edge_array = edge_array * label_array
    elif cores == 2: #Presumes you want skeleton within the core but to only 'count' the stuff around the core for volumes... because of imaging artifacts, perhaps
        edge_array = edge_array * dil_array
        label_array = dil_array - label_array

    label_count = np.count_nonzero(label_array) * xy_scale * xy_scale * z_scale

    if not length:
        edge_count = np.count_nonzero(edge_array) * xy_scale * xy_scale * z_scale # For getting the interacting skeleton
    else:
        edge_count = calculate_skeleton_lengths(
            edge_array, 
            xy_scale=xy_scale, 
            z_scale=z_scale
        )

    args = [edge_count, label_count]

    return args

def process_label(args):
    """Modified to use pre-computed bounding boxes instead of argwhere"""
    nodes, edges, label, dilate_xy, dilate_z, array_shape, bounding_boxes = args
    print(f"Processing node {label}")
    
    # Get the pre-computed bounding box for this label
    slice_obj = bounding_boxes[label-1]  # -1 because label numbers start at 1
    if slice_obj is None:
        return None, None, None
        
    z_vals, y_vals, x_vals = get_reslice_indices(slice_obj, dilate_xy, dilate_z, array_shape)
    if z_vals is None:
        return None, None, None
        
    sub_nodes = reslice_3d_array((nodes, z_vals, y_vals, x_vals))
    sub_edges = reslice_3d_array((edges, z_vals, y_vals, x_vals))
    return label, sub_nodes, sub_edges



def create_node_dictionary(nodes, edges, num_nodes, dilate_xy, dilate_z, cores=0, search = 0, fastdil = False, length = False, xy_scale = 1, z_scale = 1):
    """Modified to pre-compute all bounding boxes using find_objects"""
    node_dict = {}
    array_shape = nodes.shape
    
    # Get all bounding boxes at once
    bounding_boxes = ndimage.find_objects(nodes)
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Create args list with bounding_boxes included
        args_list = [(nodes, edges, i, dilate_xy, dilate_z, array_shape, bounding_boxes) 
                    for i in range(1, num_nodes + 1)]

        # Execute parallel tasks to process labels
        results = executor.map(process_label, args_list)

        # Process results in parallel
        for label, sub_nodes, sub_edges in results:
            executor.submit(create_dict_entry, node_dict, label, sub_nodes, sub_edges, 
                          dilate_xy, dilate_z, cores, search, fastdil, length, xy_scale, z_scale)

    return node_dict

def create_dict_entry(node_dict, label, sub_nodes, sub_edges, dilate_xy, dilate_z, cores = 0, search = 0, fastdil = False, length = False, xy_scale = 1, z_scale = 1):
    """Internal method used for the secondary algorithm to pass around args in parallel."""

    if label is None:
        pass
    else:
        node_dict[label] = _get_node_edge_dict(sub_nodes, sub_edges, label, dilate_xy, dilate_z, cores = cores, search = search, fastdil = fastdil, length = length, xy_scale = xy_scale, z_scale = z_scale)


def quantify_edge_node(nodes, edges, search = 0, xy_scale = 1, z_scale = 1, cores = 0, resize = None, save = True, skele = False, length = False, auto = True, fastdil = False):

    def save_dubval_dict(dict, index_name, val1name, val2name, filename):

        #index name goes on the left, valname on the right
        df = pd.DataFrame.from_dict(dict, orient='index', columns=[val1name, val2name])

        # Rename the index to 'Node ID'
        df.index.name = index_name

        # Save DataFrame to Excel file
        df.to_excel(filename, engine='openpyxl')

    if type(nodes) is str:
        nodes = tifffile.imread(nodes)

    if type(edges) is str:
        edges = tifffile.imread(edges)

    if skele:
        if auto:
            edges = nettracer.skeletonize(edges)
            edges = nettracer.fill_holes_3d(edges)
        edges = nettracer.skeletonize(edges)
    else:
        edges = nettracer.binarize(edges)

    if len(np.unique(nodes)) == 2:
        nodes, num_nodes = nettracer.label_objects(nodes)
    else:
        num_nodes = np.max(nodes)

    if resize is not None:
        edges = zoom(edges, resize)
        nodes = zoom(nodes, resize)
        edges = nettracer.skeletonize(edges)

    if search > 0:
        dilate_xy, dilate_z = nettracer.dilation_length_to_pixels(xy_scale, z_scale, search, search)
    else:
        dilate_xy, dilate_z = 0, 0


    edge_quants = create_node_dictionary(nodes, edges, num_nodes, dilate_xy, dilate_z, cores = cores, search = search, fastdil = fastdil, length = length, xy_scale = xy_scale, z_scale = z_scale) #Find which edges connect which nodes and put them in a dictionary.

    if save:
    
        save_dubval_dict(edge_quants, 'NodeID', 'Edge Skele Quantity', 'Search Region Volume', 'edge_node_quantity.xlsx')

    else:

        return edge_quants


# Helper methods for counting the lens of skeletons:

def calculate_skeleton_lengths(skeleton_binary, xy_scale=1.0, z_scale=1.0, skeleton_coords = None):
    """
    Calculate total length of all skeletons in a 3D binary image.
    
    skeleton_binary: 3D boolean array where True = skeleton voxel
    xy_scale, z_scale: physical units per voxel
    """

    if skeleton_coords is None:
        # Find all skeleton voxels
        skeleton_coords = np.argwhere(skeleton_binary)
        shape = skeleton_binary.shape
    else:
        shape = skeleton_binary #Very professional stuff
    
    if len(skeleton_coords) == 0:
        return 0.0
    
    # Create a mapping from coordinates to indices for fast lookup
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(skeleton_coords)}
    
    # Build adjacency graph
    adjacency_list = build_adjacency_graph(skeleton_coords, coord_to_idx, shape)
    
    # Calculate lengths using scaled distances
    total_length = calculate_graph_length(skeleton_coords, adjacency_list, xy_scale, z_scale)
    
    return total_length

def build_adjacency_graph(skeleton_coords, coord_to_idx, shape):
    """Build adjacency list for skeleton voxels using 26-connectivity."""
    adjacency_list = [[] for _ in range(len(skeleton_coords))]
    
    # 26-connectivity offsets (all combinations of -1,0,1 except 0,0,0)
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if not (dx == 0 and dy == 0 and dz == 0):
                    offsets.append((dz, dy, dx))
    
    for idx, coord in enumerate(skeleton_coords):
        z, y, x = coord
        
        # Check all 26 neighbors
        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            
            # Check bounds
            if (0 <= nz < shape[0] and 
                0 <= ny < shape[1] and 
                0 <= nx < shape[2]):
                
                neighbor_coord = (nz, ny, nx)
                if neighbor_coord in coord_to_idx:
                    neighbor_idx = coord_to_idx[neighbor_coord]
                    adjacency_list[idx].append(neighbor_idx)
    
    return adjacency_list

def calculate_graph_length(skeleton_coords, adjacency_list, xy_scale, z_scale):
    """Calculate total length by summing distances between adjacent voxels."""
    total_length = 0.0
    processed_edges = set()
    
    for idx, neighbors in enumerate(adjacency_list):
        coord = skeleton_coords[idx]
        
        for neighbor_idx in neighbors:
            # Avoid double-counting edges
            edge = tuple(sorted([idx, neighbor_idx]))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)
            
            neighbor_coord = skeleton_coords[neighbor_idx]
            
            # Calculate scaled distance
            dz = (coord[0] - neighbor_coord[0]) * z_scale
            dy = (coord[1] - neighbor_coord[1]) * xy_scale
            dx = (coord[2] - neighbor_coord[2]) * xy_scale
            
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            total_length += distance
    
    return total_length

# End helper methods



def calculate_voxel_volumes(array, xy_scale=1, z_scale=1):
    """
    Calculate voxel volumes for each uniquely labelled object in a 3D numpy array.
    
    Args:
        array: 3D numpy array where different objects are marked with different integer labels
        xy_scale: Scale factor for x and y dimensions
        z_scale: Scale factor for z dimension
        
    Returns:
        Dictionary mapping object labels to their voxel volumes
    """

    labels = np.unique(array)
    if len(labels) == 2:
        array, _ = nettracer.label_objects(array)

    del labels
    
    # Get volumes using bincount
    if 0 in array:
        volumes = np.bincount(array.ravel())[1:]
    else:
        volumes = np.bincount(array.ravel())

    
    # Apply scaling
    volumes = volumes * (xy_scale**2) * z_scale
    
    # Create dictionary with label:volume pairs
    return {label: volume for label, volume in enumerate(volumes, start=1) if volume > 0}



def search_neighbor_ids(nodes, targets, id_dict, neighborhood_dict, totals, search, xy_scale, z_scale, root, fastdil = False):

    if 0 in targets:
        targets.remove(0)
    targets = np.isin(nodes, targets)
    targets = nettracer.binarize(targets)
        
    dilated = nettracer.dilate_3D_dt(targets, search, xy_scaling = xy_scale, z_scaling = z_scale, fast_dil = fastdil)
    dilated = dilated - targets #technically we dont need the cores
    search_vol = np.count_nonzero(dilated) * xy_scale * xy_scale * z_scale #need this for density
    targets = dilated != 0
    del dilated

    
    targets = targets * nodes
    
    unique, counts = np.unique(targets, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    del count_dict[0]
    
    unique, counts = np.unique(nodes, return_counts=True)
    total_dict = dict(zip(unique, counts))

    del total_dict[0]

    for label in total_dict:
        if label in id_dict:
            if label in count_dict:
                for iden in id_dict[label]:
                    neighborhood_dict[iden] += count_dict[label]
            for iden in id_dict[label]:
                totals[iden] += total_dict[label]


    try:
        del neighborhood_dict[root]  #no good way to get this
        del totals[root] #no good way to get this
    except:
        pass
    
    volume = nodes.shape[0] * nodes.shape[1] * nodes.shape[2] * xy_scale * xy_scale * z_scale
    densities = {}
    for nodeid, amount in totals.items():
        densities[nodeid] = (neighborhood_dict[nodeid]/search_vol)/(amount/volume)

    return neighborhood_dict, totals, densities



def get_search_space_dilate(target, centroids, id_dict, search, scaling = 1):

    ymax = np.max(centroids[:, 0])
    xmax = np.max(centroids[:, 1])


    array = np.zeros((ymax + 1, xmax + 1))

    for i, row in enumerate(centroids): 
        if i + 1 in id_dict and target in id_dict[i+1]:
            y = row[0]  # get y coordinate
            x = row[1]  # get x coordinate
            array[y, x] = 1  # set value at that coordinate


    #array = downsample(array, 3)
    array = dilate_2D(array, search, search)

    search_space = np.count_nonzero(array) * scaling * scaling

    tifffile.imwrite('search_regions.tif', array)

    print(f"Search space is {search_space}")



    return array


# Methods pertaining to getting radii:

def process_object_cpu(label, objects, labeled_array, xy_scale = 1, z_scale = 1):
    """
    Process a single labeled object to estimate its radius (CPU version).
    This function is designed to be called in parallel.
    
    Parameters:
    -----------
    label : int
        The label ID to process
    objects : list
        List of slice objects from ndimage.find_objects
    labeled_array : numpy.ndarray
        The full 3D labeled array
        
    Returns:
    --------
    tuple: (label, radius, mask_volume, dimensions)
    """
    # Get the slice object (bounding box) for this label
    # Index is label-1 because find_objects returns 0-indexed results
    obj_slice = objects[label-1]
    
    if obj_slice is None:
        return label, 0, 0, np.array([0, 0, 0])
    
    # Extract subarray containing just this object (plus padding)
    # Create padded slices to ensure there's background around the object
    padded_slices = []
    for dim_idx, dim_slice in enumerate(obj_slice):
        start = max(0, dim_slice.start - 1)
        stop = min(labeled_array.shape[dim_idx], dim_slice.stop + 1)
        padded_slices.append(slice(start, stop))
    
    # Extract the subarray
    subarray = labeled_array[tuple(padded_slices)]
    
    # Create binary mask for this object within the subarray
    mask = (subarray == label)


    """
    # Determine which dimension needs resampling
    if (z_scale > xy_scale) and mask.shape[0] != 1:
        # Z dimension needs to be stretched
        zoom_factor = [z_scale/xy_scale, 1, 1]  # Scale factor for [z, y, x]
        cardinal = xy_scale
    elif (xy_scale > z_scale) and mask.shape[0] != 1:
        # XY dimensions need to be stretched
        zoom_factor = [1, xy_scale/z_scale, xy_scale/z_scale]  # Scale factor for [z, y, x]
        cardinal = z_scale
    else:
        # Already uniform scaling, no need to resample
        zoom_factor = None
        cardinal = xy_scale

    # Resample the mask if needed
    if zoom_factor:
        mask = ndimage.zoom(mask, zoom_factor, order=0)  # Use order=0 for binary masks
    """
    
    # Compute distance transform on the smaller mask
    dist_transform = compute_distance_transform_distance(mask, sampling = [z_scale, xy_scale, xy_scale])
    
    # Filter out small values near the edge to focus on more central regions
    radius = np.max(dist_transform)
    
    return label, radius

def estimate_object_radii_cpu(labeled_array, n_jobs=None, mode = 0, xy_scale = 1, z_scale = 1):
    """
    Estimate the radii of labeled objects in a 3D numpy array using distance transform.
    CPU parallel implementation.
    
    Parameters:
    -----------
    labeled_array : numpy.ndarray
        3D array where each object has a unique integer label (0 is background)
    n_jobs : int or None
        Number of parallel jobs. If None, uses all available cores.
    
    Returns:
    --------
    dict: Dictionary mapping object labels to estimated radii
    dict: (optional) Dictionary of shape statistics for each label
    """

    if mode == 1:
        return estimate_object_radii_cross_section(
        labeled_array, n_jobs=None, xy_scale=xy_scale, z_scale=z_scale,
        statistic="median", return_profiles=False)

    elif mode == 0:

        # Find bounding box for each labeled object
        objects = ndimage.find_objects(labeled_array)
        
        unique_labels = np.unique(labeled_array)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background
        
        # Create a partial function for parallel processing
        process_func = partial(process_object_cpu, objects=objects, labeled_array=labeled_array, xy_scale = xy_scale, z_scale = z_scale)
        
        # Process objects in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs
            future_to_label = {executor.submit(process_func, label): label for label in unique_labels}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_label):
                results.append(future.result())
        
        # Organize results
        radii = {}
        
        for label, radius in results:
            radii[label] = radius
        
        return radii

def estimate_object_radii_gpu(labeled_array, xy_scale = 1, z_scale = 1):
    """
    Estimate the radii of labeled objects in a 3D numpy array using distance transform.
    GPU implementation using CuPy.
    
    Parameters:
    -----------
    labeled_array : numpy.ndarray
        3D array where each object has a unique integer label (0 is background)
    
    Returns:
    --------
    dict: Dictionary mapping object labels to estimated radii
    dict: (optional) Dictionary of shape statistics for each label
    """

    try:
        if not HAS_CUPY:
            raise ImportError("CuPy is required for GPU acceleration")

        # Find bounding box for each labeled object (on CPU)
        objects = ndimage.find_objects(labeled_array)
        
        # Transfer entire labeled array to GPU once
        labeled_array_gpu = cp.asarray(labeled_array)
        
        unique_labels = cp.unique(labeled_array_gpu)
        unique_labels = cp.asnumpy(unique_labels)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background
        
        radii = {}
        
        for label in unique_labels:
            # Get the slice object (bounding box) for this label
            obj_slice = objects[label-1]
            
            if obj_slice is None:
                continue
                
            # Extract subarray from GPU array
            padded_slices = []
            for dim_idx, dim_slice in enumerate(obj_slice):
                start = max(0, dim_slice.start - 1)
                stop = min(labeled_array.shape[dim_idx], dim_slice.stop + 1)
                padded_slices.append(slice(start, stop))
            
            # Create binary mask for this object (directly on GPU)
            mask_gpu = (labeled_array_gpu[tuple(padded_slices)] == label)

            """
            # Determine which dimension needs resampling
            if (z_scale > xy_scale) and mask_gpu.shape[0] != 1:
                # Z dimension needs to be stretched
                zoom_factor = [z_scale/xy_scale, 1, 1]  # Scale factor for [z, y, x]
                cardinal = xy_scale
            elif (xy_scale > z_scale) and mask_gpu.shape[0] != 1:
                # XY dimensions need to be stretched
                zoom_factor = [1, xy_scale/z_scale, xy_scale/z_scale]  # Scale factor for [z, y, x]
                cardinal = z_scale
            else:
                # Already uniform scaling, no need to resample
                zoom_factor = None
                cardinal = xy_scale

            # Resample the mask if needed
            if zoom_factor:
                mask_gpu = cpx.zoom(mask_gpu, zoom_factor, order=0)  # Use order=0 for binary masks
            """
            
            # Compute distance transform on GPU
            dist_transform_gpu = compute_distance_transform_distance_GPU(mask_gpu, sampling = [z_scale, xy_scale, xy_scale])
        
            radius = float(cp.max(dist_transform_gpu).get())


            # Store the radius and the scaled radius
            radii[label] = radius
        
        # Clean up GPU memory
        del labeled_array_gpu
            
        return radii

    except Exception as e:
        print(f"GPU calculation failed, trying CPU instead -> {e}")
        return estimate_object_radii_cpu(labeled_array)

def compute_distance_transform_distance_GPU(nodes, sampling = [1,1,1]):

    is_pseudo_3d = nodes.shape[0] == 1
    if is_pseudo_3d:
        nodes = cp.squeeze(nodes)  # Convert to 2D for processing
        sampling = [sampling[1], sampling[2]]
    
    # Compute the distance transform on the GPU
    distance = cpx.distance_transform_edt(nodes, sampling = sampling)

    if is_pseudo_3d:
        cp.expand_dims(distance, axis = 0)
    
    return distance    


def compute_distance_transform_distance(nodes, sampling = [1,1,1]):

    is_pseudo_3d = nodes.shape[0] == 1
    if is_pseudo_3d:
        nodes = np.squeeze(nodes)  # Convert to 2D for processing
        sampling = [sampling[1], sampling[2]]

    # Fallback to CPU if there's an issue with GPU computation
    distance = ndimage.distance_transform_edt(nodes, sampling = sampling)
    if is_pseudo_3d:
        np.expand_dims(distance, axis = 0)
    return distance


_OFFSETS = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
 
 
def _skeleton_paths(skel):
    """
    Split a thin binary skeleton into ordered, non-branching paths.
 
    Vertices with degree > 2 (junctions) are removed; each remaining
    connected component is walked from one end to produce an ordered
    coordinate array. Closed loops have no endpoint and are started
    arbitrarily.
 
    Returns list of (n, 3) int arrays of voxel coordinates.
    """
    coords = np.argwhere(skel)
    if coords.shape[0] == 0:
        return []
 
    index = {tuple(c): i for i, c in enumerate(coords)}
    nbrs = [[] for _ in range(coords.shape[0])]
    for i, c in enumerate(coords):
        for o in _OFFSETS:
            j = index.get((c[0] + o[0], c[1] + o[1], c[2] + o[2]))
            if j is not None:
                nbrs[i].append(j)
 
    keep = np.array([len(n) <= 2 for n in nbrs])
    visited = np.zeros(coords.shape[0], dtype=bool)
    paths = []
 
    for seed in range(coords.shape[0]):
        if not keep[seed] or visited[seed]:
            continue
 
        comp, stack = [], [seed]
        visited[seed] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in nbrs[u]:
                if keep[v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)
 
        compset = set(comp)
        ends = [u for u in comp
                if sum(1 for v in nbrs[u] if v in compset) <= 1]
        cur = ends[0] if ends else comp[0]
 
        order, seen = [cur], {cur}
        while True:
            nxt = [v for v in nbrs[cur] if v in compset and v not in seen]
            if not nxt:
                break
            cur = nxt[0]
            seen.add(cur)
            order.append(cur)
 
        paths.append(coords[order])
 
    return paths
 
 
def _native_skeleton(mask):
    """
    Skeletonize -> fill holes -> skeletonize, on the native voxel grid.
 
    The second pass re-thins any region the fill step restored to solid.
    For structures with no enclosed voids the fill is a no-op and the second
    pass returns the first skeleton unchanged, so this costs a thinning pass
    but never changes correct results.
 
    Returns list of (n, 3) float arrays of native voxel coordinates.
    """
    skel = skeletonize(mask)
 
    skel = nettracer.fill_holes_3d(skel)
    skel = skeletonize(skel)
 
    return [np.asarray(p, dtype=np.float64) for p in _skeleton_paths(skel)]
 
 
def _snap_to_foreground(points, mask, nearest_idx):
    """
    Round to integer voxels and snap any that landed on background to the
    nearest foreground voxel. xs3d requires the seed point to be inside the
    shape; smoothing can pull a point off the medial axis on tight curves.
 
    `nearest_idx` is the return_indices output of the background EDT,
    computed once per mask by the caller.
    """
    pts = np.rint(points).astype(np.int64)
    for a in range(3):
        pts[:, a] = np.clip(pts[:, a], 0, mask.shape[a] - 1)
 
    inside = mask[pts[:, 0], pts[:, 1], pts[:, 2]]
    if not inside.all():
        bad = ~inside
        b = pts[bad]
        pts[bad] = np.stack([nearest_idx[a][b[:, 0], b[:, 1], b[:, 2]]
                             for a in range(3)], axis=1)
    return pts
 
 
def _smooth(path, window):
    """Moving average along a path, endpoints handled by edge padding."""
    if window <= 1 or path.shape[0] < 3:
        return path
    w = min(window, path.shape[0])
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return path
    pad = w // 2
    padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
    kern = np.ones(w) / w
    return np.stack([np.convolve(padded[:, a], kern, mode="valid")
                     for a in range(3)], axis=1)
 
 
# ----------------------------------------------------------------------
# per-object measurement
# ----------------------------------------------------------------------
 
def _aggregate(values, statistic):
    if values.size == 0:
        return np.nan
    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "min":
        return float(np.min(values))
    if statistic == "max":
        return float(np.max(values))
    if isinstance(statistic, (int, float)):
        return float(np.percentile(values, statistic))
    raise ValueError(f"unknown statistic: {statistic!r}")
 
 
def measure_mask_cross_sections(
    mask,
    xy_scale=1.0,
    z_scale=1.0,
    tangent_window=5,
    smoothing_window=5,
    trim_ends=2,
    min_path_length=3,
    exclude_border_contacts=True):
    """
    Cross-sectional radii along the medial axis of a single binary mask.
 
    No resampling occurs; anisotropy is applied to the tangent vector and
    passed to xs3d. `tangent_window` and `min_path_length` count skeleton
    VERTICES, so a path running along the coarse axis has fewer of them for
    the same physical length -- min_path_length is deliberately low.
 
    Returns (radii, positions) where radii is a 1-D array in physical units
    and positions is the (n, 3) array of native voxel coords they came from.
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.ndim != 3 or 1 in mask.shape or not mask.any():
        return np.array([]), np.zeros((0, 3), dtype=np.int64)
 
    anisotropy = np.array([z_scale, xy_scale, xy_scale], dtype=np.float64)
    scale_sq = anisotropy ** 2
    fmask = np.asfortranarray(mask)
 
    # one background EDT per mask, reused by every path
    _, nearest_idx = ndimage.distance_transform_edt(~mask, return_indices=True)
 
    radii, positions = [], []
 
    for path in _native_skeleton(mask):
        if path.shape[0] < min_path_length:
            continue
 
        sm = _smooth(path, smoothing_window)
        pts = _snap_to_foreground(sm, mask, nearest_idx)
        n = pts.shape[0]
 
        lo = min(trim_ends, max(0, (n - 1) // 2))
        hi = max(lo + 1, n - trim_ends)
        hi = min(hi, n)
 
        for i in range(lo, hi):
            a = max(0, i - tangent_window)
            b = min(n - 1, i + tangent_window)
            t_vox = sm[b] - sm[a]
            if not np.any(t_vox):
                continue
 
            # covector transform -- see module docstring
            normal = (t_vox * scale_sq).astype(np.float32)
 
            area, contact = xs3d.cross_sectional_area(
                fmask, pts[i].astype(np.int32), normal,
                anisotropy, return_contact=True,
            )
            if area <= 0 or not np.isfinite(area):
                continue
            if exclude_border_contacts and contact != 0:
                continue
 
            radii.append(np.sqrt(area / np.pi))
            positions.append(pts[i])
 
    if not radii:
        return np.array([]), np.zeros((0, 3), dtype=np.int64)
    return np.asarray(radii), np.asarray(positions)
 
 
def _edt_radius(mask, sampling):
    if not mask.any():
        return 0.0
    return float(ndimage.distance_transform_edt(mask, sampling=sampling).max())
 
 

def process_object_cross_section(
    label, objects, labeled_array, xy_scale=1.0, z_scale=1.0,
    statistic="median", pad=3, fallback_to_edt=True, **kwargs
):
    """
    Drop-in replacement for process_object_cpu. Same signature and return
    shape (label, radius), so the existing ThreadPoolExecutor loop is
    unchanged.
 
    `pad` is larger than the original 1 voxel: an oblique section plane can
    reach the subvolume face even when the object does not, and those
    sections get discarded as border contacts. More padding keeps them.
    """
    obj_slice = objects[label - 1]
    if obj_slice is None:
        return label, 0.0
 
    padded = tuple(
        slice(max(0, s.start - pad), min(labeled_array.shape[i], s.stop + pad))
        for i, s in enumerate(obj_slice)
    )
    mask = labeled_array[padded] == label
 
    radii, _ = measure_mask_cross_sections(
        mask, xy_scale=xy_scale, z_scale=z_scale, **kwargs
    )
 
    if radii.size:
        return label, _aggregate(radii, statistic)
    if fallback_to_edt:
        return label, _edt_radius(mask, [z_scale, xy_scale, xy_scale])
    return label, np.nan
 
 
def estimate_object_radii_cross_section(
    labeled_array, n_jobs=None, xy_scale=1.0, z_scale=1.0,
    statistic="median", pad=3, return_profiles=False, **kwargs
):
    """
    Cross-section radius per label. Mirrors estimate_object_radii_cpu.
 
    Runs serially -- xs3d releases the GIL poorly enough that threads buy
    little, and if you want real parallelism use ProcessPoolExecutor with a
    'spawn' context so a native crash cannot take down the host process.
    """
    objects = ndimage.find_objects(labeled_array)
    labels = np.unique(labeled_array)
    labels = labels[labels != 0]
 
    radii, profiles = {}, {}
    for label in labels:
        label = int(label)
        obj_slice = objects[label - 1]
        if obj_slice is None:
            radii[label] = 0.0
            if return_profiles:
                profiles[label] = (np.array([]), np.zeros((0, 3), dtype=np.int64))
            continue
 
        padded = tuple(
            slice(max(0, s.start - pad), min(labeled_array.shape[i], s.stop + pad))
            for i, s in enumerate(obj_slice)
        )
        mask = labeled_array[padded] == label
 
        prof = measure_mask_cross_sections(
            mask, xy_scale=xy_scale, z_scale=z_scale, **kwargs
        )
        if prof[0].size:
            radii[label] = _aggregate(prof[0], statistic)
        else:
            radii[label] = _edt_radius(mask, [z_scale, xy_scale, xy_scale])
 
        if return_profiles:
            profiles[label] = prof
 
    if return_profiles:
        return radii, profiles
    return radii