import tifffile
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from skimage.segmentation import watershed
import cv2
import os
try:
    import edt
    print("Parallel search functions enabled")
except:
    print("Some parallel search functions disabled (requires edt package), will fall back to single-threaded")
import math
import re
from . import nettracer
from multiprocessing import shared_memory
import multiprocessing as mp
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx
except:
    pass
import threading


def dilate_3D(tiff_array, dilated_x, dilated_y, dilated_z):
    """Internal method to dilate an array in 3D. Dilation this way is much faster than using a distance transform although the latter is theoretically more accurate.
    Arguments are an array,  and the desired pixel dilation amounts in X, Y, Z."""

    if tiff_array.shape[0] == 1:
        return nettracer.dilate_2D(tiff_array, ((dilated_x - 1) / 2))

    if dilated_x == 3 and dilated_y == 3  and dilated_z == 3:

        return dilate_3D_old(tiff_array, dilated_x, dilated_y, dilated_z)

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
    import numpy as np
    from scipy import ndimage
    
    # Handle special case for 2D arrays
    if tiff_array.shape[0] == 1:
        # Call 2D dilation function if needed
        return nettracer.dilate_2D(tiff_array, 1)  # For a 3x3 kernel, radius is 1
    
    # Create a simple 3x3x3 cubic kernel (all ones)
    kernel = np.ones((3, 3, 3), dtype=bool)
    
    # Perform binary dilation
    dilated_array = ndimage.binary_dilation(tiff_array.astype(bool), structure=kernel)
    
    return dilated_array.astype(np.uint8)


def dilate_3D_dt(array, search_distance, xy_scaling=1.0, z_scaling=1.0, GPU = False):
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

    # Determine which dimension needs resampling. the moral of the story is read documentation before you do something unecessary.
    """
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
    """

    # Resample the mask if needed
    #if zoom_factor:
        #array = ndimage.zoom(array, zoom_factor, order=0)  # Use order=0 for binary masks

    # Invert the array (find background)
    inv = array < 1
    
    if GPU:
        try:
            print("Attempting on GPU...")
            inv, indices = compute_distance_transform_GPU(inv, return_dists = True, sampling = [z_scaling, xy_scaling, xy_scaling])
        except:
            print("Failed, attempting on CPU...")
            cleanup()
            #Who would have seen this coming?:
            inv, indices = compute_distance_transform(inv, return_dists = True, sampling = [z_scaling, xy_scaling, xy_scaling])
    else:
        inv, indices = compute_distance_transform(inv, return_dists = True, sampling = [z_scaling, xy_scaling, xy_scaling])


    #inv = inv * cardinal
    
    # Threshold the distance transform to get dilated result
    inv = inv <= search_distance

    return inv.astype(np.uint8), indices, array




def binarize(image):
    """Convert an array from numerical values to boolean mask"""
    return (image != 0).astype(np.uint8)

def invert_array(array):
    """Used to flip glom array indices. 0 becomes 1 and vice versa."""
    return np.logical_not(array).astype(np.uint8)

def process_chunk(start_idx, end_idx, nodes, ring_mask, nearest_label_indices):
    nodes_chunk = nodes[:, start_idx:end_idx, :]
    ring_mask_chunk = ring_mask[:, start_idx:end_idx, :]
    dilated_nodes_with_labels_chunk = np.copy(nodes_chunk)
    
    # Get all ring indices at once
    ring_indices = np.argwhere(ring_mask_chunk)
    
    if len(ring_indices) > 0:
        # Extract coordinates
        z_coords = ring_indices[:, 0]
        y_coords = ring_indices[:, 1] 
        x_coords = ring_indices[:, 2]
        
        # Get nearest label coordinates (adjust y for chunk offset)
        nearest_coords = nearest_label_indices[:, z_coords, y_coords + start_idx, x_coords]
        nearest_z = nearest_coords[0, :]
        nearest_y = nearest_coords[1, :]
        nearest_x = nearest_coords[2, :]
        
        # Vectorized assignment
        try:
            dilated_nodes_with_labels_chunk[z_coords, y_coords, x_coords] = \
                nodes[nearest_z, nearest_y, nearest_x]
        except IndexError:
            # Fallback for any problematic indices
            valid_mask = (nearest_z < nodes.shape[0]) & \
                        (nearest_y < nodes.shape[1]) & \
                        (nearest_x < nodes.shape[2]) & \
                        (nearest_z >= 0) & (nearest_y >= 0) & (nearest_x >= 0)
            
            valid_indices = valid_mask.nonzero()[0]
            if len(valid_indices) > 0:
                dilated_nodes_with_labels_chunk[z_coords[valid_indices], y_coords[valid_indices], x_coords[valid_indices]] = \
                    nodes[nearest_z[valid_indices], nearest_y[valid_indices], nearest_x[valid_indices]]
    
    return dilated_nodes_with_labels_chunk

def smart_dilate(nodes, dilate_xy = 0, dilate_z = 0, directory = None, GPU = True, fast_dil = True, predownsample = None, use_dt_dil_amount = None, xy_scale = 1, z_scale = 1):

    if fast_dil:
        try:
            import edt
            dilated = nettracer.dilate_3D_dt(nodes, use_dt_dil_amount, xy_scale, z_scale, fast_dil = True)
            return smart_label_tiled(dilated, nodes, xy_scale=xy_scale, z_scale=z_scale, directory=None, remove_template=False, amount=use_dt_dil_amount, method = 'transform')
        except:
            import traceback
            print(traceback.format_exc())
            print("edt package not found, using alternate parallel search method...")
            try:
                return smart_dilate_short_parallel(nodes, use_dt_dil_amount, directory, xy_scale, z_scale)
            except:
                print("Parallel attempt failed, trying singleton...")
                return smart_dilate_short(nodes, use_dt_dil_amount, directory, xy_scale, z_scale)

    else:
        return smart_dilate_short(nodes, use_dt_dil_amount, directory, xy_scale, z_scale)

def smart_dilate_short(nodes, amount = None, directory = None, xy_scale = 1, z_scale = 1):

    original_shape = nodes.shape

    print("Performing distance transform for smart search...")

    dilated_binary_nodes, nearest_label_indices, nodes = dilate_3D_dt(nodes, amount, xy_scaling = xy_scale, z_scaling = z_scale)
    binary_nodes = binarize(nodes)
    ring_mask = dilated_binary_nodes & (~binary_nodes)
    del dilated_binary_nodes
    del binary_nodes

    # Step 5: Process in parallel chunks using ThreadPoolExecutor
    num_cores = mp.cpu_count()  # Use all available CPU cores
    chunk_size = nodes.shape[1] // num_cores  # Divide the array into chunks along the y-axis

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [(i * chunk_size, (i + 1) * chunk_size if i != num_cores - 1 else nodes.shape[1], nodes, ring_mask, nearest_label_indices) for i in range(num_cores)]
        results = list(executor.map(lambda args: process_chunk(*args), args_list))

    del ring_mask
    del nodes
    del nearest_label_indices

    # Combine results from chunks
    dilated_nodes_with_labels = np.concatenate(results, axis=1)

    if directory is not None:
        try:
            tifffile.imwrite(f"{directory}/search_region.tif", dilated_nodes_with_labels)
        except Exception as e:
            print(f"Could not save search region file to {directory}")

    return dilated_nodes_with_labels



def smart_dilate_short_parallel(nodes, amount=None, directory=None, xy_scale=1, z_scale=1,
                       work_ratio_limit=4.0, use_boxes=True):
    """
    Label-preserving dilation of `nodes` by `amount`, scipy only.
 
    Runs the distance transform on padded per-label bounding boxes when the
    label geometry makes that cheap, and on the whole volume otherwise.
 
    work_ratio_limit -- max summed padded-box volume, as a multiple of the array
        volume, before falling back to the whole-volume transform.
    use_boxes -- set False to force the original whole-volume path.
    """
 
    if use_boxes and amount is not None:
        try:
            if nodes.min() < 0:
                raise ValueError("negative labels are not supported by find_objects")
 
            ratio, objs, pad = estimate_box_work(nodes, amount, xy_scale, z_scale)
            n_labels = sum(1 for o in objs if o is not None)
 
            if n_labels == 0:
                return np.zeros_like(nodes)
 
            if ratio <= work_ratio_limit:
                print(f"Performing boxed distance transform for smart search "
                      f"({n_labels} labels, {ratio:.2f}x array volume)...")
                dilated_nodes_with_labels = _boxed_smart_dilate(
                    nodes, amount, objs, pad, xy_scale=xy_scale, z_scale=z_scale
                )
                if directory is not None:
                    try:
                        tifffile.imwrite(f"{directory}/search_region.tif",
                                         dilated_nodes_with_labels)
                    except Exception:
                        print(f"Could not save search region file to {directory}")
                return dilated_nodes_with_labels
 
            print(f"Boxed transform would touch {ratio:.2f}x the array volume "
                  f"(limit {work_ratio_limit}); using whole-volume transform...")
            del objs
 
        except Exception as e:
            print(f"Boxed distance transform unavailable ({e}); "
                  f"using whole-volume transform...")
 
    # ---------------- original whole-volume path, unchanged ----------------
    print("Performing distance transform for smart search...")
 
    dilated_binary_nodes, nearest_label_indices, nodes = dilate_3D_dt(
        nodes, amount, xy_scaling=xy_scale, z_scaling=z_scale
    )
    binary_nodes = binarize(nodes)
    ring_mask = dilated_binary_nodes & (~binary_nodes)
    del dilated_binary_nodes
    del binary_nodes
 
    num_cores = mp.cpu_count()
    chunk_size = nodes.shape[1] // num_cores
 
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [
            (i * chunk_size,
             (i + 1) * chunk_size if i != num_cores - 1 else nodes.shape[1],
             nodes, ring_mask, nearest_label_indices)
            for i in range(num_cores)
        ]
        results = list(executor.map(lambda args: process_chunk(*args), args_list))
 
    del ring_mask
    del nodes
    del nearest_label_indices
 
    dilated_nodes_with_labels = np.concatenate(results, axis=1)
 
    if directory is not None:
        try:
            tifffile.imwrite(f"{directory}/search_region.tif",
                             dilated_nodes_with_labels)
        except Exception:
            print(f"Could not save search region file to {directory}")
 
    return dilated_nodes_with_labels

def round_to_odd(number):
    rounded = round(number)
    # If the rounded number is even, add or subtract 1 to make it odd
    if rounded % 2 == 0:
        if number > 0:
            rounded += 1
        else:
            rounded -= 1
    return rounded

def smart_label_watershed(binary_array, label_array, directory=None,
                          remove_template=False):
    """
    Watershed-based label propagation - lower memory footprint than the
    feature transform, at the cost of being geodesic rather than Euclidean.
    """
    string_bool = isinstance(binary_array, str) or isinstance(label_array, str)
    if isinstance(binary_array, str):
        binary_array = tifffile.imread(binary_array)
    if isinstance(label_array, str):
        label_array = tifffile.imread(label_array)

    binary_array = binarize(binary_array)

    # skimage's heap stores its FIFO tie-breaker in an int32 field
    # (heap_watershed.pxi: `cnp.int32_t age`), incremented once per masked
    # voxel on this code path.  Past ~2.1e9 it wraps negative.  Because the
    # elevation here is flat, every heap comparison is a tie and `age` IS the
    # ordering -- so a wrap silently scrambles the propagation rather than
    # merely perturbing boundaries.  Nothing about the input dtypes affects
    # this; the field is baked into the compiled extension.
    masked = int(binary_array.sum())
    if masked > 2_000_000_000:
        print(f"Watershed unsafe at {masked:,} masked voxels (int32 age "
              f"counter overflows); using the feature transform instead.")
        return smart_label_whole(binary_array, label_array, directory,
                                 remove_template)

    print("Performing watershed label propagation...")

    # skimage casts markers through its np_anyint fused type and hands the
    # dtype straight back, so an integer label array is passed through as-is.
    if not np.issubdtype(label_array.dtype, np.integer):
        label_array = label_array.astype(np.int32, copy=False)

    # no elevation cast: _validate_inputs does image.astype(np.float64)
    # regardless, so pre-converting to float32 only adds a wasted copy
    out = watershed(binary_array, markers=label_array,
                    mask=binary_array, compactness=0)

    if remove_template:
        out *= binary_array

    if string_bool:
        path = (f"{directory}/smart_labelled_array.tif" if directory is not None
                else "smart_labelled_array.tif")
        try:
            tifffile.imwrite(path, out)
        except Exception:
            where = directory if directory is not None else "active directory"
            print(f"Could not save search region file to {where}")

    return out


def smart_label_single(binary_array, label_array):

    # Step 1: Binarize the labeled array
    binary_core = binarize(label_array)
    binary_array = binarize(binary_array)

    # Step 3: Isolate the ring (binary dilated mask minus original binary mask)
    ring_mask = binary_array & invert_array(binary_core)

    nearest_label_indices = compute_distance_transform(invert_array(label_array))

    # Step 5: Process the entire array without parallelization
    dilated_nodes_with_labels = process_chunk(0, label_array.shape[1], label_array, ring_mask, nearest_label_indices)

    dilated_nodes_with_labels = dilated_nodes_with_labels * binary_array


    return dilated_nodes_with_labels

def neighbor_label(binary_array, label_array):

    binary_array = binary_array != 0

    targets = binary_array - binarize(label_array)

    label_array = smart_dilate(label_array, 3, 3, GPU = False, fast_dil = False, xy_scale = 1, z_scale = 1)

    return None 

    #TBD: This requires the serial dilation strategy methinks, unfortunately. Get labeled components of all outer regions of the binary array to be labeled. For each, cut out a sub array from the old label array and pad by one in all dimensions. Apply boolean threshold to get just the label we want. Apply binary dilation on the binary subarray of just one and multiply that against the labeled sub array - we have isolated the available labelers. Finally do smart dilate on this sub array region. 






def compute_distance_transform_GPU(nodes, return_dists = False, sampling = [1, 1, 1]):
    is_pseudo_3d = nodes.shape[0] == 1
    if is_pseudo_3d:
        nodes = np.squeeze(nodes)  # Convert to 2D for processing
        sampling = [sampling[1], sampling[2]]
    
    # Convert numpy array to CuPy array
    nodes_cp = cp.asarray(nodes)
    
    # Compute the distance transform on the GPU
    dists, nearest_label_indices = cpx.distance_transform_edt(nodes_cp, return_indices=True, sampling = sampling)
    
    # Convert results back to numpy arrays
    nearest_label_indices_np = cp.asnumpy(nearest_label_indices)
    
    if is_pseudo_3d:
        # For 2D input, we get (2, H, W) but need (3, 1, H, W)
        H, W = nearest_label_indices_np[0].shape
        indices_4d = np.zeros((3, 1, H, W), dtype=nearest_label_indices_np.dtype)
        indices_4d[1:, 0] = nearest_label_indices_np  # Copy Y and X coordinates
        # indices_4d[0] stays 0 for all Z coordinates
        nearest_label_indices_np = indices_4d

    if not return_dists:

        return nearest_label_indices_np

    else:
        dists = cp.asnumpy(dists)

        return dists, nearest_label_indices_np


def compute_distance_transform(nodes, return_dists=False, sampling=[1, 1, 1]):
    is_pseudo_3d = nodes.shape[0] == 1
    if is_pseudo_3d:
        nodes = nodes[0]
        sampling = [sampling[1], sampling[2]]

    if return_dists:
        dists, nearest_label_indices = distance_transform_edt(
            nodes, return_indices=True, sampling=sampling)
    else:
        dists = None
        nearest_label_indices = distance_transform_edt(
            nodes, return_indices=True, return_distances=False, sampling=sampling)

    if is_pseudo_3d:
        H, W = nearest_label_indices[0].shape
        indices_4d = np.zeros((3, 1, H, W), dtype=nearest_label_indices.dtype)
        indices_4d[1:, 0] = nearest_label_indices
        nearest_label_indices = indices_4d
        if dists is not None:
            dists = np.expand_dims(dists, axis=0)

    return (dists, nearest_label_indices) if return_dists else nearest_label_indices



def compute_distance_transform_distance_GPU(nodes, sampling = [1, 1, 1]):

    is_pseudo_3d = nodes.shape[0] == 1
    if is_pseudo_3d:
        nodes = np.squeeze(nodes)  # Convert to 2D for processing
        sampling = [sampling[1], sampling[2]]

    # Convert numpy array to CuPy array
    nodes_cp = cp.asarray(nodes)
    
    # Compute the distance transform on the GPU
    distance = cpx.distance_transform_edt(nodes_cp, sampling = sampling)
    
    # Convert results back to numpy arrays
    distance = cp.asnumpy(distance)

    if is_pseudo_3d:
        distance = np.expand_dims(distance, axis = 0)
    
    return distance    


def _run_edt_in_process_shm(input_shm_name, output_shm_name, shape, dtype_str, sampling_tuple):
    """Helper function to run edt in a separate process using shared memory."""
    import edt  # Import here to ensure it's available in child process
    
    input_shm = shared_memory.SharedMemory(name=input_shm_name)
    output_shm = shared_memory.SharedMemory(name=output_shm_name)
    
    try:
        nodes_arr = np.ndarray(shape, dtype=dtype_str, buffer=input_shm.buf)
        
        n_cores = mp.cpu_count()
        result = edt.edt(
            nodes_arr.astype(bool),
            anisotropy=sampling_tuple,
            parallel=n_cores
        )
        
        result_array = np.ndarray(result.shape, dtype=result.dtype, buffer=output_shm.buf)
        np.copyto(result_array, result)
        
        return result.shape, str(result.dtype)
    finally:
        input_shm.close()
        output_shm.close()

def compute_distance_transform_distance(nodes, sampling=[1, 1, 1], fast_dil=False,
                                        threshold=None):
    """
    Compute distance transform with automatic parallelization when available.

    Args:
        nodes: Binary array (True/1 for objects)
        sampling: Voxel spacing [z, y, x] for anisotropic data
        threshold: If given, returns (distance <= threshold) as a uint8 mask
                   instead of the distance array, without ever materializing
                   a second float buffer.

    Returns:
        Distance transform array, or thresholded uint8 mask if threshold given.
    """
    is_pseudo_3d = nodes.shape[0] == 1

    if is_pseudo_3d:
        nodes = np.squeeze(nodes)
        sampling = [sampling[1], sampling[2]]

    # edt requires a contiguous buffer; no-op in the common case
    if not (nodes.flags['C_CONTIGUOUS'] or nodes.flags['F_CONTIGUOUS']):
        nodes = np.ascontiguousarray(nodes)

    distance = None

    if fast_dil:
        input_shm = output_shm = None
        try:
            input_shm = shared_memory.SharedMemory(create=True, size=nodes.nbytes)
            output_shm = shared_memory.SharedMemory(
                create=True, size=nodes.size * np.dtype(np.float32).itemsize)

            shm_array = np.ndarray(nodes.shape, dtype=nodes.dtype, buffer=input_shm.buf)
            np.copyto(shm_array, nodes)
            del shm_array   # drop the buffer export before close()

            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
                future = executor.submit(
                    _run_edt_in_process_shm,
                    input_shm.name,
                    output_shm.name,
                    nodes.shape,
                    str(nodes.dtype),
                    tuple(sampling),
                )
                result_shape, result_dtype = future.result()

            # free the input before allocating anything else
            input_shm.close()
            input_shm.unlink()
            input_shm = None

            view = np.ndarray(result_shape, dtype=np.dtype(result_dtype),
                              buffer=output_shm.buf)
            if threshold is not None:
                distance = (view <= threshold).view(np.uint8)   # 1N, not 4N
            else:
                distance = view.copy()
            del view

        except Exception as e:
            print(f"Parallel distance transform failed ({e}), falling back to scipy")
            try:
                import edt
                import traceback
                traceback.print_exc()
            except ImportError:
                print("edt package not found. Please use 'pip install edt' "
                      "if you would like to enable parallel searching.")
        finally:
            for shm in (input_shm, output_shm):
                if shm is not None:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass

    if distance is None:
        distance = distance_transform_edt(nodes, sampling=sampling)
        if threshold is not None:
            distance = (distance <= threshold).view(np.uint8)

    if is_pseudo_3d:
        distance = np.expand_dims(distance, axis=0)

    return distance



def gaussian(search_region, GPU = True):
    try:
        if GPU == True and cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy for guassian blur.")

            # Convert to CuPy array
            search_region_cp = cp.asarray(search_region)

            # Apply Gaussian filter
            blurred_search_cp = cpx.gaussian_filter(search_region_cp, sigma=1)

            # Convert back to NumPy array if needed
            blurred_search = cp.asnumpy(blurred_search_cp)

            return blurred_search
        else:
            print("Using CPU for guassian blur")
            blurred_search = gaussian_filter(search_region, sigma = 1)
            return blurred_search
    except Exception as e:
        print("GPU blur failed or did not detect GPU (cupy must be installed with a CUDA toolkit setup...). Computing CPU guassian blur instead.")
        print(f"Error message: {str(e)}")
        # Fallback to CPU if there's an issue with GPU computation
        blurred_search = gaussian_filter(search_region, sigma = 1)
        return blurred_search



def catch_memory(e):


    # Get the current GPU device
    device = cp.cuda.Device()

    # Get total memory in bytes
    total_memory = device.mem_info[1]

    # Capture the error message
    error_message = str(e)
    print(f"Error encountered: {error_message}")

    # Use regex to extract the memory required from the error message
    match = re.search(r'allocating ([\d,]+) bytes', error_message)

    if match:
        memory_required = int(match.group(1).replace(',', ''))

        print(f"GPU Memory required for distance transform: {memory_required}, retrying with temporary downsample")

        downsample_needed = (memory_required/total_memory)
        return (downsample_needed)

def cleanup():

    try:
        cp.get_default_memory_pool().free_all_blocks()
    except:
        pass

# Smart dilate optimizations:
 
 
def _padded_slices(sl, pad, shape):
    """Grow a find_objects slice tuple by `pad` voxels per axis, clipped to shape."""
    return tuple(
        slice(max(0, s.start - p), min(n, s.stop + p))
        for s, p, n in zip(sl, pad, shape)
    )
 
 
def estimate_box_work(nodes, amount, xy_scale=1, z_scale=1, objs=None):
    """
    Estimate the cost of the per-label bounding-box strategy.
 
    Returns (ratio, objs, pad).  `ratio` is the summed volume of the padded
    per-label bounding boxes over the volume of the whole array -- i.e. how many
    times over the boxed method will touch the image.  Well under 1.0 means the
    labels are sparse and boxing is nearly free; >> 1 means labels are large or
    heavily interleaved and one whole-volume transform is the better buy.
 
    `objs` and `pad` are returned so the caller never runs find_objects twice.
    """ 
    shape = nodes.shape
 
    # Euclidean distance bounds each axis component, so padding the box by
    # amount/spacing voxels per axis is enough to contain everything within
    # `amount` of the label.
    pad = tuple(int(math.ceil(amount / s)) + 1
                for s in (float(z_scale), float(xy_scale), float(xy_scale)))
    if shape[0] == 1:                       # pseudo-3D: never pad in z
        pad = (0, pad[1], pad[2])
 
    if objs is None:
        objs = ndimage.find_objects(nodes)
 
    total = 0.0
    for sl in objs:
        if sl is None:
            continue
        vol = 1.0
        for s in _padded_slices(sl, pad, shape):
            vol *= (s.stop - s.start)
        total += vol
 
    return total / float(nodes.size), objs, pad
 
 
def _boxed_smart_dilate(nodes, amount, objs, pad, xy_scale=1, z_scale=1,
                        num_cores=None):
    """
    Whole-volume result assembled from per-label bounding boxes.
 
    Each box is a padded crop around one label, on which the ordinary
    dilate_3D_dt + process_chunk pair is run.  A crop only sees the labels
    inside it, so a crop's answer can be wrong (too far) for voxels whose true
    nearest label lies outside.  That is fixed by merging crops on distance
    rather than trusting any one of them:
 
      For a voxel v in the dilated region, let M be its true nearest label.
      dist(v, M) <= amount, so v falls inside M's padded box, and M's own voxels
      are inside that box by construction.  M's crop therefore reports exactly
      dist(v, M).  Every other crop reports a distance to some real label, which
      can only be >= dist(v, M).  So the per-voxel minimum across crops is the
      true global answer, and no crop can beat it with a wrong label.
 
    dilate_3D_dt throws away the distances (it returns the thresholded mask),
    but they are recoverable for free from the index array it does return:
    the distance at v is just |(v - ind[v]) * spacing|.
    """
 
    if num_cores is None:
        num_cores = mp.cpu_count()
 
    shape = nodes.shape
    sampling = (float(z_scale), float(xy_scale), float(xy_scale))
 
    best_d2 = np.full(shape, np.inf, dtype=np.float32)   # squared, avoids sqrt
    out = np.zeros(shape, dtype=nodes.dtype)
    lock = threading.Lock()
 
    work = [(i + 1, sl) for i, sl in enumerate(objs) if sl is not None]
 
    def one_box(item):
        label, sl = item
        sub_sl = _padded_slices(sl, pad, shape)
        sub = np.ascontiguousarray(nodes[sub_sl])
 
        # ---- the ordinary scipy path, on the crop ----
        dilated_binary_nodes, nearest_label_indices, sub = dilate_3D_dt(
            sub, amount, xy_scaling=xy_scale, z_scaling=z_scale
        )
        ring_mask = dilated_binary_nodes & (~binarize(sub))
        labels = process_chunk(0, sub.shape[1], sub, ring_mask,
                               nearest_label_indices)
        del ring_mask
 
        # ---- recover distances from the index array ----
        grid = np.ogrid[tuple(slice(0, n) for n in sub.shape)]
        d2 = np.zeros(sub.shape, dtype=np.float32)
        for ax, scale in enumerate(sampling):
            diff = (nearest_label_indices[ax] - grid[ax]).astype(np.float32)
            diff *= scale
            diff *= diff
            d2 += diff
        del nearest_label_indices
 
        d2[dilated_binary_nodes == 0] = np.inf     # outside the dilation radius
        del dilated_binary_nodes
 
        with lock:
            bd = best_d2[sub_sl]
            better = d2 < bd
            if better.any():
                np.copyto(bd, d2, where=better)
                best_d2[sub_sl] = bd
                o = out[sub_sl]
                np.copyto(o, labels, where=better)
                out[sub_sl] = o
 
    if len(work) > 1 and num_cores > 1:
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            list(executor.map(one_box, work))
    else:
        for item in work:
            one_box(item)
 
    del best_d2
 
    # exact ties on original voxels always resolve to the original label
    core = nodes != 0
    out[core] = nodes[core]
 
    return out


# edt optimization:

 
def _tile_grid(shape, core, halo):
    """Yield (core_slices, padded_slices, offset_into_padded) for every tile."""
    ranges = []
    for n, c in zip(shape, core):
        starts = list(range(0, n, c)) or [0]
        ranges.append([(s, min(s + c, n)) for s in starts])
 
    for z0, z1 in ranges[0]:
        for y0, y1 in ranges[1]:
            for x0, x1 in ranges[2]:
                lo = (z0, y0, x0)
                hi = (z1, y1, x1)
                pad_lo = tuple(max(0, a - h) for a, h in zip(lo, halo))
                pad_hi = tuple(min(n, b + h) for b, h, n in zip(hi, halo, shape))
                core_sl = tuple(slice(a, b) for a, b in zip(lo, hi))
                pad_sl = tuple(slice(a, b) for a, b in zip(pad_lo, pad_hi))
                inner = tuple(slice(a - p, b - p)
                              for a, b, p in zip(lo, hi, pad_lo))
                yield core_sl, pad_sl, inner
 
 
def _resolve_tile_payload(payload):
    """Resolve one tile."""

 
    method, sub_nodes, sub_mask, inner, sampling, core_mask, idx = payload
 
    if method == "watershed":
        from skimage.segmentation import watershed as _ws
        vals = _ws(sub_mask.astype(np.float32),
                   markers=sub_nodes.astype(np.int32),
                   mask=sub_mask, compactness=0)[inner]
    else:
        _, ind = ndimage.distance_transform_edt(
            sub_nodes == 0, sampling=sampling, return_indices=True)
        vals = sub_nodes[tuple(ind)][inner]
 
    return idx, np.where(core_mask, vals, 0).astype(sub_nodes.dtype)
 
 
def estimate_tile_work(binary_array, shape, core, halo):
    """
    Fraction of the array each strategy actually touches.
 
    Returns (n_tiles, n_live, ratio) where ratio is summed padded volume of
    non-empty tiles over the array volume.  Empty tiles are skipped entirely,
    so on sparse masks this lands well below 1 even with generous halos.
    """
    n_tiles = n_live = 0
    total = 0.0
    for core_sl, pad_sl, _ in _tile_grid(shape, core, halo):
        n_tiles += 1
        if not binary_array[core_sl].any():
            continue
        n_live += 1
        vol = 1.0
        for s in pad_sl:
            vol *= (s.stop - s.start)
        total += vol
    size = 1.0
    for n in shape:
        size *= n
    return n_tiles, n_live, total / size
 
 
def smart_label_tiled(binary_array, label_array, directory=None,
                      remove_template=False, amount=None,
                      xy_scale=1, z_scale=1, method="transform",
                      tile=None, workers=None):
    """
    Assign labels to a precomputed dilated mask using halo-padded tiles.
 
    First four arguments match smart_label_watershed() positionally.
 
    amount  -- the dilation radius the mask was built with.  Required: it sets
               the halo width.  Passing it too small silently corrupts tile
               borders, so it is not optional.
    method  -- "transform" : scipy feature transform per tile.  Exact Euclidean,
                             matches a whole-volume transform voxel for voxel.
               "watershed" : skimage watershed per tile.  Cheaper per voxel,
                             geodesic rather than Euclidean, and a flood that
                             would need to leave the tile gets clipped.
    tile    -- core edge length in voxels.  None picks a size that keeps halo
               overhead near 2x while staying small enough to spread across
               threads.
    workers -- worker count, defaults to cpu_count().  scipy's feature
               transform and skimage's watershed both release the GIL, so
               threads give real parallelism for those two.
    """

    if amount is None:
        raise ValueError("smart_label_tiled needs `amount` to size the halo")
 
    string_bool = isinstance(binary_array, str) or isinstance(label_array, str)
    if isinstance(binary_array, str):
        binary_array = tifffile.imread(binary_array)
    if isinstance(label_array, str):
        label_array = tifffile.imread(label_array)
 
    binary_array = binarize(binary_array)
    shape = binary_array.shape
    sampling = [float(z_scale), float(xy_scale), float(xy_scale)]
 
    halo = tuple(int(math.ceil(amount / s)) + 1 for s in sampling)
    if shape[0] == 1:
        halo = (0, halo[1], halo[2])
 
    if workers is None:
        workers = mp.cpu_count()
 
    if tile is None:
        # Two competing pulls: bigger cores shrink halo overhead (per axis the
        # cost is (core + 2h)/core), smaller cores give the pool more tiles to
        # balance across.  Aim for ~4 tiles per worker but never let a core get
        # narrower than 6 halos, which caps overhead near 1.8x.
        vol = 1.0
        for n in shape:
            vol *= n
        by_workers = int((vol / max(1, 4 * workers)) ** (1.0 / 3.0))
        tile = max(32, 6 * max(halo), by_workers)
    core = tuple(min(tile, n) if h else n for n, h in zip(shape, halo))
    core = tuple(max(1, c) for c in core)
    if shape[0] == 1:
        core = (1, core[1], core[2])
 
    out = np.zeros(shape, dtype=label_array.dtype)
    n_live = [0, 0, 0]      # live tiles, shortcut tiles, resolved tiles
 
    def do_tile(item):
        core_sl, pad_sl, inner = item
 
        core_mask = binary_array[core_sl]
        if not core_mask.any():
            return                                  # nothing dilated here
        n_live[0] += 1
 
        sub_nodes = label_array[pad_sl]
        present = np.unique(sub_nodes)
        present = present[present != 0]
 
        if present.size == 0:
            return
        if present.size == 1:
            # only one label can reach this tile: no transform needed at all
            n_live[1] += 1
            out[core_sl] = np.where(core_mask, present[0], 0).astype(out.dtype)
            return
 
        n_live[2] += 1
        payload = (method, sub_nodes, binary_array[pad_sl], inner, sampling,
                   core_mask, core_sl)
        _, vals = _resolve_tile_payload(payload)
 
        # tile cores are disjoint, so this write needs no lock
        out[core_sl] = vals.astype(out.dtype)
 
    tiles = list(_tile_grid(shape, core, halo))
 
    if workers > 1 and len(tiles) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(do_tile, tiles))
    else:
        for item in tiles:
            do_tile(item)
 
    print(f"Labelled by tiles: {len(tiles)} tiles of {core}, {n_live[0]} live "
          f"({n_live[1]} single-label, {n_live[2]} resolved by {method})")
 
    # original labels always survive intact, and nothing escapes edt's mask
    core_vox = label_array != 0
    out[core_vox] = label_array[core_vox]
    out *= binary_array.astype(out.dtype)
 
    if string_bool:
        path = (f"{directory}/smart_labelled_array.tif" if directory
                else "smart_labelled_array.tif")
        try:
            tifffile.imwrite(path, out)
        except Exception:
            print(f"Could not save search region file to {path}")
 
    return out


# Smart label
 
 
def smart_label_whole(binary_array, label_array, directory=None,
                      remove_template=False, xy_scale=1, z_scale=1):
    """
    Whole-volume scipy feature transform, then parallel chunked assignment.
 
    This is the original smart_label path with the GPU and downsample branches
    removed: they existed to survive a transform too large for GPU memory, at
    the cost of resampling labels and blurring their boundaries.  mode 0 handles
    large arrays by tiling instead, so nothing is traded away.
    """
 
    string_bool = isinstance(binary_array, str) or isinstance(label_array, str)
    if isinstance(binary_array, str):
        binary_array = tifffile.imread(binary_array)
    if isinstance(label_array, str):
        label_array = tifffile.imread(label_array)
 
    binary_array = binarize(binary_array)
 
    print("Performing distance transform for smart label...")
 
    binary_core = binarize(label_array)
    nearest_label_indices = compute_distance_transform(
        invert_array(binary_core),
        sampling=[float(z_scale), float(xy_scale), float(xy_scale)]
    )
    ring_mask = binary_array & invert_array(binary_core)
    del binary_core
 
    num_cores = mp.cpu_count()
    chunk_size = max(1, label_array.shape[1] // num_cores)
 
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [
            (i * chunk_size,
             (i + 1) * chunk_size if i != num_cores - 1 else label_array.shape[1],
             label_array, ring_mask, nearest_label_indices)
            for i in range(num_cores)
        ]
        results = list(executor.map(lambda a: process_chunk(*a), args_list))
 
    del ring_mask, nearest_label_indices
 
    out = np.concatenate(results, axis=1)
    del results
 
    out *= binary_array.astype(out.dtype)
 
    if string_bool:
        _write(out, directory)
    return out
 
 
def _write(out, directory):
    path = (f"{directory}/smart_labelled_array.tif" if directory is not None
            else "smart_labelled_array.tif")
    try:
        tifffile.imwrite(path, out)
    except Exception:
        where = directory if directory is not None else "active directory"
        print(f"Could not save search region file to {where}")
 
 
def smart_label(binary_array, label_array, directory=None, GPU=True,
                predownsample=None, remove_template=False, mode=0,
                xy_scale=1, z_scale=1, tile=None, workers=None):
    """
    Propagate labels from label_array through binary_array.
 
    mode 1 -- edt reach field sizes per-tile halos, scipy feature transform per
              tile, threaded.  Exact Euclidean and self-sizing: no dilation
              radius has to be supplied or measured.  Falls back to
              smart_label_watershed if edt is unavailable or the tiled path
              raises.
    mode 0 -- whole-volume scipy feature transform (smart_label_whole).
 
    GPU, predownsample -- accepted and ignored, kept so existing call sites keep
        working.
    """
    if mode == 1: # Note that watershedding is faster but can only be applied for contiguous labels.
        return smart_label_watershed(binary_array, label_array, directory,
                                     remove_template)
    else:
        return smart_label_whole(binary_array, label_array, directory,
                                 remove_template, xy_scale, z_scale)


# Multi label dt:

def _run_multilabel_edt_in_process_shm(input_shm_name, output_shm_name, shape,
                                       dtype_str, sampling_tuple, black_border,
                                       parallel, squared):
    """Run edt on a labelled array in shared memory.
 
    NOTE: edt dispatches on dtype.  Boolean arrays hit the binary
    specialisation; integer arrays hit the multi-label code path.  So no
    .astype(bool) here -- that cast is exactly what the binary version wanted
    and exactly what this version must not do.
    """
    import numpy as np
    import edt
    from multiprocessing import shared_memory
 
    input_shm = shared_memory.SharedMemory(name=input_shm_name)
    output_shm = shared_memory.SharedMemory(name=output_shm_name)
    nodes_arr = result_array = result = None
    try:
        nodes_arr = np.ndarray(shape, dtype=dtype_str, buffer=input_shm.buf)
        fn = edt.edtsq if squared else edt.edt
        # order='C': the parent copies into a C-ordered shm view, so
        # anisotropy[i] lines up with array axis i.
        result = fn(nodes_arr, anisotropy=sampling_tuple,
                    black_border=black_border, order='C', parallel=parallel)
        result_array = np.ndarray(result.shape, dtype=result.dtype,
                                  buffer=output_shm.buf)
        np.copyto(result_array, result)
        return result.shape, str(result.dtype)
    finally:
        # drop every buffer export before close(), else close() raises
        del nodes_arr, result_array, result
        input_shm.close()
        output_shm.close()
 
 
def _apply_threshold(view, nodes, thr, keep_labels):
    """Voxels within `thr` of their own label's boundary.
 
    Background is excluded explicitly: it sits at distance 0, so a bare
    `view <= thr` would sweep all of it in.
    """
    mask = view <= thr          # 1N bool, not 4N float
    mask &= nodes != 0
    if keep_labels:
        return np.where(mask, nodes, 0)
    return mask.view(np.uint8)
 
 
def compute_multilabel_distance_transform(nodes, sampling=[1, 1, 1],
                                          threshold=None, black_border=False,
                                          keep_labels=False, parallel=0):
    """Multi-label distance transform via edt, in an isolated subprocess.
 
    Args:
        nodes: Integer label array (0 = background).  Bool is accepted and
               behaves like the binary version.
        sampling: Voxel spacing [z, y, x] for anisotropic data.
        threshold: If given, returns voxels within `threshold` of their own
                   label's boundary as a uint8 mask, without materialising a
                   second float buffer.
        keep_labels: With `threshold`, return label IDs in the band instead of
                     a 0/1 mask.
        black_border: True treats the outside of the volume as background.
        parallel: Threads for edt; <= 0 uses the CPU count.
    Returns:
        float32 distance array, the thresholded mask / label band, or None if
        the transform failed.
    """
    is_pseudo_3d = nodes.shape[0] == 1
    if is_pseudo_3d:
        nodes = np.squeeze(nodes)
        sampling = [sampling[1], sampling[2]]
    if not (nodes.flags['C_CONTIGUOUS'] or nodes.flags['F_CONTIGUOUS']):
        nodes = np.ascontiguousarray(nodes)
 
    # Skip the sqrt when the result only gets compared to a cutoff.
    squared = threshold is not None
    thr = threshold ** 2 if squared else threshold
 
    distance = None
    input_shm = output_shm = None
    try:
        input_shm = shared_memory.SharedMemory(create=True, size=nodes.nbytes)
        output_shm = shared_memory.SharedMemory(
            create=True, size=nodes.size * np.dtype(np.float32).itemsize)
        shm_array = np.ndarray(nodes.shape, dtype=nodes.dtype,
                               buffer=input_shm.buf)
        np.copyto(shm_array, nodes)
        del shm_array   # drop the buffer export before close()
 
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(
                _run_multilabel_edt_in_process_shm,
                input_shm.name,
                output_shm.name,
                nodes.shape,
                str(nodes.dtype),
                tuple(sampling),
                bool(black_border),
                int(parallel),
                squared,
            )
            result_shape, result_dtype = future.result()
 
        # free the input before allocating anything else
        input_shm.close()
        input_shm.unlink()
        input_shm = None
 
        view = np.ndarray(result_shape, dtype=np.dtype(result_dtype),
                          buffer=output_shm.buf)
        try:
            if threshold is not None:
                distance = _apply_threshold(view, nodes, thr, keep_labels)
            else:
                distance = view.copy()
        finally:
            del view     # must go before output_shm.close() below
    except ImportError:
        print("edt package not found. Please use 'pip install edt' "
              "to enable the multi-label distance transform.")
    except MemoryError:
        # int32/int64 labels are 4-8x the old bool buffer, and the child holds
        # the labels, edt's float32 result and the output buffer at once.
        print("Multi-label distance transform ran out of memory. Try a "
              "smaller label dtype, or crop the volume.")
    except Exception as e:
        print(f"Multi-label distance transform failed ({e})")
        traceback.print_exc()
    finally:
        for shm in (input_shm, output_shm):
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
 
    if distance is not None and is_pseudo_3d:
        distance = np.expand_dims(distance, axis=0)
    return distance