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
            return smart_label_watershed(dilated, nodes, directory = None, remove_template = False)
        except:
            print("edt package not found. Please use 'pip install edt' if you would like to enable parallel searching.")
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

def round_to_odd(number):
    rounded = round(number)
    # If the rounded number is even, add or subtract 1 to make it odd
    if rounded % 2 == 0:
        if number > 0:
            rounded += 1
        else:
            rounded -= 1
    return rounded

def smart_label_watershed(binary_array, label_array, directory = None, remove_template = False):
    """
    Watershed-based version - much lower memory footprint
    """
    original_shape = binary_array.shape
    
    if type(binary_array) == str or type(label_array) == str:
        string_bool = True
    else:
        string_bool = None
    if type(binary_array) == str:
        binary_array = tifffile.imread(binary_array)
    if type(label_array) == str:
        label_array = tifffile.imread(label_array)
    
    # Binarize
    binary_array = binarize(binary_array)
    
    print("Performing watershed label propagation...")
    
    # Watershed approach: propagate existing labels into the dilated region
    # The labels themselves are the "markers" (seeds)
    # We use the binary mask to define where labels can spread
    
    # Simple elevation map: distance from edges (lower = closer to labeled regions)
    # This makes watershed flow from labeled regions outward
    elevation = binary_array.astype(np.float32)
    
    # Apply watershed - labels propagate into binary_array region
    dilated_nodes_with_labels = watershed(
        elevation,           # Elevation map (flat works fine)
        markers=label_array, # Seed labels
        mask=binary_array,   # Where to propagate
        compactness=0        # Pure distance-based (not shape-based)
    )
    
    if remove_template:
        dilated_nodes_with_labels *= binary_array
    
    if string_bool:
        if directory is not None:
            try:
                tifffile.imwrite(f"{directory}/smart_labelled_array.tif", dilated_nodes_with_labels)
            except Exception as e:
                print(f"Could not save search region file to {directory}")
        else:
            try:
                tifffile.imwrite("smart_labelled_array.tif", dilated_nodes_with_labels)
            except Exception as e:
                print(f"Could not save search region file to active directory")
    
    return dilated_nodes_with_labels

def smart_label(binary_array, label_array, directory = None, GPU = True, predownsample = None, remove_template = False, mode = 0):

    if mode == 1:
        return smart_label_watershed(binary_array, label_array, directory, remove_template)

    original_shape = binary_array.shape

    if type(binary_array) == str or type(label_array) == str:
        string_bool = True
    else:
        string_bool = None
    if type(binary_array) == str:
        binary_array = tifffile.imread(binary_array)
    if type(label_array) == str:
        label_array = tifffile.imread(label_array)

    # Binarize inputs
    binary_array = binarize(binary_array)
    
    print("Performing distance transform for smart label...")

    downsample_needed = None  # Track if we downsampled
    
    try:
        if GPU == True and cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy for distance transform.")

            try:
                if predownsample is None:
                    # Compute binary_core only when needed
                    binary_core = binarize(label_array)
                    nearest_label_indices = compute_distance_transform_GPU(invert_array(binary_core))
                    del binary_core  # Free immediately after use
                else:
                    raise ZeroDivisionError  # Force downsample path

            except (cp.cuda.memory.OutOfMemoryError, ZeroDivisionError) as e:
                if predownsample is None:
                    down_factor = catch_memory(e)
                else:
                    down_factor = (predownsample)**3

                while True:
                    downsample_needed = down_factor**(1./3.)
                    small_array = nettracer.downsample(label_array, downsample_needed)
                    try:
                        binary_core = binarize(small_array)
                        nearest_label_indices = compute_distance_transform_GPU(invert_array(binary_core))
                        print(f"Using {down_factor} downsample ({downsample_needed} in each dim)")
                        del small_array  # Don't need small_array anymore
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        del small_array, binary_core  # Clean up before retry
                        down_factor += 1
                
                # Update label_array for later use
                label_array = nettracer.downsample(label_array, downsample_needed)
                binary_small = nettracer.downsample(binary_array, downsample_needed)
                binary_small = nettracer.dilate_3D_old(binary_small)
                ring_mask = binary_small & invert_array(binary_core)
                del binary_small, binary_core  # Free after creating ring_mask
        else:
            raise Exception("GPU not available")
            
    except Exception as e:
        if GPU:
            print("GPU dt failed or did not detect GPU. Computing CPU distance transform instead.")
            print(f"Error message: {str(e)}")
            import traceback
            print(traceback.format_exc())
        binary_core = binarize(label_array)
        nearest_label_indices = compute_distance_transform(invert_array(binary_core))
        del binary_core

    # Compute ring_mask only if not already computed in downsample path
    if 'ring_mask' not in locals():
        binary_core = binarize(label_array)
        ring_mask = binary_array & invert_array(binary_core)
        del binary_core

    # Step 5: Process in parallel chunks
    num_cores = mp.cpu_count()
    chunk_size = label_array.shape[1] // num_cores

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [(i * chunk_size, (i + 1) * chunk_size if i != num_cores - 1 else label_array.shape[1], 
                     label_array, ring_mask, nearest_label_indices) for i in range(num_cores)]
        results = list(executor.map(lambda args: process_chunk(*args), args_list))

    # Free large arrays no longer needed
    del label_array, ring_mask, nearest_label_indices

    # Combine results
    dilated_nodes_with_labels = np.concatenate(results, axis=1)
    del results  # Free the list of chunks

    if downsample_needed is not None:  # If downsample was used
        dilated_nodes_with_labels = nettracer.upsample_with_padding(dilated_nodes_with_labels, downsample_needed, original_shape)
        dilated_nodes_with_labels *= binary_array  # In-place multiply if possible
    elif remove_template:
        dilated_nodes_with_labels *= binary_array  # In-place multiply if possible

    del binary_array  # Done with this

    if string_bool:
        if directory is not None:
            try:
                tifffile.imwrite(f"{directory}/smart_labelled_array.tif", dilated_nodes_with_labels)
            except Exception as e:
                print(f"Could not save search region file to {directory}")
        else:
            try:
                tifffile.imwrite("smart_labelled_array.tif", dilated_nodes_with_labels)
            except Exception as e:
                print(f"Could not save search region file to active directory")

    return dilated_nodes_with_labels

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


def compute_distance_transform(nodes, return_dists = False, sampling = [1, 1, 1]):
    #print("(Now doing distance transform...)")
    is_pseudo_3d = nodes.shape[0] == 1
    if is_pseudo_3d:
        nodes = np.squeeze(nodes)  # Convert to 2D for processing
        sampling = [sampling[1], sampling[2]]

    dists, nearest_label_indices = distance_transform_edt(nodes, return_indices=True, sampling = sampling)

    if is_pseudo_3d:
        # For 2D input, we get (2, H, W) but need (3, 1, H, W)
        H, W = nearest_label_indices[0].shape
        indices_4d = np.zeros((3, 1, H, W), dtype=nearest_label_indices.dtype)
        indices_4d[1:, 0] = nearest_label_indices  # Copy Y and X coordinates
        # indices_4d[0] stays 0 for all Z coordinates
        nearest_label_indices = indices_4d

    if not return_dists:

        return nearest_label_indices

    else:

        return dists, nearest_label_indices



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

def compute_distance_transform_distance(nodes, sampling=[1, 1, 1], fast_dil=False):
    """
    Compute distance transform with automatic parallelization when available.
    
    Args:
        nodes: Binary array (True/1 for objects)
        sampling: Voxel spacing [z, y, x] for anisotropic data
    
    Returns:
        Distance transform array
    """
    is_pseudo_3d = nodes.shape[0] == 1
    
    if is_pseudo_3d:
        nodes = np.squeeze(nodes)
        sampling = [sampling[1], sampling[2]]
    
    if fast_dil:
        try:
            # Use shared memory for all array sizes
            input_shm = shared_memory.SharedMemory(create=True, size=nodes.nbytes)
            output_size = nodes.size * np.dtype(np.float64).itemsize
            output_shm = shared_memory.SharedMemory(create=True, size=output_size)
            
            try:
                shm_array = np.ndarray(nodes.shape, dtype=nodes.dtype, buffer=input_shm.buf)
                np.copyto(shm_array, nodes)
                
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _run_edt_in_process_shm,
                        input_shm.name,
                        output_shm.name,
                        nodes.shape,
                        str(nodes.dtype),
                        tuple(sampling)
                    )
                    result_shape, result_dtype = future.result()
                
                distance = np.ndarray(result_shape, dtype=result_dtype, buffer=output_shm.buf).copy()
                
            finally:
                input_shm.close()
                input_shm.unlink()
                output_shm.close()
                output_shm.unlink()
                
        except Exception as e:
            print(f"Parallel distance transform failed ({e}), falling back to scipy")
            try:
                import edt
                import traceback
                traceback.print_exc()  # See the full error
            except:
                print("edt package not found. Please use 'pip install edt' if you would like to enable parallel searching.")
            distance = distance_transform_edt(nodes, sampling=sampling)
    else:
        distance = distance_transform_edt(nodes, sampling=sampling)
    
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



if __name__ == "__main__":
    nodes = input("Labelled Nodes tiff?: ")
    nodes = tifffile.imread(nodes)

    # Step 1: Binarize the labeled array
    binary_nodes = binarize(nodes)

    # Step 2: Dilate the binarized array
    dilated_binary_nodes = dilate_3D(binary_nodes, 10, 10, 10)

    # Step 3: Isolate the ring (binary dilated mask minus original binary mask)
    ring_mask = dilated_binary_nodes & invert_array(binary_nodes)

    # Step 4: Find the nearest label for each voxel in the ring
    distance, nearest_label_indices = distance_transform_edt(invert_array(nodes), return_indices=True)

    # Step 5: Process in parallel chunks using ThreadPoolExecutor
    num_cores = mp.cpu_count()  # Use all available CPU cores
    chunk_size = nodes.shape[0] // num_cores  # Divide the array into chunks along the z-axis

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [(i * chunk_size, (i + 1) * chunk_size if i != num_cores - 1 else nodes.shape[0], nodes, ring_mask, nearest_label_indices) for i in range(num_cores)]
        results = list(executor.map(lambda args: process_chunk(*args), args_list))

    # Combine results from chunks
    dilated_nodes_with_labels = np.concatenate(results, axis=0)

    # Save the result
    output_file = "dilated_nodes_with_labels.tif"
    tifffile.imwrite(output_file, dilated_nodes_with_labels)
    print(f"Result saved to {output_file}")