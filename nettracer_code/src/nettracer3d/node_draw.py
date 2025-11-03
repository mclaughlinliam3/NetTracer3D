import numpy as np
import tifffile
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
import cv2

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

def upsample_with_padding(data, factor, original_shape):
    # Upsample the input binary array while adding padding to match the original shape

    # Get the dimensions of the original and upsampled arrays
    original_shape = np.array(original_shape)
    binary_array = zoom(data, factor, order=0)
    upsampled_shape = np.array(binary_array.shape)

    # Calculate the positive differences in dimensions
    difference_dims = original_shape - upsampled_shape

    # Calculate the padding amounts for each dimension
    padding_dims = np.maximum(difference_dims, 0)
    padding_before = padding_dims // 2
    padding_after = padding_dims - padding_before

    # Pad the binary array along each dimension
    padded_array = np.pad(binary_array, [(padding_before[0], padding_after[0]),
                                         (padding_before[1], padding_after[1]),
                                         (padding_before[2], padding_after[2])], mode='constant', constant_values=0)

    # Calculate the subtraction amounts for each dimension
    sub_dims = np.maximum(-difference_dims, 0)
    sub_before = sub_dims // 2
    sub_after = sub_dims - sub_before

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

    # Remove columns from the beginning and end
    if sub_dims[2] == 0:
        trimmed_array = trimmed_rows
    else:
        trimmed_array = trimmed_rows[:, :, sub_before[2]:-sub_after[2]]

    return trimmed_array

def draw_nodes(nodes, num_nodes):
    # Find centroids
    centroids = np.array([np.mean(np.argwhere(nodes == i), axis=0) for i in range(1, num_nodes + 1)])

    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(nodes, dtype=np.uint8)

    # Use the default font from ImageFont
    font_size = None

    # Iterate through each centroid
    for idx, centroid in enumerate(centroids, start=1):
        z, y, x = centroid.astype(int)

        try:
            draw_array = _draw_at_plane(z, y, x, draw_array, idx, font_size)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z + 1, y, x, draw_array, idx, font_size)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z - 1, y, x, draw_array, idx, font_size)
        except IndexError:
            pass

    # Save the draw_array as a 3D TIFF file
    tifffile.imwrite("labelled_nodes.tif", draw_array)

def draw_from_centroids(nodes, num_nodes, centroids, twod_bool, directory=None):
    """Optimized version using OpenCV"""
    print("Drawing node IDs...")
    draw_array = np.zeros_like(nodes, dtype=np.uint8)
    
    # Draw text using OpenCV (no PIL conversions needed)
    for idx in centroids.keys():
        centroid = centroids[idx]
        z, y, x = centroid.astype(int)
        
        for z_offset in [0, 1, -1]:
            z_target = z + z_offset
            if 0 <= z_target < draw_array.shape[0]:
                cv2.putText(draw_array[z_target], str(idx), (x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA)
    
    if twod_bool:
        draw_array = draw_array[0,:,:] | draw_array[1,:,:]
    
    filename = f'{directory}/labelled_node_indices.tif' if directory else 'labelled_node_indices.tif'
    try:
        tifffile.imwrite(filename, draw_array)
    except Exception as e:
        print(f"Could not save node indices to {filename}")
    
    return draw_array

def degree_draw(degree_dict, centroid_dict, nodes):
    """Draw node degrees at centroid locations using OpenCV"""
    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(nodes, dtype=np.uint8)
    
    for node in centroid_dict:
        # Skip if node not in degree_dict
        if node not in degree_dict:
            continue
            
        degree = degree_dict[node]
        z, y, x = centroid_dict[node].astype(int)
        
        # Draw on current z-plane and adjacent planes
        for z_offset in [0, 1, -1]:
            z_target = z + z_offset
            # Check bounds
            if 0 <= z_target < draw_array.shape[0]:
                cv2.putText(
                    draw_array[z_target],  # Image to draw on
                    str(degree),            # Text to draw
                    (x, y),                 # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    0.4,                    # Font scale
                    255,                    # Color (white)
                    1,                      # Thickness
                    cv2.LINE_AA             # Anti-aliasing
                )
    
    return draw_array

def degree_infect(degree_dict, nodes, make_floats = False):

    if not make_floats:
        return_nodes = np.zeros_like(nodes)  # Start with all zeros
    else:
        return_nodes = np.zeros(nodes.shape, dtype=np.float32)
    
    if not degree_dict:  # Handle empty dict
        return return_nodes
    
    # Create arrays for old and new values
    old_vals = np.array(list(degree_dict.keys()))
    new_vals = np.array(list(degree_dict.values()))
    
    # Sort for searchsorted to work correctly
    sort_idx = np.argsort(old_vals)
    old_vals_sorted = old_vals[sort_idx]
    new_vals_sorted = new_vals[sort_idx]
    
    # Find which nodes exist in the dictionary
    mask = np.isin(nodes, old_vals_sorted)
    
    # Only process nodes that exist in the dictionary
    if np.any(mask):
        indices = np.searchsorted(old_vals_sorted, nodes[mask])
        return_nodes[mask] = new_vals_sorted[indices]
    
    return return_nodes


def _draw_at_plane(z_loc, y_loc, x_loc, array, num, font_size=None):
    # Get the 2D slice at the specified Z position
    slice_to_draw = array[z_loc, :, :]

    # Create an image from the 2D slice
    image = Image.fromarray(slice_to_draw.astype(np.uint8) * 255)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Load the default font with the specified font size

    font = ImageFont.load_default()

    # Draw the number at the centroid index
    draw.text((x_loc, y_loc), str(num), fill='white', font=font)

    # Save the modified 2D slice into draw_array at the specified Z position
    array[z_loc, :, :] = np.array(image)

    return array

def compute_centroid(binary_stack, label):
    """
    Finds centroid of labelled object in array
    """
    indices = np.argwhere(binary_stack == label)
    centroid = np.round(np.mean(indices, axis=0)).astype(int)

    return centroid

if __name__ == "__main__":

    nodes = tifffile.imread("nodes_for_networks.tif")

    node_shape = nodes.shape

    nodes = downsample(nodes, 5)

    # Label the connected components
    structure_3d = np.ones((3, 3, 3), dtype=int)
    nodes, num_nodes = ndimage.label(nodes, structure=structure_3d)

    # Find centroids
    centroids = np.array([np.mean(np.argwhere(node_labels == i), axis=0) for i in range(1, num_nodes + 1)])

    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(nodes, dtype=np.uint8)

    # Use the default font from ImageFont
    font = ImageFont.load_default()

    # Iterate through each centroid
    for idx, centroid in enumerate(centroids, start=1):
        z, y, x = centroid.astype(int)

        # Get the 2D slice at the specified Z position
        slice_to_draw = draw_array[z, :, :]

        # Create an image from the 2D slice
        image = Image.fromarray(slice_to_draw.astype(np.uint8) * 255)

        # Create a drawing object
        draw = ImageDraw.Draw(image)

        # Draw the number at the centroid index
        draw.text((x, y), str(idx), fill='white', font=font)

        # Save the modified 2D slice into draw_array at the specified Z position
        draw_array[z, :, :] = np.array(image)

    if len(node_shape) == 2:
        draw_array = draw_array[0,:,:]

    # Save the draw_array as a 3D TIFF file
    tifffile.imwrite("draw_array.tif", draw_array)
