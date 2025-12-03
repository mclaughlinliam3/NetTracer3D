import tifffile
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import zoom
try:
    import cupy as cp
except:
    pass
    
from . import network_analysis

def read_excel_to_lists(file_path, sheet_name=0):
    """Convert a pd dataframe to lists"""
    # Read the Excel file into a DataFrame without headers
    df = pd.read_excel(file_path, header=None, sheet_name=sheet_name)

    df = df.drop(0)

    # Initialize an empty list to store the lists of values
    data_lists = []

    # Iterate over each column in the DataFrame
    for column_name, column_data in df.items():
        # Convert the column values to a list and append to the data_lists
        data_lists.append(column_data.tolist())

    master_list = [[], [], []]


    for i in range(0, len(data_lists), 3):

        master_list[0].extend(data_lists[i])
        master_list[1].extend(data_lists[i+1])

        try:
            master_list[2].extend(data_lists[i+2])
        except IndexError:
            pass

    return master_list

def remove_dupes(pair1, pair2):
    # Combine pairs into a set of tuples for faster membership check
    pairwise_set = set(zip(pair1, pair2))

    # Initialize sets to store unique pairs and their reversed forms
    unique_pairs = set()
    reversed_pairs = set()

    # Iterate through the pairs, adding them to unique_pairs if not already present,
    # or to reversed_pairs if they're in reversed order
    for pair in pairwise_set:
        if pair not in unique_pairs and pair[::-1] not in reversed_pairs:
            unique_pairs.add(pair)
            reversed_pairs.add(pair[::-1])

    # Unpack the unique pairs into separate lists
    pair1_unique, pair2_unique = zip(*unique_pairs)

    return pair1_unique, pair2_unique

def compute_centroid(binary_stack, label):
    """
    Finds centroid of labelled object in array
    """
    indices = np.argwhere(binary_stack == label)
    if indices.shape[0] == 0:
        return None
    else:
        centroid = np.round(np.mean(indices, axis=0)).astype(int)
        return centroid

def draw_line_inplace(start, end, array):
    """
    Draws a white line between two points in a 3D array.
    """
    
    # Calculate the distances between start and end coordinates
    try:
        distances = end - start
    except:
        end = np.array(end)
        start = np.array(start)
        distances = end - start

    # Determine the number of steps along the line
    num_steps = int(max(np.abs(distances)) + 1)

    # Generate linearly spaced coordinates along each dimension
    x_coords = np.linspace(start[0], end[0], num_steps, endpoint=True).round().astype(int)
    y_coords = np.linspace(start[1], end[1], num_steps, endpoint=True).round().astype(int)
    z_coords = np.linspace(start[2], end[2], num_steps, endpoint=True).round().astype(int)

    # Clip coordinates to ensure they are within the valid range
    x_coords = np.clip(x_coords, 0, array.shape[0] - 1)
    y_coords = np.clip(y_coords, 0, array.shape[1] - 1)
    z_coords = np.clip(z_coords, 0, array.shape[2] - 1)

    # Set the line coordinates to 255 in the existing array
    array[x_coords, y_coords, z_coords] = 255

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

def draw_network_lattice(nodes, network, label_nodes = None):

    print("Drawing network lattice...")
    if type(nodes) == str:
        nodes = tifffile.imread(nodes)

    if label_nodes is None:
        label_nodes = True

    if label_nodes:
        structure_3d = np.ones((3, 3, 3), dtype=int)
        nodes, num_nodes = ndimage.label(nodes, structure=structure_3d)

    network = read_excel_to_lists(network)
    pair1 = network[0]
    pair2 = network[1]
    centroid_dic = {}
    print("removing duplicates")
    pair1, pair2 = remove_dupes(pair1, pair2)
    pair1 = list(pair1)
    pair2 = list(pair2)
    network = set(pair1 + pair2)
    print(network)
    print("Finding centroids")
    for item in network:
        centroid = compute_centroid(nodes, item)
        if centroid is not None:
            centroid_dic[item] = centroid
    output_stack = np.zeros(np.shape(nodes), dtype=np.uint8)

    for i, pair1_val in enumerate(pair1):
        pair2_val = pair2[i]
        try:
            pair1_centroid = centroid_dic[pair1_val]
            pair2_centroid = centroid_dic[pair2_val]
            draw_line_inplace(pair1_centroid, pair2_centroid, output_stack)
        except KeyError:
            print(f"Missing centroid {i}")
            pass

    tifffile.imwrite("drawn_network.tif", output_stack)
    print("done")

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

def draw_network_from_centroids(nodes, network, centroids, twod_bool, directory = None):

    print("Drawing network lattice")
    if type(network) == str:
        network = read_excel_to_lists(network)
    pair1 = network[0]
    pair2 = network[1]
    centroid_dic = {}

    pair1, pair2 = remove_dupes(pair1, pair2)
    pair1 = list(pair1)
    pair2 = list(pair2)
    network = set(pair1 + pair2)

    for item in network:
        try:
            centroid = centroids[item]
            centroid_dic[item] = centroid

        except KeyError:
            pass
            #print(f"Centroid {item} missing")
    output_stack = np.zeros(np.shape(nodes), dtype=np.uint8)

    for i, pair1_val in enumerate(pair1):

        pair2_val = pair2[i]
        try:
            pair1_centroid = centroid_dic[pair1_val]
            pair2_centroid = centroid_dic[pair2_val]
            draw_line_inplace(pair1_centroid, pair2_centroid, output_stack)
        except KeyError:
            #print(f"Missing centroid {i}")
            pass

    if twod_bool:
        output_stack = output_stack[0,:,:] | output_stack[0,:,:]

    if directory is None:
        try:
            tifffile.imwrite("drawn_network.tif", output_stack)
        except Exception as e:
            print("Could not save network lattice to active directory")
            print("Network lattice saved as drawn_network.tif")

    if directory is not None:
        try:
            tifffile.imwrite(f"{directory}/drawn_network.tif", output_stack)
            print(f"Network lattice saved to {directory}/drawn_network.tif")
        except Exception as e:
            print(f"Could not save network lattice to {directory}")

    return output_stack

def draw_network_from_centroids_GPU(nodes, network, centroids, twod_bool, directory = None):

    def draw_line_inplace_GPU(start, end, array):
        """
        Draws a white line between two points in a 3D array.
        """
        
        # Calculate the distances between start and end coordinates
        distances = (end-start)

        # Determine the number of steps along the line
        num_steps = int(cp.max(cp.absolute(distances)) + 1)

        # Generate linearly spaced coordinates along each dimension
        x_coords = cp.linspace(start[0], end[0], num_steps, endpoint=True).round().astype(int)
        y_coords = cp.linspace(start[1], end[1], num_steps, endpoint=True).round().astype(int)
        z_coords = cp.linspace(start[2], end[2], num_steps, endpoint=True).round().astype(int)

        # Set the line coordinates to 255 in the existing array
        array[x_coords, y_coords, z_coords] = 255

    nodes = cp.asarray(nodes)
    print("Drawing network lattice")
    if type(network) == str:
        network = read_excel_to_lists(network)
    pair1 = network[0]
    pair2 = network[1]
    centroid_dic = {}

    pair1, pair2 = remove_dupes(pair1, pair2)
    pair1 = list(pair1)
    pair2 = list(pair2)
    network = set(pair1 + pair2)

    for item in network:
        try:
            centroid = centroids[item]
            centroid_dic[item] = centroid

        except KeyError:
            print(f"Centroid {item} missing")
    output_stack = cp.zeros(cp.shape(nodes), dtype=np.uint8)

    for i, pair1_val in enumerate(pair1):

        pair2_val = pair2[i]
        try:
            pair1_centroid = cp.asarray(centroid_dic[pair1_val])
            pair2_centroid = cp.asarray(centroid_dic[pair2_val])
            draw_line_inplace_GPU(pair1_centroid, pair2_centroid, output_stack)
        except KeyError:
            print(f"Missing centroid {i}")
            pass

    if twod_bool:
        output_stack = output_stack[0,:,:] | output_stack[0,:,:]

    output_stack = cp.asnumpy(output_stack)

    if directory is None:
        try:
            tifffile.imwrite("drawn_network.tif", output_stack)
        except Exception as e:
            print("Could not save network lattice to active directory")
            print("Network lattice saved as drawn_network.tif")

    if directory is not None:
        try:
            tifffile.imwrite(f"{directory}/drawn_network.tif", output_stack)
            print(f"Network lattice saved to {directory}/drawn_network.tif")
        except Exception as e:
            print(f"Could not save network lattice to {directory}")


if __name__ == '__main__':

    nodes = input("node file?: ")
    while True:
    	Q = input("Label nodes (Y/N)? Label if they are binary. Do not label if they already have grayscale labels: ")
    	if Q == 'Y' or Q == 'N':
    		break
    network = input("excel with node network info?: ")

    nodes = tifffile.imread(nodes)

    node_shape = nodes.shape



    if Q == 'Y':
        print("labelling nodes...")
        structure_3d = np.ones((3, 3, 3), dtype=int)
        nodes, num_nodes = ndimage.label(nodes, structure=structure_3d)

    #nodes = downsample(nodes, 10)


    network = read_excel_to_lists(network)
    pair1 = network[0]
    pair2 = network[1]
    centroid_dic = {}
    print("removing duplicates")
    pair1, pair2 = remove_dupes(pair1, pair2)
    pair1 = list(pair1)
    pair2 = list(pair2)
    network = set(pair1 + pair2)
    print(network)
    print("Finding centroids")
    for item in network:
        centroid = compute_centroid(nodes, item)
        if centroid is not None:
            centroid_dic[item] = centroid
    output_stack = np.zeros(np.shape(nodes), dtype=np.uint8)

    for i, pair1_val in enumerate(pair1):
        print(f"Drawing line for pair {i}...")
        pair2_val = pair2[i]
        try:
            pair1_centroid = centroid_dic[pair1_val]
            pair2_centroid = centroid_dic[pair2_val]
            draw_line_inplace(pair1_centroid, pair2_centroid, output_stack)
        except KeyError:
            print("Missing centroid")
            pass

    if len(node_shape) == 2:
        output_stack = output_stack[0,:,:]


    tifffile.imwrite("drawn_network.tif", output_stack)
    print("done")


