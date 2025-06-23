import pandas as pd
import networkx as nx
import json
import tifffile
import numpy as np
from networkx.algorithms import community
from scipy.ndimage import zoom
from scipy import ndimage
from . import node_draw
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from . import nettracer
from . import modularity
import multiprocessing as mp
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx
except:
    pass

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

def create_bar_graph(data_dict, title, x_label, y_label, directory=None):
    """
    Create a bar graph from a dictionary where keys are bar names and values are heights.
    
    Parameters:
    data_dict (dict): Dictionary with bar names as keys and heights as values
    title (str): Title of the graph
    x_label (str): Label for x-axis
    y_label (str): Label for y-axis
    directory (str, optional): Directory path to save the plot. If None, plot is not saved
    """
    import matplotlib.pyplot as plt
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Create bars
    plt.bar(list(data_dict.keys()), list(data_dict.values()))
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Rotate x-axis labels if there are many bars
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    try:
    
        # Save plot if directory is specified
        if directory:
            plt.savefig(f"{directory}/bar_graph.png")

    except:
        pass

    try:
        
        # Display the plot
        plt.show()
    except:
        pass

def open_network(excel_file_path):
    """opens an unweighted network from the network excel file"""

    if type(excel_file_path) == str:
        # Read the Excel file into a pandas DataFrame
        master_list = read_excel_to_lists(excel_file_path)
    else:
        master_list = excel_file_path

    # Create a graph
    G = nx.Graph()

    nodes_a = master_list[0]
    nodes_b = master_list[1]

    # Add edges to the graph
    for i in range(len(nodes_a)):
        G.add_edge(nodes_a[i], nodes_b[i])

    return G

def read_excel_to_lists(file_path, sheet_name=0):
    """Convert a pd dataframe to lists. Handles both .xlsx and .csv files"""
    def load_json_to_list(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert only numeric strings to integers, leave other strings as is
        converted_data = [[],[],[]]
        for i in data[0]:
            try:
                converted_data[0].append(int(data[0][i]))
                converted_data[1].append(int(data[1][i]))
                try:
                    converted_data[2].append(int(data[2][i]))
                except IndexError:
                    converted_data[2].append(0)
            except ValueError:
                converted_data[k] = v
        
        return converted_data

    if type(file_path) == str:
        # Check file extension
        if file_path.lower().endswith('.xlsx'):
            # Read the Excel file into a DataFrame without headers
            df = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
            df = df.drop(0)
        elif file_path.lower().endswith('.csv'):
            # Read the CSV file into a DataFrame without headers
            df = pd.read_csv(file_path, header=None)
            df = df.drop(0)
        elif file_path.lower().endswith('.json'):
            df = load_json_to_list(file_path)
            return df
        else:
            raise ValueError("File must be either .xlsx, .csv, or .json format")
    else:
        df = file_path

    # Initialize an empty list to store the lists of values
    data_lists = []
    # Iterate over each column in the DataFrame
    for column_name, column_data in df.items():
        # Convert the column values to a list and append to the data_lists
        data_lists.append(column_data.tolist())
    master_list = [[], [], []]
    for i in range(0, len(data_lists), 3):
        master_list[0].extend([int(x) for x in data_lists[i]])
        master_list[1].extend([int(x) for x in data_lists[i+1]])
        try:
            master_list[2].extend([int(x) for x in data_lists[i+2]])
        except IndexError:
            master_list[2].extend([0])  # Note: Changed to list with single int 0
            
    return master_list

def master_list_to_excel(master_list, excel_name):

    nodesA = master_list[0]
    nodesB = master_list[1]
    edgesC = master_list[2]
    # Create a DataFrame from the lists
    df = pd.DataFrame({
    'Nodes A': nodesA,
    'Nodes B': nodesB,
    'Edges C': edgesC
    })

    # Save the DataFrame to an Excel file
    df.to_excel(excel_name, index=False)

def weighted_network(excel_file_path):
    """creates a network where the edges have weights proportional to the number of connections they make between the same structure"""

    if type(excel_file_path) == list:
        master_list = excel_file_path
    else:
        # Read the Excel file into a pandas DataFrame
        master_list = read_excel_to_lists(excel_file_path)

    # Create a graph
    G = nx.Graph()

    # Create a dictionary to store edge weights based on node pairs
    edge_weights = {}

    nodes_a = master_list[0]
    nodes_b = master_list[1]

    # Iterate over the DataFrame rows and update edge weights
    for i in range(len(nodes_a)):
        node1, node2 = nodes_a[i], nodes_b[i]
        edge = (node1, node2) if node1 < node2 else (node2, node1)  # Ensure consistent order
        edge_weights[edge] = edge_weights.get(edge, 0) + 1

    # Add edges to the graph with weights
    for edge, weight in edge_weights.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    return G, edge_weights


def _color_code(grayscale_image):
    """Color code a grayscale array. Currently expects linearly ascending grayscale labels, will crash if there are gaps. (Main use case is grayscale anyway)"""

    def generate_colormap(num_labels):
        # Generate a colormap with visually distinct colors using the new method
        cmap = plt.colormaps['hsv']
        colors = cmap(np.linspace(0, 1, num_labels))
        return colors

    def grayscale_to_rgb(grayscale_image):
        # Get the number of labels
        num_labels = np.max(grayscale_image) + 1
        
        # Generate a colormap
        colormap = generate_colormap(num_labels)
        
        # Create an empty RGB image
        rgb_image = np.zeros((*grayscale_image.shape, 3), dtype=np.uint8)
        
        # Assign colors to each label
        for label in range(1, num_labels):
            color = (colormap[label][:3] * 255).astype(np.uint8)  # Convert to RGB and ensure dtype is uint8
            rgb_image[grayscale_image == label] = color
            
        return rgb_image

    # Convert the grayscale image to RGB
    rgb_image = grayscale_to_rgb(grayscale_image)

    return rgb_image

def read_centroids_to_dict(file_path):
    """
    Read centroid data from either Excel (.xlsx) or CSV file into a dictionary.
    
    Parameters:
    file_path (str): Path to the input file (.xlsx or .csv)
    
    Returns:
    dict: Dictionary with first column as keys and next three columns as numpy array values
    """
    def load_json_to_dict(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert only numeric strings to integers, leave other strings as is
        converted_data = {}
        for k, v in data.items():
            try:
                converted_data[int(k)] = v
            except ValueError:
                converted_data[k] = v
        
        return converted_data
    # Check file extension
    if file_path.lower().endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith('.json'):
        df = load_json_to_dict(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide either .xlsx, .csv, or .json file")
    
    # Initialize an empty dictionary
    data_dict = {}
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # First column is the key
        key = row.iloc[0]
        # Next three columns are the values
        value = np.array(row.iloc[1:4])
        # Add the key-value pair to the dictionary
        data_dict[key] = value
        
    return data_dict

def read_excel_to_singval_dict(file_path):
    """
    Read data from either Excel (.xlsx) or CSV file into a dictionary with single values.
    
    Parameters:
    file_path (str): Path to the input file (.xlsx or .csv)
    
    Returns:
    dict: Dictionary with first column as keys and second column as values
    """
    def load_json_to_dict(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert only numeric strings to integers, leave other strings as is
        converted_data = {}
        for k, v in data.items():
            try:
                converted_data[int(k)] = v
            except ValueError:
                converted_data[k] = v
        
        return converted_data

    # Check file extension and read accordingly
    if file_path.lower().endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith('.json'):
        df = load_json_to_dict(file_path)
        return df
    else:
        raise ValueError("Unsupported file format. Please provide either .xlsx, .csv, or .json file")
    
    # Convert the DataFrame to a dictionary
    data_dict = {}
    for idx, row in df.iterrows():
        key = row.iloc[0]  # First column as key
        value = row.iloc[1]  # Second column as value
        data_dict[key] = value
        
    return data_dict

def combine_lists_to_sublists(master_list):

    def fill_if_empty(lst, num_zeros):
        """
        Checks if a list is empty and fills it with zeros if it is.
        
        Args:
            lst (list): The list to check and potentially fill
            num_zeros (int): Number of zeros to add if list is empty
        
        Returns:
            list: The original list if not empty, or new list filled with zeros if empty
        """
        if not lst:  # This checks if the list is empty
            lst.extend([0] * num_zeros)
        return lst

    list1 = master_list[0]
    list2 = master_list[1]
    list3 = master_list[2]
    list3 = fill_if_empty(list3, len(list1))

    # Combine the lists into one list of sublists
    combined_list = [list(sublist) for sublist in zip(list1, list2, list3)]
    
    return combined_list


def find_centroids(nodes, down_factor = None, network = None):

    """Can be used to save an excel file containing node IDs and centroids in a network. Inputs are a node.tif or node np array, an optional network excel file, and optional downsample factor"""

    if type(nodes) == str: #Open into numpy array if filepath
        nodes = tifffile.imread(nodes)

    if len(np.unique(nodes)) == 2: #Label if binary
        structure_3d = np.ones((3, 3, 3), dtype=int)
        nodes, num_nodes = ndimage.label(nodes)

    if down_factor is not None:
        nodes = downsample(nodes, down_factor)
    else: 
        down_factor = 1

    centroid_dict = {}

    if network is not None:

        G = open_network(network)

        node_ids = list(G.nodes)

        for nodeid in node_ids:
            centroid = compute_centroid(nodes, nodeid)
            if centroid is not None:
                centroid = down_factor * centroid
                centroid_dict[nodeid] = centroid

    else:
        node_max = np.max(nodes)
        for nodeid in range(1, node_max + 1):
            centroid = compute_centroid(nodes, nodeid)
            if centroid is not None:
                centroid = down_factor * centroid
                centroid_dict[nodeid] = centroid

    _save_centroid_dictionary(centroid_dict)

    return centroid_dict

def _save_centroid_dictionary(centroid_dict, filepath=None, index='Node ID'):
    # Convert dictionary to DataFrame with keys as index and values as a column
    df = pd.DataFrame.from_dict(centroid_dict, orient='index', columns=['Z', 'Y', 'X'])
    
    # Rename the index to specified name
    df.index.name = index
    
    if filepath is None:
        base_path = 'centroids'
    else:
        # Remove file extension if present to use as base path
        base_path = filepath.rsplit('.', 1)[0]
    
    # First try to save as CSV
    try:
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path)
        print(f"Successfully saved centroids to {csv_path}")
        return
    except Exception as e:
        print(f"Could not save centroids as CSV: {str(e)}")
        
        # If CSV fails, try to save as Excel
        try:
            xlsx_path = f"{base_path}.xlsx"
            df.to_excel(xlsx_path, engine='openpyxl')
            print(f"Successfully saved centroids to {xlsx_path}")
        except Exception as e:
            print(f"Could not save centroids as XLSX: {str(e)}")

def _find_centroids_GPU(nodes, node_list=None, down_factor=None):
    """Internal use version to get centroids without saving"""

    def _compute_centroid_GPU(binary_stack, label):
        """
        Finds centroid of labelled object in array
        """
        indices = cp.argwhere(binary_stack == label)
        if indices.shape[0] == 0:
            return None
        else:
            centroid = cp.round(np.mean(indices, axis=0)).astype(int)
    
        centroid = centroid.tolist()
        return centroid

    nodes = cp.asarray(nodes)
    if isinstance(nodes, str):  # Open into numpy array if filepath
        nodes = tifffile.imread(nodes)

        if len(cp.unique(nodes)) == 2:  # Label if binary
            structure_3d = cp.ones((3, 3, 3), dtype=int)
            nodes, num_nodes = cpx.label(nodes)

    if down_factor is not None:
        nodes = cp.asnumpy(nodes)
        nodes = downsample(nodes, down_factor)
        nodes = cp.asarray(nodes)
    else:
        down_factor = 1

    centroid_dict = {}

    if node_list is None:
        node_list = cp.unique(nodes)
        node_list = node_list.tolist()
        if node_list[0] == 0:
            del node_list[0]

    for label in node_list:
        centroid = _compute_centroid_GPU(nodes, label)
        if centroid is not None:
            centroid_dict[label] = centroid

    return centroid_dict

def _find_centroids_old(nodes, node_list = None, down_factor = None):

    """Internal use version to get centroids without saving"""


    if type(nodes) == str: #Open into numpy array if filepath
        nodes = tifffile.imread(nodes)

        if len(np.unique(nodes)) == 2: #Label if binary
            structure_3d = np.ones((3, 3, 3), dtype=int)
            nodes, num_nodes = ndimage.label(nodes)

    if down_factor is not None:
        nodes = downsample(nodes, down_factor)
    else:
        down_factor = 1

    centroid_dict = {}

    if node_list is None:

        node_max = np.max(nodes)

        for nodeid in range(1, node_max + 1):
            centroid = compute_centroid(nodes, nodeid)
            if centroid is not None:
                #centroid = down_factor * centroid
                centroid_dict[nodeid] = centroid

    else:
        for nodeid in node_list:
            centroid = compute_centroid(nodes, nodeid)
            if centroid is not None:
                #centroid = down_factor * centroid
                centroid_dict[nodeid] = centroid

    return centroid_dict

def _find_centroids(nodes, node_list=None, down_factor=None):
    """Internal use version to get centroids without saving"""
    def get_label_indices(binary_stack, label, y_offset):
        """
        Finds indices of labelled object in array and adjusts for the Y-offset.
        """
        indices = np.argwhere(binary_stack == label)
        # Adjust the Y coordinate by the y_offset
        indices[:, 1] += y_offset
        return indices
    
    def compute_indices_in_chunk(chunk, y_offset):
        """
        Get indices for all labels in a given chunk of the 3D array.
        Adjust Y-coordinate based on the y_offset for each chunk.
        """
        indices_dict_chunk = {}
        label_list = np.unique(chunk)
        try:
            if label_list[0] == 0:
                label_list = np.delete(label_list, 0)
        except:
            pass
        
        for label in label_list:
            indices = get_label_indices(chunk, label, y_offset)
            indices_dict_chunk[label] = indices
        return indices_dict_chunk
    
    def chunk_3d_array(array, num_chunks):
        """
        Split the 3D array into smaller chunks along the y-axis.
        """
        y_slices = np.array_split(array, num_chunks, axis=1)
        return y_slices
    
    if isinstance(nodes, str):  # Open into numpy array if filepath
        nodes = tifffile.imread(nodes)
        if len(np.unique(nodes)) == 2:  # Label if binary
            structure_3d = np.ones((3, 3, 3), dtype=int)
            nodes, num_nodes = ndimage.label(nodes)
    
    if down_factor is not None:
        nodes = downsample(nodes, down_factor)
    else:
        down_factor = 1
    
    indices_dict = {}
    num_cpus = mp.cpu_count()
    
    # Chunk the 3D array along the y-axis into smaller subarrays
    node_chunks = chunk_3d_array(nodes, num_cpus)
    
    # Calculate Y offset for each chunk
    chunk_sizes = [chunk.shape[1] for chunk in node_chunks]
    y_offsets = np.cumsum([0] + chunk_sizes[:-1])
    
    # Parallel computation of indices across chunks
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(compute_indices_in_chunk, chunk, y_offset): chunk_id
                  for chunk_id, (chunk, y_offset) in enumerate(zip(node_chunks, y_offsets))}
        
        for future in as_completed(futures):
            indices_chunk = future.result()
            # Merge indices for each label
            for label, indices in indices_chunk.items():
                if label in indices_dict:
                    indices_dict[label] = np.vstack((indices_dict[label], indices))
                else:
                    indices_dict[label] = indices
    
    # Compute centroids from collected indices
    centroid_dict = {}
    for label, indices in indices_dict.items():
        centroid = np.round(np.mean(indices, axis=0)).astype(int)
        centroid_dict[label] = centroid
    
    try:
        del centroid_dict[0]
    except:
        pass
    
    return centroid_dict


def get_degrees(nodes, network, down_factor = None, directory = None, centroids = None, called = False, no_img = 0):

    print("Generating table containing degree of each node...")
    if type(nodes) == str:
        nodes = tifffile.imread(nodes)

    if len(np.unique(nodes)) < 3:
        
        structure_3d = np.ones((3, 3, 3), dtype=int)
        nodes, num_nodes = ndimage.label(nodes, structure=structure_3d)

    if type(network) == str:

        G, weights = weighted_network(network)
    else:
        G = network

    node_list = list(G.nodes)
    node_dict = {}

    for node in node_list:
        node_dict[node] = (G.degree(node))

    if not called:

        # Convert dictionary to DataFrame with keys as index and values as a column
        df = pd.DataFrame.from_dict(node_dict, orient='index', columns=['Degree'])

        # Rename the index to 'Node ID'
        df.index.name = 'Node ID'

    if not called:

        if directory is None:
            # Save DataFrame to Excel file
            df.to_excel('degrees.xlsx', engine='openpyxl')
            print("Degrees saved to degrees.xlsx")

        else:
            df.to_excel(f'{directory}/degrees.xlsx', engine='openpyxl')
            print(f"Degrees saved to {directory}/degrees.xlsx")


    print("Drawing overlay containing degree labels for each node...")

    if down_factor is not None:

        for item in nodes.shape:
            if item < 5:
                down_factor = 1
                break


    if no_img == 1:

        if centroids is None:

            centroids = _find_centroids(nodes, down_factor = down_factor)

        nodes = node_draw.degree_draw(node_dict, centroids, nodes)

        if not called:

            if directory is None:

                tifffile.imwrite("degree_labels.tif", labels)
                print(f"Degree labels saved to degree_labels.tif")


            else:
                tifffile.imwrite(f"{directory}/degree_labels.tif", labels)
                print(f"Degree labels saved to {directory}/degree_labels.tif")


    elif no_img == 2:

        print("Drawing overlay containing grayscale degree labels for each node...")

        nodes = node_draw.degree_infect(node_dict, nodes)

        if not called:

            if directory is None:

                tifffile.imwrite("degree_labels_grayscale.tif", masks)

            else:
                tifffile.imwrite(f"{directory}/degree_labels_grayscale.tif", masks)

    return node_dict, nodes



def remove_dupes(network):
    """Remove Duplicates using numpy arrays"""    
    if type(network) == str:
        network = read_excel_to_lists
    
    nodesA = np.array(network[0])
    nodesB = np.array(network[1])
    edgesC = np.array(network[2])
    
    # Create normalized edges (smaller node first)
    edges = np.column_stack([np.minimum(nodesA, nodesB), np.maximum(nodesA, nodesB)])
    
    # Find unique edges and their indices
    _, unique_indices = np.unique(edges, axis=0, return_index=True)
    
    # Sort indices to maintain original order
    unique_indices = np.sort(unique_indices)
    
    # Extract unique connections
    filtered_nodesA = nodesA[unique_indices].tolist()
    filtered_nodesB = nodesB[unique_indices].tolist()
    filtered_edgesC = edgesC[unique_indices].tolist()
    
    return [filtered_nodesA, filtered_nodesB, filtered_edgesC]






#Concerning radial analysis:
def radial_analysis(nodes, network, rad_dist, xy_scale = None, z_scale = None, centroids = None, directory = None, down_factor = None):
    print("Performing Radial Distribution Analysis...")

    print("Generating excel notebook containing degree of each node...")
    if type(nodes) == str:
        nodes = tifffile.imread(nodes)

    if len(np.unique(nodes)) < 3:
        
        structure_3d = np.ones((3, 3, 3), dtype=int)
        nodes, num_nodes = ndimage.label(nodes, structure=structure_3d)

    if type(network) == str:

        network = read_excel_to_lists(network)

    if xy_scale is None:
        xy_scale = 1

    if z_scale is None:
        z_scale = 1

    if down_factor is not None:
        xy_scale = xy_scale * down_factor
        z_scale = z_scale * down_factor
        nodes = downsample(nodes, down_factor)


    num_objects = np.max(nodes)

    if centroids is None:
        centroids = _find_centroids(nodes)

    dist_list = get_distance_list(centroids, network, xy_scale, z_scale)
    x_vals, y_vals = buckets(dist_list, num_objects, rad_dist, directory = directory)
    histogram(x_vals, y_vals, directory = directory)
    output = {}
    for i in range(len(x_vals)):
        output[y_vals[i]] = x_vals[i]
    return output

def buckets(dists, num_objects, rad_dist, directory = None):
    y_vals = []
    x_vals = []
    radius = 0
    max_dist = max(dists)

    while radius < max_dist:
        radius2 = radius + rad_dist
        radial_objs = 0
        for item in dists:
            if item >= radius and item <= radius2:
                radial_objs += 1
        radial_avg = radial_objs/num_objects
        radius = radius2
        x_vals.append(radial_avg)
        y_vals.append(radius)

    # Create a DataFrame from the lists
    data = {'Radial Distance From Any Node': y_vals, 'Average Number of Neighboring Nodes': x_vals}
    df = pd.DataFrame(data)

    try:

        if directory is None:
            # Save the DataFrame to an Excel file
            df.to_excel('radial_distribution.xlsx', index=False)
            print("Radial distribution saved to radial_distribution.xlsx")
        else:
            df.to_excel(f'{directory}/radial_distribution.xlsx', index=False)
            print(f"Radial distribution saved to {directory}/radial_distribution.xlsx")
    except:
        pass

    return x_vals, y_vals

def histogram(counts, y_vals, directory = None):
    # Calculate the bin edges based on the y_vals
    bins = np.linspace(min(y_vals), max(y_vals), len(y_vals) + 1)

    # Create a histogram
    plt.hist(x=y_vals, bins=bins, weights=counts, edgecolor='black')

    # Adding labels and title (Optional, but recommended for clarity)
    plt.title('Radial Distribution of Network')
    plt.xlabel('Distance from any node')
    plt.ylabel('Avg Number of Neigbhoring Vertices')

    try:
        if directory is not None:
            plt.savefig(f'{directory}/radial_plot.png')
    except:
        pass

    # Show the plot
    plt.show()

def get_distance_list(centroids, network, xy_scale, z_scale):
    print("Generating radial distribution...")

    distance_list = [] #init empty list to contain all distance vals

    nodesa = network[0]
    nodesb = network[1]

    for i in range(len(nodesa)):
        try:
            z1, y1, x1 = centroids[nodesa[i]]
            z1, y1, x1 = z1 * z_scale, y1 * xy_scale, x1 * xy_scale
            z2, y2, x2 = centroids[nodesb[i]]
            z2, y2, x2 = z2 * z_scale, y2 * xy_scale, x2 * xy_scale

            dist = np.sqrt((z2 - z1)**2 + (y2 - y1)**2 + (x2 - x1)**2)
            distance_list.append(dist)
        except:
            pass

    return distance_list


def prune_samenode_connections(networkfile, nodeIDs):
    """Even faster numpy-based version for very large datasets"""
    import numpy as np
    
    # Handle nodeIDs input
    if type(nodeIDs) == str:
        df = pd.read_excel(nodeIDs)
        data_dict = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0]).to_dict()
    else:
        data_dict = nodeIDs
    
    # Handle networkfile input
    if type(networkfile) == str:
        master_list = read_excel_to_lists(networkfile)
    else:
        master_list = networkfile
    
    nodesA = np.array(master_list[0])
    nodesB = np.array(master_list[1])
    
    # Handle edgesC safely
    if len(master_list) > 2 and master_list[2]:
        edgesC = np.array(master_list[2], dtype=object)
    else:
        edgesC = np.array([None] * len(nodesA), dtype=object)
    
    # Vectorized lookup of node IDs
    idsA = np.array([data_dict.get(node) for node in nodesA])
    idsB = np.array([data_dict.get(node) for node in nodesB])
    
    # Create boolean mask - keep where IDs are different
    keep_mask = idsA != idsB
    
    # Apply filter
    filtered_nodesA = nodesA[keep_mask].tolist()
    filtered_nodesB = nodesB[keep_mask].tolist()
    filtered_edgesC = edgesC[keep_mask].tolist()
    
    # Create save_list
    save_list = [[filtered_nodesA[i], filtered_nodesB[i], filtered_edgesC[i]] 
                 for i in range(len(filtered_nodesA))]
    
    # Handle file saving
    if type(networkfile) == str:
        filename = 'network_pruned_away_samenode_connections.xlsx'
        nettracer.create_and_save_dataframe(save_list, filename)
        print(f"Pruned network saved to {filename}")
    
    # Create output_dict
    nodes_in_filtered = set(filtered_nodesA + filtered_nodesB)
    output_dict = {node: data_dict[node] for node in nodes_in_filtered 
                   if node in data_dict}
    
    # Handle identity file saving
    if type(networkfile) == str:
        filename = 'Node_identities_pruned_away_samenode_connections.xlsx'
        save_singval_dict(output_dict, 'NodeID', 'Identity', filename)
        print(f"Pruned network identities saved to {filename}")
    
    master_list = [filtered_nodesA, filtered_nodesB, filtered_edgesC]
    return master_list, output_dict


def isolate_internode_connections(networkfile, nodeIDs, ID1, ID2):
    """Even faster numpy-based version for very large datasets"""
    import numpy as np
    
    # Handle nodeIDs input
    if type(nodeIDs) == str:
        df = pd.read_excel(nodeIDs)
        data_dict = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0]).to_dict()
    else:
        data_dict = nodeIDs
    
    # Handle networkfile input
    if type(networkfile) == str:
        master_list = read_excel_to_lists(networkfile)
    else:
        master_list = networkfile
    
    nodesA = np.array(master_list[0])
    nodesB = np.array(master_list[1])
    edgesC = np.array(master_list[2])
    
    # Vectorized lookup of node values
    valuesA = np.array([str(data_dict.get(node, '')) for node in nodesA])
    valuesB = np.array([str(data_dict.get(node, '')) for node in nodesB])
    
    # Create boolean mask for filtering
    legalIDs_set = {str(ID1), str(ID2)}
    maskA = np.array([val in legalIDs_set for val in valuesA])
    maskB = np.array([val in legalIDs_set for val in valuesB])
    keep_mask = maskA & maskB
    
    # Apply filter
    filtered_nodesA = nodesA[keep_mask].tolist()
    filtered_nodesB = nodesB[keep_mask].tolist()
    filtered_edgesC = edgesC[keep_mask].tolist()
    
    # Create save_list
    save_list = [[filtered_nodesA[i], filtered_nodesB[i], filtered_edgesC[i]] 
                 for i in range(len(filtered_nodesA))]
    
    # Handle file saving
    if type(networkfile) == str:
        filename = f'network_isolated_{ID1}_{ID2}_connections.xlsx'
        nettracer.create_and_save_dataframe(save_list, filename)
        print(f"Isolated internode network saved to {filename}")
    
    # Create output_dict
    nodes_in_filtered = set(filtered_nodesA + filtered_nodesB)
    output_dict = {node: data_dict[node] for node in nodes_in_filtered 
                   if node in data_dict}
    
    # Handle identity file saving
    if type(networkfile) == str:
        filename = f'Node_identities_for_isolated_{ID1}_{ID2}_network.xlsx'
        save_singval_dict(output_dict, 'NodeID', 'Identity', filename)
        print(f"Isolated network identities saved to {filename}")
    
    master_list = [filtered_nodesA, filtered_nodesB, filtered_edgesC]
    return master_list, output_dict

def edge_to_node(network, node_identities=None, maxnode=None):
    """Even faster numpy-based version for very large datasets"""
    import numpy as np
    
    # Handle node_identities input
    if node_identities is not None and type(node_identities) == str:
        df = pd.read_excel(node_identities)
        identity_dict = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0]).to_dict()
    elif node_identities is not None and type(node_identities) != str:
        identity_dict = node_identities.copy()
    else:
        identity_dict = {}
    
    # Handle network input
    if type(network) == str:
        master_list = read_excel_to_lists(network)
    else:
        master_list = network
    
    # Convert to numpy arrays for vectorized operations
    nodesA = np.array(master_list[0])
    nodesB = np.array(master_list[1])
    edgesC = np.array(master_list[2])
    
    # Get all unique nodes efficiently
    allnodes = set(np.concatenate([nodesA, nodesB]).tolist())
    
    # Calculate maxnode if not provided
    if maxnode is None:
        maxnode = int(np.max(np.concatenate([nodesA, nodesB])))
    
    print(f"Transposing all edge vals by {maxnode} to prevent ID overlap with preexisting nodes")
    
    # Vectorized edge transposition
    transposed_edges = edgesC + maxnode
    
    # Create new_network using vectorized operations
    # Create arrays for the two types of connections
    connections1 = np.column_stack([nodesA, transposed_edges, np.zeros(len(nodesA))])
    connections2 = np.column_stack([transposed_edges, nodesB, np.zeros(len(nodesB))])
    
    # Combine and convert to list format
    new_network_array = np.vstack([connections1, connections2])
    new_network = new_network_array.astype(int).tolist()
    
    # Update identity_dict efficiently
    # Add missing nodes
    for node in allnodes:
        if node not in identity_dict:
            identity_dict[node] = 'Node'
    
    # Add all edges at once
    for edge in transposed_edges.tolist():
        identity_dict[edge] = 'Edge'
    
    # Handle output
    if type(network) == str:
        save_singval_dict(identity_dict, 'NodeID', 'Identity', 'edge_to_node_identities.xlsx')
        nettracer.create_and_save_dataframe(new_network, 'edge-node_network.xlsx')
    else:
        df = nettracer.create_and_save_dataframe(new_network)
        return df, identity_dict, maxnode



def save_singval_dict(dict, index_name, valname, filename):
    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(dict, orient='index', columns=[valname])
    
    # Rename the index
    df.index.name = index_name
    
    # Remove file extension if present to use as base path
    base_path = filename.rsplit('.', 1)[0]
    
    # First try to save as CSV
    try:
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path)
        print(f"Successfully saved {valname} data to {csv_path}")
        return
    except Exception as e:
        print(f"Could not save as CSV: {str(e)}")
        
        # If CSV fails, try to save as Excel
        try:
            xlsx_path = f"{base_path}.xlsx"
            df.to_excel(xlsx_path, engine='openpyxl')
            print(f"Successfully saved {valname} data to {xlsx_path}")
        except Exception as e:
            print(f"Could not save as XLSX: {str(e)}")


def rand_net_weighted(num_rows, num_nodes, nodes):
    """Optimized weighted random network generation - allows duplicate edges"""
    nodes_array = np.array(nodes)
    n_nodes = len(nodes)
    
    # Pre-generate all random indices at once
    node_indices = np.random.randint(0, n_nodes, num_rows)
    partner_indices = np.random.randint(0, n_nodes, num_rows)
    
    # Fix self-connections by regenerating only where needed
    self_connection_mask = node_indices == partner_indices
    while np.any(self_connection_mask):
        partner_indices[self_connection_mask] = np.random.randint(0, n_nodes, np.sum(self_connection_mask))
        self_connection_mask = node_indices == partner_indices
    
    # Create network efficiently using vectorized operations
    random_network = np.column_stack([
        nodes_array[node_indices],
        nodes_array[partner_indices], 
        np.zeros(num_rows, dtype=int)
    ]).tolist()
    
    df = nettracer.create_and_save_dataframe(random_network)
    G, edge_weights = weighted_network(df)
    return G, df


def rand_net(num_rows, num_nodes, nodes):
    """Optimized unweighted random network generation - prevents duplicate edges"""
    random_network = []
    seen_edges = set()
    nodes_set = set(nodes)
    n_nodes = len(nodes)
    
    # Pre-calculate maximum possible unique edges
    max_possible_edges = n_nodes * (n_nodes - 1)  # No self-connections, but allows both directions
    
    if num_rows > max_possible_edges:
        raise ValueError(f"Cannot generate {num_rows} unique edges with {n_nodes} nodes. Maximum possible: {max_possible_edges}")
    
    attempts = 0
    max_attempts = num_rows * 10  # Prevent infinite loops
    
    while len(random_network) < num_rows and attempts < max_attempts:
        # Generate batch of random pairs
        batch_size = min(1000, num_rows - len(random_network))
        
        node_indices = np.random.randint(0, n_nodes, batch_size)
        partner_indices = np.random.randint(0, n_nodes, batch_size)
        
        for i in range(batch_size):
            node_idx = node_indices[i]
            partner_idx = partner_indices[i]
            
            # Skip self-connections
            if node_idx == partner_idx:
                attempts += 1
                continue
                
            node = nodes[node_idx]
            partner = nodes[partner_idx]
            
            # Create normalized edge tuple to check for duplicates (both directions)
            edge_tuple = tuple(sorted([node, partner]))
            
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                random_network.append([node, partner, 0])
                
            attempts += 1
            
            if len(random_network) >= num_rows:
                break
    
    if len(random_network) < num_rows:
        print(f"Warning: Only generated {len(random_network)} edges out of requested {num_rows}")
    
    df = nettracer.create_and_save_dataframe(random_network)
    G, edge_weights = weighted_network(df)
    return G, df


def generate_random(G, net_lists, weighted=True):
    """Optimized random network generation dispatcher"""
    nodes = list(G.nodes)
    num_nodes = len(nodes)
    num_rows = len(net_lists[0])
    
    if weighted:
        return rand_net_weighted(num_rows, num_nodes, nodes)
    else:
        return rand_net(num_rows, num_nodes, nodes)


def list_trim(list1, list2, component):

    list1_copy = list1
    indices_to_delete = []
    for i in range(len(list1)):

        if list1[i] not in component and list2[i] not in component:
            indices_to_delete.append(i)

    for i in reversed(indices_to_delete):
        del list1_copy[i]

    return list1_copy

def degree_distribution(G, directory = None):

    def create_incremental_list(length, start=1):
        return list(range(start, start + length))

    node_list = list(G.nodes)
    degree_dict = {}

    for node in node_list:
        degree = G.degree(node)
        if degree not in degree_dict:
            degree_dict[degree] = 1
        else:
            degree_dict[degree] += 1

    high_degree = max(degree_dict.keys())
    proportion_list = [0] * high_degree

    for item in degree_dict:
        proportion_list[item - 1] = float(degree_dict[item]/len(node_list))
    degrees = create_incremental_list(high_degree)


    df = pd.DataFrame({
        'Degree (k)': degrees,
        'Proportion of nodes with degree (p(k))': proportion_list
    })

    try:

        if directory is None:
            # Save the DataFrame to an Excel file
            df.to_excel('degree_dist.xlsx', index=False)
            print("Degree distribution saved to degree_dist.xlsx")
        else:
            df.to_excel(f'{directory}/degree_dist.xlsx', index=False)
            print(f"Degree distribution saved to {directory}/degree_dist.xlsx")
    except:
        pass


    power_trendline(degrees, proportion_list, directory = directory)

    return_dict = {}
    for i in range(len(degrees)):
        return_dict[degrees[i]] = proportion_list[i]

    return return_dict


def power_trendline(x, y, directory = None):
    # Handle zeros in y for logarithmic transformations
    """
    y = np.array(y)
    x = np.array(x)
    y[y == 0] += 0.001

    # Define the power function
    def power_func(x, a, b):
        return a * (x ** b)

    # Fit the power function to the data
    popt, pcov = curve_fit(power_func, x, y)
    a, b = popt

    # Create a range of x values for the trendline
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = power_func(x_fit, a, b)

    # Calculate R-squared value
    y_pred = power_func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    """
    # ^ I commented out this power trendline stuff because I decided I no longer want it to do that so.

    # Create a scatterplot
    plt.scatter(x, y, label='Data')
    plt.xlabel('Degree (k)')
    plt.ylabel('Proportion of nodes with degree (p(k))')
    plt.title('Degree Distribution of Network')

    # Plot the power trendline
    #plt.plot(x_fit, y_fit, color='red', label=f'Power Trendline: $y = {a:.2f}x^{{{b:.2f}}}$')

    # Annotate the plot with the trendline equation and R-squared value
    """
    plt.text(
        0.05, 0.95, 
        f'$y = {a:.2f}x^{{{b:.2f}}}$\n$R^2 = {r2:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top'
    )
    """

    try:

        if directory is not None:
            plt.savefig(f'{directory}/degree_plot.png')
    except:
        pass


    plt.show()

