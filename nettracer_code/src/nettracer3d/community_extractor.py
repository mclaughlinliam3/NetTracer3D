import pandas as pd
import networkx as nx
import tifffile
import numpy as np
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict, Counter
from networkx.algorithms import community
from scipy import ndimage
from scipy.ndimage import zoom
from networkx.algorithms import community
import random
from . import node_draw


def binarize(image):
    """Convert an array from numerical values to boolean mask"""
    image = image != 0

    image = image.astype(np.uint8)

    return image

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

def weighted_network(excel_file_path):
    """creates a network where the edges have weights proportional to the number of connections they make between the same structure"""
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

def compute_centroid(binary_stack, label):
    """
    Finds centroid of labelled object in array
    """
    indices = np.argwhere(binary_stack == label)
    centroid = np.round(np.mean(indices, axis=0)).astype(int)

    return centroid



def get_border_nodes(partition, G):
# Find nodes that border nodes in other communities
    border_nodes = set()
    intercom_connections = 0
    connected_coms = []
    for edge in G.edges():
        if partition[edge[0]] != partition[edge[1]]:
            border_nodes.add(edge[0])
            border_nodes.add(edge[1])
            connected_coms.append(partition[edge[0]])
            connected_coms.append(partition[edge[1]])
            intercom_connections += 1

    return border_nodes, intercom_connections, set(connected_coms)

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

def labels_to_boolean(label_array, labels_list):
    # Use np.isin to create a boolean array with a single operation
    boolean_array = np.isin(label_array, labels_list)
    
    return boolean_array

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


def open_network(excel_file_path):
    """opens an unweighted network from the network excel file"""

    # Read the Excel file into a pandas DataFrame
    master_list = read_excel_to_lists(excel_file_path)

    # Create a graph
    G = nx.Graph()

    nodes_a = master_list[0]
    nodes_b = master_list[1]

    # Add edges to the graph
    for i in range(len(nodes_a)):
        G.add_edge(nodes_a[i], nodes_b[i])

    return G


def _isolate_connected(G, key = None):

    if key is None:
        connected_components = list(nx.connected_components(G))
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Gcc[0])
        return G0

    else:
        # Get the connected component containing the specific node label
        connected_component = nx.node_connected_component(G, key)

        G0 = G.subgraph(connected_component)
        return G0


def extract_mothers(nodes, G, partition, centroid_dic = None, directory = None, ret_nodes = False, called = False):


    my_nodes, intercom_connections, connected_coms = get_border_nodes(partition, G)
    some_communities = partition.keys()

    print(f"Number of intercommunity connections: {intercom_connections}")
    print(f"{len(connected_coms)} communities with any connectivity of {len(some_communities)} communities")

    mother_nodes = list(my_nodes)

    if ret_nodes or called:
        
        # Create a list to store nodes to be removed
        nodes_to_remove = []

        # Iterate through all nodes in the graph
        for node in G.nodes():
            # Check if the node's ID is not in the id_list
            if node not in mother_nodes:
                nodes_to_remove.append(node)

        # Remove the identified nodes from the graph
        G.remove_nodes_from(nodes_to_remove)

        if ret_nodes:

            return G


    if not ret_nodes:

        if centroid_dic is None:
            for item in nodes.shape:
                if item < 5:
                    down_factor = 1
                    break
                else:
                    down_factor = 5

            smalls2 = downsample(nodes, down_factor)

            centroid_dic = {}

            for item in mother_nodes:
                centroid = compute_centroid(smalls2, item)
                centroid_dic[item] = centroid

        mother_dict = {}


        for node in mother_nodes:
            mother_dict[node] = G.degree(node)

        #mask2 = labels_to_boolean(nodes, mother_nodes)

        smalls = labels_to_boolean(nodes, mother_nodes)

        if not called:

            # Convert boolean values to 0 and 255
            mask = smalls * nodes

            labels = node_draw.degree_draw(mother_dict, centroid_dic, smalls)

            # Convert dictionary to DataFrame with keys as index and values as a column
            df = pd.DataFrame.from_dict(mother_dict, orient='index', columns=['Degree'])

            # Rename the index to 'Node ID'
            df.index.name = 'Node ID'

            if directory is None:

                # Save DataFrame to Excel file
                df.to_excel('mothers.xlsx', engine='openpyxl')
                print("Mother list saved to mothers.xlsx")
            else:
                df.to_excel(f'{directory}/mothers.xlsx', engine='openpyxl')
                print(f"Mother list saved to {directory}/mothers.xlsx")

            if directory is None:

                tifffile.imwrite("mother_nodes.tif", mask)
                print("Mother nodes saved to mother_nodes.tif")
                tifffile.imwrite("mother_degree_labels.tif", labels)
                print(f"Mother degree labels saved to mother_degree_labels.tif")

            else:
                tifffile.imwrite(f"{directory}/mother_nodes.tif", mask)
                print(f"Mother nodes saved to {directory}/mother_nodes.tif")
                tifffile.imwrite(f"{directory}/mother_degree_labels.tif", labels)
                print(f"Mother degree labels saved to {directory}/mother_degree_labels.tif")


            smalls = node_draw.degree_infect(mother_dict, mask)

            if directory is None:

                tifffile.imwrite("mother_degree_labels_grayscale.tif", smalls)
                print("Mother graycale degree labels saved to mother_degree_labels_grayscale.tif")

            else:
                tifffile.imwrite(f"{directory}/mother_degree_labels_grayscale.tif", smalls)
                print(f"Mother graycale degree labels saved to {directory}/mother_degree_labels_grayscale.tif")


            return mother_nodes, smalls
        else:
            smalls = smalls * nodes
            return G, smalls



def find_hub_nodes(G: nx.Graph, proportion: float = 0.1) -> List:
    """
    Identifies hub nodes in a network based on average shortest path length,
    handling multiple connected components.
    
    Args:
        G (nx.Graph): NetworkX graph (can have multiple components)
        proportion (float): Proportion of top nodes to return (0.0 to 1.0)
        
    Returns:
        List of nodes identified as hubs across all components
    """
    if not 0 < proportion <= 1:
        raise ValueError("Proportion must be between 0 and 1")
    
    # Get connected components
    components = list(nx.connected_components(G))
    
    # Dictionary to store average path lengths for all nodes
    avg_path_lengths: Dict[int, float] = {}
    
    output = []

    # Process each component separately
    for component in components:
        # Create subgraph for this component
        subgraph = G.subgraph(component)
        if not (len(subgraph.nodes()) * proportion >= 0.75): #Skip components that are too small
            continue
        
        # Calculate average shortest path length for each node in this component
        for node in subgraph.nodes():
            # Get shortest paths from this node to all others in the component
            path_lengths = nx.single_source_shortest_path_length(subgraph, node)
            # Calculate average path length within this component
            avg_length = sum(path_lengths.values()) / (len(subgraph.nodes()) - 1)
            avg_path_lengths[node] = avg_length
    
        # Sort nodes by average path length (ascending)
        sorted_nodes = sorted(avg_path_lengths.items(), key=lambda x: x[1])
        
        # Calculate number of nodes to return
        num_nodes = int(np.ceil(len(G.nodes()) * proportion))
        
        # Return the top nodes (those with lowest average path lengths)
        hub_nodes = [node for node, _ in sorted_nodes[:num_nodes]]
        output.extend(hub_nodes)
        avg_path_lengths: Dict[int, float] = {}
    
    return output

def get_color_name_mapping():
    """Return a dictionary of common colors and their RGB values."""
    return {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'brown': (165, 42, 42),
        'pink': (255, 192, 203),
        'navy': (0, 0, 128),
        'teal': (0, 128, 128),
        'olive': (128, 128, 0),
        'maroon': (128, 0, 0),
        'lime': (50, 205, 50),
        'indigo': (75, 0, 130),
        'violet': (238, 130, 238),
        'coral': (255, 127, 80),
        'turquoise': (64, 224, 208),
        'gold': (255, 215, 0)
    }

def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Convert an RGB tuple to its nearest color name.
    
    Args:
        rgb: Tuple of (r, g, b) values
        
    Returns:
        str: Name of the closest matching color
    """
    color_map = get_color_name_mapping()
    
    # Convert input RGB to numpy array
    rgb_array = np.array(rgb)
    
    # Calculate Euclidean distance to all known colors
    min_distance = float('inf')
    closest_color = None
    
    for color_name, color_rgb in color_map.items():
        distance = np.sqrt(np.sum((rgb_array - np.array(color_rgb)) ** 2))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
            
    return closest_color

def convert_node_colors_to_names(node_to_color: Dict[int, Tuple[int, int, int]]) -> Dict[int, str]:
    """
    Convert a dictionary of node-to-RGB mappings to node-to-color-name mappings.
    
    Args:
        node_to_color: Dictionary mapping node IDs to RGB tuples
        
    Returns:
        Dictionary mapping node IDs to color names
    """
    return {node: rgb_to_color_name(color) for node, color in node_to_color.items()}

def generate_distinct_colors(n_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate visually distinct RGB colors using HSV color space.
    Colors are generated with maximum saturation and value, varying only in hue.
    
    Args:
        n_colors: Number of distinct colors needed
    
    Returns:
        List of RGB tuples
    """
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        # Convert HSV to RGB (assuming S=V=1)
        h = hue * 6
        c = int(255)
        x = int(255 * (1 - abs(h % 2 - 1)))
        
        if h < 1:
            rgb = (c, x, 0)
        elif h < 2:
            rgb = (x, c, 0)
        elif h < 3:
            rgb = (0, c, x)
        elif h < 4:
            rgb = (0, x, c)
        elif h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
            
        colors.append(rgb)
    return colors

def assign_node_colors(node_list: List[int], labeled_array: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Assign distinct colors to nodes and create an RGBA image.
    
    Args:
        node_list: List of node IDs
        labeled_array: 3D numpy array with labels corresponding to node IDs
    
    Returns:
        Tuple of (RGBA-coded numpy array (H, W, D, 4), dictionary mapping nodes to color names)
    """
    
    # Sort communities by size (descending)
    sorted_nodes= sorted(node_list, reverse=True)
    
    # Generate distinct colors
    colors = generate_distinct_colors(len(node_list))
    random.shuffle(colors) #Randomly sorted to make adjacent structures likely stand out
    
    # Convert RGB colors to RGBA by adding alpha channel
    colors_rgba = [(r, g, b, 255) for r, g, b in colors]  # Full opacity for colored regions
    
    # Create mapping from community to color
    node_to_color = {node: colors_rgba[i] for i, node in enumerate(sorted_nodes)}
    
    # Create RGBA array (initialize with transparent background)
    rgba_array = np.zeros((*labeled_array.shape, 4), dtype=np.uint8)
    
    # Assign colors to each voxel based on its label
    for label in np.unique(labeled_array):
        if label in node_to_color:  # Skip background (usually label 0)
            mask = labeled_array == label
            for i in range(4):  # RGBA channels
                rgba_array[mask, i] = node_to_color[label][i]

    # Convert the RGB portion of community_to_color back to RGB for color naming
    node_to_color_rgb = {k: tuple(v[:3]) for k, v in node_to_color.items()}
    node_to_color_names = convert_node_colors_to_names(node_to_color_rgb)
    
    return rgba_array, node_to_color_names

def assign_community_colors(community_dict: Dict[int, int], labeled_array: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    """Ultra-fast version using lookup table approach."""
    
    # Same setup as before
    communities = set(community_dict.values())
    community_sizes = Counter(community_dict.values())
    sorted_communities = sorted(communities, key=lambda x: community_sizes[x], reverse=True)
    
    colors = generate_distinct_colors(len(communities))
    colors_rgba = np.array([(r, g, b, 255) for r, g, b in colors], dtype=np.uint8)
    
    community_to_color = {comm: colors_rgba[i] for i, comm in enumerate(sorted_communities)}
    node_to_color = {node: community_to_color[comm] for node, comm in community_dict.items()}
    
    # Create lookup table - this is the key optimization
    max_label = max(max(labeled_array.flat), max(node_to_color.keys()) if node_to_color else 0)
    color_lut = np.zeros((max_label + 1, 4), dtype=np.uint8)  # Transparent by default
    
    for node_id, color in node_to_color.items():
        color_lut[node_id] = color
    
    # Single vectorized operation - this is much faster!
    rgba_array = color_lut[labeled_array]
    
    # Rest remains the same
    community_to_color_rgb = {k: tuple(v[:3]) for k, v in community_to_color.items()}
    node_to_color_names = convert_node_colors_to_names(community_to_color_rgb)
    
    return rgba_array, node_to_color_names

def assign_community_grays(community_dict: Dict[int, Union[int, str, Any]], labeled_array: np.ndarray) -> np.ndarray:
    """
    Assign grayscale values to communities. For numeric communities, uses the community
    number directly. For string/other communities, assigns sequential values.
    
    Args:
        community_dict: Dictionary mapping node IDs to community identifiers (numbers or strings)
        labeled_array: 3D numpy array with labels corresponding to node IDs
    
    Returns:
        tuple: (grayscale numpy array, mapping of node IDs to assigned values)
    """
    # Determine if we're dealing with numeric or string communities
    sample_value = next(iter(community_dict.values()))
    is_numeric = isinstance(sample_value, (int, float))
    
    if is_numeric:
        # For numeric communities, use values directly
        node_to_gray = community_dict
        max_val = max(community_dict.values())
    else:
        # For string/other communities, assign sequential values
        unique_communities = sorted(set(community_dict.values()))
        community_to_value = {comm: i+1 for i, comm in enumerate(unique_communities)}
        node_to_gray = {node: community_to_value[comm] for node, comm in community_dict.items()}
        max_val = len(unique_communities)
    
    # Choose appropriate dtype based on maximum value
    if max_val <= 255:
        dtype = np.uint8
    elif max_val <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32
    
    # Create output array
    gray_array = np.zeros_like(labeled_array, dtype=dtype)
    
    # Create mapping of unique communities to their grayscale values
    if is_numeric:
        community_to_gray = {comm: comm for comm in set(community_dict.values())}
    else:
        community_to_gray = {comm: i+1 for i, comm in enumerate(sorted(set(community_dict.values())))}
    
    # Use numpy's vectorized operations for faster assignment
    unique_labels = np.unique(labeled_array)
    for label in unique_labels:
        if label in node_to_gray:
            gray_array[labeled_array == label] = node_to_gray[label]
    
    return gray_array, community_to_gray


if __name__ == "__main__":

    # Read the Excel file into a pandas DataFrame
    excel_file_path = input("Excel file?: ")
    masks = input("watershedded, dilated glom mask?: ")
    masks = tifffile.imread(masks)
    masks = masks.astype(np.uint16)

    G = open_network(excel_file_path)

    # Get a list of connected components
    connected_components = list(nx.connected_components(G))

    largest_component = max(connected_components, key=len)


    # Choose a specific connected component (let's say, the first one)
    #selected_component = connected_components[0]

    # Convert the set of nodes to a list
    #nodes_in_component = list(selected_component)

    nodes_in_largest_component = list(largest_component)

    mask2 = labels_to_boolean(masks, nodes_in_largest_component)

    # Convert boolean values to 0 and 255
    mask2 = mask2.astype(np.uint8) * 255

    tifffile.imwrite("isolated_community.tif", mask2)