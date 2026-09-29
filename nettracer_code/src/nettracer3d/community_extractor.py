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
import copy
from . import node_draw
import math
from . import color_schemes as _cschemes

# These overlays are composited over the main window, which is black.
# The palette engine uses this to avoid emitting near-black colors.
_COLOR_BACKGROUND = 'black'

# Okabe-Ito. Their black is dropped (it IS the background); white takes its slot.
_CB_PALETTE_HEX = [
    "FFFFFF",  # white          (stands in for Okabe-Ito's black)
    "E69F00",  # orange
    "56B4E9",  # sky blue
    "009E73",  # bluish green
    "F0E442",  # yellow
    "0072B2",  # blue
    "D55E00",  # vermillion
    "CC79A7",  # reddish purple
    # Paul Tol "muted" extras, also CVD-conscious, for n > 8.
    "88CCEE",  # cyan
    "999933",  # olive
    "882255",  # wine
    "44AA99",  # teal
    "DDCC77",  # sand
    "AA4499",  # purple
]
 
# Deuteranopia simulation matrix (Vienot et al. 1999), applied to LINEAR RGB.
_DEUTAN_MATRIX = (
    (0.33066007, 0.66933993, 0.0),
    (0.33066007, 0.66933993, 0.0),
    (-0.02785538, 0.02785538, 1.0),
)
 
# Lightness weights. Under normal vision hue/chroma carry the categorical
# signal, so lightness is discounted. A deuteranope has lost the red-green
# axis, leaving lightness as the main surviving channel -- so it is weighted
# UP there. This is exactly why Okabe-Ito works: its colors are staggered in
# lightness, not just in hue.
_CB_L_WEIGHT_NORMAL = 0.75
_CB_L_WEIGHT_DEUTAN = 1.6
 
# Separation below which two colors read as "the same" to a deuteranope.
# Calibrated empirically: the tightest pair in Okabe-Ito -- a palette
# published as CVD-safe -- scores 0.083 under the metric below.
CB_CONFUSABLE = 0.083

RGB = Tuple[int, int, int]
 
# Kelly's 22 colors of maximum contrast, minus black.
# Includes white, grays, tans and browns on purpose.
_BASE_PALETTE_HEX = [
    "F2F3F4",  # white
    "F3C300",  # vivid yellow
    "875692",  # strong purple
    "F38400",  # vivid orange
    "A1CAF1",  # very light blue
    "BE0032",  # vivid red
    "C2B280",  # grayish yellow / buff
    "848482",  # medium gray
    "008856",  # vivid green
    "E68FAC",  # strong purplish pink
    "0067A5",  # strong blue
    "F99379",  # strong yellowish pink
    "604E97",  # strong violet
    "F6A600",  # vivid orange yellow
    "B3446C",  # strong purplish red
    "DCD300",  # vivid greenish yellow
    "882D17",  # strong reddish brown
    "8DB600",  # vivid yellowish green
    "654522",  # deep yellowish brown
    "E25822",  # vivid reddish orange
    "2B3D26",  # dark olive green
]


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
        try:
            if partition[edge[0]] != partition[edge[1]]:
                border_nodes.add(edge[0])
                border_nodes.add(edge[1])
                connected_coms.append(partition[edge[0]])
                connected_coms.append(partition[edge[1]])
                intercom_connections += 1
        except:
            pass

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
    """Return a dictionary of descriptive color names and their RGB values."""
    return {
        # Reds
        'crimson_red': (220, 20, 60),
        'bright_red': (255, 0, 0),
        'dark_red': (139, 0, 0),
        'coral_red': (255, 127, 80),
        'rose_red': (255, 102, 102),
        'burgundy': (128, 0, 32),
        'cherry_red': (222, 49, 99),
        
        # Greens
        'forest_green': (34, 139, 34),
        'lime_green': (50, 205, 50),
        'bright_green': (0, 255, 0),
        'dark_green': (0, 100, 0),
        'mint_green': (152, 255, 152),
        'sage_green': (159, 183, 121),
        'emerald_green': (80, 200, 120),
        'olive_green': (128, 128, 0),
        
        # Blues
        'royal_blue': (65, 105, 225),
        'bright_blue': (0, 0, 255),
        'navy_blue': (0, 0, 128),
        'sky_blue': (135, 206, 235),
        'steel_blue': (70, 130, 180),
        'powder_blue': (176, 224, 230),
        'midnight_blue': (25, 25, 112),
        'cobalt_blue': (0, 71, 171),
        
        # Purples
        'deep_purple': (75, 0, 130),
        'royal_purple': (120, 81, 169),
        'lavender': (230, 230, 250),
        'plum_purple': (221, 160, 221),
        'violet_purple': (238, 130, 238),
        'magenta': (255, 0, 255),
        'orchid': (218, 112, 214),
        
        # Yellows & Golds
        'bright_yellow': (255, 255, 0),
        'golden_yellow': (255, 215, 0),
        'lemon_yellow': (255, 247, 0),
        'amber': (255, 191, 0),
        'mustard_yellow': (255, 219, 88),
        'cream': (255, 253, 208),
        'wheat': (245, 222, 179),
        
        # Oranges
        'bright_orange': (255, 165, 0),
        'burnt_orange': (204, 85, 0),
        'peach': (255, 218, 185),
        'tangerine': (255, 163, 67),
        'pumpkin_orange': (255, 117, 24),
        'apricot': (251, 206, 177),
        
        # Pinks
        'hot_pink': (255, 105, 180),
        'light_pink': (255, 192, 203),
        'deep_pink': (255, 20, 147),
        'salmon_pink': (250, 128, 114),
        'blush_pink': (255, 182, 193),
        'fuchsia': (255, 0, 255),
        
        # Cyans & Teals
        'bright_cyan': (0, 255, 255),
        'dark_teal': (0, 128, 128),
        'turquoise': (64, 224, 208),
        'seafoam': (159, 226, 191),
        'teal_blue': (54, 117, 136),
        
        # Browns & Earth Tones
        'chocolate_brown': (210, 105, 30),
        'saddle_brown': (139, 69, 19),
        'light_brown': (205, 133, 63),
        'tan': (210, 180, 140),
        'beige': (245, 245, 220),
        'coffee_brown': (111, 78, 55),
        'rust_brown': (183, 65, 14),
        
        # Grays & Neutrals
        'charcoal_gray': (54, 69, 79),
        'light_gray': (211, 211, 211),
        'silver': (192, 192, 192),
        'slate_gray': (112, 128, 144),
        'ash_gray': (178, 190, 181),
        'smoke_gray': (152, 152, 152),
        
        # Additional Distinctive Colors
        'lime_yellow': (191, 255, 0),
        'electric_blue': (125, 249, 255),
        'neon_green': (57, 255, 20),
        'wine_red': (114, 47, 55),
        'copper': (184, 115, 51),
        'ivory': (255, 255, 240),
        'periwinkle': (204, 204, 255),
        'mint': (189, 252, 201)
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
            #closest_color = color_name + f" {str(rgb_array)}" # <- if we want RGB names
            closest_color = color_name

    return closest_color

def convert_node_colors_to_names(node_to_color: Dict[int, Tuple[int, int, int]], 
                                show_legend: bool = True,
                                figsize: Tuple[int, int] = (10, 8),
                                save_path: str = None) -> Dict[int, str]:
    """
    Convert a dictionary of node-to-RGB mappings to node-to-color-name mappings.
    Optionally displays a matplotlib legend showing the mappings.
    
    Args:
        node_to_color: Dictionary mapping node IDs to RGB tuples
        show_legend: Whether to display the color legend plot
        figsize: Figure size as (width, height) for the legend
        save_path: Optional path to save the legend figure
        
    Returns:
        Dictionary mapping node IDs to color names
    """
    # Convert colors to names
    node_to_names = {node: rgb_to_color_name(color) for node, color in node_to_color.items()}
    
    # Create legend if requested
    if show_legend:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        num_entries = len(node_to_color)
        
        # Calculate text widths to determine optimal figure size
        sorted_nodes = sorted(node_to_color.keys())
        
        # Create a temporary figure to measure text widths
        temp_fig, temp_ax = plt.subplots(figsize=(1, 1))
        
        max_node_width = 0
        max_color_width = 0
        
        for node in sorted_nodes:
            color_name = node_to_names[node]
            
            # Measure node ID text width
            node_text = temp_ax.text(0, 0, str(node), fontsize=12, fontweight='bold')
            node_bbox = node_text.get_window_extent(renderer=temp_fig.canvas.get_renderer())
            node_width = node_bbox.width
            max_node_width = max(max_node_width, node_width)
            
            # Measure color name text width
            color_text = temp_ax.text(0, 0, color_name.replace('_', ' ').title(), fontsize=11)
            color_bbox = color_text.get_window_extent(renderer=temp_fig.canvas.get_renderer())
            color_width = color_bbox.width
            max_color_width = max(max_color_width, color_width)
        
        plt.close(temp_fig)
        
        # Convert pixel widths to figure units (approximate conversion)
        # This is a rough conversion - matplotlib uses 72 DPI by default
        dpi = 72
        max_node_width_fig = max_node_width / dpi
        max_color_width_fig = max_color_width / dpi
        
        # Calculate optimal figure dimensions
        entry_height = 0.6  # Reduced for tighter spacing
        margin = 0.3
        swatch_width = 0.8
        spacing = 0.2
        
        # Calculate total width needed
        total_width = (margin + max_node_width_fig + spacing + 
                       swatch_width + spacing + max_color_width_fig + margin)
        
        # Ensure minimum width for readability
        total_width = max(total_width, 4.0)
        
        # Calculate total height
        title_height = 0.8
        total_height = num_entries * entry_height + title_height + 2 * margin
        
        # Create the actual figure with calculated dimensions
        fig, ax = plt.subplots(figsize=(total_width, total_height))
        
        # Set axis limits to match our calculated dimensions
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.axis('off')
        
        # Title
        ax.text(total_width/2, total_height - margin - 0.2, 'Color Legend', 
                fontsize=14, fontweight='bold', ha='center', va='top')
        
        # Create legend entries
        for i, node in enumerate(sorted_nodes):
            y_pos = total_height - title_height - margin - (i + 1) * entry_height + entry_height/2
            rgb = node_to_color[node]
            color_name = node_to_names[node]
            
            # Normalize RGB values for matplotlib (0-1 range)
            norm_rgb = tuple(c/255.0 for c in rgb)
            
            # Position calculations
            node_x = margin
            swatch_x = margin + max_node_width_fig + spacing
            color_x = swatch_x + swatch_width + spacing
            
            # Node ID (left-aligned)
            ax.text(node_x, y_pos, str(node), fontsize=12, fontweight='bold', 
                    va='center', ha='left')
            
            # Draw color swatch
            swatch_y = y_pos - entry_height/4
            swatch = Rectangle((swatch_x, swatch_y), swatch_width, entry_height/2, 
                              facecolor=norm_rgb, edgecolor='black', linewidth=1)
            ax.add_patch(swatch)
            
            # Color name
            formatted_name = color_name.replace('_', ' ').title()
            # Truncate very long color names to prevent layout issues
            if len(formatted_name) > 25:
                formatted_name = formatted_name[:22] + "..."
                
            ax.text(color_x, y_pos, formatted_name, 
                    fontsize=11, va='center', ha='left')
        
        # Add a subtle border around the entire legend
        border_margin = 0.1
        border = Rectangle((border_margin, border_margin), 
                          total_width - 2*border_margin, 
                          total_height - 2*border_margin, 
                          fill=False, edgecolor='lightgray', linewidth=1.5)
        ax.add_patch(border)
        
        # Remove any extra whitespace
        plt.tight_layout(pad=0.1)
        
        # Adjust the figure to eliminate whitespace
        ax.margins(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            
        plt.show()
    
    return node_to_names

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

def assign_node_colors(node_list: List[int], labeled_array: np.ndarray,
                       color_mode = 0, custom_map = None,
                       use_previous = False,
                       parent = None) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Color each node individually.

    Thin wrapper: normalises its arguments into the shared contract in
    color_schemes.resolve_palette, then does the LUT work.

    This function takes its instruction as a single integer `color_mode`:

        0 = default          2 = colorblind      4 = use previous
        1 = alt              3 = custom

    Scheme name strings ('Custom', 'Previous', ...) work too. `use_previous`
    is kept as an explicit override for callers that prefer a flag; when set
    it wins over color_mode.

    color_mode 3 TRIGGERS the custom color editor -- the caller does not need
    to build a palette first. Pass custom_map={node_id: '#rrggbb'} to skip the
    prompt and apply a prepared map instead. If the editor cannot be shown
    (headless, no PyQt6) or is cancelled, this falls back to the default
    scheme rather than failing.

    color_mode 4 reuses the last palette applied to NODES -- a separate
    registry slot from communities and identities. Nodes seen before keep
    their exact colors; anything new is colored from the same scheme. With
    nothing recorded yet it falls back to the default scheme.

    Note: these are individual nodes, not communities or identities, so there
    is no outlier group -- node 0 is not special-cased to brown here.
    """
    scheme = (_cschemes.SCHEME_PREVIOUS if use_previous
              else _cschemes.scheme_from_legacy(color_mode))

    node_to_hex = _cschemes.resolve_palette_interactive(
        node_list,
        scheme=scheme,
        background=_COLOR_BACKGROUND,
        custom_map=custom_map,
        parent=parent,
        category_name='Node',
        shuffle='colors',        # historical: shuffle colors, unseeded
        sort_reverse=True,       # historical: nodes sorted descending
        outlier_label=None,      # individual nodes have no outlier group
        domain=_cschemes.DOMAIN_NODES,
    )

    node_to_color = {n: np.array(_cschemes.hex_to_rgba(h), dtype=np.uint8)
                     for n, h in node_to_hex.items()}

    # Create lookup table
    max_label = max(max(labeled_array.flat), max(node_list) if node_list else 0)
    color_lut = np.zeros((int(max_label) + 1, 4), dtype=np.uint8)  # Transparent by default

    for node_id, color in node_to_color.items():
        color_lut[node_id] = color

    # Single vectorized operation - eliminates all loops!
    rgba_array = color_lut[labeled_array]

    # Convert colors for naming
    node_to_color_rgb = {k: tuple(v[:3]) for k, v in node_to_color.items()}
    node_to_color_names = convert_node_colors_to_names(node_to_color_rgb, show_legend = False)

    return rgba_array, node_to_color_names

def _detect_color_domain(values) -> str:
    """
    Work out whether these labels are communities or identities.

    assign_community_colors handles both, and the caller often does not say
    which, so the registry slot has to be inferred from the data:

        list/tuple/set values  -> identities  ({node: ['type A', 'type B']})
        anything else          -> communities ({node: 3})

    Getting this wrong is not a crash, it is worse: identities silently reuse
    the community palette (and vice versa), so 'previous' returns colors from
    the wrong render. Pass domain= explicitly to override.
    """
    for v in values:
        if isinstance(v, (list, tuple, set, frozenset)):
            return _cschemes.DOMAIN_IDENTITIES
    return _cschemes.DOMAIN_COMMUNITIES


def assign_community_colors(community_dict, labeled_array,
                            alt_color_schema = False, color_blind_schema = False,
                            custom = False,
                            use_previous_communities = False,
                            use_previous_identities = False,
                            color_scheme = None, custom_map = None,
                            domain = None, parent = None):
    # --- which registry slot ---
    if domain is not None:
        effective_domain = domain
    elif use_previous_identities:
        effective_domain = _cschemes.DOMAIN_IDENTITIES
    elif use_previous_communities:
        effective_domain = _cschemes.DOMAIN_COMMUNITIES
    else:
        effective_domain = _detect_color_domain(community_dict.values())

    # --- which scheme ---
    if custom:
        scheme = _cschemes.SCHEME_CUSTOM
    elif use_previous_communities or use_previous_identities:
        scheme = _cschemes.SCHEME_PREVIOUS
    elif color_scheme is not None:
        scheme = _cschemes.scheme_from_legacy(color_scheme)
    elif alt_color_schema:
        scheme = _cschemes.SCHEME_ALT
    elif color_blind_schema:
        scheme = _cschemes.SCHEME_COLORBLIND
    else:
        scheme = _cschemes.SCHEME_DEFAULT

    community_to_hex = _cschemes.resolve_palette_interactive(
        community_dict.values(),
        scheme=scheme,
        background=_COLOR_BACKGROUND,
        custom_map=custom_map,
        parent=parent,
        category_name=('Identity'
                       if effective_domain == _cschemes.DOMAIN_IDENTITIES
                       else 'Community'),
        shuffle='seeded',
        outlier_label=0,
        outlier_consumes_slot=False,
        domain=effective_domain,
    )

    community_to_color = {c: np.array(_cschemes.hex_to_rgba(h), dtype=np.uint8)
                          for c, h in community_to_hex.items()}

    # normalize_label collapses an identity list to the single member the
    # palette was keyed by, so the lookup matches instead of missing (a miss
    # here leaves the node fully transparent).
    node_to_color = {}
    for node, comm in community_dict.items():
        key = _cschemes.normalize_label(comm, None)
        if key in community_to_color:
            node_to_color[node] = community_to_color[key]

    max_label = max(max(labeled_array.flat),
                    max(node_to_color.keys()) if node_to_color else 0)
    color_lut = np.zeros((int(max_label) + 1, 4), dtype=np.uint8)
    for node_id, color in node_to_color.items():
        color_lut[node_id] = color

    rgba_array = color_lut[labeled_array]
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
        try:
            unique_communities = sorted(set(community_dict.values()))
        except:
            community_dict = {node: str(comm) for node, comm in community_dict.items()}
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
    
    # Create lookup table
    max_label = max(max(labeled_array.flat), max(node_to_gray.keys()) if node_to_gray else 0)
    gray_lut = np.zeros(int(max_label) + 1, dtype=dtype)
    
    for node_id, gray_val in node_to_gray.items():
        gray_lut[node_id] = gray_val
    
    gray_array = gray_lut[labeled_array]
    
    return gray_array, community_to_gray
    

# New color stuff:

def _hex_to_rgb(h: str) -> RGB:
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
 
 
def _srgb_to_linear(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
 
 
def _linear_to_srgb(c: float) -> int:
    c = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
    return max(0, min(255, round(c * 255)))
 
 
def _rgb_to_oklab(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v) for v in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )
 
 
def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance in OKLab, with lightness weighted down slightly so
    hue/chroma differences count for more (better for categorical legends)."""
    dl = (a[0] - b[0]) * 0.75
    return math.sqrt(dl * dl + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
 
 
def _blend_toward(rgb: RGB, target: RGB, t: float) -> RGB:
    """Blend in linear light, which keeps the result from looking muddy."""
    out = []
    for c, tc in zip(rgb, target):
        v = _srgb_to_linear(c) * (1 - t) + _srgb_to_linear(tc) * t
        out.append(_linear_to_srgb(v))
    return tuple(out)
 
 
def _ensure_lightness(rgb: RGB, min_l: float, toward: RGB = (255, 255, 255)) -> RGB:
    """Lift a color until its OKLab lightness clears min_l."""
    if _rgb_to_oklab(rgb)[0] >= min_l:
        return rgb
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if _rgb_to_oklab(_blend_toward(rgb, toward, mid))[0] < min_l:
            lo = mid
        else:
            hi = mid
    return _blend_toward(rgb, toward, hi)
 
 
def _candidate_grid(step_levels: int = 8) -> List[RGB]:
    levels = [round(i * 255 / (step_levels - 1)) for i in range(step_levels)]
    return [(r, g, b) for r in levels for g in levels for b in levels]
 
 
def generate_distinct_colors_alternative(
    n_colors: int,
    background: RGB = (0, 0, 0),
    min_lightness: float = 0.45,
    min_separation: float = 0.20,
) -> List[RGB]:
    """
    Generate visually distinct RGB colors suitable for plotting on `background`.
 
    Args:
        n_colors:       how many colors to return
        background:     canvas color; nothing returned will sit near it
        min_lightness:  OKLab L floor (0-1). On black, ~0.45 keeps everything
                        readable; raise it for thin lines or small markers.
        min_separation: OKLab distance a generated color must keep from every
                        already-chosen color and from the background.
 
    Returns:
        List of (r, g, b) tuples, 0-255.
    """
    if n_colors <= 0:
        return []
 
    bg_lab = _rgb_to_oklab(background)
    chosen: List[RGB] = []
    chosen_lab: List[Tuple[float, float, float]] = []
 
    def accept(rgb: RGB, threshold: float) -> bool:
        lab = _rgb_to_oklab(rgb)
        if _dist(lab, bg_lab) < min_separation:
            return False
        if any(_dist(lab, c) < threshold for c in chosen_lab):
            return False
        chosen.append(rgb)
        chosen_lab.append(lab)
        return True
 
    # 1. Curated palette first -- these beat anything an algorithm picks.
    #    Their ordering is already tuned for contrast, so only reject true
    #    near-duplicates (which lightness-lifting can occasionally create).
    for h in _BASE_PALETTE_HEX:
        if len(chosen) >= n_colors:
            return chosen
        accept(_ensure_lightness(_hex_to_rgb(h), min_lightness), 0.10)
 
    # 2. Extend by greedy farthest-point sampling over an RGB grid.
    candidates = [
        c for c in _candidate_grid()
        if _rgb_to_oklab(c)[0] >= min_lightness
        and _dist(_rgb_to_oklab(c), bg_lab) >= min_separation
    ]
    cand_lab = [_rgb_to_oklab(c) for c in candidates]
 
    while len(chosen) < n_colors and candidates:
        best_i, best_d = -1, -1.0
        for i, lab in enumerate(cand_lab):
            d = min(_dist(lab, c) for c in chosen_lab) if chosen_lab else _dist(lab, bg_lab)
            if d > best_d:
                best_i, best_d = i, d
        chosen.append(candidates.pop(best_i))
        chosen_lab.append(cand_lab.pop(best_i))
 
    return chosen
 
 
def to_hex(colors: List[RGB]) -> List[str]:
    return ["#%02X%02X%02X" % c for c in colors]



# Color blind stuff
 
def simulate_deuteranopia(rgb: "RGB") -> "RGB":
    """Simulate how an RGB color appears to a deuteranope."""
    lin = [_srgb_to_linear(c) for c in rgb]
    return tuple(
        _linear_to_srgb(max(0.0, min(1.0, sum(_DEUTAN_MATRIX[i][j] * lin[j]
                                              for j in range(3)))))
        for i in range(3)
    )
 
 
def _cb_dist_weighted(a, b, l_weight: float) -> float:
    """Euclidean distance in OKLab with an explicit lightness weight."""
    dl = (a[0] - b[0]) * l_weight
    return math.sqrt(dl * dl + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
 
 
def _cb_labs(rgb: "RGB"):
    """OKLab coords of a color under (normal vision, deuteranopia)."""
    return (_rgb_to_oklab(rgb), _rgb_to_oklab(simulate_deuteranopia(rgb)))
 
 
def _cb_dist(a, b) -> float:
    """Worst-case separation across normal and deuteranopic vision. A pair is
    only 'far apart' if it looks far apart to BOTH viewers."""
    return min(
        _cb_dist_weighted(a[0], b[0], _CB_L_WEIGHT_NORMAL),
        _cb_dist_weighted(a[1], b[1], _CB_L_WEIGHT_DEUTAN),
    )
 
 
def generate_distinct_colors_colorblind(
    n_colors: int,
    background: "RGB" = (0, 0, 0),
    min_lightness: float = 0.45,
    min_separation: float = 0.20,
) -> "List[RGB]":
    """
    Generate distinct RGB colors that stay distinguishable under deuteranopia,
    for plotting on `background`.
 
    Same signature and contract as generate_distinct_colors(), so it can be
    swapped in directly.
 
    Args:
        n_colors:       how many colors to return
        background:     canvas color; nothing returned will sit near it
        min_lightness:  OKLab L floor (0-1). ~0.45 keeps everything readable
                        on black; raise it for thin lines or small markers.
        min_separation: distance every color must keep from the background.
 
    Returns:
        List of (r, g, b) tuples, 0-255. n_colors <= 8 returns Okabe-Ito
        unmodified.
 
    Note:
        A deuteranope perceives a 2D color space rather than 3D, so past
        ~12 colors separation is carried increasingly by lightness alone.
        See max_safe_colors_colorblind() and encode redundantly beyond that
        (marker shape, dash pattern, direct labels).
    """
    if n_colors <= 0:
        return []
 
    bg = _cb_labs(background)
    chosen: "List[RGB]" = []
    chosen_v = []
 
    def accept(rgb) -> bool:
        v = _cb_labs(rgb)
        if _cb_dist(v, bg) < min_separation:
            return False
        if any(_cb_dist(v, c) < CB_CONFUSABLE for c in chosen_v):
            return False
        chosen.append(rgb)
        chosen_v.append(v)
        return True
 
    # 1. Curated CVD-safe palette first -- hand-tuned, beats sampling at low n.
    for h in _CB_PALETTE_HEX:
        if len(chosen) >= n_colors:
            return chosen
        accept(_ensure_lightness(_hex_to_rgb(h), min_lightness))
 
    # 2. Extend by greedy farthest-point sampling under worst-case vision.
    candidates, cand_v = [], []
    for c in _candidate_grid():
        if _rgb_to_oklab(c)[0] < min_lightness:
            continue
        v = _cb_labs(c)
        if _cb_dist(v, bg) < min_separation:
            continue
        candidates.append(c)
        cand_v.append(v)
 
    while len(chosen) < n_colors and candidates:
        best_i, best_d = -1, -1.0
        for i, v in enumerate(cand_v):
            d = min(_cb_dist(v, c) for c in chosen_v) if chosen_v else _cb_dist(v, bg)
            if d > best_d:
                best_i, best_d = i, d
        chosen.append(candidates.pop(best_i))
        chosen_v.append(cand_v.pop(best_i))
 
    return chosen
 
 
def audit_colorblind(colors, threshold: float = CB_CONFUSABLE):
    """
    Check a palette for pairs a deuteranope would confuse.
 
    Returns a list of (distance, hex_a, hex_b), closest first. Empty is good.
    Works on any palette, including hand-picked ones.
    """
    vs = [_cb_labs(c) for c in colors]
    out = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            d = _cb_dist(vs[i], vs[j])
            if d < threshold:
                out.append((round(d, 3),
                            "#%02X%02X%02X" % tuple(colors[i]),
                            "#%02X%02X%02X" % tuple(colors[j])))
    return sorted(out)
 
 
def max_safe_colors_colorblind(
    background: "RGB" = (0, 0, 0),
    min_lightness: float = 0.45,
    min_separation: float = 0.20,
    margin: float = 1.25,
    limit: int = 30,
) -> int:
    """
    Largest n whose worst-case pair still clears CB_CONFUSABLE * margin.
 
    The default margin asks for 25% more headroom than the bare confusability
    floor, since scatter markers and thin lines are read under worse
    conditions than color swatches. Use this to decide when to stop relying
    on color alone.
    """
    colors = generate_distinct_colors_colorblind(limit, background,
                                                 min_lightness, min_separation)
    vs = [_cb_labs(c) for c in colors]
    target = CB_CONFUSABLE * margin
    best = 1
    for n in range(2, len(colors) + 1):
        worst = min(_cb_dist(vs[i], vs[j])
                    for i in range(n) for j in range(i + 1, n))
        if worst < target:
            return best
        best = n
    return best