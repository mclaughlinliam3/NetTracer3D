import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
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

def custom_circular_layout(G, num_rings):
    pos = {}
    nodes = list(G.nodes())
    ring_size = len(nodes) // num_rings
    remaining_nodes = len(nodes) % num_rings

    for ring in range(num_rings):
        start = ring * ring_size
        end = start + ring_size
        if ring == num_rings - 1:
            end += remaining_nodes
        
        ring_nodes = nodes[start:end]
        angle_step = 2 * 3.14159 / len(ring_nodes)
        radius = 1 + ring  # Increasing radius for each ring
        
        for i, node in enumerate(ring_nodes):
            angle = i * angle_step
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
    
    return pos

def ball_and_stick_layout(G, edges_list, nodes_list):
    pos = {}
    layers = []

    # Find the edge with the highest degree
    edge_degree = 0
    center_node = None
    for edge in edges_list:
        if G.degree(edge) > edge_degree:
            edge_degree = G.degree(edge)
            center_node = edge

    pos[center_node] = (0, 0)
    layers.append([center_node])
    
    # BFS to assign positions in concentric circles
    visited = set([center_node])
    current_layer = [center_node]
    radius = 1
    
    while current_layer:
        next_layer = []
        angle_step = 2 * np.pi / len(current_layer)
        
        for i, node in enumerate(current_layer):
            angle = i * angle_step
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            pos[node] = (x, y)
            
            neighbors = [n for n in G.neighbors(node) if n not in visited]
            next_layer.extend(neighbors)
            visited.update(neighbors)
        
        current_layer = next_layer
        radius += 1
        if current_layer:
            layers.append(current_layer)
    
    return pos, layers

def geometric_positions(centroids, shape):
    xy_pos = {}
    z_pos = {}
    z_max = shape[0]
    y_max = shape[1]
    shifter = float(shape[2]/(shape[2] * 10))
    for item in centroids:
        centroid = centroids[item]
        new_pos = tuple((centroid[2], y_max - centroid[1]))
        if new_pos in xy_pos.values():
            new_pos = list(new_pos)
            new_pos[0] = new_pos[0] + shifter
            new_pos = tuple(new_pos)

        xy_pos[item] = new_pos
        z_pos[item] = 300 * float(float(z_max - centroid[0])/z_max)

    return xy_pos, z_pos


def show_simple_network(excel_file_path, geometric = False, geo_info = None, directory = None, show_labels = True):

    if type(excel_file_path) == str:
        master_list = read_excel_to_lists(excel_file_path)
    else:
        master_list = excel_file_path

    edges = zip(master_list[0], master_list[1])

    # Create a graph
    G = nx.Graph()

    # Add edges from the DataFrame
    G.add_edges_from(edges)

    if geometric:
        for node in list(G.nodes()):
            if node not in geo_info[0]:
                G.remove_node(node)
                print(f"Removing node {node} from network visualization (no centroid - likely due to downsampling when finding centroids)")

        pos, z_pos  = geometric_positions(geo_info[0], geo_info[1])
        node_sizes_list = [z_pos[node] for node in G.nodes()]
        nx.draw(G, pos, with_labels=show_labels, font_color='black', font_weight='bold', node_size= node_sizes_list, alpha=0.8, font_size = 12)
    else:
        # Visualize the graph with different edge colors for each community
        pos = nx.spring_layout(G, iterations = 15)
        nx.draw(G, pos, with_labels=show_labels, font_color='red', font_weight='bold', node_size=10)

    if directory is not None:
        plt.savefig(f'{directory}/network_plot.png')

    plt.show()


def show_identity_network(excel_file_path, node_identities, geometric=False, geo_info=None, directory=None, show_labels = True):
    if type(node_identities) == str:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(node_identities)
        # Convert the DataFrame to a dictionary
        identity_dict = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0]).to_dict()
    else:
        identity_dict = node_identities

    if type(excel_file_path) == str:
        master_list = read_excel_to_lists(excel_file_path)
    else:
        master_list = excel_file_path

    edges = zip(master_list[0], master_list[1])
    
    # Create a graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # ENHANCED COLOR LOGIC - Generate bright, contrasting colors
    unique_categories = list(set(identity_dict.values()))
    num_categories = len(unique_categories)
    
    def generate_distinct_colors(n):
        """Generate visually distinct, bright colors with maximum contrast"""
        if n <= 12:
            # Use carefully selected high-contrast colors for small sets
            base_colors = [
                '#FF0000',  # Bright Red
                '#0066FF',  # Bright Blue
                '#00CC00',  # Bright Green
                '#FF8800',  # Bright Orange
                '#8800FF',  # Bright Purple
                '#FFFF00',  # Bright Yellow
                '#FF0088',  # Bright Pink
                '#00FFFF',  # Bright Cyan
                '#88FF00',  # Bright Lime
                '#FF4400',  # Red-Orange
                '#0088FF',  # Sky Blue
                '#CC00FF'   # Magenta
            ]
            return base_colors[:n]
        else:
            # For larger sets, use HSV color space for maximum separation
            colors = []
            import colorsys
            for i in range(n):
                hue = (i * 360 / n) % 360
                # Alternate saturation and value to create more distinction
                sat = 0.85 if i % 2 == 0 else 0.95
                val = 0.95 if i % 3 != 0 else 0.85
                
                # Convert HSV to RGB
                rgb = colorsys.hsv_to_rgb(hue/360, sat, val)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
                )
                colors.append(hex_color)
            return colors
    
    # Generate the enhanced color palette
    colors = generate_distinct_colors(num_categories)
    color_map = dict(zip(unique_categories, colors))

    # Node size handling
    node_dict = {}
    for node in G.nodes():
        try: #Perhaps remove this
            if identity_dict[node] == 'Edge':
                node_dict[node] = 10
            else:
                node_dict[node] = 10
        except:
            node_dict[node] = 10

    if geometric:
        # Handle geometric positioning
        for node in list(G.nodes()):
            if node not in geo_info[0]:
                G.remove_node(node)
                print(f"Removing node {node} from network visualization "
                      f"(no centroid - likely due to downsampling when finding centroids)")
        
        pos, z_pos = geometric_positions(geo_info[0], geo_info[1])
        node_sizes_list = [z_pos[node] for node in G.nodes()]
    else:
        pos = nx.spring_layout(G)
        node_sizes_list = [node_dict[node] for node in G.nodes()]

    # Create figure with custom size
    plt.figure(figsize=(12, 8))
    
    # Create separate axes for the graph and legend
    graph_ax = plt.gca()
    
    # Draw the network with enhanced font styling
    misc = False
    node_colors = []
    for node in G.nodes():
        try:
            node_colors.append(color_map[identity_dict[node]])
        except:
            misc = True
            node_colors.append((1, 1, 1))

    #node_colors = [color_map[identity_dict[node]] for node in G.nodes()]
    nx.draw(G, pos, ax=graph_ax, with_labels=show_labels, font_color='black', 
            font_weight='bold', node_size=node_sizes_list, 
            node_color=node_colors, alpha=0.8, font_size=11, font_family='sans-serif')

    # Create custom legend with multiple columns if needed
    legend_handles = [Patch(color=color, label=category) 
                     for category, color in color_map.items()]

    if misc:
        legend_handles.append(Patch(color = (1, 1, 1,), label = 'Unassigned'))
    
    # Adjust number of columns based on number of categories
    if len(unique_categories) > 20:
        ncol = 3
        bbox_to_anchor = (1.2, 1)
    elif len(unique_categories) > 10:
        ncol = 2
        bbox_to_anchor = (1.1, 1)
    else:
        ncol = 1
        bbox_to_anchor = (1.05, 1)

    # Add legend with adjusted parameters
    legend = plt.legend(handles=legend_handles, 
                       bbox_to_anchor=bbox_to_anchor,
                       loc='upper left',
                       title='Categories',
                       ncol=ncol,
                       fontsize='small',
                       title_fontsize='medium')
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    
    # Save if directory provided
    if directory is not None:
        plt.savefig(f'{directory}/identity_network_plot.png',
                    bbox_inches='tight',
                    dpi=300)
    
    plt.show()


if __name__ == "__main__":

    pass