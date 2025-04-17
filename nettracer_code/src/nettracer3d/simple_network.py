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


def show_community_network(excel_file_path, geometric = False, geo_info = None, directory = None):

    if type(excel_file_path) == str:
        master_list = read_excel_to_lists(excel_file_path)
    else:
        master_list = excel_file_path

    edges = list(zip(master_list[0], master_list[1]))

    # Create a graph
    G = nx.Graph()

    # Add edges from the DataFrame
    G.add_edges_from(edges)

    # Print basic information about the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

    # Calculate the average degree connectivity
    average_degree_connectivity = nx.average_degree_connectivity(G)
    print("Average degree connectivity:", average_degree_connectivity)

    # Calculate the average number of edges attached to a node
    average_edges_per_node = num_nodes/num_edges
    print("Average edges per node:", average_edges_per_node)

    # Calculate and display modularity for each connected component
    connected_components = list(nx.connected_components(G))
    for i, component in enumerate(connected_components):
        subgraph = G.subgraph(component)
        component_communities = list(community.label_propagation_communities(subgraph))
        modularity = community.modularity(subgraph, component_communities)
        print(f"Label propogation modularity for component with {len(component)} nodes: {modularity}")

    # Visualize the graph with nodes colored by community
    pos = nx.spring_layout(G)

    # Detect communities using label propagation
    communities = list(community.label_propagation_communities(G))

    print(f"label prop communities: {communities}")

    # Assign a color to each node based on its community
    node_colors = {}
    for i, community_nodes in enumerate(communities):
        color = mcolors.to_hex(plt.cm.tab10(i / len(communities))[:3])
        for node in community_nodes:
            node_colors[node] = color

    if geometric:
        for node in list(G.nodes()):
            if node not in geo_info[0]:
                G.remove_node(node)
                print(f"Removing node {node} from network visualization (no centroid - likely due to downsampling when finding centroids)")

        pos, z_pos  = geometric_positions(geo_info[0], geo_info[1])
        node_sizes_list = [z_pos[node] for node in G.nodes()]
        node_color_list = [node_colors[node] for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, font_color='black', font_weight='bold', node_size = node_sizes_list, node_color = node_color_list, alpha=0.8, font_size = 12)
    else:
        # Get color list for nodes
        node_color_list = [node_colors[node] for node in G.nodes()]
        # Draw the graph with node colors
        nx.draw(G, pos, with_labels=True, font_color='black', font_weight='bold', node_size=100, node_color = node_color_list, alpha = 0.8)

    if directory is not None:
        plt.savefig(f'{directory}/community_label_propogation_network_plot.png')

    plt.show()


def modularity(G, solo_mod):

    if not solo_mod:

        # Calculate and display modularity for each connected component
        return_dict = {}
        connected_components = list(nx.connected_components(G))
        for i, component in enumerate(connected_components):
            subgraph = G.subgraph(component)
            component_communities = list(community.label_propagation_communities(subgraph))
            modularity = community.modularity(subgraph, component_communities)
            print(f"Label propogation modularity for component with {len(component)} nodes: {modularity}")
            return_dict[len(component)] = modularity

        return return_dict

    else:
        # Step 1: Detect communities using label propagation
        communities = list(community.label_propagation_communities(G))

        # Step 2: Calculate modularity
        modularity = community.modularity(G, communities)

        print(f"Modularity of network is {modularity}")

        return modularity


def show_simple_network(excel_file_path, geometric = False, geo_info = None, directory = None):

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
        nx.draw(G, pos, with_labels=True, font_color='black', font_weight='bold', node_size= node_sizes_list, alpha=0.8, font_size = 12)
    else:
        # Visualize the graph with different edge colors for each community
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_color='red', font_weight='bold', node_size=10)

    if directory is not None:
        plt.savefig(f'{directory}/network_plot.png')

    plt.show()


def show_identity_network(excel_file_path, node_identities, geometric=False, geo_info=None, directory=None):
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

    # Create a more sophisticated color palette using a combination of colormap sequences
    unique_categories = list(set(identity_dict.values()))
    num_categories = len(unique_categories)
    
    # Create a color palette that combines multiple colormaps for more distinct colors
    if num_categories <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_categories))
    elif num_categories <= 20:
        colors1 = plt.cm.tab20(np.linspace(0, 1, min(num_categories, 20)))
        colors = colors1[:num_categories]
    else:
        # For large number of categories, combine multiple distinct colormaps
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors3 = plt.cm.Pastel1(np.linspace(0, 1, 9))
        colors4 = plt.cm.Paired(np.linspace(0, 1, 12))
        
        # Combine and take needed number of colors
        all_colors = np.vstack([colors1, colors2, colors3, colors4])
        # Shuffle the colors to ensure adjacent categories have distinct colors
        np.random.seed(42)  # For consistency
        np.random.shuffle(all_colors)
        colors = all_colors[:num_categories]

    color_map = {category: mcolors.to_hex(color[:3]) 
                 for category, color in zip(unique_categories, colors)}

    # Node size handling
    node_dict = {node: 30 if identity_dict[node] == 'Edge' else 100 
                 for node in G.nodes()}
    
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
    
    # Draw the network
    node_colors = [color_map[identity_dict[node]] for node in G.nodes()]
    nx.draw(G, pos, ax=graph_ax, with_labels=True, font_color='black', 
            font_weight='bold', node_size=node_sizes_list, 
            node_color=node_colors, alpha=0.8, font_size=12)

    # Create custom legend with multiple columns if needed
    legend_handles = [Patch(color=color, label=category) 
                     for category, color in color_map.items()]
    
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

    # Read the Excel file into a pandas DataFrame
    excel_file_path = input("Excel file?: ")

    master_list = read_excel_to_lists(excel_file_path)

    edges = zip(master_list[0], master_list[1])

    #df = pd.read_excel(excel_file_path)

    # Create a graph
    G = nx.Graph()

    # Add edges from the DataFrame
    #edges = df.values  # Assuming the columns are named "Node1" and "Node2"
    G.add_edges_from(edges)

    # Print basic information about the graph
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Calculate the average degree connectivity
    #average_degree_connectivity = nx.average_degree_connectivity(G)
    #print("Average degree connectivity:", average_degree_connectivity)

    # Calculate the average number of edges attached to a node
    #average_edges_per_node = sum(k * v for k, v in average_degree_connectivity.items()) / G.number_of_nodes()
    #print("Average edges per node:", average_edges_per_node)

    # Visualize the graph with different edge colors for each community
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_color='red', font_weight='bold', node_size=10)

    #connected_components = list(nx.connected_components(G))
    #for i, component in enumerate(connected_components):
        #communities = community.label_propagation_communities(G.subgraph(component))
        
        # Assign a different color to each community
        #colors = [mcolors.to_hex(plt.cm.tab10(i / len(connected_components))[:3]) for _ in range(len(component))]
        
        #nx.draw_networkx_edges(G, pos, edgelist=G.subgraph(component).edges(), edge_color=colors)

        #num_nodes = len(component)
        #modularity = community.modularity(G.subgraph(component), communities)
        #print(f"Modularity for component with {num_nodes} nodes:", modularity)

    plt.show()