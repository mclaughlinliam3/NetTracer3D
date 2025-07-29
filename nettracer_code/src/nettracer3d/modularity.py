from networkx.algorithms import community
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from . import network_analysis
from . import simple_network
import numpy as np
import itertools

def open_network(excel_file_path):

    df = pd.read_excel(excel_file_path)

    # Create a graph
    G = nx.Graph()

    # Add edges from the DataFrame
    edges = df.values  # Assuming the columns are named "Node1" and "Node2"
    G.add_edges_from(edges)

    return G

def save_figure(directory_path, identifier):
    file_name = f"Network {identifier}"
    full_path = os.path.join(directory_path, file_name)
    plt.savefig(full_path)

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

def get_edge_weights(G):
    edge_weights = {}
    weight_G = G.edges(data = True)

    for item in weight_G:
        nodea = item[0]
        nodeb = item[1]
        weight_dict = item[2]
        try:
            weight = weight_dict['weight']
        except KeyError:
            weight = 1

        key = (nodea, nodeb)
        edge_weights[key] = weight

    return edge_weights

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



def show_communities_flex(G, master_list, normalized_weights, geo_info=None, geometric=False, directory=None, weighted=True, partition=None, style=0):

    if normalized_weights is None:
        G, edge_weights = network_analysis.weighted_network(master_list)

        if edge_weights is None:
            edge_weights = get_edge_weights(G)
        # Find the maximum and minimum edge weights
        max_weight = max(weight for edge, weight in edge_weights.items())
        min_weight = min(weight for edge, weight in edge_weights.items())

        if max_weight > 1:
            # Normalize edge weights to the range [0.1, 1.0]
            normalized_weights = {edge: 0.1 + 0.9 * ((weight - min_weight) / (max_weight - min_weight)) for edge, weight in edge_weights.items()}
        else:
            normalized_weights = {edge: 0.1 for edge, weight in edge_weights.items()}

        # Add edges to the graph with normalized weights
        for edge, normalized_weight in normalized_weights.items():
            G.add_edge(edge[0], edge[1], weight=normalized_weight)

    if geometric:
        for node in list(G.nodes()):
            if node not in geo_info[0]:
                G.remove_node(node)
                print(f"Removing node {node} from network visualization (no centroid - likely due to downsampling when finding centroids)")

    # Group nodes by their community
    communities = {}
    for node, community_id in partition.items():
        communities.setdefault(community_id, []).append(node)

    # Create a mapping of community IDs to sequential indices
    unique_communities = sorted(set(partition.values()))
    
    # Use the same color generation method as the overlay system
    # Get community sizes for sorting (largest first)
    from collections import Counter
    community_sizes = Counter(partition.values())
    sorted_communities = sorted(unique_communities, key=lambda x: community_sizes[x], reverse=True)
    
    from . import community_extractor

    # Generate distinct colors using the same method as assign_community_colors
    colors_rgb = community_extractor.generate_distinct_colors(len(unique_communities))
    
    # Create community to color mapping (same order as the overlay system)
    community_to_color = {comm: colors_rgb[i] for i, comm in enumerate(sorted_communities)}
    
    # Convert RGB tuples to matplotlib format (0-1 range)
    colors_matplotlib = {}
    for comm, rgb in community_to_color.items():
        colors_matplotlib[comm] = tuple(c/255.0 for c in rgb)

    if weighted:
        G = nx.Graph()

        # Add edges to the graph with normalized weights
        for edge, normalized_weight in normalized_weights.items():
            G.add_edge(edge[0], edge[1], weight=normalized_weight)

        if geometric:
            pos, z_pos = simple_network.geometric_positions(geo_info[0], geo_info[1])
            
            # Draw the nodes, coloring them according to their community
            for community_id, nodes in communities.items():
                node_sizes_list = [z_pos[node] for node in nodes]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=[colors_matplotlib[community_id]], 
                                     node_size=node_sizes_list, alpha=0.8)

            # Draw edges with normalized weights
            for edge in G.edges():
                normalized_weight = G[edge[0]][edge[1]]['weight']
                nx.draw_networkx_edges(G, pos, edgelist=[edge], width=5 * normalized_weight, edge_color='black')

            nx.draw_networkx_labels(G, pos)

        else:
            pos = nx.spring_layout(G)
            
            # Draw the nodes, coloring them according to their community
            for community_id, nodes in communities.items():
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=[colors_matplotlib[community_id]], 
                                     node_size=100, alpha=0.8)

            # Draw edges with normalized weights
            for edge in G.edges():
                normalized_weight = G[edge[0]][edge[1]]['weight']
                nx.draw_networkx_edges(G, pos, edgelist=[edge], width=5 * normalized_weight, edge_color='black')

            nx.draw_networkx_labels(G, pos)

    else:
        # Create node color list based on partition and the same color mapping
        node_colors = [colors_matplotlib[partition[node]] for node in G.nodes()]

        if geometric:
            pos, z_pos = simple_network.geometric_positions(geo_info[0], geo_info[1])
            node_sizes_list = [z_pos[node] for node in G.nodes()]
            nx.draw(G, pos, with_labels=True, font_color='black', font_weight='bold', 
                   node_size=node_sizes_list, node_color=node_colors, alpha=0.8, font_size=12)
        else:
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, font_color='black', font_weight='bold', 
                   node_size=100, node_color=node_colors, alpha=0.8)

    plt.axis('off')
    plt.show()

    if directory is not None:
        plt.savefig(f'{directory}/network_plot.png')

    return partition, normalized_weights



def community_partition(master_list, weighted = False, style = 0, dostats = True, seed = None):

    def calculate_network_stats(G, communities):
        """
        Calculate comprehensive network statistics for the graph and its communities.
        
        Parameters:
        -----------
        G : networkx.Graph
            The input graph
        communities : list
            List of sets/lists containing node ids for each community
        
        Returns:
        --------
        dict
            Dictionary containing various network statistics
        """
        stats = {}

        try:
        
            # Overall network modularity
            stats['Modularity Entire Network'] = community.modularity(G, communities)
        except:
            pass

        try:        
            # Component-level modularity
            connected_components = list(nx.connected_components(G))
            if len(connected_components) > 1:
                for i, component in enumerate(connected_components):
                    subgraph = G.subgraph(component)
                    component_communities = list(community.label_propagation_communities(subgraph))
                    modularity = community.modularity(subgraph, component_communities)
                    num_nodes = len(component)
                    stats[f'Modularity of component with {num_nodes} nodes'] = modularity
        except:
            pass

        try:        
            # Community size statistics
            stats['Number of Communities'] = len(communities)
            community_sizes = [len(com) for com in communities]
            stats['Community Sizes'] = community_sizes
            stats['Average Community Size'] = np.mean(community_sizes)
        except:
            pass

        try:
        
            # Per-community statistics
            for i, com in enumerate(communities):
                subgraph = G.subgraph(com)
                
                # Basic community metrics
                stats[f'Community {i+1} Density'] = nx.density(subgraph)
                stats[f'Community {i+1} Conductance'] = nx.conductance(G, com)
                stats[f'Community {i+1} Avg Clustering'] = nx.average_clustering(subgraph)
                
                # Degree centrality
                degree_cent = nx.degree_centrality(subgraph)
                stats[f'Community {i+1} Avg Degree Centrality'] = np.mean(list(degree_cent.values()))
                
                # Average path length (only for connected subgraphs)
                if nx.is_connected(subgraph):
                    stats[f'Community {i+1} Avg Path Length'] = nx.average_shortest_path_length(subgraph)
        except:
            pass

        try:        
            # Global network metrics
            stats['Global Clustering Coefficient'] = nx.average_clustering(G)
        except:
            pass
        try:
            stats['Assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            pass

        def count_inter_community_edges(G, communities):
            inter_edges = 0
            for com1, com2 in itertools.combinations(communities, 2):
                inter_edges += len(list(nx.edge_boundary(G, com1, com2)))
            return inter_edges

        try:
            stats['Inter-community Edges'] = count_inter_community_edges(G, communities)
        except:
            pass

        # Calculate mixing parameter (ratio of external to total edges for nodes)
        def mixing_parameter(G, communities):
            external_edges = 0
            total_edges = 0
            for com in communities:
                subgraph = G.subgraph(com)
                internal_edges = subgraph.number_of_edges()
                total_com_edges = sum(G.degree(node) for node in com)
                external_edges += total_com_edges - (2 * internal_edges)
                total_edges += total_com_edges
            return external_edges / total_edges

        try:
            stats['Mixing Parameter'] = mixing_parameter(G, communities)
        except:
            pass
        
        return stats

    def calculate_louvain_network_stats(G, partition, seed):
        """
        Calculate comprehensive network statistics for the graph using Louvain community detection.
        
        Parameters:
        -----------
        G : networkx.Graph
            The input graph
        partition : dict
            Dictionary mapping node -> community id from Louvain detection
        
        Returns:
        --------
        dict
            Dictionary containing various network statistics
        """
        stats = {}
        
        
        try:
            # Overall network modularity using Louvain
            stats['Modularity Entire Network'] = community.modularity(G, partition)
        except:
            pass

        try:        
            # Component-level modularity
            connected_components = list(nx.connected_components(G))
            if len(connected_components) > 1:
                for i, component in enumerate(connected_components):
                    subgraph = G.subgraph(component)
                    subgraph_partition = nx.community.louvain_communities(G, weight='weight', seed = seed)
                    modularity = community.modularity(subgraph, subgraph_partition)
                    num_nodes = len(component)
                    stats[f'Modularity of component with {num_nodes} nodes'] = modularity
        except:
            pass

        try:            
            # Community size statistics
            stats['Number of Communities'] = len(communities)
            community_sizes = [len(com) for com in communities]
            stats['Community Sizes'] = community_sizes
            import numpy as np
            stats['Average Community Size'] = np.mean(community_sizes)
        except:
            pass

        #try:        
            # Add some Louvain-specific statistics
            #stats['Partition Resolution'] = 1.0  # Default resolution parameter
        #except:
            #pass
        
        # Global network metrics
        try:
            stats['Global Clustering Coefficient'] = nx.average_clustering(G)
        except:
            pass
        try:
            stats['Assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            pass

        def count_inter_community_edges(G, communities):
            inter_edges = 0
            for com1, com2 in itertools.combinations(communities, 2):
                inter_edges += len(list(nx.edge_boundary(G, com1, com2)))
            return inter_edges

        try:
            stats['Inter-community Edges'] = count_inter_community_edges(G, communities)
        except:
            pass

        # Calculate mixing parameter (ratio of external to total edges for nodes)
        def mixing_parameter(G, communities):
            external_edges = 0
            total_edges = 0
            for com in communities:
                subgraph = G.subgraph(com)
                internal_edges = subgraph.number_of_edges()
                total_com_edges = sum(G.degree(node) for node in com)
                external_edges += total_com_edges - (2 * internal_edges)
                total_edges += total_com_edges
            return external_edges / total_edges

        try:
            stats['Mixing Parameter'] = mixing_parameter(G, communities)
        except:
            pass

        try:        
            # Per-community statistics
            for i, com in enumerate(communities):
                subgraph = G.subgraph(com)
                
                # Basic community metrics
                stats[f'Community {i+1} Density'] = nx.density(subgraph)
                stats[f'Community {i+1} Conductance'] = nx.conductance(G, com)
                stats[f'Community {i+1} Avg Clustering'] = nx.average_clustering(subgraph)
                
                # Degree centrality
                degree_cent = nx.degree_centrality(subgraph)
                import numpy as np
                stats[f'Community {i+1} Avg Degree Centrality'] = np.mean(list(degree_cent.values()))
                
                # Average path length (only for connected subgraphs)
                if nx.is_connected(subgraph):
                    stats[f'Community {i+1} Avg Path Length'] = nx.average_shortest_path_length(subgraph)
        except:
            import traceback
            pass
        
        return stats

    stats = {}

    if weighted:
        G, edge_weights = network_analysis.weighted_network(master_list)
        edge_weights = get_edge_weights(G)

    if style == 1 and weighted:
        G = nx.Graph()

        # Find the maximum and minimum edge weights
        max_weight = max(weight for edge, weight in edge_weights.items())
        min_weight = min(weight for edge, weight in edge_weights.items())

        if max_weight > 1:
            # Normalize edge weights to the range [0.1, 1.0]
            normalized_weights = {edge: 0.1 + 0.9 * ((weight - min_weight) / (max_weight - min_weight)) for edge, weight in edge_weights.items()}
        else:
            normalized_weights = {edge: 0.1 for edge, weight in edge_weights.items()}

        # Add edges to the graph with normalized weights
        for edge, normalized_weight in normalized_weights.items():
            G.add_edge(edge[0], edge[1], weight=normalized_weight)

        # Replace Louvain with NetworkX's implementation
        communities = list(nx.community.louvain_communities(G, weight='weight', seed = seed))
        
        # Convert to the same format as community_louvain.best_partition
        output = {}
        for i, com in enumerate(communities):
            for node in com:
                output[node] = i + 1

        if dostats:
            stats = calculate_louvain_network_stats(G, communities, seed)

        return output, normalized_weights, stats

    elif style == 1:
        edges = list(zip(master_list[0], master_list[1]))
        G = nx.Graph()
        G.add_edges_from(edges)

        # Louvain with NetworkX's implementation
        communities = list(nx.community.louvain_communities(G, seed = seed))

        # Convert to the same format as community_louvain.best_partition
        output = {}
        for i, com in enumerate(communities):
            for node in com:
                output[node] = i + 1

        if dostats:
            stats = calculate_louvain_network_stats(G, communities, seed)

        return output, None, stats

    elif style == 0 and weighted:

        G = nx.Graph()

        # Find the maximum and minimum edge weights
        max_weight = max(weight for edge, weight in edge_weights.items())
        min_weight = min(weight for edge, weight in edge_weights.items())

        if max_weight > 1:
            # Normalize edge weights to the range [0.1, 1.0]
            normalized_weights = {edge: 0.1 + 0.9 * ((weight - min_weight) / (max_weight - min_weight)) for edge, weight in edge_weights.items()}
        else:
            normalized_weights = {edge: 0.1 for edge, weight in edge_weights.items()}

        # Add edges to the graph with normalized weights
        for edge, normalized_weight in normalized_weights.items():
            G.add_edge(edge[0], edge[1], weight=normalized_weight)

        if seed is not None:
            import random
            import numpy as np
            # Set seeds
            random.seed(seed)
            np.random.seed(seed)

        # Detect communities using label propagation
        communities = list(community.label_propagation_communities(G))
        output = {}
        for i, com in enumerate(communities):
            for node in com:
                output[node] = i + 1

        if dostats:

            stats = calculate_network_stats(G, communities)

        return output, normalized_weights, stats

    elif style == 0:


        edges = list(zip(master_list[0], master_list[1]))

        # Create a graph
        G = nx.Graph()

        # Add edges from the DataFrame
        G.add_edges_from(edges)


        # Detect communities using label propagation

        if seed is not None:
            import random
            import numpy as np
            # Set seeds
            random.seed(seed)
            np.random.seed(seed)

        communities = list(community.label_propagation_communities(G))
        output = {}
        for i, com in enumerate(communities):
            for node in com:
                output[node] = i + 1

        if dostats:

            stats = calculate_network_stats(G, communities)

        return output, None, stats


def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except Exception as e:
        pass

def find_threshold(G):
    largest_component = max(nx.connected_components(G), key=len)
    largest_component_subgraph = G.subgraph(largest_component)
    largest_component = float(largest_component_subgraph.number_of_nodes())
    threshold = int(round(0.1 * largest_component))

    return threshold


def remove_small_components(G, threshold):
    """
    Remove connected components with fewer nodes than the threshold from the graph G.

    Parameters:
    - G: A NetworkX graph
    - threshold: An integer, the minimum size required for a component to be retained

    Returns:
    - A modified graph G with small components removed.
    """
    for component in list(nx.connected_components(G)):
        if len(component) < threshold:
            # Remove the nodes of the component from the graph
            G.remove_nodes_from(component)
    return G


if __name__ == "__main__":
    pass