import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Optional
import umap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as mcolors
from collections import Counter
from . import community_extractor
import random
import re
import matplotlib.patches as mpatches




import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def cluster_arrays_dbscan(data_input, seed=42):
    """
    Simple DBSCAN clustering of 1D arrays with sensible defaults.
    
    Parameters:
    -----------
    data_input : dict or List[List[float]]
        Dictionary {key: array} or list of arrays to cluster
    seed : int
        Random seed for reproducibility (used for parameter estimation)
        
    Returns:
    --------
    list: [[key1, key2], [key3, key4, key5]] - List of clusters, each containing keys/indices
          Note: Outliers are excluded from the output
    """
    
    # Handle both dict and list inputs
    if isinstance(data_input, dict):
        keys = list(data_input.keys())
        array_values = list(data_input.values())
    else:
        keys = list(range(len(data_input)))
        array_values = data_input
    
    # Convert to numpy
    data = np.array(array_values)
    n_samples = len(data)
    
    # Simple heuristics for DBSCAN parameters
    min_samples = max(3, int(np.sqrt(n_samples) * 0.2))  # Roughly sqrt(n)/5, minimum 3
    
    # Estimate eps using 4th nearest neighbor distance (common heuristic)
    k = min(4, n_samples - 1)
    if k > 0:
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(data)
        distances, _ = nbrs.kneighbors(data)
        # Use 80th percentile of k-nearest distances as eps
        eps = np.percentile(distances[:, k], 80)
    else:
        eps = 0.1  # fallback
    
    print(f"Using DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Organize results into clusters (excluding outliers)
    clusters = []
    
    # Add only the main clusters (non-noise points)
    unique_labels = np.unique(labels)
    main_clusters = [label for label in unique_labels if label != -1]
    
    for label in main_clusters:
        cluster_indices = np.where(labels == label)[0]
        cluster_keys = [keys[i] for i in cluster_indices]
        clusters.append(cluster_keys)
    
    n_main_clusters = len(main_clusters)
    n_outliers = np.sum(labels == -1)
    
    print(f"Found {n_main_clusters} main clusters and {n_outliers} outliers (go in neighborhood 0)")
    
    return clusters

def cluster_arrays(data_input, n_clusters=None, seed=42):
    """
    Simple clustering of 1D arrays with key tracking and automatic cluster count detection.
    
    Parameters:
    -----------
    data_input : dict or List[List[float]]
        Dictionary {key: array} or list of arrays to cluster
    n_clusters : int or None
        How many groups you want. If None, will automatically determine optimal k
    seed : int
        Random seed for reproducibility
    max_k : int
        Maximum number of clusters to consider when auto-detecting (default: 10)
        
    Returns:
    --------
    list: [[key1, key2], [key3, key4, key5]] - List of clusters, each containing keys/indices
    """
    
    # Handle both dict and list inputs
    if isinstance(data_input, dict):
        keys = list(data_input.keys())
        array_values = list(data_input.values())
    else:
        keys = list(range(len(data_input)))
        array_values = data_input
    
    # Convert to numpy
    data = np.array(array_values)
    
    # Auto-detect optimal number of clusters if not specified
    if n_clusters is None:
        n_clusters = _find_optimal_clusters(data, seed, max_k = (len(keys) // 3))
        print(f"Auto-detected optimal number of clusters: {n_clusters}")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(data)
    
    # Organize results into clusters - simple list of lists with keys only
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(keys[i])
    
    return clusters

def _find_optimal_clusters(data, seed, max_k):
    """
    Find optimal number of clusters using Calinski-Harabasz index.
    """
    n_samples = len(data)
    
    # Need at least 2 samples to cluster
    if n_samples < 2:
        return 1
    
    # Limit max_k to reasonable bounds
    max_k = min(max_k, n_samples - 1, 20)
    print(f"Max_k: {max_k}, n_samples: {n_samples}")
    
    if max_k < 2:
        return 1
    
    # Use Calinski-Harabasz index to find optimal k
    ch_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        try:
            print(f"Testing {k} clusters")
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(data)
            
            # Check if we got the expected number of clusters
            if len(np.unique(labels)) == k:
                score = calinski_harabasz_score(data, labels)
                ch_scores.append(score)
            else:
                ch_scores.append(0)  # Penalize solutions that didn't achieve k clusters
                
        except Exception:
            ch_scores.append(0)
    
    # Find k with highest Calinski-Harabasz score
    if ch_scores and max(ch_scores) > 0:
        optimal_k = k_range[np.argmax(ch_scores)]
        print(f"Using {optimal_k} neighborhoods")
        return optimal_k
    
def plot_dict_heatmap(unsorted_data_dict, id_set, figsize=(12, 8), title="Neighborhood Heatmap", 
                     center_at_one=False, center_at_zero=False, sublabel = "Community"):
    """
    Create a heatmap from a dictionary of numpy arrays.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are identifiers and values are 1D numpy arrays of floats (0-1)
    id_set : list or set
        List or set of strings describing what each index in the numpy arrays represents
    figsize : tuple, optional
        Figure size (width, height)
    title : str, optional
        Title for the heatmap
    center_at_one : bool, optional
        If True, uses a diverging colormap centered at 1 with nonlinear scaling:
        - 0 to 1: blue to white (underrepresentation to normal)
        - 1+: white to red (overrepresentation)
        If False (default), uses standard white-to-red scaling from 0 to 1
    center_at_zero : bool, optional
        If True, uses a diverging colormap centered at 0 with symmetric scaling:
        - Negative values: blue (below zero)
        - 0: white (center)
        - Positive values: red (above zero)
        Takes priority over center_at_one if both are True
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    data_dict = {k: unsorted_data_dict[k] for k in sorted(unsorted_data_dict.keys())}
    # Convert dict to 2D array for heatmap
    keys = list(data_dict.keys())
    data_matrix = np.array([data_dict[key] for key in keys])

    # Convert id_set to sorted list if it's a set or unsorted list
    if isinstance(id_set, set):
        sorted_id_set = sorted(list(id_set))
    else:
        sorted_id_set = sorted(id_set)
    
    # Create mapping from original to sorted order
    original_id_list = list(id_set) if isinstance(id_set, set) else id_set
    sorted_indices = [original_id_list.index(id_val) for id_val in sorted_id_set]
    
    # Reorder data columns to match sorted id_set
    data_matrix = data_matrix[:, sorted_indices]

    # Move key 0 to the bottom if it exists as the first key
    if keys and keys[0] == 0:
        keys.append(keys.pop(0))
        data_matrix = np.vstack([data_matrix[1:], data_matrix[0:1]])

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if center_at_zero:
        # Custom colormap and scaling for center_at_zero mode
        # Find the actual data range
        data_min = np.min(data_matrix)
        data_max = np.max(data_matrix)
        
        # Create a custom colormap: blue -> white -> red
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
                 '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
        
        # Create symmetric nonlinear transformation
        def transform_data(data):
            transformed = np.zeros_like(data)
            
            # Get max absolute value for symmetric scaling
            max_abs = max(abs(data_min), abs(data_max))
            
            if max_abs == 0:
                return transformed + 0.5
            
            # For negative values: map to [0, 0.5)
            mask_neg = data < 0
            if np.any(mask_neg):
                # Use sqrt of absolute value for more resolution near zero
                transformed[mask_neg] = 0.5 * (1 - np.sqrt(np.abs(data[mask_neg])) / np.sqrt(max_abs))
            
            # For positive values: map to (0.5, 1.0]
            mask_pos = data > 0
            if np.any(mask_pos):
                transformed[mask_pos] = 0.5 + 0.5 * np.sqrt(data[mask_pos]) / np.sqrt(max_abs)
            
            # Zero maps to 0.5 (center)
            mask_zero = data == 0
            transformed[mask_zero] = 0.5
            
            return transformed
        
        # Transform the data for visualization
        transformed_matrix = transform_data(data_matrix)
        
        # Create heatmap with custom colormap
        im = ax.imshow(transformed_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Create custom colorbar with original values
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Set colorbar ticks to show meaningful values
        # Create symmetric ticks around zero
        max_abs = max(abs(data_min), abs(data_max))
        tick_values = []
        
        if max_abs <= 1:
            step_values = [0, 0.25, 0.5, 0.75, 1.0]
        elif max_abs <= 2:
            step_values = [0, 0.5, 1.0, 1.5, 2.0]
        else:
            step_values = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Add negative and positive versions of each step
        for val in step_values:
            if -val >= data_min and val != 0:
                tick_values.append(-val)
        tick_values.append(0)
        for val in step_values:
            if val <= data_max and val != 0:
                tick_values.append(val)
        
        tick_values = sorted(set(tick_values))
        
        # Transform tick values for colorbar positioning
        transformed_ticks = transform_data(np.array(tick_values))
        cbar.set_ticks(transformed_ticks)
        cbar.set_ticklabels([f'{v:.2f}' for v in tick_values])
        cbar.ax.set_ylabel('Value (centered at 0)', rotation=-90, va="bottom")
        
    elif center_at_one:
        # Custom colormap and scaling for center_at_one mode
        # Find the actual data range
        data_min = np.min(data_matrix)
        data_max = np.max(data_matrix)
        
        # Create a custom colormap: blue -> white -> red
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
                 '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
        
        # Create nonlinear transformation
        # Map 0->1 with more resolution, 1+ with less resolution
        def transform_data(data):
            transformed = np.zeros_like(data)
            
            # For values 0 to 1: use square root for faster approach to middle
            mask_low = data <= 1
            transformed[mask_low] = 0.5 * np.sqrt(data[mask_low])
            
            # For values > 1: use slower logarithmic scaling
            mask_high = data > 1
            if np.any(mask_high):
                # Scale from 0.5 to 1.0 based on log of excess above 1
                max_excess = np.max(data[mask_high] - 1) if np.any(mask_high) else 0
                if max_excess > 0:
                    excess_normalized = np.log1p(data[mask_high] - 1) / np.log1p(max_excess)
                    transformed[mask_high] = 0.5 + 0.5 * excess_normalized
                else:
                    transformed[mask_high] = 0.5
            
            return transformed
        
        # Transform the data for visualization
        transformed_matrix = transform_data(data_matrix)
        
        # Create heatmap with custom colormap
        im = ax.imshow(transformed_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Create custom colorbar with original values
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Set colorbar ticks to show meaningful values
        if data_max > 1:
            tick_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            tick_values = [v for v in tick_values if data_min <= v <= data_max]
        else:
            tick_values = [0, 0.25, 0.5, 0.75, 1.0]
            tick_values = [v for v in tick_values if data_min <= v <= data_max]
        
        # Transform tick values for colorbar positioning
        transformed_ticks = transform_data(np.array(tick_values))
        cbar.set_ticks(transformed_ticks)
        cbar.set_ticklabels([f'{v:.2f}' for v in tick_values])
        cbar.ax.set_ylabel('Representation Ratio', rotation=-90, va="bottom")
        
    else:
        # Default behavior: white-to-red colormap
        im = ax.imshow(data_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
        
        # Add standard colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Intensity', rotation=-90, va="bottom")
    
    # Set ticks and labels (use sorted_id_set)
    ax.set_xticks(np.arange(len(sorted_id_set)))
    ax.set_yticks(np.arange(len(keys)))
    ax.set_xticklabels(sorted_id_set)
    labels = list(keys)
    if labels and labels[-1] == 0:
        labels[-1] = 'Excluded (0)'
    ax.set_yticklabels(labels)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations only if id_set is not too large
    if len(sorted_id_set) <= 20:
        for i in range(len(keys)):
            for j in range(len(sorted_id_set)):
                # Use original data values for annotations
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)

    ret_dict = {}

    for i, row in enumerate(data_matrix):
        ret_dict[keys[i]] = row
    
    # Set labels and title
    if center_at_zero:
        ax.set_xlabel('Value Relative to Zero')
    elif center_at_one:
        ax.set_xlabel('Representation Factor of Node Type')
    else:
        ax.set_xlabel('Proportion of Node Type')

    ax.set_ylabel(f'{sublabel}')
    ax.set_title(title)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    return ret_dict, sorted_id_set
    


def generate_distinct_colors(n_colors: int) -> List[str]:
    """
    Generate visually distinct colors using matplotlib's tab and Set colormaps.
    Falls back to HSV generation for large numbers of colors.
    
    Args:
        n_colors: Number of distinct colors needed
    
    Returns:
        List of color strings compatible with matplotlib
    """
    if n_colors <= 10:
        # Use tab10 colormap for up to 10 colors
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_colors, 10)))
        return [mcolors.rgb2hex(color) for color in colors[:n_colors]]
    elif n_colors <= 20:
        # Use tab20 for up to 20 colors
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
        return [mcolors.rgb2hex(color) for color in colors[:n_colors]]
    else:
        # For larger numbers, use HSV space
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            color = mcolors.hsv_to_rgb([hue, 0.8, 0.9])  # High saturation and value
            colors.append(mcolors.rgb2hex(color))
        return colors


def visualize_cluster_composition_umap(cluster_data: Dict[int, np.ndarray], 
                                     class_names: Set[str],
                                     label: bool = False,
                                     n_components: int = 2,
                                     random_state: int = 42,
                                     id_dictionary: Optional[Dict[int, str]] = None, 
                                     graph_label = "Community ID",
                                     title = 'UMAP Visualization of Community Compositions',
                                     neighborhoods: Optional[Dict[int, int]] = None,
                                     original_communities = None,
                                     subname = 'Supercommunity'):
    """
    Convert cluster composition data to UMAP visualization.
    
    Parameters:
    -----------
    cluster_data : dict
        Dictionary where keys are cluster IDs (int) and values are 1D numpy arrays
        representing the composition of each cluster
    class_names : set
        Set of strings representing the class names (order corresponds to array indices)
    label : bool
        Whether to show cluster ID labels on the plot
    n_components : int
        Number of UMAP components (default: 2 for 2D visualization)
    random_state : int
        Random state for reproducibility
    id_dictionary : dict, optional
        Dictionary mapping cluster IDs to identity names. If provided, colors will be
        assigned by identity rather than cluster ID, and a legend will be shown.
    neighborhoods : dict, optional
        Dictionary mapping node IDs to neighborhood IDs {node_id: neighborhood_id}.
        If provided, points will be colored by neighborhood using community coloration methods.
    
    Returns:
    --------
    embedding : numpy.ndarray
        UMAP embedding of the cluster compositions
    """
    
    # Convert set to sorted list for consistent ordering
    try:
        class_labels = sorted(list(class_names))
    except:
        class_labels = None
    
    # Extract cluster IDs and compositions
    cluster_ids = list(cluster_data.keys())
    compositions = np.array([cluster_data[cluster_id] for cluster_id in cluster_ids])
    
    # Create UMAP reducer
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    
    # Fit and transform the composition data
    embedding = reducer.fit_transform(compositions)
    
    # Determine coloring scheme based on parameters
    if neighborhoods is not None and original_communities is not None:
        # Use neighborhood coloring - import the community extractor methods
        from . import community_extractor
        from collections import Counter
        
        # Use original_communities (which is {node: neighborhood}) for color generation
        # This ensures we use the proper node counts for sorting
        
        # Separate outliers (neighborhood 0) from regular neighborhoods in ORIGINAL structure
        outlier_neighborhoods = {node: neighborhood for node, neighborhood in original_communities.items() if neighborhood == 0}
        non_outlier_neighborhoods = {node: neighborhood for node, neighborhood in original_communities.items() if neighborhood != 0}
        
        # Get neighborhoods excluding outliers
        unique_neighborhoods = sorted(set(non_outlier_neighborhoods.values())) if non_outlier_neighborhoods else list()
        
        # Generate colors for non-outlier neighborhoods only (same as assign_community_colors)
        colors = community_extractor.generate_distinct_colors(len(unique_neighborhoods)) if unique_neighborhoods else []
        
        # Sort neighborhoods by size for consistent color assignment (same logic as assign_community_colors)
        # Use the ORIGINAL node counts from original_communities
        if non_outlier_neighborhoods:
            neighborhood_sizes = Counter(non_outlier_neighborhoods.values())
            sorted_neighborhoods = random.Random(42).sample(list(unique_neighborhoods), len(unique_neighborhoods))
            neighborhood_to_color = {neighborhood: colors[i] for i, neighborhood in enumerate(sorted_neighborhoods)}
        else:
            neighborhood_to_color = {}
        
        # Add brown color for outliers (neighborhood 0) - same as assign_community_colors
        if outlier_neighborhoods:
            neighborhood_to_color[0] = (139, 69, 19)  # Brown color (RGB, not RGBA here)
        
        # Map each cluster to its neighborhood color using 'neighborhoods' ({community: neighborhood}) for assignment
        point_colors = []
        neighborhood_labels = []
        for cluster_id in cluster_ids:
            if cluster_id in neighborhoods:
                neighborhood_id = neighborhoods[cluster_id]  # This is {community: neighborhood}
                if neighborhood_id in neighborhood_to_color:
                    point_colors.append(neighborhood_to_color[neighborhood_id])
                    neighborhood_labels.append(neighborhood_id)
                else:
                    # Default color for neighborhoods not found
                    point_colors.append((128, 128, 128))  # Gray
                    neighborhood_labels.append("Unknown")
            else:
                # Default color for nodes not in any neighborhood
                point_colors.append((128, 128, 128))  # Gray
                neighborhood_labels.append("Unknown")
        
        # Normalize RGB values for matplotlib (0-1 range)
        point_colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in point_colors]
        
        # Get unique neighborhoods for legend
        unique_neighborhoods_for_legend = sorted(list(set(neighborhood_to_color.keys())))
        
        use_neighborhood_coloring = True
        
    elif id_dictionary is not None:
        # Use identity coloring (existing logic)
        identities = [id_dictionary.get(cluster_id, "Unknown") for cluster_id in cluster_ids]
        unique_identities = sorted(list(set(identities)))
        colors = generate_distinct_colors(len(unique_identities))
        identity_to_color = {identity: colors[i] for i, identity in enumerate(unique_identities)}
        point_colors = [identity_to_color[identity] for identity in identities]
        use_identity_coloring = True
        use_neighborhood_coloring = False
    else:
        # Use default cluster ID coloring
        point_colors = cluster_ids
        use_identity_coloring = False
        use_neighborhood_coloring = False
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    if n_components == 2:
        if use_neighborhood_coloring:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=point_colors, s=100, alpha=0.7)
        elif use_identity_coloring:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=point_colors, s=100, alpha=0.7)
        else:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=point_colors, cmap='viridis', s=100, alpha=0.7)
        
        if label:
            # Add cluster ID labels
            for i, cluster_id in enumerate(cluster_ids):
                display_label = f'{cluster_id}'
                if use_neighborhood_coloring and cluster_id in neighborhoods:
                    neighborhood_id = neighborhoods[cluster_id]
                    display_label = f'{cluster_id}\n(N{neighborhood_id})'
                elif id_dictionary is not None:
                    identity = id_dictionary.get(cluster_id, "Unknown")
                    display_label = f'{cluster_id}\n({identity})'
                
                plt.annotate(display_label, 
                            (embedding[i, 0], embedding[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, alpha=0.8, ha='left')
        
        # Add appropriate legend/colorbar
        if use_neighborhood_coloring:
            # Create custom legend for neighborhoods
            legend_elements = []
            for neighborhood_id in unique_neighborhoods_for_legend:
                color = neighborhood_to_color[neighborhood_id]
                norm_color = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=norm_color, 
                              markersize=10, label=f'{subname} {neighborhood_id}')
                )
            plt.legend(handles=legend_elements, title=f'{subname}', 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        elif use_identity_coloring:
            # Create custom legend for identities
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=identity_to_color[identity], 
                                        markersize=10, label=identity)
                             for identity in unique_identities]
            plt.legend(handles=legend_elements, title='Identity', 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.colorbar(scatter, label=graph_label)
        
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        if use_neighborhood_coloring:
            title += f' (Colored by {subname})'
        elif use_identity_coloring:
            title += ' (Colored by Identity)'
        plt.title(title)
        
    elif n_components == 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if use_neighborhood_coloring:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=point_colors, s=100, alpha=0.7)
        elif use_identity_coloring:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=point_colors, s=100, alpha=0.7)
        else:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=point_colors, cmap='viridis', s=100, alpha=0.7)
        
        if label:
            # Add cluster ID labels
            for i, cluster_id in enumerate(cluster_ids):
                display_label = f'C{cluster_id}'
                if use_neighborhood_coloring and cluster_id in neighborhoods:
                    neighborhood_id = neighborhoods[cluster_id]
                    display_label = f'C{cluster_id}\n(N{neighborhood_id})'
                elif id_dictionary is not None:
                    identity = id_dictionary.get(cluster_id, "Unknown")
                    display_label = f'C{cluster_id}\n({identity})'
                
                ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2],
                       display_label, fontsize=8)
        
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        title = '3D UMAP Visualization of Cluster Compositions'
        if use_neighborhood_coloring:
            title += ' (Colored by Neighborhood)'
        elif use_identity_coloring:
            title += ' (Colored by Identity)'
        ax.set_title(title)
        
        # Add appropriate legend/colorbar
        if use_neighborhood_coloring:
            # Create custom legend for neighborhoods
            legend_elements = []
            for neighborhood_id in unique_neighborhoods_for_legend:
                color = neighborhood_to_color[neighborhood_id]
                norm_color = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=norm_color, 
                              markersize=10, label=f'Neighborhood {neighborhood_id}')
                )
            ax.legend(handles=legend_elements, title='Neighborhoods', 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
        elif use_identity_coloring:
            # Create custom legend for identities
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=identity_to_color[identity], 
                                        markersize=10, label=identity)
                             for identity in unique_identities]
            ax.legend(handles=legend_elements, title='Identity', 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.colorbar(scatter, label='Cluster ID')
    
    plt.tight_layout()
    plt.show()
    
    if class_labels is not None:
        # Print composition details
        print("Cluster Compositions:")
        print(f"Classes: {class_labels}")
        for i, cluster_id in enumerate(cluster_ids):
            composition = compositions[i]
            additional_info = ""
            if use_neighborhood_coloring and cluster_id in neighborhoods:
                neighborhood_id = neighborhoods[cluster_id]
                additional_info = f" (Neighborhood: {neighborhood_id})"
            elif id_dictionary is not None:
                identity = id_dictionary.get(cluster_id, "Unknown")
                additional_info = f" (Identity: {identity})"
            
            print(f"Cluster {cluster_id}{additional_info}: {composition}")
            # Show which classes dominate this cluster
            dominant_indices = np.argsort(composition)[::-1][:2]  # Top 2
            dominant_classes = [class_labels[idx] for idx in dominant_indices]
            dominant_values = [composition[idx] for idx in dominant_indices]
            print(f"  Dominant: {dominant_classes[0]} ({dominant_values[0]:.3f}), {dominant_classes[1]} ({dominant_values[1]:.3f})")
    
    return embedding

def create_community_heatmap(community_intensity, node_community, node_centroids, shape=None, is_3d=True, 
                           labeled_array=None, figsize=(12, 8), point_size=50, alpha=0.7, 
                           colorbar_label="Community Intensity", title="Community Intensity Heatmap"):
    """
    Create a 2D or 3D heatmap showing nodes colored by their community intensities.
    Can return either matplotlib plot or numpy RGB array for overlay purposes.
    
    Parameters:
    -----------
    community_intensity : dict
        Dictionary mapping community IDs to intensity values
        Keys can be np.int64 or regular ints
        
    node_community : dict
        Dictionary mapping node IDs to community IDs
        
    node_centroids : dict
        Dictionary mapping node IDs to centroids
        Centroids should be [Z, Y, X] for 3D or [1, Y, X] for pseudo-3D
        
    shape : tuple, optional
        Shape of the output array in [Z, Y, X] format
        If None, will be inferred from node_centroids
        
    is_3d : bool, default=True
        If True, create 3D plot/array. If False, create 2D plot/array.
        
    labeled_array : np.ndarray, optional
        If provided, returns numpy RGB array overlay using this labeled array template
        instead of matplotlib plot. Uses lookup table approach for efficiency.
        
    figsize : tuple, default=(12, 8)
        Figure size (width, height) - only used for matplotlib
        
    point_size : int, default=50
        Size of scatter plot points - only used for matplotlib
        
    alpha : float, default=0.7
        Transparency of points (0-1) - only used for matplotlib
        
    colorbar_label : str, default="Community Intensity"
        Label for the colorbar - only used for matplotlib
        
    title : str, default="Community Intensity Heatmap"
        Title for the plot
        
    Returns:
    --------
    If labeled_array is None: fig, ax (matplotlib figure and axis objects)
    If labeled_array is provided: np.ndarray (RGB heatmap array with community intensity colors)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert numpy int64 keys to regular ints for consistency
    community_intensity_clean = {}
    for k, v in community_intensity.items():
        if hasattr(k, 'item'):  # numpy scalar
            community_intensity_clean[k.item()] = v
        else:
            community_intensity_clean[k] = v
    
    # Prepare data for plotting
    node_positions = []
    node_intensities = []
    
    for node_id, centroid in node_centroids.items():
        try:
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
                
            # Get community for this node
            community_id = node_community[node_id]
            
            # Convert community_id to regular int if it's numpy
            if hasattr(community_id, 'item'):
                community_id = community_id.item()
                
            # Get intensity for this community
            intensity = community_intensity_clean[community_id]
            
            node_positions.append(centroid)
            node_intensities.append(intensity)
        except KeyError:
            # Skip nodes that don't have community assignments or community intensities
            pass
    
    # Convert to numpy arrays
    positions = np.array(node_positions)
    intensities = np.array(node_intensities)
    
    # Determine shape if not provided
    if shape is None:
        if len(positions) > 0:
            max_coords = np.max(positions, axis=0).astype(int)
            shape = tuple(max_coords + 1)
        else:
            shape = (100, 100, 100) if is_3d else (1, 100, 100)
    
    # Determine min and max intensities for scaling
    if len(intensities) > 0:
        min_intensity = np.min(intensities)
        max_intensity = np.max(intensities)
    else:
        min_intensity, max_intensity = 0, 1
    
    if labeled_array is not None:
        # Create numpy RGB array output using labeled array and lookup table approach
        
        # Create mapping from node ID to community intensity value
        node_to_community_intensity = {}
        for node_id, centroid in node_centroids.items():
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
            
            try:
                # Get community for this node
                community_id = node_community[node_id]
                
                # Convert community_id to regular int if it's numpy
                if hasattr(community_id, 'item'):
                    community_id = community_id.item()
                    
                # Get intensity for this community
                if community_id in community_intensity_clean:
                    node_to_community_intensity[node_id] = community_intensity_clean[community_id]
            except KeyError:
                # Skip nodes that don't have community assignments
                pass
        
        # Create colormap function (RdBu_r - red for high, blue for low, yellow/white for middle)
        def intensity_to_rgb(intensity, min_val, max_val):
            """Convert intensity value to RGB using RdBu_r colormap logic, centered at 0"""
            
            # Handle edge case where all values are the same
            if max_val == min_val:
                if intensity == 0:
                    return np.array([255, 255, 255], dtype=np.uint8)  # White for 0
                elif intensity > 0:
                    return np.array([255, 200, 200], dtype=np.uint8)  # Light red for positive
                else:
                    return np.array([200, 200, 255], dtype=np.uint8)  # Light blue for negative
            
            # Find the maximum absolute value for symmetric scaling around 0
            max_abs = max(abs(min_val), abs(max_val))
            
            # If max_abs is 0, everything is 0, so return white
            if max_abs == 0:
                return np.array([255, 255, 255], dtype=np.uint8)  # White
            
            # Normalize intensity to -1 to 1 range, centered at 0
            normalized = intensity / max_abs
            normalized = np.clip(normalized, -1, 1)
            
            if normalized > 0:
                # Positive values: white to red (intensity 0 = white, max positive = red)
                r = 255
                g = int(255 * (1 - normalized))
                b = int(255 * (1 - normalized))
            elif normalized < 0:
                # Negative values: white to blue (intensity 0 = white, max negative = blue)  
                r = int(255 * (1 + normalized))
                g = int(255 * (1 + normalized))
                b = 255
            else:
                # Exactly 0: white
                r, g, b = 255, 255, 255
            
            return np.array([r, g, b], dtype=np.uint8)
        
        # Create lookup table for RGB colors
        max_label = max(max(labeled_array.flat), max(node_to_community_intensity.keys()) if node_to_community_intensity else 0)
        color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)  # Default to black (0,0,0)
        
        # Fill lookup table with RGB colors based on community intensity
        for node_id, intensity in node_to_community_intensity.items():
            rgb_color = intensity_to_rgb(intensity, min_intensity, max_intensity)
            color_lut[int(node_id)] = rgb_color
        
        # Apply lookup table to labeled array - single vectorized operation
        if is_3d:
            # Return full 3D RGB array [Z, Y, X, 3]
            heatmap_array = color_lut[labeled_array]
        else:
            # Return 2D RGB array
            if labeled_array.ndim == 3:
                # Take middle slice for 2D representation
                middle_slice = labeled_array.shape[0] // 2
                heatmap_array = color_lut[labeled_array[middle_slice]]
            else:
                # Already 2D
                heatmap_array = color_lut[labeled_array]
        
        return heatmap_array
    
    else:
        # Create matplotlib plot
        fig = plt.figure(figsize=figsize)
        
        if is_3d:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract coordinates (assuming [Z, Y, X] format)
            z_coords = positions[:, 0]
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title}')
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[0])
            
        else:
            # 2D plot (using Y, X coordinates, ignoring Z/first dimension)
            ax = fig.add_subplot(111)
            
            # Extract Y, X coordinates
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'{title}')
            ax.grid(True, alpha=0.3)
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            
            # Set origin to top-left (invert Y-axis)
            ax.invert_yaxis()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label)
        
        # Add text annotations for min/max values
        cbar.ax.text(1.05, 0, f'Min: {min_intensity:.3f}\n(Blue)', 
                    transform=cbar.ax.transAxes, va='bottom')
        cbar.ax.text(1.05, 1, f'Max: {max_intensity:.3f}\n(Red)', 
                    transform=cbar.ax.transAxes, va='top')
        
        plt.tight_layout()
        plt.show()


def create_node_heatmap(node_intensity, node_centroids, shape=None, is_3d=True, 
                        labeled_array=None, figsize=(12, 8), point_size=50, alpha=0.7, 
                        colorbar_label="Node Intensity", title="Node Clustering Intensity Heatmap"):
    """
    Create a 2D or 3D heatmap showing nodes colored by their individual intensities.
    Can return either matplotlib plot or numpy array for overlay purposes.
    
    Parameters:
    -----------
    node_intensity : dict
        Dictionary mapping node IDs to intensity values
        Keys can be np.int64 or regular ints
        
    node_centroids : dict
        Dictionary mapping node IDs to centroids
        Centroids should be [Z, Y, X] for 3D or [1, Y, X] for pseudo-3D
        
    shape : tuple, optional
        Shape of the output array in [Z, Y, X] format
        If None, will be inferred from node_centroids
        
    is_3d : bool, default=True
        If True, create 3D plot/array. If False, create 2D plot/array.
        
    labeled_array : np.ndarray, optional
        If provided, returns numpy array overlay using this labeled array template
        instead of matplotlib plot. Uses lookup table approach for efficiency.
        
    figsize : tuple, default=(12, 8)
        Figure size (width, height) - only used for matplotlib
        
    point_size : int, default=50
        Size of scatter plot points - only used for matplotlib
        
    alpha : float, default=0.7
        Transparency of points (0-1) - only used for matplotlib
        
    colorbar_label : str, default="Node Intensity"
        Label for the colorbar - only used for matplotlib
        
    Returns:
    --------
    If labeled_array is None: fig, ax (matplotlib figure and axis objects)
    If labeled_array is provided: np.ndarray (heatmap array with intensity values)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert numpy int64 keys to regular ints for consistency
    node_intensity_clean = {}
    for k, v in node_intensity.items():
        if hasattr(k, 'item'):  # numpy scalar
            node_intensity_clean[k.item()] = v
        else:
            node_intensity_clean[k] = v
    
    # Prepare data for plotting/array creation
    node_positions = []
    node_intensities = []
    
    for node_id, centroid in node_centroids.items():
        try:
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
                
            # Get intensity for this node
            intensity = node_intensity_clean[node_id]
            
            node_positions.append(centroid)
            node_intensities.append(intensity)
        except KeyError:
            # Skip nodes that don't have intensity values
            pass
    
    # Convert to numpy arrays
    positions = np.array(node_positions)
    intensities = np.array(node_intensities)
    
    # Determine shape if not provided
    if shape is None:
        if len(positions) > 0:
            max_coords = np.max(positions, axis=0).astype(int)
            shape = tuple(max_coords + 1)
        else:
            shape = (100, 100, 100) if is_3d else (1, 100, 100)
    
    # Determine min and max intensities for scaling
    if len(intensities) > 0:
        min_intensity = np.min(intensities)
        max_intensity = np.max(intensities)
    else:
        min_intensity, max_intensity = 0, 1
    
    if labeled_array is not None:
        # Create numpy RGB array output using labeled array and lookup table approach
        
        # Create mapping from node ID to intensity value (keep original float values)
        node_to_intensity = {}
        for node_id, centroid in node_centroids.items():
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
            
            # Only include nodes that have intensity values
            if node_id in node_intensity_clean:
                node_to_intensity[node_id] = node_intensity_clean[node_id]
        
        # Create colormap function (RdBu_r - red for high, blue for low, yellow/white for middle)
        def intensity_to_rgba(intensity, min_val, max_val):
            """Convert intensity value to RGBA using RdBu_r colormap logic, centered at 0"""
            
            # Handle edge case where all values are the same
            if max_val == min_val:
                if intensity == 0:
                    return np.array([255, 255, 255, 0], dtype=np.uint8)  # Transparent white for 0
                elif intensity > 0:
                    return np.array([255, 200, 200, 255], dtype=np.uint8)  # Opaque light red for positive
                else:
                    return np.array([200, 200, 255, 255], dtype=np.uint8)  # Opaque light blue for negative
            
            # Find the maximum absolute value for symmetric scaling around 0
            max_abs = max(abs(min_val), abs(max_val))
            
            # If max_abs is 0, everything is 0, so return transparent
            if max_abs == 0:
                return np.array([255, 255, 255, 0], dtype=np.uint8)  # Transparent white
            
            # Normalize intensity to -1 to 1 range, centered at 0
            normalized = intensity / max_abs
            normalized = np.clip(normalized, -1, 1)
            
            if normalized > 0:
                # Positive values: white to red (intensity 0 = transparent, max positive = red)
                r = 255
                g = int(255 * (1 - normalized))
                b = int(255 * (1 - normalized))
                alpha = 255  # Fully opaque for all non-zero values
            elif normalized < 0:
                # Negative values: white to blue (intensity 0 = transparent, max negative = blue)  
                r = int(255 * (1 + normalized))
                g = int(255 * (1 + normalized))
                b = 255
                alpha = 255  # Fully opaque for all non-zero values
            else:
                # Exactly 0: transparent
                r, g, b, alpha = 255, 255, 255, 0
            
            return np.array([r, g, b, alpha], dtype=np.uint8)

        # Modified usage in your main function:
        # Create lookup table for RGBA colors (note the 4 channels now)
        max_label = max(max(labeled_array.flat), max(node_to_intensity.keys()) if node_to_intensity else 0)
        color_lut = np.zeros((max_label + 1, 4), dtype=np.uint8)  # Default to transparent (0,0,0,0)

        # Fill lookup table with RGBA colors based on intensity
        for node_id, intensity in node_to_intensity.items():
            rgba_color = intensity_to_rgba(intensity, min_intensity, max_intensity)
            color_lut[int(node_id)] = rgba_color

        # Apply lookup table to labeled array - single vectorized operation
        if is_3d:
            # Return full 3D RGBA array [Z, Y, X, 4]
            heatmap_array = color_lut[labeled_array]
        else:
            # Return 2D RGBA array
            if labeled_array.ndim == 3:
                # Take middle slice for 2D representation
                middle_slice = labeled_array.shape[0] // 2
                heatmap_array = color_lut[labeled_array[middle_slice]]
            else:
                # Already 2D
                heatmap_array = color_lut[labeled_array]

        return heatmap_array
    
    else:
        # Create matplotlib plot
        fig = plt.figure(figsize=figsize)
        
        if is_3d:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract coordinates (assuming [Z, Y, X] format)
            z_coords = positions[:, 0]
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title}')
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[0])
            
        else:
            # 2D plot (using Y, X coordinates, ignoring Z/first dimension)
            ax = fig.add_subplot(111)
            
            # Extract Y, X coordinates
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'{title}')
            ax.grid(True, alpha=0.3)
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            
            # Set origin to top-left (invert Y-axis)
            ax.invert_yaxis()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label)
        
        # Add text annotations for min/max values
        cbar.ax.text(1.05, 0, f'Min: {min_intensity:.3f}\n(Blue)', 
                    transform=cbar.ax.transAxes, va='bottom')
        cbar.ax.text(1.05, 1, f'Max: {max_intensity:.3f}\n(Red)', 
                    transform=cbar.ax.transAxes, va='top')
        
        plt.tight_layout()
        plt.show()

def create_violin_plots(data_dict, graph_title="Violin Plots"):
    """
    Create violin plots from dictionary data with distinct colors and IQR lines.
    
    Parameters:
    data_dict (dict): Dictionary where keys are column headers (strings) and 
                     values are lists of floats
    graph_title (str): Title for the overall plot
    """
    if not data_dict:
        print("No data to plot")
        return
    
    # Prepare data
    data_dict = dict(sorted(data_dict.items()))
    labels = list(data_dict.keys())
    data_lists = list(data_dict.values())
    
    # Generate colors
    try:
        final_colors = generate_distinct_colors(len(labels))
    except Exception as e:
        print(f"Color generation failed, using default colors: {e}")
        final_colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    
    # Create violin plots
    violin_parts = ax.violinplot(
        data_lists, positions=range(len(labels)),
        showmeans=False, showmedians=True, showextrema=True
    )
    
    # Color violins
    for i, pc in enumerate(violin_parts['bodies']):
        if i < len(final_colors):
            pc.set_facecolor(final_colors[i])
            pc.set_alpha(0.7)
    
    # Color other violin parts
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in violin_parts:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_linewidth(1)
    
    # Set y-limits using percentiles to reduce extreme outlier influence
    all_data = [val for sublist in data_lists for val in sublist]
    if all_data:
        y_min = np.percentile(all_data, 5)
        y_max = np.percentile(all_data, 95)
        y_range = y_max - y_min
        y_padding = y_range * 0.15
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add IQR and median text annotations and dotted IQR lines
    for i, data in enumerate(data_lists):
        if len(data) > 0:
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1

            # Add dotted green lines for IQR
            ax.hlines(
                [q1, q3],
                i - 0.25, i + 0.25,
                colors='green',
                linestyles='dotted',
                linewidth=1.5,
                zorder=3,
                label='IQR (25th–75th)' if i == 0 else None  # Add label once
            )
            
            # Text annotation below the violins
            y_min_current = ax.get_ylim()[0]
            y_text = y_min_current - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15
            ax.text(
                i, y_text, f'Median: {median:.2f}\nIQR: {iqr:.2f}', 
                ha='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
    
    # Customize appearance
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(graph_title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Values (Z-score-like)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add baseline
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Identity Basepoint')
    ax.legend(loc='upper right')
    
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.show()

    # --- Outlier Detection ---
    outliers_info = []
    non_outlier_data = []

    for i, data in enumerate(data_lists):
        if len(data) > 0:
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [val for val in data if val < lower_bound or val > upper_bound]
            non_outliers = [val for val in data if lower_bound <= val <= upper_bound]

            outliers_info.append({
                'label': labels[i],
                'outliers': outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'total_count': len(data)
            })
            non_outlier_data.append(non_outliers)
        else:
            outliers_info.append({
                'label': labels[i],
                'outliers': [],
                'lower_bound': None,
                'upper_bound': None,
                'total_count': 0
            })
            non_outlier_data.append([])

    print("\n" + "="*60)
    print("OUTLIER DETECTION SUMMARY")
    print("="*60)
    total_outliers = 0
    for info in outliers_info:
        n_outliers = len(info['outliers'])
        total_outliers += n_outliers
        if n_outliers > 0:
            print(f"{info['label']}: {n_outliers} outliers out of {info['total_count']} points "
                  f"({n_outliers/info['total_count']*100:.1f}%)")
            print(f" Outlier Removed Range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
    if total_outliers == 0:
        print("No outliers detected in any dataset.")
    else:
        print(f"\nTotal outliers across all datasets: {total_outliers}")
    print("="*60 + "\n")


def create_neighbor_heatmap(distance_dict, dpi=300):
    """
    Create a professional heatmap from nearest neighbor distance data.
    
    Parameters:
    -----------
    distance_dict : dict
        Dictionary with keys like 'Root Type vs Neighbors: Neighbor Type'
        and values as distances
    dpi : int
        Resolution of the output image
    """
    
    # Parse the dictionary to extract cell types and create matrix
    cell_types = set()
    data_tuples = []
    
    for key, value in distance_dict.items():
        # Extract root and neighbor from keys like:
        # 'Includes: Tumor+ vs Neighbors: Includes Fibroblast+'
        match = re.search(r':\s*(.+?)\s+vs\s+Neighbors:\s*(.+?)$', key)
        if match:
            root = match.group(1).strip()
            neighbor = match.group(2).strip()
            
            # Remove 'Includes' or 'Includes:' prefix if present
            root = re.sub(r'^Includes:?\s*', '', root).strip()
            neighbor = re.sub(r'^Includes:?\s*', '', neighbor).strip()
            
            cell_types.add(root)
            cell_types.add(neighbor)
            data_tuples.append((root, neighbor, float(value)))
    
    # Sort cell types for consistent ordering
    cell_types = sorted(list(cell_types))
    n = len(cell_types)
    
    # Create matrix
    matrix = np.full((n, n), np.nan)
    for root, neighbor, value in data_tuples:
        i = cell_types.index(root)
        j = cell_types.index(neighbor)
        matrix[i, j] = value
    
    # Create figure with specific styling
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')
    
    # Create custom colormap: red (close/clustered) -> yellow -> blue (distant)
    colors = ['#d32f2f', '#e57373', '#ff9800', '#ffc107', '#64b5f6', '#2196f3', '#1565c0']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    cmap.set_bad(color='#1a1a1a')  # Set color for NaN values to match background
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cell_types, color='#cccccc', fontsize=11, fontweight='500')
    ax.set_yticklabels(cell_types, color='#cccccc', fontsize=11, fontweight='500')
    
    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add axis labels to clarify root vs neighbor
    ax.set_xlabel('Neighbor Node Type (searching to)', 
                   fontsize=13, fontweight='600', color='#64c8ff', labelpad=15)
    ax.set_ylabel('Root Node Type (searching from)', 
                   fontsize=13, fontweight='600', color='#64c8ff', labelpad=15)
    
    # Add title
    title = ax.text(0.5, 1.08, 'Nearest Neighbor Distance Matrix',
                    transform=ax.transAxes, fontsize=18, fontweight='700',
                    color='#f5f5f5', ha='center')
    
    subtitle = ax.text(0.5, 1.03, 'Average spatial distances between node populations',
                       transform=ax.transAxes, fontsize=11,
                       color='#888888', ha='center')
    
    # Add values as text in each cell only if n is not too large
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    # Always use black text
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                 ha='center', va='center', color='#1a1a1a',
                                 fontsize=12, fontweight='700', fontfamily='monospace')
    
    # Add colorbar with custom styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Distance', rotation=270, labelpad=20, 
                   fontsize=11, color='#aaaaaa', fontweight='500')
    cbar.ax.yaxis.set_tick_params(color='#666666', labelcolor='#aaaaaa')
    cbar.outline.set_edgecolor('#333333')
    cbar.ax.set_facecolor('#1a1a1a')
    
    # Style the grid
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which='minor', color='#333333', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)
    ax.tick_params(which='major', size=0)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()