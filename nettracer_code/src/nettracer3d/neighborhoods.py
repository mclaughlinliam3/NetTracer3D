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
                     center_at_one=False):
    """
    Create a heatmap from a dictionary of numpy arrays.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are identifiers and values are 1D numpy arrays of floats (0-1)
    id_set : list
        List of strings describing what each index in the numpy arrays represents
    figsize : tuple, optional
        Figure size (width, height)
    title : str, optional
        Title for the heatmap
    center_at_one : bool, optional
        If True, uses a diverging colormap centered at 1 with nonlinear scaling:
        - 0 to 1: blue to white (underrepresentation to normal)
        - 1+: white to red (overrepresentation)
        If False (default), uses standard white-to-red scaling from 0 to 1
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    data_dict = {k: unsorted_data_dict[k] for k in sorted(unsorted_data_dict.keys())}
    # Convert dict to 2D array for heatmap
    # Each row represents one key from the dict
    keys = list(data_dict.keys())
    data_matrix = np.array([data_dict[key] for key in keys])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if center_at_one:
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
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(id_set)))
    ax.set_yticks(np.arange(len(keys)))
    ax.set_xticklabels(id_set)
    ax.set_yticklabels(keys)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations showing the actual values
    for i in range(len(keys)):
        for j in range(len(id_set)):
            # Use original data values for annotations
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)


    ret_dict = {}

    for i, row in enumerate(data_matrix):
        ret_dict[keys[i]] = row
    
    # Set labels and title
    if center_at_one:
        ax.set_xlabel('Representation Factor of Node Type')
    else:
        ax.set_xlabel('Proportion of Node Type')

    ax.set_ylabel('Neighborhood')
    ax.set_title(title)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    return ret_dict
    


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
                                     neighborhoods: Optional[Dict[int, int]] = None):
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
    if neighborhoods is not None:
        # Use neighborhood coloring - import the community extractor methods
        from . import community_extractor
        
        # Filter neighborhoods to only include cluster_ids that exist in our data
        filtered_neighborhoods = {node_id: neighborhood_id 
                                for node_id, neighborhood_id in neighborhoods.items() 
                                if node_id in cluster_ids}
        
        # Create a dummy labeled array just for the coloring function
        # We only need the coloring logic, not actual clustering
        dummy_array = np.array(cluster_ids)
        
        # Get colors using the community coloration method
        _, neighborhood_color_names = community_extractor.assign_community_colors(
            filtered_neighborhoods, dummy_array
        )
        
        # Create color mapping for our points
        unique_neighborhoods = sorted(list(set(filtered_neighborhoods.values())))
        colors = community_extractor.generate_distinct_colors(len(unique_neighborhoods))
        neighborhood_to_color = {neighborhood: colors[i] for i, neighborhood in enumerate(unique_neighborhoods)}
        
        # Map each cluster to its neighborhood color
        point_colors = []
        neighborhood_labels = []
        for cluster_id in cluster_ids:
            if cluster_id in filtered_neighborhoods:
                neighborhood_id = filtered_neighborhoods[cluster_id]
                point_colors.append(neighborhood_to_color[neighborhood_id])
                neighborhood_labels.append(neighborhood_id)
            else:
                # Default color for nodes not in any neighborhood
                point_colors.append((128, 128, 128))  # Gray
                neighborhood_labels.append("Unknown")
        
        # Normalize RGB values for matplotlib (0-1 range)
        point_colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in point_colors]
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
                if use_neighborhood_coloring and cluster_id in filtered_neighborhoods:
                    neighborhood_id = filtered_neighborhoods[cluster_id]
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
            for neighborhood_id in unique_neighborhoods:
                color = neighborhood_to_color[neighborhood_id]
                norm_color = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=norm_color, 
                              markersize=10, label=f'Neighborhood {neighborhood_id}')
                )
            plt.legend(handles=legend_elements, title='Neighborhoods', 
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
            title += ' (Colored by Neighborhood)'
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
                if use_neighborhood_coloring and cluster_id in filtered_neighborhoods:
                    neighborhood_id = filtered_neighborhoods[cluster_id]
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
            for neighborhood_id in unique_neighborhoods:
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
            if use_neighborhood_coloring and cluster_id in filtered_neighborhoods:
                neighborhood_id = filtered_neighborhoods[cluster_id]
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
        max_label = max(max(labeled_array.flat), max(node_to_intensity.keys()) if node_to_intensity else 0)
        color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)  # Default to black (0,0,0)
        
        # Fill lookup table with RGB colors based on intensity
        for node_id, intensity in node_to_intensity.items():
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

# Example usage:
if __name__ == "__main__":
    # Sample data for demonstration
    sample_dict = {
        'category_A': np.array([0.1, 0.5, 0.8, 0.3, 0.9]),
        'category_B': np.array([0.7, 0.2, 0.6, 0.4, 0.1]),
        'category_C': np.array([0.9, 0.8, 0.2, 0.7, 0.5])
    }
    
    sample_id_set = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Create the heatmap
    fig, ax = plot_dict_heatmap(sample_dict, sample_id_set, 
                               title="Sample Heatmap Visualization")
    
    plt.show()