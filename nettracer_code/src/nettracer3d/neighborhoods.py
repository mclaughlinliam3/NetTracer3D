import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Dict, Set
import umap



import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

def cluster_arrays(data_input, n_clusters, seed = 42):
    """
    Simple clustering of 1D arrays with key tracking.
    
    Parameters:
    -----------
    data_input : dict or List[List[float]]
        Dictionary {key: array} or list of arrays to cluster
    n_clusters : int  
        How many groups you want
        
    Returns:
    --------
    dict: {cluster_id: {'keys': [keys], 'arrays': [arrays]}}
    """
    
    # Handle both dict and list inputs
    if isinstance(data_input, dict):
        keys = list(data_input.keys())
        array_values = list(data_input.values())  # Use .values() to get the arrays
    else:
        keys = list(range(len(data_input)))  # Use indices as keys for lists
        array_values = data_input
    
    # Convert to numpy and cluster
    data = np.array(array_values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(data)
    

    clusters = [[] for _ in range(n_clusters)]

    for i, label in enumerate(labels):
        clusters[label].append(keys[i])

    return clusters
    
def plot_dict_heatmap(unsorted_data_dict, id_set, figsize=(12, 8), title="Neighborhood Heatmap"):
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
    
    # Create heatmap with white-to-red colormap
    im = ax.imshow(data_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    
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
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Intensity', rotation=-90, va="bottom")
    
    # Set labels and title
    ax.set_xlabel('Proportion of Node Type')
    ax.set_ylabel('Neighborhood')
    ax.set_title(title)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()


def visualize_cluster_composition_umap(cluster_data: Dict[int, np.ndarray], 
                                     class_names: Set[str],
                                     label = False,
                                     n_components: int = 2,
                                     random_state: int = 42):
    """
    Convert cluster composition data to UMAP visualization.
    
    Parameters:
    -----------
    cluster_data : dict
        Dictionary where keys are cluster IDs (int) and values are 1D numpy arrays
        representing the composition of each cluster
    class_names : set
        Set of strings representing the class names (order corresponds to array indices)
    n_components : int
        Number of UMAP components (default: 2 for 2D visualization)
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    embedding : numpy.ndarray
        UMAP embedding of the cluster compositions
    """
    
    # Convert set to sorted list for consistent ordering
    class_labels = sorted(list(class_names))
    
    # Extract cluster IDs and compositions
    cluster_ids = list(cluster_data.keys())
    compositions = np.array([cluster_data[cluster_id] for cluster_id in cluster_ids])
    
    # Create UMAP reducer
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    
    # Fit and transform the composition data
    embedding = reducer.fit_transform(compositions)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=cluster_ids, cmap='viridis', s=100, alpha=0.7)
        
        if label:
            # Add cluster ID labels
            for i, cluster_id in enumerate(cluster_ids):
                plt.annotate(f'{cluster_id}', 
                            (embedding[i, 0], embedding[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, alpha=0.8)
        
        plt.colorbar(scatter, label='Community ID')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('UMAP Visualization of Community Compositions')
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           c=cluster_ids, cmap='viridis', s=100, alpha=0.7)
        
        # Add cluster ID labels
        for i, cluster_id in enumerate(cluster_ids):
            ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2],
                   f'C{cluster_id}', fontsize=8)
        
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        ax.set_title('3D UMAP Visualization of Cluster Compositions')
        plt.colorbar(scatter, label='Cluster ID')
    
    plt.tight_layout()
    plt.show()
    
    # Print composition details
    print("Cluster Compositions:")
    print(f"Classes: {class_labels}")
    for i, cluster_id in enumerate(cluster_ids):
        composition = compositions[i]
        print(f"Cluster {cluster_id}: {composition}")
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
            """Convert intensity value to RGB using RdBu_r colormap logic"""
            if max_val == min_val:
                # All same value, use neutral color
                return np.array([255, 255, 255], dtype=np.uint8)  # White
            
            # Normalize to -1 to 1 range (like RdBu_r colormap)
            normalized = 2 * (intensity - min_val) / (max_val - min_val) - 1
            normalized = np.clip(normalized, -1, 1)
            
            if normalized > 0:
                # Positive values: white to red
                r = 255
                g = int(255 * (1 - normalized))
                b = int(255 * (1 - normalized))
            else:
                # Negative values: white to blue
                r = int(255 * (1 + normalized))
                g = int(255 * (1 + normalized))
                b = 255
            
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
            """Convert intensity value to RGB using RdBu_r colormap logic"""
            if max_val == min_val:
                # All same value, use neutral color
                return np.array([255, 255, 255], dtype=np.uint8)  # White
            
            # Normalize to -1 to 1 range (like RdBu_r colormap)
            normalized = 2 * (intensity - min_val) / (max_val - min_val) - 1
            normalized = np.clip(normalized, -1, 1)
            
            if normalized > 0:
                # Positive values: white to red
                r = 255
                g = int(255 * (1 - normalized))
                b = int(255 * (1 - normalized))
            else:
                # Negative values: white to blue
                r = int(255 * (1 + normalized))
                g = int(255 * (1 + normalized))
                b = 255
            
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