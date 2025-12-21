import numpy as np
import networkx as nx
from . import nettracer as n3d
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_fill_holes
from scipy.spatial import cKDTree
from . import smart_dilate as sdl
from skimage.morphology import remove_small_objects, skeletonize
import warnings
warnings.filterwarnings('ignore')


class VesselDenoiser:
    """
    Denoise vessel segmentations using graph-based geometric features
    """
    
    def __init__(self, 
                 score_thresh = 2,
                 xy_scale = 1,
                 z_scale = 1):

        self.score_thresh = score_thresh
        self.xy_scale = xy_scale
        self.z_scale = z_scale


    def select_kernel_points_topology(self, data, skeleton):
        """
        ENDPOINTS ONLY version: Returns only skeleton endpoints (degree=1 nodes)
        """
        skeleton_coords = np.argwhere(skeleton)
        if len(skeleton_coords) == 0:
            return skeleton_coords
        
        # Map coord -> index
        coord_to_idx = {tuple(c): i for i, c in enumerate(skeleton_coords)}
        
        # Build full 26-connected skeleton graph
        skel_graph = nx.Graph()
        for i, c in enumerate(skeleton_coords):
            skel_graph.add_node(i, pos=c)
        
        nbr_offsets = [(dz, dy, dx)
                       for dz in (-1, 0, 1)
                       for dy in (-1, 0, 1)
                       for dx in (-1, 0, 1)
                       if not (dz == dy == dx == 0)]
        
        for i, c in enumerate(skeleton_coords):
            cz, cy, cx = c
            for dz, dy, dx in nbr_offsets:
                nb = (cz + dz, cy + dy, cx + dx)
                j = coord_to_idx.get(nb, None)
                if j is not None and j > i:
                    skel_graph.add_edge(i, j)
        
        # Get degree per voxel
        deg = dict(skel_graph.degree())
        
        # ONLY keep endpoints (degree=1)
        endpoints = {i for i, d in deg.items() if d == 1}
        
        # Return endpoint coordinates
        kernel_coords = np.array([skeleton_coords[i] for i in endpoints])
        return kernel_coords
        
    
    def extract_kernel_features(self, skeleton, distance_map, kernel_pos, radius=5):
        """Extract geometric features for a kernel at a skeleton point"""
        z, y, x = kernel_pos
        shape = skeleton.shape
        
        features = {}
        
        # Vessel radius at this point
        features['radius'] = distance_map[z, y, x]
        
        # Local skeleton density (connectivity measure)
        z_min = max(0, z - radius)
        z_max = min(shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(shape[2], x + radius + 1)
        
        local_region = skeleton[z_min:z_max, y_min:y_max, x_min:x_max]
        features['local_density'] = np.sum(local_region) / max(local_region.size, 1)
        
        # Local direction vector
        features['direction'] = self._compute_local_direction(
            skeleton, kernel_pos, radius
        )
        
        # Position
        features['pos'] = np.array(kernel_pos)
        
        # ALL kernels are endpoints in this version
        features['is_endpoint'] = True
        
        return features

    
    def _compute_local_direction(self, skeleton, pos, radius=5):
        """Compute principal direction of skeleton in local neighborhood"""
        z, y, x = pos
        shape = skeleton.shape
        
        z_min = max(0, z - radius)
        z_max = min(shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(shape[2], x + radius + 1)
        
        local_skel = skeleton[z_min:z_max, y_min:y_max, x_min:x_max]
        coords = np.argwhere(local_skel)
        
        if len(coords) < 2:
            return np.array([0., 0., 1.])
        
        # PCA to find principal direction
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_direction = eigenvectors[:, -1]  # largest eigenvalue
        
        return principal_direction / (np.linalg.norm(principal_direction) + 1e-10)
    
    def group_endpoints_by_vertex(self, skeleton_points, verts):
        """
        Group endpoints by which vertex (labeled blob) they belong to
        
        Returns:
        --------
        vertex_to_endpoints : dict
            Dictionary mapping vertex_label -> [list of endpoint indices]
        """
        vertex_to_endpoints = {}
        
        for idx, pos in enumerate(skeleton_points):
            z, y, x = pos.astype(int)
            vertex_label = int(verts[z, y, x])
            
            # Skip if endpoint is not in any vertex (label=0)
            if vertex_label == 0:
                continue
            
            if vertex_label not in vertex_to_endpoints:
                vertex_to_endpoints[vertex_label] = []
            
            vertex_to_endpoints[vertex_label].append(idx)
        
        return vertex_to_endpoints
    
    def compute_edge_features(self, feat_i, feat_j):
        """
        Compute features for potential connection between two endpoints
        NO DISTANCE-BASED FEATURES - only radius and direction
        """
        features = {}
        
        # Euclidean distance (for reference only, not used in scoring)
        pos_diff = feat_j['pos'] - feat_i['pos']
        features['distance'] = np.linalg.norm(pos_diff)
        
        # Radius similarity
        r_i, r_j = feat_i['radius'], feat_j['radius']
        features['radius_diff'] = abs(r_i - r_j)
        features['radius_ratio'] = min(r_i, r_j) / (max(r_i, r_j) + 1e-10)
        features['mean_radius'] = (r_i + r_j) / 2.0
        
        # Direction alignment
        direction_vec = pos_diff / (features['distance'] + 1e-10)
        
        # Alignment with both local directions
        align_i = abs(np.dot(feat_i['direction'], direction_vec))
        align_j = abs(np.dot(feat_j['direction'], direction_vec))
        features['alignment'] = (align_i + align_j) / 2.0
        
        # Smoothness: how well does connection align with both local directions
        features['smoothness'] = min(align_i, align_j)
        
        # Density similarity
        features['density_diff'] = abs(feat_i['local_density'] - feat_j['local_density'])
        
        return features
    
    def score_connection(self, edge_features):
        score = 0.0

        # HARD REJECT for definite forks/sharp turns
        if edge_features['smoothness'] < 0.5:  # At least one endpoint pointing away
            return -999
        
        # Base similarity scoring
        score += edge_features['radius_ratio'] * 10.0
        score += edge_features['alignment'] * 8.0
        score += edge_features['smoothness'] * 6.0
        score -= edge_features['density_diff'] * 0.5
        
        # PENALTY for poor directional alignment (punish forks!)
        # Alignment < 0.5 means vessels are pointing in different directions
        # This doesn't trigger that often so it might be redundant with the above step
        if edge_features['alignment'] < 0.5:
            penalty = (0.5 - edge_features['alignment']) * 15.0
            score -= penalty
        
        # ADDITIONAL PENALTY for sharp turns/forks --- no longer in use since we now hard reject these, but I left this in here to reverse it later potentially
        # Smoothness < 0.4 means at least one endpoint points away
        #if edge_features['smoothness'] < 0.4:
         #   penalty = (0.4 - edge_features['smoothness']) * 20.0
          #  score -= penalty
        
        # Size bonus: ONLY if vessels already match well
        
        if edge_features['radius_ratio'] > 0.7 and edge_features['alignment'] > 0.5:
            mean_radius = edge_features['mean_radius']
            score += mean_radius * 1.5
        
        return score
    
    def connect_vertices_across_gaps(self, skeleton_points, kernel_features, 
                                     labeled_skeleton, vertex_to_endpoints, verbose=False):
        """
        Connect vertices by finding best endpoint pair across each vertex
        Each vertex makes at most one connection
        """
        # Initialize label dictionary: label -> label (identity mapping)
        unique_labels = np.unique(labeled_skeleton[labeled_skeleton > 0])
        label_dict = {int(label): int(label) for label in unique_labels}
        
        # Map endpoint index to its skeleton label
        endpoint_to_label = {}
        for idx, pos in enumerate(skeleton_points):
            z, y, x = pos.astype(int)
            label = int(labeled_skeleton[z, y, x])
            endpoint_to_label[idx] = label
        
        # Find root label (union-find helper)
        def find_root(label):
            root = label
            while label_dict[root] != root:
                root = label_dict[root]
            return root
        
        # Iterate through each vertex
        for vertex_label, endpoint_indices in vertex_to_endpoints.items():
            if len(endpoint_indices) < 2:
                # Need at least 2 endpoints to make a connection
                continue
            
            if verbose and len(endpoint_indices) > 0:
                print(f"\nVertex {vertex_label}: {len(endpoint_indices)} endpoints")
            
            # Find best pair of endpoints to connect
            best_i = None
            best_j = None
            best_score = -np.inf
            
            # Try all pairs of endpoints within this vertex
            for i in range(len(endpoint_indices)):
                for j in range(i + 1, len(endpoint_indices)):
                    idx_i = endpoint_indices[i]
                    idx_j = endpoint_indices[j]
                    
                    feat_i = kernel_features[idx_i]
                    feat_j = kernel_features[idx_j]
                    
                    label_i = endpoint_to_label[idx_i]
                    label_j = endpoint_to_label[idx_j]
                    
                    root_i = find_root(label_i)
                    root_j = find_root(label_j)
                    
                    # Skip if already unified
                    if root_i == root_j:
                        continue
                    
                    # Compute edge features (no skeleton needed, no distance penalty)
                    edge_feat = self.compute_edge_features(feat_i, feat_j)
                    
                    # Score this connection
                    score = self.score_connection(edge_feat)
                    
                    # Apply threshold
                    if score > self.score_thresh and score > best_score:
                        best_score = score
                        best_i = idx_i
                        best_j = idx_j
            
            # Make the best connection for this vertex
            if best_i is not None and best_j is not None:
                label_i = endpoint_to_label[best_i]
                label_j = endpoint_to_label[best_j]
                
                root_i = find_root(label_i)
                root_j = find_root(label_j)
                
                # Unify labels: point larger label to smaller label
                if root_i < root_j:
                    label_dict[root_j] = root_i
                    unified_label = root_i
                else:
                    label_dict[root_i] = root_j
                    unified_label = root_j
                
                if verbose:
                    feat_i = kernel_features[best_i]
                    feat_j = kernel_features[best_j]
                    print(f"  ✓ Connected labels {label_i} <-> {label_j} (unified as {unified_label})")
                    print(f"    Score: {best_score:.2f} | Radii: {feat_i['radius']:.1f}, {feat_j['radius']:.1f}")
        
        return label_dict
    
    def denoise(self, data, skeleton, labeled_skeleton, verts, verbose=False):
        """
        Main pipeline: unify skeleton labels by connecting endpoints at vertices
        
        Parameters:
        -----------
        data : ndarray
            3D binary segmentation (for distance transform)
        skeleton : ndarray
            3D binary skeleton
        labeled_skeleton : ndarray
            Labeled skeleton (each branch has unique label)
        verts : ndarray
            Labeled vertices (blobs where branches meet)
        verbose : bool
            Print progress
            
        Returns:
        --------
        label_dict : dict
            Dictionary mapping old labels to unified labels
        """
        if verbose:
            print("Starting skeleton label unification...")
            print(f"Initial unique labels: {len(np.unique(labeled_skeleton[labeled_skeleton > 0]))}")
        
        # Compute distance transform
        if verbose:
            print("Computing distance transform...")
        distance_map = sdl.compute_distance_transform_distance(data, fast_dil = True)
        
        # Extract endpoints
        if verbose:
            print("Extracting skeleton endpoints...")
        kernel_points = self.select_kernel_points_topology(data, skeleton)
        
        if verbose:
            print(f"Found {len(kernel_points)} endpoints")
        
        # Group endpoints by vertex
        if verbose:
            print("Grouping endpoints by vertex...")
        vertex_to_endpoints = self.group_endpoints_by_vertex(kernel_points, verts)
        
        if verbose:
            print(f"Found {len(vertex_to_endpoints)} vertices with endpoints")
            vertices_with_multiple = sum(1 for v in vertex_to_endpoints.values() if len(v) >= 2)
            print(f"  {vertices_with_multiple} vertices have 2+ endpoints (connection candidates)")
        
        # Extract features for each endpoint
        if verbose:
            print("Extracting endpoint features...")
        kernel_features = []
        for pt in kernel_points:
            feat = self.extract_kernel_features(skeleton, distance_map, pt)
            kernel_features.append(feat)
        
        # Connect vertices
        if verbose:
            print("Connecting endpoints at vertices...")
        label_dict = self.connect_vertices_across_gaps(
            kernel_points, kernel_features, labeled_skeleton, 
            vertex_to_endpoints, verbose
        )
        
        # Compress label dictionary (path compression for union-find)
        if verbose:
            print("\nCompressing label mappings...")
        for label in list(label_dict.keys()):
            root = label
            while label_dict[root] != root:
                root = label_dict[root]
            label_dict[label] = root
        
        # Count final unified components
        final_labels = set(label_dict.values())
        if verbose:
            print(f"Final unified labels: {len(final_labels)}")
            print(f"Reduced from {len(label_dict)} to {len(final_labels)} components")
        
        return label_dict


def trace(data, labeled_skeleton, verts, score_thresh=10, xy_scale = 1, z_scale = 1, verbose=False):
    """
    Trace and unify skeleton labels using vertex-based endpoint grouping
    """
    skeleton = n3d.binarize(labeled_skeleton)
    
    # Create denoiser
    denoiser = VesselDenoiser(score_thresh=score_thresh, xy_scale = xy_scale, z_scale = z_scale)
    
    # Run label unification
    label_dict = denoiser.denoise(data, skeleton, labeled_skeleton, verts, verbose=verbose)
    
    # Apply unified labels efficiently (SINGLE PASS)
    # Create lookup array: index by old label, get new label
    max_label = np.max(labeled_skeleton)
    label_map = np.arange(max_label + 1)  # Identity mapping by default
    
    for old_label, new_label in label_dict.items():
        label_map[old_label] = new_label
    
    # Single array indexing operation
    relabeled_skeleton = label_map[labeled_skeleton]
    
    return relabeled_skeleton


if __name__ == "__main__":
    print("Test area")