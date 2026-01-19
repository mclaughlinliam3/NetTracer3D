import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from collections import deque
from . import smart_dilate as sdl


class VesselDenoiser:
    """
    Denoise vessel segmentations using graph-based geometric features
    IMPROVED: Uses skeleton topology to compute endpoint directions
    """
    
    def __init__(self, 
                 score_thresh = 2,
                 xy_scale = 1,
                 z_scale = 1,
                 trace_length = 10):
        self.score_thresh = score_thresh
        self.xy_scale = xy_scale
        self.z_scale = z_scale
        self.trace_length = trace_length  # How far to trace from endpoint

    def _build_skeleton_graph(self, skeleton):
        """
        Build a graph from skeleton where nodes are voxel coordinates
        and edges connect 26-connected neighbors
        """
        skeleton_coords = np.argwhere(skeleton)
        if len(skeleton_coords) == 0:
            return None, None
        
        # Map coordinate tuple -> node index
        coord_to_idx = {tuple(c): i for i, c in enumerate(skeleton_coords)}
        
        # Build graph
        skel_graph = nx.Graph()
        for i, c in enumerate(skeleton_coords):
            skel_graph.add_node(i, pos=c)
        
        # 26-connected neighborhood
        nbr_offsets = [(dz, dy, dx)
                       for dz in (-1, 0, 1)
                       for dy in (-1, 0, 1)
                       for dx in (-1, 0, 1)
                       if not (dz == dy == dx == 0)]
        
        # Add edges
        for i, c in enumerate(skeleton_coords):
            cz, cy, cx = c
            for dz, dy, dx in nbr_offsets:
                nb = (cz + dz, cy + dy, cx + dx)
                j = coord_to_idx.get(nb)
                if j is not None and j > i:
                    skel_graph.add_edge(i, j)
        
        return skel_graph, coord_to_idx

    def select_kernel_points_topology(self, data, skeleton):
        """
        Returns only skeleton endpoints (degree=1 nodes)
        """
        skel_graph, coord_to_idx = self._build_skeleton_graph(skeleton)
        
        if skel_graph is None:
            return np.array([]), None, None
        
        # Get degree per node
        deg = dict(skel_graph.degree())
        
        # ONLY keep endpoints (degree=1)
        endpoints = [i for i, d in deg.items() if d == 1]
        
        # Get coordinates
        skeleton_coords = np.argwhere(skeleton)
        kernel_coords = np.array([skeleton_coords[i] for i in endpoints])
        
        return kernel_coords, skel_graph, coord_to_idx

    def _compute_endpoint_direction(self, skel_graph, endpoint_idx, trace_length=None):
        """
        Compute direction by tracing along skeleton from endpoint.
        Returns direction vector pointing INTO the skeleton (away from endpoint).
        
        Parameters:
        -----------
        skel_graph : networkx.Graph
            Skeleton graph with node positions
        endpoint_idx : int
            Node index of the endpoint
        trace_length : int
            How many steps to trace along skeleton
            
        Returns:
        --------
        direction : ndarray
            Normalized direction vector pointing into skeleton from endpoint
        """
        if trace_length is None:
            trace_length = self.trace_length
        
        # Get endpoint position
        endpoint_pos = skel_graph.nodes[endpoint_idx]['pos']
        
        # BFS from endpoint to collect positions along skeleton path
        visited = {endpoint_idx}
        queue = deque([endpoint_idx])
        path_positions = []
        
        while queue and len(path_positions) < trace_length:
            current = queue.popleft()
            
            # Get neighbors
            for neighbor in skel_graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # Add this position to path
                    neighbor_pos = skel_graph.nodes[neighbor]['pos']
                    path_positions.append(neighbor_pos)
                    
                    if len(path_positions) >= trace_length:
                        break
        
        # If we couldn't trace far enough, use what we have
        if len(path_positions) == 0:
            # Isolated endpoint, return arbitrary direction
            return np.array([0., 0., 1.])
        
        # Compute direction as average vector from endpoint to traced positions
        # This gives us the direction the skeleton is "extending" from the endpoint
        path_positions = np.array(path_positions)
        
        # Weight more distant points more heavily (they better represent overall direction)
        weights = np.linspace(1.0, 2.0, len(path_positions))
        weights = weights / weights.sum()
        
        # Weighted average position along the path
        weighted_target = np.sum(path_positions * weights[:, None], axis=0)
        
        # Direction from endpoint toward this position
        direction = weighted_target - endpoint_pos
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return np.array([0., 0., 1.])
        
        return direction / norm

    def extract_kernel_features(self, skeleton, distance_map, kernel_pos, 
                                skel_graph, coord_to_idx, endpoint_idx):
        """Extract geometric features for a kernel at a skeleton endpoint"""
        z, y, x = kernel_pos
        
        features = {}
        
        # Vessel radius at this point
        features['radius'] = distance_map[z, y, x]
                        
        # Direction vector using topology-based tracing
        features['direction'] = self._compute_endpoint_direction(
            skel_graph, endpoint_idx, self.trace_length
        )
        
        # Position
        features['pos'] = np.array(kernel_pos)
        
        # All kernels are endpoints
        features['is_endpoint'] = True
        
        return features

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
        Compute features for potential connection between two endpoints.
        IMPROVED: Uses proper directional alignment (not abs value).
        
        Two endpoints should connect if:
        - Their skeletons are pointing TOWARD each other (negative dot product of directions)
        - They have similar radii
        - The connection vector aligns with both skeleton directions
        """
        features = {}
        
        # Vector from endpoint i to endpoint j
        pos_diff = feat_j['pos'] - feat_i['pos']
        features['distance'] = np.linalg.norm(pos_diff)
        
        if features['distance'] < 1e-10:
            # Same point, shouldn't happen
            features['connection_vector'] = np.array([0., 0., 1.])
        else:
            features['connection_vector'] = pos_diff / features['distance']
        
        # Radius similarity
        r_i, r_j = feat_i['radius'], feat_j['radius']
        features['radius_diff'] = abs(r_i - r_j)
        features['radius_ratio'] = min(r_i, r_j) / (max(r_i, r_j) + 1e-10)
        features['mean_radius'] = (r_i + r_j) / 2.0
        
        # CRITICAL: Check if skeletons point toward each other
        # If both directions point into their skeletons (away from endpoints),
        # they should point in OPPOSITE directions across the gap
        dir_i = feat_i['direction']
        dir_j = feat_j['direction']
        connection_vec = features['connection_vector']
        
        # How well does endpoint i's skeleton direction align with the gap vector?
        # (positive = pointing toward j)
        align_i = np.dot(dir_i, connection_vec)
        
        # How well does endpoint j's skeleton direction align AGAINST the gap vector?
        # (negative = pointing toward i)
        align_j = np.dot(dir_j, connection_vec)
        
        # For good connection: align_i should be positive (i pointing toward j)
        # and align_j should be negative (j pointing toward i)
        # So align_i - align_j should be large and positive
        features['approach_score'] = align_i - align_j
        
        # Individual alignment scores (for diagnostics)
        features['align_i'] = align_i
        features['align_j'] = align_j
        
        # How parallel/antiparallel are the two skeleton directions?
        # -1 = pointing toward each other (good for connection)
        # +1 = pointing in same direction (bad, parallel branches)
        features['direction_similarity'] = np.dot(dir_i, dir_j)
        
        return features
    
    def score_connection(self, edge_features):
        """
        Score potential connection between two endpoints.
        FIXED: Directions point INTO skeletons (away from endpoints)
        """
        score = 0.0
        
        # For good connections when directions point INTO skeletons:
        # - align_i should be NEGATIVE (skeleton i extends away from j)
        # - align_j should be POSITIVE (skeleton j extends away from i)  
        # - Both skeletons extend away from the gap (good!)
        
        # HARD REJECT: If skeletons point in same direction (parallel branches)
        if edge_features['direction_similarity'] > 0.7:
            return -999
        
        # HARD REJECT: If both skeletons extend TOWARD the gap (diverging structure)
        # This means: align_i > 0 and align_j < 0 (both point at gap = fork/divergence)
        if edge_features['align_i'] > 0.3 and edge_features['align_j'] < -0.3:
            return -999
        
        # HARD REJECT: If either skeleton extends the wrong way
        # align_i should be negative, align_j should be positive
        if edge_features['align_i'] > 0.3 or edge_features['align_j'] < -0.3:
            return -999
        
        # Base similarity scoring
        score += edge_features['radius_ratio'] * 15.0
        
        # REWARD: Skeletons extending away from each other across gap
        # When directions point into skeletons:
        # Good connection has align_i < 0 and align_j > 0
        # So we want to MAXIMIZE: -align_i + align_j (both terms positive)
        extension_score = (-edge_features['align_i'] + edge_features['align_j'])
        score += extension_score * 10.0
        
        # REWARD: Skeletons pointing in opposite directions (antiparallel)
        # direction_similarity should be negative
        antiparallel_bonus = max(0, -edge_features['direction_similarity']) * 5.0
        score += antiparallel_bonus

        # SIZE BONUS: Reward large, well-matched vessels
        if edge_features['radius_ratio'] > 0.7 and extension_score > 1.0:
            mean_radius = edge_features['mean_radius']
            score += mean_radius * 1.5
        
        return score
    
    def connect_vertices_across_gaps(self, skeleton_points, kernel_features, 
                                     labeled_skeleton, vertex_to_endpoints, verbose=False):
        """
        Connect vertices by finding best endpoint pair across each vertex.
        Each vertex makes at most one connection.
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
                    
                    # Compute edge features
                    edge_feat = self.compute_edge_features(feat_i, feat_j)
                    
                    # Score this connection
                    score = self.score_connection(edge_feat)
                    #print(score)
                    
                    if verbose and score > -900:
                        print(f"  Pair {idx_i}-{idx_j}: score={score:.2f}, "
                              f"approach={edge_feat['approach_score']:.2f}, "
                              f"dir_sim={edge_feat['direction_similarity']:.2f}")
                    
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
                
                # Unify labels
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
        """
        if verbose:
            print("Starting skeleton label unification (IMPROVED VERSION)...")
            print(f"Initial unique labels: {len(np.unique(labeled_skeleton[labeled_skeleton > 0]))}")
        
        # Compute distance transform
        if verbose:
            print("Computing distance transform...")
        distance_map = sdl.compute_distance_transform_distance(data, fast_dil = True)
        
        # Extract endpoints and build skeleton graph
        if verbose:
            print("Extracting skeleton endpoints and building graph...")
        kernel_points, skel_graph, coord_to_idx = self.select_kernel_points_topology(data, skeleton)
        
        if verbose:
            print(f"Found {len(kernel_points)} endpoints")
        
        if len(kernel_points) == 0:
            # No endpoints, return identity mapping
            unique_labels = np.unique(labeled_skeleton[labeled_skeleton > 0])
            return {int(label): int(label) for label in unique_labels}
        
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
            print("Extracting endpoint features with topology-based directions...")
        
        # Create reverse mapping: position -> node index in graph
        skeleton_coords = np.argwhere(skeleton)
        kernel_features = []
        
        for pt in kernel_points:
            # Find this endpoint in the graph
            pt_tuple = tuple(pt)
            endpoint_idx = coord_to_idx.get(pt_tuple)
            
            if endpoint_idx is None:
                # Shouldn't happen, but handle gracefully
                print(f"Warning: Endpoint {pt} not found in graph")
                continue
            
            feat = self.extract_kernel_features(
                skeleton, distance_map, pt, skel_graph, coord_to_idx, endpoint_idx
            )
            kernel_features.append(feat)
        
        # Connect vertices
        if verbose:
            print("Connecting endpoints at vertices...")
        label_dict = self.connect_vertices_across_gaps(
            kernel_points, kernel_features, labeled_skeleton, 
            vertex_to_endpoints, verbose
        )
        
        # Compress label dictionary
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


def trace(data, labeled_skeleton, verts, score_thresh=10, xy_scale=1, z_scale=1, 
          trace_length=10, verbose=False):
    """
    Trace and unify skeleton labels using vertex-based endpoint grouping.
    IMPROVED: Uses topology-based direction calculation.
    
    Parameters:
    -----------
    trace_length : int
        How many voxels to trace from each endpoint to determine direction
    """
    skeleton = (labeled_skeleton > 0).astype(np.uint8)
    
    # Create denoiser with trace_length parameter
    denoiser = VesselDenoiser(
        score_thresh=score_thresh, 
        xy_scale=xy_scale, 
        z_scale=z_scale,
        trace_length=trace_length
    )
    
    # Run label unification
    label_dict = denoiser.denoise(data, skeleton, labeled_skeleton, verts, verbose=verbose)
    
    # Apply unified labels
    max_label = np.max(labeled_skeleton)
    label_map = np.arange(max_label + 1)
    
    for old_label, new_label in label_dict.items():
        label_map[old_label] = new_label
    
    relabeled_skeleton = label_map[labeled_skeleton]
    
    return relabeled_skeleton


if __name__ == "__main__":
    print("Improved branch stitcher with topology-based direction calculation")