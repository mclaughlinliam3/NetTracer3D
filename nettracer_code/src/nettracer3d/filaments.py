import numpy as np
import networkx as nx
from . import nettracer as n3d
from scipy.ndimage import gaussian_filter, binary_fill_holes
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_objects, skeletonize
import warnings
from . import smart_dilate as sdl
warnings.filterwarnings('ignore')


class VesselDenoiser:
    """
    Denoise vessel segmentations using graph-based geometric features
    """
    
    def __init__(self, 
                 kernel_spacing=1,
                 max_connection_distance=20,
                 min_component_size=20,
                 gap_tolerance=5.0,
                 blob_sphericity = 1.0,
                 blob_volume = 200,
                 spine_removal=0,
                 score_thresh = 2,
                 xy_scale = 1,
                 z_scale = 1,
                 radius_aware_distance=True):
        """
        Parameters:
        -----------
        kernel_spacing : int
            Spacing between kernel sampling points on skeleton
        max_connection_distance : float
            Maximum distance to consider connecting two kernels (base distance)
        min_component_size : int
            Minimum number of kernels to keep a component
        gap_tolerance : float
            Maximum gap size relative to vessel radius
        radius_aware_distance : bool
            If True, scale connection distance based on vessel radius
        """
        self.kernel_spacing = kernel_spacing
        self.max_connection_distance = max_connection_distance
        self.min_component_size = min_component_size
        self.gap_tolerance = gap_tolerance
        self.blob_sphericity = blob_sphericity
        self.blob_volume = blob_volume
        self.spine_removal = spine_removal
        self.radius_aware_distance = radius_aware_distance
        self.score_thresh = score_thresh
        self.xy_scale = xy_scale
        self.z_scale = z_scale

        self._sphere_cache = {}  # Cache sphere masks for different radii

    def filter_large_spherical_blobs(self, binary_array, 
                                      min_volume=200,
                                      min_sphericity=1.0,
                                      verbose=True):
        """
        Remove large spherical artifacts prior to denoising.
        Vessels are elongated; large spherical blobs are likely artifacts.
        
        Parameters:
        -----------
        binary_array : ndarray
            3D binary segmentation
        min_volume : int
            Minimum volume (voxels) to consider for removal
        min_sphericity : float
            Minimum sphericity (0-1) to consider for removal
            Objects with BOTH large volume AND high sphericity are removed
            
        Returns:
        --------
        filtered : ndarray
            Binary array with large spherical blobs removed
        """
        from scipy.ndimage import label
        
        if verbose:
            print("Filtering large spherical blobs...")
        
        # Label connected components
        labeled, num_features = n3d.label_objects(binary_array)
        
        if num_features == 0:
            return binary_array.copy()
        
        # Calculate volumes using bincount (very fast)
        volumes = np.bincount(labeled.ravel())
        
        # Calculate surface areas efficiently by counting exposed faces
        surface_areas = np.zeros(num_features + 1, dtype=np.int64)
        
        # Check each of 6 face directions (±x, ±y, ±z)
        # A voxel contributes to surface area if any neighbor is different
        for axis in range(3):
            for direction in [-1, 1]:
                # Pad with zeros only on the axis we're checking
                pad_width = [(1, 1) if i == axis else (0, 0) for i in range(3)]
                padded = np.pad(labeled, pad_width, mode='constant', constant_values=0)
                
                # Roll the padded array
                shifted = np.roll(padded, direction, axis=axis)
                
                # Extract the center region (original size) from shifted
                slices = [slice(1, -1) if i == axis else slice(None) for i in range(3)]
                shifted_cropped = shifted[tuple(slices)]
                
                # Find exposed faces
                exposed_faces = (labeled != shifted_cropped) & (labeled > 0)
                
                face_counts = np.bincount(labeled[exposed_faces], 
                                         minlength=num_features + 1)
                surface_areas += face_counts
        del padded
        
        # Calculate sphericity for each component
        # Sphericity = (surface area of sphere with same volume) / (actual surface area)
        # For a sphere: A = π^(1/3) * (6V)^(2/3)
        # Perfect sphere = 1.0, elongated objects < 1.0
        sphericities = np.zeros(num_features + 1)
        valid_mask = (volumes > 0) & (surface_areas > 0)
        
        # Ideal surface area for a sphere of this volume
        ideal_surface = np.pi**(1/3) * (6 * volumes[valid_mask])**(2/3)
        sphericities[valid_mask] = ideal_surface / surface_areas[valid_mask]
        
        # Identify components to remove: BOTH large AND spherical
        to_remove = (volumes >= min_volume) & (sphericities >= min_sphericity)
        
        if verbose:
            num_removed = np.sum(to_remove[1:])  # Exclude background label 0
            total_voxels_removed = np.sum(volumes[to_remove])
            
            if num_removed > 0:
                print(f"  Found {num_removed} large spherical blob(s) to remove:")
                removed_indices = np.where(to_remove)[0]
                for idx in removed_indices[1:5]:  # Show first few, skip background
                    if idx > 0:
                        print(f"    Blob {idx}: volume={volumes[idx]} voxels, "
                              f"sphericity={sphericities[idx]:.3f}")
                if num_removed > 4:
                    print(f"    ... and {num_removed - 4} more")
                print(f"  Total voxels removed: {total_voxels_removed}")
            else:
                print(f"  No large spherical blobs found (criteria: volume≥{min_volume}, "
                      f"sphericity≥{min_sphericity})")
        
        # Create output array, removing unwanted blobs
        keep_mask = ~to_remove[labeled]
        filtered = binary_array & keep_mask
        
        return filtered.astype(binary_array.dtype)


    def _get_sphere_mask(self, radius):
        """
        Get a cached sphere mask for the given radius
        This avoids recomputing the same sphere mask many times
        """
        # Round radius to nearest 0.5 to limit cache size
        cache_key = round(radius * 2) / 2
        
        if cache_key not in self._sphere_cache:
            r = max(1, int(np.ceil(cache_key)))
            
            # Create coordinate grids for a box
            size = 2 * r + 1
            center = r
            zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
            
            # Create sphere mask
            dist_sq = zz**2 + yy**2 + xx**2
            mask = dist_sq <= cache_key**2
            
            # Store the mask and its size info
            self._sphere_cache[cache_key] = {
                'mask': mask,
                'radius_int': r,
                'center': center
            }
        
        return self._sphere_cache[cache_key]


    def _draw_sphere_3d_cached(self, array, center, radius):
        """Draw a filled sphere using cached mask (much faster)"""
        sphere_data = self._get_sphere_mask(radius)
        mask = sphere_data['mask']
        r = sphere_data['radius_int']
        
        z, y, x = center
        
        # Bounding box in the array
        z_min = max(0, int(z - r))
        z_max = min(array.shape[0], int(z + r + 1))
        y_min = max(0, int(y - r))
        y_max = min(array.shape[1], int(y + r + 1))
        x_min = max(0, int(x - r))
        x_max = min(array.shape[2], int(x + r + 1))
        
        # Calculate actual slice sizes
        array_z_size = z_max - z_min
        array_y_size = y_max - y_min
        array_x_size = x_max - x_min
        
        # Skip if completely out of bounds
        if array_z_size <= 0 or array_y_size <= 0 or array_x_size <= 0:
            return
        
        # Calculate mask offset (where sphere center maps to in mask coords)
        # Mask center is at index r
        mask_z_start = max(0, r - int(z) + z_min)
        mask_y_start = max(0, r - int(y) + y_min)
        mask_x_start = max(0, r - int(x) + x_min)
        
        # Mask end is start + array size (ensure exact match)
        mask_z_end = mask_z_start + array_z_size
        mask_y_end = mask_y_start + array_y_size
        mask_x_end = mask_x_start + array_x_size
        
        # Clip mask if it goes beyond mask boundaries
        mask_z_end = min(mask_z_end, mask.shape[0])
        mask_y_end = min(mask_y_end, mask.shape[1])
        mask_x_end = min(mask_x_end, mask.shape[2])
        
        # Recalculate array slice to match actual mask slice
        actual_z_size = mask_z_end - mask_z_start
        actual_y_size = mask_y_end - mask_y_start
        actual_x_size = mask_x_end - mask_x_start
        
        z_max = z_min + actual_z_size
        y_max = y_min + actual_y_size
        x_max = x_min + actual_x_size
        
        # Now they should match!
        try:
            array[z_min:z_max, y_min:y_max, x_min:x_max] |= \
                mask[mask_z_start:mask_z_end, mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        except ValueError as e:
            # Debug info if it still fails
            print(f"WARNING: Sphere drawing mismatch at pos ({z:.1f},{y:.1f},{x:.1f}), radius {radius}")
            print(f"  Array slice: {array[z_min:z_max, y_min:y_max, x_min:x_max].shape}")
            print(f"  Mask slice: {mask[mask_z_start:mask_z_end, mask_y_start:mask_y_end, mask_x_start:mask_x_end].shape}")
            # Skip this sphere rather than crash
            pass


    def draw_vessel_lines_optimized(self, G, shape):
        """
        OPTIMIZED: Reconstruct vessel structure by drawing tapered cylinders
        Uses sphere caching for ~5-10x speedup
        """
        result = np.zeros(shape, dtype=np.uint8)
        
        # Draw cylinders between connected kernels
        for i, j in G.edges():
            pos_i = G.nodes[i]['pos']
            pos_j = G.nodes[j]['pos']
            radius_i = G.nodes[i]['radius']
            radius_j = G.nodes[j]['radius']
            
            # Draw tapered cylinder (using cached sphere method)
            self._draw_cylinder_3d_cached(result, pos_i, pos_j, radius_i, radius_j)
        
        # Also draw spheres at kernel centers to ensure continuity
        for node in G.nodes():
            pos = G.nodes[node]['pos']
            radius = G.nodes[node]['radius']
            self._draw_sphere_3d_cached(result, pos, radius)
        
        return result


    def _draw_cylinder_3d_cached(self, array, pos1, pos2, radius1, radius2):
        """
        Draw a tapered cylinder using cached sphere masks
        This is much faster than recomputing sphere masks each time
        """
        distance = np.linalg.norm(pos2 - pos1)
        if distance < 0.5:
            self._draw_sphere_3d_cached(array, pos1, max(radius1, radius2))
            return
        
        # Adaptive sampling: more samples for large radius changes
        radius_change = abs(radius2 - radius1)
        samples_per_unit = 2.0  # Default: 2 samples per voxel
        if radius_change > 2:
            samples_per_unit = 3.0  # More samples for tapered vessels
        
        num_samples = max(3, int(distance * samples_per_unit))
        t_values = np.linspace(0, 1, num_samples)
        
        # Interpolate and draw
        for t in t_values:
            pos = pos1 * (1 - t) + pos2 * t
            radius = radius1 * (1 - t) + radius2 * t
            self._draw_sphere_3d_cached(array, pos, radius)

    def select_kernel_points_topology(self, skeleton):
        """
        Topology-aware kernel selection.
        Keeps endpoints + branchpoints, and samples along chains between them.
        Prevents missing internal connections when subsampling.
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

        # Degree per voxel
        deg = dict(skel_graph.degree())

        # Critical nodes: endpoints (deg=1) or branchpoints (deg>=3)
        # Store endpoints and branchpoints separately to ensure preservation
        endpoints = {i for i, d in deg.items() if d == 1}
        branchpoints = {i for i, d in deg.items() if d >= 3}
        critical = endpoints | branchpoints

        kernels = set(critical)

        # Walk each chain starting from critical nodes
        visited_edges = set()

        for c_idx in critical:
            for nb in skel_graph.neighbors(c_idx):
                edge = tuple(sorted((c_idx, nb)))
                if edge in visited_edges:
                    continue

                # Start a chain
                chain = [c_idx, nb]
                visited_edges.add(edge)
                prev = c_idx
                cur = nb

                while cur not in critical:
                    # degree==2 node: continue straight
                    nbs = list(skel_graph.neighbors(cur))
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
                    edge2 = tuple(sorted((cur, nxt)))
                    if edge2 in visited_edges:
                        break
                    visited_edges.add(edge2)

                    chain.append(nxt)
                    prev, cur = cur, nxt

                # Now chain goes critical -> ... -> critical (or end)
                # Sample every kernel_spacing along the chain, but keep ends
                for k in chain[::self.kernel_spacing]:
                    kernels.add(k)
                kernels.add(chain[0])
                kernels.add(chain[-1])

        # CRITICAL FIX FOR ISSUE 2: Explicitly ensure ALL endpoints and branchpoints
        # are in the final kernel set, even if chain walking had any issues
        kernels.update(endpoints)
        kernels.update(branchpoints)

        # Return kernel coordinates
        kernel_coords = np.array([skeleton_coords[i] for i in kernels])
        return kernel_coords
        
    def _is_skeleton_endpoint(self, skeleton, pos, radius=3):
        """
        Determine if a skeleton point is an endpoint or internal node
        Endpoints have few neighbors, internal nodes are well-connected
        """
        z, y, x = pos
        shape = skeleton.shape
        
        # Check local neighborhood
        z_min = max(0, z - radius)
        z_max = min(shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(shape[2], x + radius + 1)
        
        local_skel = skeleton[z_min:z_max, y_min:y_max, x_min:x_max]
        local_coords = np.argwhere(local_skel)
        
        if len(local_coords) <= 1:
            return True  # Isolated point is an endpoint
        
        # Translate to global coordinates
        offset = np.array([z_min, y_min, x_min])
        global_coords = local_coords + offset
        
        # Find neighbors within small radius
        center = np.array([z, y, x])
        distances = np.linalg.norm(global_coords - center, axis=1)
        
        # Count neighbors within immediate vicinity (excluding self)
        neighbor_mask = (distances > 0.1) & (distances < radius)
        num_neighbors = np.sum(neighbor_mask)
        
        # Endpoint: has 1-2 neighbors (tip or along a thin path)
        # Internal/branch: has 3+ neighbors (well-connected)
        is_endpoint = num_neighbors <= 2
        
        return is_endpoint
    
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
        
        # Determine if this is an endpoint
        features['is_endpoint'] = self._is_skeleton_endpoint(skeleton, kernel_pos)
        
        # Local direction vector (principal direction of nearby skeleton points)
        features['direction'] = self._compute_local_direction(
            skeleton, kernel_pos, radius
        )
        
        # Position
        features['pos'] = np.array(kernel_pos)
        
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
    
    def compute_edge_features(self, feat_i, feat_j, skeleton):
        """Compute features for potential connection between two kernels"""
        features = {}
        
        # Euclidean distance
        pos_diff = feat_j['pos'] - feat_i['pos']
        features['distance'] = np.linalg.norm(pos_diff)
        
        # Radius similarity
        r_i, r_j = feat_i['radius'], feat_j['radius']
        features['radius_diff'] = abs(r_i - r_j)
        features['radius_ratio'] = min(r_i, r_j) / (max(r_i, r_j) + 1e-10)
        features['mean_radius'] = (r_i + r_j) / 2.0
        
        # Gap size relative to vessel radius
        features['gap_ratio'] = features['distance'] / (features['mean_radius'] + 1e-10)
        
        # Direction alignment
        direction_vec = pos_diff / (features['distance'] + 1e-10)
        
        # Alignment with both local directions
        align_i = abs(np.dot(feat_i['direction'], direction_vec))
        align_j = abs(np.dot(feat_j['direction'], direction_vec))
        features['alignment'] = (align_i + align_j) / 2.0
        
        # Smoothness: how well does connection align with both local directions
        features['smoothness'] = min(align_i, align_j)
        
        # Path support: count skeleton points along the path (only if skeleton provided)
        if skeleton is not None:
            features['path_support'] = self._count_skeleton_along_path(
                feat_i['pos'], feat_j['pos'], skeleton
            )
        else:
            features['path_support'] = 0.0
        
        # Density similarity
        features['density_diff'] = abs(feat_i['local_density'] - feat_j['local_density'])

        features['endpoint_count'] = 0
        if feat_j['is_endpoint']:
            features['endpoint_count'] += 1
        if feat_i['is_endpoint']:
            features['endpoint_count'] += 1
        
        return features
    
    def _count_skeleton_along_path(self, pos1, pos2, skeleton, num_samples=10):
        """Count how many skeleton points exist along the path"""
        t = np.linspace(0, 1, num_samples)
        path_points = pos1[:, None] * (1 - t) + pos2[:, None] * t
        
        count = 0
        for i in range(num_samples):
            coords = np.round(path_points[:, i]).astype(int)
            if (0 <= coords[0] < skeleton.shape[0] and
                0 <= coords[1] < skeleton.shape[1] and
                0 <= coords[2] < skeleton.shape[2]):
                if skeleton[tuple(coords)]:
                    count += 1
        
        return count / num_samples
    
    def build_skeleton_backbone(self, skeleton_points, kernel_features, skeleton):
        """
        Connect kernels to their true immediate neighbors along each continuous skeleton path.
        No distance caps. If skeleton is continuous, kernels WILL connect.
        """
        G = nx.Graph()
        for i, feat in enumerate(kernel_features):
            G.add_node(i, **feat)

        skeleton_coords = np.argwhere(skeleton)
        coord_to_idx = {tuple(c): i for i, c in enumerate(skeleton_coords)}

        # full 26-connected skeleton graph
        skel_graph = nx.Graph()
        nbr_offsets = [(dz, dy, dx)
                       for dz in (-1, 0, 1)
                       for dy in (-1, 0, 1)
                       for dx in (-1, 0, 1)
                       if not (dz == dy == dx == 0)]

        for i, c in enumerate(skeleton_coords):
            skel_graph.add_node(i, pos=c)
        for i, c in enumerate(skeleton_coords):
            cz, cy, cx = c
            for dz, dy, dx in nbr_offsets:
                nb = (cz + dz, cy + dy, cx + dx)
                j = coord_to_idx.get(nb)
                if j is not None and j > i:
                    skel_graph.add_edge(i, j)

        # map kernels into skeleton index space
        skel_idx_to_kernel = {}
        kernel_to_skel_idx = {}
        for k_id, k_pos in enumerate(skeleton_points):
            t = tuple(k_pos)
            if t in coord_to_idx:
                s_idx = coord_to_idx[t]
                kernel_to_skel_idx[k_id] = s_idx
                skel_idx_to_kernel[s_idx] = k_id

        visited_edges = set()

        for k_id, s_idx in kernel_to_skel_idx.items():
            for nb in skel_graph.neighbors(s_idx):
                e = tuple(sorted((s_idx, nb)))
                if e in visited_edges:
                    continue
                visited_edges.add(e)

                prev, cur = s_idx, nb
                steps = 1

                # walk until next kernel or stop
                while cur not in skel_idx_to_kernel:
                    nbs = list(skel_graph.neighbors(cur))
                    
                    # If this node has degree != 2, it should be a branchpoint or endpoint
                    # If it's not a kernel, something is wrong, but we should still try
                    # to walk through it to find the next kernel
                    if len(nbs) == 1:
                        # True endpoint with no kernel - this shouldn't happen but handle it
                        break
                    elif len(nbs) == 2:
                        # Normal degree-2 node, continue straight
                        nxt = nbs[0] if nbs[1] == prev else nbs[1]
                    else:
                        # Junction (degree >= 3) that's not a kernel - try to continue
                        # in a consistent direction. This is a rare case but we handle it.
                        # Find the neighbor that's not prev
                        candidates = [n for n in nbs if n != prev]
                        if not candidates:
                            break
                        # Pick the first available path
                        nxt = candidates[0]
                    
                    e2 = tuple(sorted((cur, nxt)))
                    if e2 in visited_edges:
                        break
                    visited_edges.add(e2)
                    prev, cur = cur, nxt
                    steps += 1
                    
                    # Safety check: don't walk forever
                    if steps > 10000:
                        break

                if cur in skel_idx_to_kernel:
                    j_id = skel_idx_to_kernel[cur]
                    if j_id != k_id and not G.has_edge(k_id, j_id):
                        edge_feat = self.compute_edge_features(
                            kernel_features[k_id],
                            kernel_features[j_id],
                            skeleton
                        )
                        edge_feat["skeleton_steps"] = steps
                        G.add_edge(k_id, j_id, **edge_feat)

        # This ensures ALL kernels that are neighbors in the skeleton are connected
        for k_id, s_idx in kernel_to_skel_idx.items():
            # Check all neighbors of this kernel in the skeleton
            for nb_s_idx in skel_graph.neighbors(s_idx):
                # If the neighbor is also a kernel, connect them
                if nb_s_idx in skel_idx_to_kernel:
                    j_id = skel_idx_to_kernel[nb_s_idx]
                    if j_id != k_id and not G.has_edge(k_id, j_id):
                        edge_feat = self.compute_edge_features(
                            kernel_features[k_id],
                            kernel_features[j_id],
                            skeleton
                        )
                        edge_feat["skeleton_steps"] = 1
                        G.add_edge(k_id, j_id, **edge_feat)

        return G
    
    def connect_endpoints_across_gaps(self, G, skeleton_points, kernel_features, skeleton):
        """
        Second stage: Let endpoints reach out to connect across gaps
        Optimized version using Union-Find for fast connectivity checks
        """
        from scipy.cluster.hierarchy import DisjointSet
        
        # Identify all endpoints
        endpoint_nodes = [i for i, feat in enumerate(kernel_features) if feat['is_endpoint']]
        
        if len(endpoint_nodes) == 0:
            return G
        
        # Initialize Union-Find with existing graph connections
        ds = DisjointSet(range(len(skeleton_points)))
        for u, v in G.edges():
            ds.merge(u, v)
        
        # Build KD-tree for all points
        tree = cKDTree(skeleton_points)
        
        for endpoint_idx in endpoint_nodes:
            feat_i = kernel_features[endpoint_idx]
            pos_i = skeleton_points[endpoint_idx]
            direction_i = feat_i['direction']
            
            # Use radius-aware connection distance
            if self.radius_aware_distance:
                local_radius = feat_i['radius']
                connection_dist = max(self.max_connection_distance, local_radius * 3)
            else:
                connection_dist = self.max_connection_distance
            
            # Find all points within connection distance
            nearby_indices = tree.query_ball_point(pos_i, connection_dist)
            
            for j in nearby_indices:
                if endpoint_idx == j:
                    continue
                
                # FAST connectivity check - O(1) amortized instead of O(V+E)
                if ds.connected(endpoint_idx, j):
                    continue
                
                feat_j = kernel_features[j]
                pos_j = skeleton_points[j]
                is_endpoint_j = feat_j['is_endpoint']
                
                # Check if they're in the same component (using union-find)
                same_component = ds.connected(endpoint_idx, j)
                
                # Check directionality
                to_target = pos_j - pos_i
                to_target_normalized = to_target / (np.linalg.norm(to_target) + 1e-10)
                direction_dot = np.dot(direction_i, to_target_normalized)
                
                # Compute edge features
                edge_feat = self.compute_edge_features(feat_i, feat_j, skeleton)
                
                # Decide based on component membership
                should_connect = False
                
                if same_component:
                    should_connect = True
                else:
                    # Different components - require STRONG evidence
                    if edge_feat['path_support'] > 0.5:
                        should_connect = True
                    elif direction_dot > 0.3 and edge_feat['radius_ratio'] > 0.5:
                        score = self.score_connection(edge_feat)
                        if score > self.score_thresh:
                            should_connect = True
                    elif edge_feat['radius_ratio'] > 0.7:
                        score = self.score_connection(edge_feat)
                        if score > self.score_thresh:
                            should_connect = True
                
                # Special check: if j is internal node, require alignment
                if should_connect and not is_endpoint_j:
                    if edge_feat['alignment'] < 0.5:
                        should_connect = False
                
                if should_connect:
                    G.add_edge(endpoint_idx, j, **edge_feat)
                    # Update union-find structure immediately
                    ds.merge(endpoint_idx, j)
        
        return G
    
    def score_connection(self, edge_features):
        """
        Scoring function for endpoint gap connections
        Used when endpoints reach out to bridge gaps
        """
        score = 0.0
        
        # Prefer similar radii (vessels maintain consistent width)
        score += edge_features['radius_ratio'] * 3.0
        
        # Prefer reasonable gap sizes relative to vessel radius
        if edge_features['gap_ratio'] < self.gap_tolerance:
            score += (self.gap_tolerance - edge_features['gap_ratio']) * 2.0
        else:
            # Penalize very large gaps
            score -= (edge_features['gap_ratio'] - self.gap_tolerance) * 1.0
        # Prefer similar local properties
        score -= edge_features['density_diff'] * 0.5
        
        # Prefer aligned directions (smooth connections)
        score += edge_features['alignment'] * 2.0
        score += edge_features['smoothness'] * 1.5
        
        # Bonus for any existing skeleton path support
        if edge_features['path_support'] > 0.3:
            score += 5.0  # Strong bonus for existing path
        

        
        return score
    
    def screen_noise_filaments(self, G):
        """
        Final stage: Screen entire connected filaments for noise
        Remove complete filaments that are likely noise based on their properties
        """
        components = list(nx.connected_components(G))
        
        if len(components) == 0:
            return G
        
        # Extract component features
        nodes_to_remove = []
        
        for component in components:
            positions = np.array([G.nodes[n]['pos'] for n in component])
            radii = [G.nodes[n]['radius'] for n in component]
            degrees = [G.degree(n) for n in component]
            
            # Component statistics
            size = len(component) * self.kernel_spacing
            mean_radius = np.mean(radii)
            max_radius = np.max(radii)
            avg_degree = np.mean(degrees)
            
            # Measure linearity using PCA
            if len(positions) > 2:
                centered = positions - positions.mean(axis=0)
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                # Ratio of largest to smallest eigenvalue indicates linearity
                linearity = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            else:
                linearity = 1.0
            
            # Measure elongation (max distance / mean deviation from center)
            if len(positions) > 1:
                mean_pos = positions.mean(axis=0)
                deviations = np.linalg.norm(positions - mean_pos, axis=1)
                mean_deviation = np.mean(deviations)
                
                # FAST APPROXIMATION: Use bounding box diagonal
                # This is O(n) instead of O(n²) and uses minimal memory
                bbox_min = positions.min(axis=0)
                bbox_max = positions.max(axis=0)
                max_dist = np.linalg.norm(bbox_max - bbox_min)
                
                elongation = max_dist / (mean_deviation + 1) if mean_deviation > 0 else max_dist
            else:
                elongation = 0
            
            # Decision: Remove this filament if it's noise
            is_noise = False
            
            # Very small components with no special features
            if size < self.min_component_size:
                # Keep if large radius (real vessel)
                #if max_radius < 3.0:
                    # Keep if linear arrangement
                 #   if linearity < 3.0:
                        # Keep if well connected
                  #      if avg_degree < 1.5:
                is_noise = True
            
            # Blob-like structures (not elongated, not linear)
            if elongation < 1.5 and linearity < 2.0:
                if size < 30 and max_radius < 5.0:
                    is_noise = True

            # Isolated single points
            if size == 1:
                if max_radius < 2.0:
                    is_noise = True

            
            if is_noise:
                nodes_to_remove.extend(component)
        
        # Remove noise filaments
        G.remove_nodes_from(nodes_to_remove)
        
        return G
    
    def draw_vessel_lines(self, G, shape):
        """Reconstruct vessel structure by drawing lines between connected kernels"""
        result = np.zeros(shape, dtype=np.uint8)
        
        for i, j in G.edges():
            pos_i = G.nodes[i]['pos']
            pos_j = G.nodes[j]['pos']
            
            # Draw line between kernels
            self._draw_line_3d(result, pos_i, pos_j)
        
        # Also mark kernel centers
        for node in G.nodes():
            pos = G.nodes[node]['pos']
            z, y, x = np.round(pos).astype(int)
            if (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
                result[z, y, x] = 1
        
        return result
    
    def _draw_line_3d(self, array, pos1, pos2, num_points=None):
        """Draw a line in 3D array between two points"""
        if num_points is None:
            num_points = int(np.linalg.norm(pos2 - pos1) * 2) + 1
        
        t = np.linspace(0, 1, num_points)
        line_points = pos1[:, None] * (1 - t) + pos2[:, None] * t
        
        for i in range(num_points):
            coords = np.round(line_points[:, i]).astype(int)
            if (0 <= coords[0] < array.shape[0] and
                0 <= coords[1] < array.shape[1] and
                0 <= coords[2] < array.shape[2]):
                array[tuple(coords)] = 1
    
    def denoise(self, binary_segmentation, verbose=True):
        """
        Main denoising pipeline
        
        Parameters:
        -----------
        binary_segmentation : ndarray
            3D binary array of vessel segmentation
        verbose : bool
            Print progress information
            
        Returns:
        --------
        denoised : ndarray
            Cleaned vessel segmentation
        """
        if verbose:
            print("Starting vessel denoising pipeline...")
            print(f"Input shape: {binary_segmentation.shape}")
        
        # Step 1: Remove very small objects (obvious noise)
        if verbose:
            print("Step 1: Removing small noise objects...")
        cleaned = remove_small_objects(
            binary_segmentation.astype(bool), 
            min_size=10
        )
        
        # Step 2: Skeletonize
        if verbose:
            print("Step 2: Computing skeleton...")

        skeleton = n3d.skeletonize(cleaned)
        if len(skeleton.shape) == 3 and skeleton.shape[0] != 1:
            skeleton = n3d.fill_holes_3d(skeleton)
            skeleton = n3d.skeletonize(skeleton)
        if self.spine_removal > 0:
            skeleton = n3d.remove_branches_new(skeleton, self.spine_removal)
            skeleton = n3d.dilate_3D(skeleton, 3, 3, 3)
            skeleton = n3d.skeletonize(skeleton)
        
        if verbose:
            print("Step 3: Computing distance transform...")
        distance_map = sdl.compute_distance_transform_distance(cleaned, fast_dil = True)
        
        # Step 3: Sample kernels along skeleton
        if verbose:
            print("Step 4: Sampling kernels along skeleton...")
        
        skeleton_points = np.argwhere(skeleton)

        # Topology-aware subsampling (safe)
        kernel_points = self.select_kernel_points_topology(skeleton)

        if verbose:
            print(f"  Extracted {len(kernel_points)} kernel points "
                  f"(topology-aware, spacing={self.kernel_spacing})")
        
        # Step 4: Extract features
        if verbose:
            print("Step 5: Extracting kernel features...")
        kernel_features = []
        for pt in kernel_points:
            feat = self.extract_kernel_features(skeleton, distance_map, pt)
            kernel_features.append(feat)
        
        if verbose:
            num_endpoints = sum(1 for f in kernel_features if f['is_endpoint'])
            num_internal = len(kernel_features) - num_endpoints
            print(f"  Identified {num_endpoints} endpoints, {num_internal} internal nodes")
        
        # Step 5: Build graph - Stage 1: Connect skeleton backbone
        if verbose:
            print("Step 6: Building skeleton backbone (all immediate neighbors)...")
        G = self.build_skeleton_backbone(kernel_points, kernel_features, skeleton)
        
        if verbose:
            num_components = nx.number_connected_components(G)
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            print(f"  Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"  Average degree: {avg_degree:.2f} (branch points have 3+)")
            print(f"  Connected components: {num_components}")
            
            # Check for isolated nodes after all passes
            isolated = [n for n in G.nodes() if G.degree(n) == 0]
            if len(isolated) > 0:
                print(f"  WARNING: {len(isolated)} isolated nodes remain (truly disconnected)")
            else:
                print(f"  ✓ All nodes connected to neighbors")
            
            # Check component sizes
            comp_sizes = [len(c) for c in nx.connected_components(G)]
            if len(comp_sizes) > 0:
                print(f"  Component sizes: min={min(comp_sizes)}, max={max(comp_sizes)}, mean={np.mean(comp_sizes):.1f}")
        
        # Step 6: Connect endpoints across gaps
        if verbose:
            print("Step 7: Connecting endpoints across gaps...")
        initial_edges = G.number_of_edges()
        G = self.connect_endpoints_across_gaps(G, kernel_points, kernel_features, skeleton)
        
        if verbose:
            new_edges = G.number_of_edges() - initial_edges
            print(f"  Added {new_edges} gap-bridging connections")
            num_components = nx.number_connected_components(G)
            print(f"  Components after bridging: {num_components}")
        
        # Step 7: Screen entire filaments for noise
        if verbose:
            print("Step 8: Screening noise filaments...")
        initial_nodes = G.number_of_nodes()
        G = self.screen_noise_filaments(G)
        
        if verbose:
            removed = initial_nodes - G.number_of_nodes()
            print(f"  Removed {removed} noise nodes")
            print(f"  Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Step 8: Reconstruct
        if verbose:
            print("Step 9: Reconstructing vessel structure...")
        result = self.draw_vessel_lines_optimized(G, binary_segmentation.shape)

        if self.blob_sphericity < 1 and self.blob_sphericity > 0:
            if verbose:
                print("Step 10: Filtering large spherical artifacts...")
            result = self.filter_large_spherical_blobs(
                result,
                min_volume=self.blob_volume,
                min_sphericity=self.blob_sphericity,  
                verbose=verbose
            )
        
        if verbose:
            print("Denoising complete!")
            print(f"Output voxels: {np.sum(result)} (input: {np.sum(binary_segmentation)})")
        
        return result


def trace(data, kernel_spacing = 1, max_distance = 20, min_component = 20, gap_tolerance = 5, blob_sphericity = 1.0, blob_volume = 200, spine_removal = 0, score_thresh = 2, xy_scale = 1, z_scale = 1):

    """Main function with user prompts"""
    
    # Convert to binary if needed
    if data.dtype != bool and data.dtype != np.uint8:
        print("Converting to binary...")
        data = (data > 0).astype(np.uint8)
    
    # Create denoiser
    denoiser = VesselDenoiser(
        kernel_spacing=kernel_spacing,
        max_connection_distance=max_distance,
        min_component_size=min_component,
        gap_tolerance=gap_tolerance,
        blob_sphericity = blob_sphericity,
        blob_volume = blob_volume,
        spine_removal = spine_removal,
        score_thresh = score_thresh,
        xy_scale = xy_scale,
        z_scale = z_scale
    )
    
    # Run denoising
    result = denoiser.denoise(data, verbose=True)
    
    return result


if __name__ == "__main__":
    
    print("Test area")