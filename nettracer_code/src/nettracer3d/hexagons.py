import numpy as np
from typing import Optional, Tuple


def generate_hexagonal_labels(
    side_length: float,
    dims: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
    xy_scale: float = 1.0,
    z_scale: float = 1.0,
    shape_3d: str = 'prism'
) -> np.ndarray:
    """
    Generate a labeled array of hexagons (2D) or hexagonal prisms/rhombic dodecahedrons (3D).
    
    Parameters:
    -----------
    side_length : float
        Base side length unit for hexagons and prisms.
    dims : tuple of int
        Output dimensions (Z, Y, X). For 2D case, Z should be 1.
    mask : np.ndarray, optional
        Boolean mask with same shape as dims. True values indicate regions 
        where hexagons should NOT be created.
    xy_scale : float, optional
        Scaling factor for XY dimensions (default: 1.0).
        Effective hexagon side length = side_length / xy_scale.
        Use this for anisotropic data where XY resolution differs from Z.
    z_scale : float, optional
        Scaling factor for Z dimension (default: 1.0).
        Effective prism height = side_length / z_scale.
        Use this for anisotropic data where Z resolution differs from XY.
    shape_3d : str, optional
        Shape to use for 3D tessellation (default: 'prism').
        Options:
        - 'prism': Hexagonal prisms (hexagons extruded in Z)
        - 'dodecahedron': Rhombic dodecahedrons (optimal 3D space fillers)
    
    Returns:
    --------
    np.ndarray
        Labeled array where each hexagon/prism/dodecahedron has a unique label starting from 1.
        Masked regions (if provided) will be labeled 0.
        For 2D case (dims[0]==1), returns shape (1, Y, X).
    """
    z_dim, y_dim, x_dim = dims
    is_2d = (z_dim == 1)
    
    # Apply scaling factors
    effective_xy_side = side_length / xy_scale
    effective_z_side = side_length / z_scale
    
    if is_2d:
        # Generate 2D hexagonal tessellation using EDT
        labels = _generate_2d_hexagons_edt(effective_xy_side, y_dim, x_dim, mask)
        # Expand to 3D with Z=1
        labels = labels[np.newaxis, :, :]
    else:
        if shape_3d == 'dodecahedron':
            # Generate 3D rhombic dodecahedron tessellation using EDT
            labels = _generate_3d_dodecahedrons_edt(effective_xy_side, effective_z_side, 
                                                   z_dim, y_dim, x_dim, mask)
        else:  # 'prism'
            # Generate 3D hexagonal prism tessellation (fast replication method)
            labels = _generate_3d_hexagonal_prisms_optimized(effective_xy_side, effective_z_side, 
                                                            z_dim, y_dim, x_dim)
            # Apply mask if provided (for prisms, we use old method)
            if mask is not None:
                labels[mask] = 0
                labels = _relabel_consecutive_fast(labels)
    
    return labels
    
    # Apply mask if provided (most efficient to do at the end)
    if mask is not None:
        labels[mask] = 0
        # Relabel consecutively starting from 1
        labels = _relabel_consecutive(labels)
    
    return labels


def _generate_2d_hexagons_edt(side_length: float, height: int, width: int, 
                              mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ultra-fast 2D hexagonal tessellation using Euclidean Distance Transform.
    O(n_pixels) complexity - much faster than KDTree for large images.
    
    Integrates masking directly into seed placement for optimal performance.
    """
    from scipy.ndimage import distance_transform_edt, label as nd_label
    
    # Hexagon geometry (flat-top orientation)
    hex_height = np.sqrt(3) * side_length
    col_spacing = 1.5 * side_length
    row_spacing = hex_height
    
    # Create sparse array to mark hexagon centers
    seeds = np.zeros((height, width), dtype=bool)
    
    max_cols = int(np.ceil(width / col_spacing)) + 2
    max_rows = int(np.ceil(height / row_spacing)) + 2
    
    # Place seed points at hexagon centers
    for r in range(-1, max_rows):
        for q in range(-1, max_cols):
            cx = q * col_spacing
            cy = r * row_spacing + (q % 2) * (row_spacing / 2)
            
            xi, yi = int(round(cx)), int(round(cy))
            if 0 <= xi < width and 0 <= yi < height:
                # Check mask if provided
                if mask is None or not mask[0, yi, xi]:
                    seeds[yi, xi] = True
    
    # Label seed points consecutively
    seed_labels, num_seeds = nd_label(seeds)
    
    # Use EDT to find nearest seed for each pixel
    _, indices = distance_transform_edt(
        seed_labels == 0,
        return_indices=True
    )
    
    # Look up labels from seed positions
    labels = seed_labels[indices[0], indices[1]]
    
    # Re-mask to clean up any edge leaks (very fast operation)
    if mask is not None:
        labels[mask[0]] = 0
    
    return labels.astype(np.int32)


def _generate_3d_dodecahedrons_edt(xy_side_length: float, z_side_length: float,
                                   depth: int, height: int, width: int,
                                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ultra-fast rhombic dodecahedron generation using Euclidean Distance Transform.
    O(n_voxels) complexity - much faster than KDTree!
    
    Uses FCC lattice with integrated masking for optimal performance.
    """
    from scipy.ndimage import distance_transform_edt, label as nd_label
    
    # FCC lattice spacing
    lattice_xy = xy_side_length * np.sqrt(2)
    lattice_z = z_side_length * np.sqrt(2)
    
    # Create sparse array to mark FCC lattice centers
    seeds = np.zeros((depth, height, width), dtype=bool)
    
    max_x = int(np.ceil(width / lattice_xy)) + 2
    max_y = int(np.ceil(height / lattice_xy)) + 2
    max_z = int(np.ceil(depth / lattice_z)) + 2
    
    # Place seed points at FCC lattice positions
    for i in range(-1, max_x):
        for j in range(-1, max_y):
            for k in range(-1, max_z):
                # FCC lattice = base cubic + 3 face centers
                positions = [
                    (i * lattice_xy, j * lattice_xy, k * lattice_z),
                    ((i + 0.5) * lattice_xy, (j + 0.5) * lattice_xy, k * lattice_z),
                    ((i + 0.5) * lattice_xy, j * lattice_xy, (k + 0.5) * lattice_z),
                    (i * lattice_xy, (j + 0.5) * lattice_xy, (k + 0.5) * lattice_z),
                ]
                
                for x, y, z in positions:
                    xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
                    if 0 <= xi < width and 0 <= yi < height and 0 <= zi < depth:
                        # Check mask if provided
                        if mask is None or not mask[zi, yi, xi]:
                            seeds[zi, yi, xi] = True
    
    # Label seed points consecutively (KEY OPTIMIZATION!)
    seed_labels, num_seeds = nd_label(seeds)
    
    # Use EDT to propagate labels to all voxels
    # This is the magic: O(n_voxels) instead of O(n_voxels × log(n_seeds))
    _, indices = distance_transform_edt(
        seed_labels == 0,
        return_indices=True,
        sampling=[1.0, 1.0, 1.0]  # Isotropic for now, can adjust for anisotropy
    )
    
    # Look up labels from seed positions
    labels = seed_labels[indices[0], indices[1], indices[2]]
    
    # Re-mask to clean up any edge leaks
    if mask is not None:
        labels[mask] = 0
    
    return labels.astype(np.int32)


def _generate_2d_hexagons_optimized(side_length: float, height: int, width: int) -> np.ndarray:
    """
    Optimized 2D hexagonal tessellation using KDTree for O(n_pixels * log(n_centers)) performance.
    """
    from scipy.spatial import cKDTree
    
    # For small images, use direct approach; for large, use KDTree
    n_pixels = height * width
    
    if n_pixels < 100000:  # Threshold for KDTree overhead
        return _generate_2d_hexagons_cube(side_length, height, width)
    
    # Hexagon geometry (flat-top orientation)
    hex_height = np.sqrt(3) * side_length
    col_spacing = 1.5 * side_length
    row_spacing = hex_height
    
    # Generate grid of hexagon centers
    max_cols = int(np.ceil(width / col_spacing)) + 2
    max_rows = int(np.ceil(height / row_spacing)) + 2
    
    centers = []
    
    for r in range(-1, max_rows):
        for q in range(-1, max_cols):
            # Calculate center position
            cx = q * col_spacing
            cy = r * row_spacing + (q % 2) * (row_spacing / 2)
            centers.append([cx, cy])
    
    centers = np.array(centers)
    
    # Build KDTree
    tree = cKDTree(centers)
    
    # Create pixel coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(height, dtype=np.float32),
                                     np.arange(width, dtype=np.float32),
                                     indexing='ij')
    
    # Stack into (n_pixels, 2) array
    pixel_coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
    
    # Query nearest center for each pixel (batch for memory efficiency)
    batch_size = 1000000
    indices_list = []
    
    for i in range(0, len(pixel_coords), batch_size):
        batch = pixel_coords[i:i+batch_size]
        _, batch_indices = tree.query(batch, k=1, workers=-1)  # Use all CPU cores
        indices_list.append(batch_indices)
    
    indices = np.concatenate(indices_list)
    
    # Reshape to 2D array
    labels = (indices + 1).reshape(height, width).astype(np.int32)
    
    return labels


def _generate_3d_hexagonal_prisms_optimized(xy_side_length: float, z_side_length: float,
                                           depth: int, height: int, width: int) -> np.ndarray:
    """
    Optimized 3D hexagonal prism generation using EDT for base pattern.
    Generate 2D pattern once using EDT, then replicate with layer offsets.
    """
    # Generate base 2D pattern using EDT (super fast!)
    base_pattern = _generate_2d_hexagons_edt(xy_side_length, height, width, mask=None)
    max_base_label = base_pattern.max()
    
    # Prism height
    prism_height = z_side_length
    
    labels = np.zeros((depth, height, width), dtype=np.int32)
    
    # Determine number of unique prism layers
    num_layers = int(np.ceil(depth / prism_height))
    
    # Assign labels by layer (simple replication with offset)
    for layer in range(num_layers):
        z_start = int(layer * prism_height)
        z_end = min(int((layer + 1) * prism_height), depth)
        
        # Same base pattern, different label offset per layer
        layer_labels = base_pattern + (layer * max_base_label)
        labels[z_start:z_end, :, :] = layer_labels
    
    return labels


def _generate_3d_dodecahedrons_optimized(xy_side_length: float, z_side_length: float,
                                        depth: int, height: int, width: int) -> np.ndarray:
    """
    Optimized rhombic dodecahedron generation using KDTree for fast nearest-neighbor lookup.
    This is MUCH faster - O(n_voxels * log(n_centers)) instead of O(n_voxels * n_centers).
    Uses parallel processing for additional speedup.
    """
    from scipy.spatial import cKDTree
    
    # FCC lattice spacing
    lattice_xy = xy_side_length * np.sqrt(2)
    lattice_z = z_side_length * np.sqrt(2)
    
    # Generate FCC lattice centers
    max_x = int(np.ceil(width / lattice_xy)) + 2
    max_y = int(np.ceil(height / lattice_xy)) + 2
    max_z = int(np.ceil(depth / lattice_z)) + 2
    
    centers = []
    
    # Generate FCC lattice points
    for i in range(-1, max_x):
        for j in range(-1, max_y):
            for k in range(-1, max_z):
                # Base cubic lattice
                centers.append([i * lattice_xy, j * lattice_xy, k * lattice_z])
                
                # FCC offsets (face centers)
                centers.append([(i + 0.5) * lattice_xy, (j + 0.5) * lattice_xy, k * lattice_z])
                centers.append([(i + 0.5) * lattice_xy, j * lattice_xy, (k + 0.5) * lattice_z])
                centers.append([i * lattice_xy, (j + 0.5) * lattice_xy, (k + 0.5) * lattice_z])
    
    centers = np.array(centers, dtype=np.float32)
    
    # Build KDTree for fast nearest-neighbor queries
    tree = cKDTree(centers)
    
    # Create coordinate grid for all voxels
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing='ij'
    )
    
    # Stack into (n_voxels, 3) array
    voxel_coords = np.stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], axis=1)
    
    # Query in batches to manage memory
    batch_size = 2000000  # Process 2M voxels at a time
    indices_list = []
    
    for i in range(0, len(voxel_coords), batch_size):
        batch = voxel_coords[i:i+batch_size]
        # Use parallel workers for speedup
        _, batch_indices = tree.query(batch, k=1, workers=-1)
        indices_list.append(batch_indices)
    
    indices = np.concatenate(indices_list)
    
    # Reshape to 3D array
    labels = (indices + 1).reshape(depth, height, width).astype(np.int32)
    
    return labels


def _generate_2d_hexagons(side_length: float, height: int, width: int) -> np.ndarray:
    """
    Generate 2D hexagonal tessellation using axial coordinates.
    Uses flat-top hexagon orientation.
    """
    # Hexagon geometry (flat-top orientation)
    # Width between opposite sides: 2 * side_length
    # Height: sqrt(3) * side_length
    hex_width = 2.0 * side_length
    hex_height = np.sqrt(3) * side_length
    
    # Column and row spacing
    col_spacing = 1.5 * side_length  # horizontal spacing between hex centers
    row_spacing = hex_height  # vertical spacing
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[0:height, 0:width]
    
    # Initialize label array
    labels = np.zeros((height, width), dtype=np.int32)
    
    # For each pixel, determine which hexagon it belongs to
    # We'll use axial coordinates (q, r) for hexagons
    
    # Estimate which hexagon a pixel might belong to
    q_float = x_coords / col_spacing
    r_float = (y_coords - (x_coords / col_spacing % 1) * row_spacing / 2) / row_spacing
    
    # Check neighboring hexagons to find the closest
    label_id = 1
    hex_map = {}
    
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            q = np.floor(q_float).astype(int) + dx
            r = np.floor(r_float).astype(int) + dy
            
            # Calculate hexagon center positions
            hex_x = q * col_spacing
            hex_y = r * row_spacing + (q % 2) * (row_spacing / 2)
            
            # Calculate distance from pixel to hexagon center
            dist = _hex_distance(x_coords, y_coords, hex_x, hex_y, side_length)
            
            # Update labels for pixels within this hexagon
            for q_val, r_val in zip(q.flat, r.flat):
                if (q_val, r_val) not in hex_map:
                    hex_map[(q_val, r_val)] = label_id
                    label_id += 1
    
    # Assign labels based on closest hexagon
    for q_val, r_val in hex_map.keys():
        q_mask = (np.floor(q_float) == q_val)
        r_mask = (np.floor(r_float) == r_val)
        
        # Refine by checking if pixel is actually in hexagon
        hex_x = q_val * col_spacing
        hex_y = r_val * row_spacing + (q_val % 2) * (row_spacing / 2)
        
        in_hex = _point_in_hexagon(x_coords, y_coords, hex_x, hex_y, side_length)
        labels[in_hex] = hex_map[(q_val, r_val)]
    
    # Better approach: use cube coordinates
    labels = _generate_2d_hexagons_cube(side_length, height, width)
    
    return labels


def _generate_2d_hexagons_cube(side_length: float, height: int, width: int) -> np.ndarray:
    """
    More efficient hexagon generation using cube coordinates and nearest hex search.
    """
    # Flat-top hexagon orientation
    hex_height = np.sqrt(3) * side_length
    hex_width = 2.0 * side_length
    
    labels = np.zeros((height, width), dtype=np.int32)
    
    # For each pixel, find the nearest hexagon center
    hex_map = {}
    label_counter = 1
    
    for y in range(height):
        for x in range(width):
            # Convert pixel to axial coordinates (approximate)
            q = (2./3. * x) / side_length
            r = (-1./3. * x + np.sqrt(3)/3. * y) / side_length
            
            # Round to nearest hexagon (cube coordinate rounding)
            q_int, r_int = _hex_round(q, r)
            
            # Get or create label for this hexagon
            if (q_int, r_int) not in hex_map:
                hex_map[(q_int, r_int)] = label_counter
                label_counter += 1
            
            labels[y, x] = hex_map[(q_int, r_int)]
    
    return labels


def _generate_3d_hexagonal_prisms(xy_side_length: float, z_side_length: float, 
                                   depth: int, height: int, width: int) -> np.ndarray:
    """
    Generate 3D hexagonal prism tessellation.
    Hexagons tile the XY plane, prisms extend in Z direction.
    
    Parameters:
    -----------
    xy_side_length : float
        Side length for base hexagons in XY plane
    z_side_length : float
        Height of prisms in Z direction
    """
    # Generate base 2D hexagonal pattern
    base_pattern = _generate_2d_hexagons_cube(xy_side_length, height, width)
    
    # Prism height from z_side_length
    prism_height = z_side_length
    
    labels = np.zeros((depth, height, width), dtype=np.int32)
    
    # Stack hexagonal layers
    label_offset = 0
    
    for z in range(depth):
        # Determine which prism layer we're in
        layer_idx = int(z / prism_height)
        
        # Each layer gets a new set of labels
        if z % prism_height == 0 and z > 0:
            label_offset = labels[z-1].max()
        
        labels[z] = base_pattern + label_offset
    
    return labels


def _hex_round(q: float, r: float) -> Tuple[int, int]:
    """
    Round fractional axial coordinates to nearest hexagon.
    Uses cube coordinate conversion for accurate rounding.
    """
    # Convert axial to cube coordinates
    x = q
    z = r
    y = -x - z
    
    # Round to integers
    rx = round(x)
    ry = round(y)
    rz = round(z)
    
    # Fix rounding errors (ensure rx + ry + rz = 0)
    x_diff = abs(rx - x)
    y_diff = abs(ry - y)
    z_diff = abs(rz - z)
    
    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry
    
    # Convert back to axial
    return int(rx), int(rz)


def _point_in_hexagon(x: np.ndarray, y: np.ndarray, cx: float, cy: float, 
                      side_length: float) -> np.ndarray:
    """
    Check if points (x, y) are inside a flat-top hexagon centered at (cx, cy).
    """
    # Translate to hexagon-centered coordinates
    dx = x - cx
    dy = y - cy
    
    # For flat-top hexagon, check against 6 edges
    # This is a simplified check using bounding regions
    h = np.sqrt(3) * side_length / 2
    
    # Rough bounding check
    in_hex = (np.abs(dx) <= side_length) & (np.abs(dy) <= h)
    
    return in_hex


def _hex_distance(x: np.ndarray, y: np.ndarray, cx: float, cy: float, 
                  side_length: float) -> np.ndarray:
    """
    Calculate distance from points to hexagon center.
    """
    return np.sqrt((x - cx)**2 + (y - cy)**2)


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """
    Fast relabeling using vectorized lookup table.
    About 10-100x faster than loop-based approach.
    0 is preserved (typically for background/mask).
    """
    # Get unique labels excluding 0
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    
    if len(unique_labels) == 0:
        return labels
    
    # Create lookup table (vectorized!)
    max_label = unique_labels.max()
    if max_label > 100000000:  # Avoid huge arrays
        # Fall back to slower method for pathological cases
        new_labels = np.zeros_like(labels)
        for new_id, old_id in enumerate(unique_labels, start=1):
            new_labels[labels == old_id] = new_id
        return new_labels
    
    # Create mapping using lookup table
    lookup = np.zeros(max_label + 1, dtype=labels.dtype)
    lookup[unique_labels] = np.arange(1, len(unique_labels) + 1, dtype=labels.dtype)
    
    # Apply mapping (single vectorized operation!)
    new_labels = lookup[labels]
    
    return new_labels

