import numpy as np
from scipy.spatial import cKDTree
import warnings
from . import nettracer as n3d
from . import smart_dilate as sdl
warnings.filterwarnings('ignore')


class EndpointConnector:
    """
    Simple endpoint connector - finds skeleton endpoints and connects them
    if they're within a specified distance.
    """
    
    def __init__(self, connection_distance=20, spine_removal = 0):
        """
        Parameters:
        -----------
        connection_distance : float
            Maximum distance to connect two endpoints
        """
        self.connection_distance = connection_distance
        self._sphere_cache = {}  # Cache sphere masks for different radii
        self.spine_removal = spine_removal

    def _get_sphere_mask(self, radius):
        """Get a cached sphere mask for the given radius"""
        cache_key = round(radius * 2) / 2
        
        if cache_key not in self._sphere_cache:
            r = max(1, int(np.ceil(cache_key)))
            
            size = 2 * r + 1
            center = r
            zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
            
            dist_sq = zz**2 + yy**2 + xx**2
            mask = dist_sq <= cache_key**2
            
            self._sphere_cache[cache_key] = {
                'mask': mask,
                'radius_int': r,
                'center': center
            }
        
        return self._sphere_cache[cache_key]

    def _draw_sphere_3d_cached(self, array, center, radius):
        """Draw a filled sphere using cached mask"""
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
        
        if array_z_size <= 0 or array_y_size <= 0 or array_x_size <= 0:
            return
        
        # Calculate mask offset
        mask_z_start = max(0, r - int(z) + z_min)
        mask_y_start = max(0, r - int(y) + y_min)
        mask_x_start = max(0, r - int(x) + x_min)
        
        mask_z_end = mask_z_start + array_z_size
        mask_y_end = mask_y_start + array_y_size
        mask_x_end = mask_x_start + array_x_size
        
        mask_z_end = min(mask_z_end, mask.shape[0])
        mask_y_end = min(mask_y_end, mask.shape[1])
        mask_x_end = min(mask_x_end, mask.shape[2])
        
        actual_z_size = mask_z_end - mask_z_start
        actual_y_size = mask_y_end - mask_y_start
        actual_x_size = mask_x_end - mask_x_start
        
        z_max = z_min + actual_z_size
        y_max = y_min + actual_y_size
        x_max = x_min + actual_x_size
        
        try:
            array[z_min:z_max, y_min:y_max, x_min:x_max] |= \
                mask[mask_z_start:mask_z_end, mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        except ValueError:
            pass

    def _draw_cylinder_3d_cached(self, array, pos1, pos2, radius1, radius2):
        """Draw a tapered cylinder using cached sphere masks"""
        distance = np.linalg.norm(pos2 - pos1)
        if distance < 0.5:
            self._draw_sphere_3d_cached(array, pos1, max(radius1, radius2))
            return
        
        radius_change = abs(radius2 - radius1)
        samples_per_unit = 2.0
        if radius_change > 2:
            samples_per_unit = 3.0
        
        num_samples = max(3, int(distance * samples_per_unit))
        t_values = np.linspace(0, 1, num_samples)
        
        for t in t_values:
            pos = pos1 * (1 - t) + pos2 * t
            radius = radius1 * (1 - t) + radius2 * t
            self._draw_sphere_3d_cached(array, pos, radius)

    def _find_endpoints(self, skeleton):
        """
        Find skeleton endpoints by checking connectivity
        Endpoints have degree 1 (only one neighbor)
        """
        endpoints = []
        skeleton_coords = np.argwhere(skeleton)
        
        if len(skeleton_coords) == 0:
            return np.array([])
        
        # 26-connectivity offsets
        nbr_offsets = [(dz, dy, dx)
                       for dz in (-1, 0, 1)
                       for dy in (-1, 0, 1)
                       for dx in (-1, 0, 1)
                       if not (dz == dy == dx == 0)]
        
        for coord in skeleton_coords:
            z, y, x = coord
            
            # Count neighbors
            neighbor_count = 0
            for dz, dy, dx in nbr_offsets:
                nz, ny, nx = z + dz, y + dy, x + dx
                
                if (0 <= nz < skeleton.shape[0] and
                    0 <= ny < skeleton.shape[1] and
                    0 <= nx < skeleton.shape[2]):
                    if skeleton[nz, ny, nx]:
                        neighbor_count += 1
            
            # Endpoint has exactly 1 neighbor
            if neighbor_count == 1:
                endpoints.append(coord)
        
        return np.array(endpoints)

    def connect_endpoints(self, binary_image, verbose=True):
        """
        Main function: connect endpoints within specified distance
        
        Parameters:
        -----------
        binary_image : ndarray
            3D binary segmentation
        verbose : bool
            Print progress information
            
        Returns:
        --------
        result : ndarray
            Original image with endpoint connections drawn
        """
        if verbose:
            print(f"Starting endpoint connector...")
            print(f"Input shape: {binary_image.shape}")
        
        # Make a copy to modify
        result = binary_image.copy()
        
        # Compute skeleton
        if verbose:
            print("Computing skeleton...")
        skeleton = n3d.skeletonize(binary_image)
        if len(skeleton.shape) == 3 and skeleton.shape[0] != 1:
            skeleton = n3d.fill_holes_3d(skeleton)
            skeleton = n3d.skeletonize(skeleton)
        if self.spine_removal > 0:
            print(f"removing spines: {self.spine_removal}")
            skeleton = n3d.remove_branches_new(skeleton, self.spine_removal)
            skeleton = n3d.dilate_3D(skeleton, 3, 3, 3)
            skeleton = n3d.skeletonize(skeleton)


        # Compute distance transform (for radii)
        if verbose:
            print("Computing distance transform...")
        distance_map = sdl.compute_distance_transform_distance(binary_image, fast_dil = True)
        
        # Find endpoints
        if verbose:
            print("Finding skeleton endpoints...")
        endpoints = self._find_endpoints(skeleton)
        
        if len(endpoints) == 0:
            if verbose:
                print("No endpoints found!")
            return result
        
        if verbose:
            print(f"Found {len(endpoints)} endpoints")
        
        # Get radius at each endpoint
        endpoint_radii = []
        for ep in endpoints:
            radius = distance_map[tuple(ep)]
            endpoint_radii.append(radius)
        endpoint_radii = np.array(endpoint_radii)
        
        # Build KD-tree for fast distance queries
        if verbose:
            print(f"Connecting endpoints within {self.connection_distance} voxels...")
        tree = cKDTree(endpoints)
        
        # Find all pairs within connection distance
        connections_made = 0
        for i, ep1 in enumerate(endpoints):
            # Query all points within connection distance
            nearby_indices = tree.query_ball_point(ep1, self.connection_distance)
            
            for j in nearby_indices:
                if j <= i:  # Skip self and already processed pairs
                    continue
                
                ep2 = endpoints[j]
                radius1 = endpoint_radii[i]
                radius2 = endpoint_radii[j]
                
                # Draw tapered cylinder connection
                self._draw_cylinder_3d_cached(
                    result,
                    ep1.astype(float),
                    ep2.astype(float),
                    radius1,
                    radius2
                )
                connections_made += 1
        
        if verbose:
            print(f"Made {connections_made} connections")
            print(f"Done! Output voxels: {np.sum(result)} (input: {np.sum(binary_image)})")
        
        return result


def connect_endpoints(binary_image, connection_distance=20, spine_removal = 0, verbose=True):
    """
    Simple function to connect skeleton endpoints
    
    Parameters:
    -----------
    binary_image : ndarray
        3D binary segmentation
    connection_distance : float
        Maximum distance to connect endpoints
    verbose : bool
        Print progress
        
    Returns:
    --------
    result : ndarray
        Image with endpoint connections
    """
    # Convert to binary if needed
    if verbose:
        print("Converting to binary...")
    binary_image = (binary_image > 0).astype(np.uint8)
    
    # Create connector and run
    connector = EndpointConnector(connection_distance=connection_distance, spine_removal = spine_removal)
    result = connector.connect_endpoints(binary_image, verbose=verbose)
    
    return result


if __name__ == "__main__":
    print("Endpoint connector ready")