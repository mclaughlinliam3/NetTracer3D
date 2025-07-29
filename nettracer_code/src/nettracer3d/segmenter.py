from sklearn.ensemble import RandomForestClassifier
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy import ndimage
import multiprocessing
from collections import defaultdict
from typing import List, Dict, Tuple, Any

class InteractiveSegmenter:
    def __init__(self, image_3d, use_gpu=False):
        self.image_3d = image_3d
        self.patterns = []

        self.use_gpu = False

        self.model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            max_depth=None
        )

        self.feature_cache = None
        self.lock = threading.Lock()
        self._currently_segmenting = None

        # Current position attributes
        self.current_z = None
        self.current_x = None
        self.current_y = None

        self.realtimechunks = None
        self.current_speed = False

        # Tracking if we're using 2d or 3d segs
        self.use_two = False
        self.two_slices = []
        self.speed = True
        self.cur_gpu = False
        self.prev_z = None
        self.previewing = False

        #  flags to track state
        self._currently_processing = False
        self._skip_next_update = False
        self._last_processed_slice = None
        self.mem_lock = False

        #Adjustable feature map params:
        self.sigmas = [1,2,4,8]
        self.windows = 10
        self.dogs = [(1, 2), (2, 4), (4, 8)]
        self.master_chunk = 49
        self.twod_chunk_size = 262144
        self.batch_amplifier = 1

        #Data when loading prev model:
        self.previous_foreground = None
        self.previous_background = None
        self.previous_z_fore = None
        self.previous_z_back = None

    def get_minimal_chunks_for_coordinates_cpu(self, coordinates_by_z):
        """
        Get minimal set of 2D chunks needed to cover the given coordinates
        Uses same chunking logic as create_2d_chunks() - CPU version
        """
        MAX_CHUNK_SIZE = self.twod_chunk_size
        needed_chunks = {}
        
        for z in coordinates_by_z:
            y_coords = [coord[0] for coord in coordinates_by_z[z]]
            x_coords = [coord[1] for coord in coordinates_by_z[z]]
            
            # Find bounding box of coordinates in this Z-slice
            y_min, y_max = min(y_coords), max(y_coords)
            x_min, x_max = min(x_coords), max(x_coords)
            
            # Create chunks using same logic as create_2d_chunks
            y_dim = self.image_3d.shape[1]
            x_dim = self.image_3d.shape[2]
            total_pixels = y_dim * x_dim
            
            if total_pixels <= MAX_CHUNK_SIZE:
                # Single chunk for entire Z slice
                needed_chunks[z] = [[z, 0, y_dim, 0, x_dim]]
            else:
                # Multiple chunks - find which ones contain our coordinates
                largest_dim = 'y' if y_dim >= x_dim else 'x'
                num_divisions = int(np.ceil(total_pixels / MAX_CHUNK_SIZE))
                
                chunks_for_z = []
                
                if largest_dim == 'y':
                    div_size = int(np.ceil(y_dim / num_divisions))
                    for i in range(0, y_dim, div_size):
                        end_i = min(i + div_size, y_dim)
                        # Check if this chunk contains any of our coordinates
                        if any(i <= y <= end_i-1 for y in y_coords):
                            chunks_for_z.append([z, i, end_i, 0, x_dim])
                else:
                    div_size = int(np.ceil(x_dim / num_divisions))
                    for i in range(0, x_dim, div_size):
                        end_i = min(i + div_size, x_dim)
                        # Check if this chunk contains any of our coordinates
                        if any(i <= x <= end_i-1 for x in x_coords):
                            chunks_for_z.append([z, 0, y_dim, i, end_i])
                
                needed_chunks[z] = chunks_for_z
        
        return needed_chunks

    def compute_features_for_chunk_2d_cpu(self, chunk_coords, speed):
        """
        Compute features for a 2D chunk - CPU version
        chunk_coords: [z, y_start, y_end, x_start, x_end]
        """
        z, y_start, y_end, x_start, x_end = chunk_coords
        
        # Extract 2D subarray for this chunk
        subarray_2d = self.image_3d[z, y_start:y_end, x_start:x_end]
        
        # Compute features for this chunk
        if speed:
            feature_map = self.compute_feature_maps_cpu_2d(image_2d=subarray_2d)
        else:
            feature_map = self.compute_deep_feature_maps_cpu_2d(image_2d=subarray_2d)
        
        return feature_map, (y_start, x_start)  # Return offset for coordinate mapping

    def compute_deep_feature_maps_cpu_2d(self, z=None, image_2d = None):
        """Vectorized detailed version with Gaussian gradient magnitudes, Laplacians, and largest Hessian eigenvalue for 2D images"""
        if z is None:
            z = self.image_3d.shape[0] // 2  # Use middle slice if not specified
        
        if image_2d is None:
            image_2d = self.image_3d[z, :, :]

        if image_2d.ndim == 3 and image_2d.shape[-1] == 3:
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(3):
                channel_features = self.compute_deep_feature_maps_cpu_2d(image_2d = image_2d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return np.concatenate(features_per_channel, axis=-1)
        
        
        # Calculate total number of features
        num_basic_features = 1 + len(self.sigmas) + len(self.dogs)  # original + gaussians + dogs
        num_gradient_features = len(self.sigmas)  # gradient magnitude for each sigma
        num_laplacian_features = len(self.sigmas)  # laplacian for each sigma
        num_hessian_features = len(self.sigmas) * 1  # 1 eigenvalue (largest) for each sigma
        
        total_features = num_basic_features + num_gradient_features + num_laplacian_features + num_hessian_features
        
        # Pre-allocate result array
        features = np.empty(image_2d.shape + (total_features,), dtype=image_2d.dtype)
        features[..., 0] = image_2d
        
        feature_idx = 1
        
        # Cache for Gaussian filters - only compute each sigma once
        gaussian_cache = {}
        
        # Compute all unique sigmas needed (from both sigmas and dogs)
        all_sigmas = set(self.sigmas)
        for s1, s2 in self.dogs:
            all_sigmas.add(s1)
            all_sigmas.add(s2)
        
        # Pre-compute all Gaussian filters
        for sigma in all_sigmas:
            gaussian_cache[sigma] = ndimage.gaussian_filter(image_2d, sigma)
        
        # Gaussian smoothing - use cached results
        for sigma in self.sigmas:
            features[..., feature_idx] = gaussian_cache[sigma]
            feature_idx += 1
        
        # Difference of Gaussians - use cached results
        for s1, s2 in self.dogs:
            features[..., feature_idx] = gaussian_cache[s1] - gaussian_cache[s2]
            feature_idx += 1
        
        # Gaussian gradient magnitudes for each sigma (vectorized, 2D version)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            gx = ndimage.sobel(gaussian_img, axis=1, mode='reflect')  # x direction
            gy = ndimage.sobel(gaussian_img, axis=0, mode='reflect')  # y direction
            features[..., feature_idx] = np.sqrt(gx**2 + gy**2)
            feature_idx += 1
        
        # Laplacian of Gaussian for each sigma (vectorized, 2D version)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            features[..., feature_idx] = ndimage.laplace(gaussian_img, mode='reflect')
            feature_idx += 1
        
        # Largest Hessian eigenvalue for each sigma (analytical 2D version)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            
            # Compute second derivatives (Hessian components) - all vectorized for 2D
            hxx = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 2], mode='reflect')
            hyy = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[2, 0], mode='reflect')
            hxy = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 1], mode='reflect')
            
            # Analytical eigenvalue computation for 2x2 symmetric matrices
            # For matrix [[hxx, hxy], [hxy, hyy]], eigenvalues are:
            # λ = (trace ± sqrt(trace² - 4*det)) / 2
            
            trace = hxx + hyy
            det = hxx * hyy - hxy * hxy
            
            # Calculate discriminant and ensure it's non-negative
            discriminant = trace * trace - 4 * det
            discriminant = np.maximum(discriminant, 0)  # Handle numerical errors
            
            sqrt_discriminant = np.sqrt(discriminant)
            
            # Calculate both eigenvalues
            eigenval1 = (trace + sqrt_discriminant) / 2
            eigenval2 = (trace - sqrt_discriminant) / 2
            
            # Take the larger eigenvalue (most positive/least negative)
            largest_eigenvalues = np.maximum(eigenval1, eigenval2)
            
            # Add the largest eigenvalue as a feature
            features[..., feature_idx] = largest_eigenvalues
            feature_idx += 1
        
        # Normalize only morphological features, keep intensity features raw
        intensity_features = features[..., :num_basic_features]  # original + gaussians + DoGs
        morphology_features = features[..., num_basic_features:]  # gradients + laplacians + eigenvalues

        # Normalize only morphological features
        morph_means = np.mean(morphology_features, axis=(0, 1), keepdims=True)
        morph_stds = np.std(morphology_features, axis=(0, 1), keepdims=True)
        morph_stds = np.where(morph_stds == 0, 1, morph_stds)
        morphology_features = (morphology_features - morph_means) / morph_stds

        # Recombine
        features = np.concatenate([intensity_features, morphology_features], axis=-1)
        
        return features


    def compute_feature_maps_cpu_2d(self, z=None, image_2d = None):
        """Compute feature maps for 2D images using CPU with caching optimization"""
        if image_2d is None:
            image_2d = self.image_3d[z, :, :]

        if image_2d.ndim == 3 and image_2d.shape[-1] == 3:
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(3):
                channel_features = self.compute_feature_maps_cpu_2d(image_2d = image_2d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return np.concatenate(features_per_channel, axis=-1)
        
        # Pre-allocate result array
        num_features = len(self.sigmas) + len(self.dogs) + 2  # +2 for original image + gradient
        features = np.empty(image_2d.shape + (num_features,), dtype=image_2d.dtype)
        
        # Include original image as first feature
        features[..., 0] = image_2d
        feature_idx = 1
        
        # Cache for Gaussian filters - only compute each sigma once
        gaussian_cache = {}
        
        # Compute all unique sigmas needed (from both sigmas and dogs)
        all_sigmas = set(self.sigmas)
        for s1, s2 in self.dogs:
            all_sigmas.add(s1)
            all_sigmas.add(s2)
        
        # Pre-compute all Gaussian filters
        for sigma in all_sigmas:
            gaussian_cache[sigma] = ndimage.gaussian_filter(image_2d, sigma)
        
        # Gaussian smoothing - use cached results
        for sigma in self.sigmas:
            features[..., feature_idx] = gaussian_cache[sigma]
            feature_idx += 1
        
        # Difference of Gaussians - use cached results
        for s1, s2 in self.dogs:
            features[..., feature_idx] = gaussian_cache[s1] - gaussian_cache[s2]
            feature_idx += 1
        
        # Gradient magnitude (2D version)
        gx = ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
        gy = ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
        features[..., feature_idx] = np.sqrt(gx**2 + gy**2)
        
        return features

    def compute_deep_feature_maps_cpu(self, image_3d=None):
        """Vectorized detailed version with Gaussian gradient magnitudes, Laplacians, and largest Hessian eigenvalue only"""
        if image_3d is None:
            image_3d = self.image_3d

        if image_3d.ndim == 4 and image_3d.shape[-1] == 3:
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(3):
                channel_features = self.compute_deep_feature_maps_cpu(image_3d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return np.concatenate(features_per_channel, axis=-1)
        
        
        # Calculate total number of features
        num_basic_features = 1 + len(self.sigmas) + len(self.dogs)  # original + gaussians + dogs
        num_gradient_features = len(self.sigmas)  # gradient magnitude for each sigma
        num_laplacian_features = len(self.sigmas)  # laplacian for each sigma
        num_hessian_features = len(self.sigmas) * 1  # 1 eigenvalue (largest) for each sigma
        
        total_features = num_basic_features + num_gradient_features + num_laplacian_features + num_hessian_features
        
        # Pre-allocate result array
        features = np.empty(image_3d.shape + (total_features,), dtype=image_3d.dtype)
        features[..., 0] = image_3d
        
        feature_idx = 1
        
        # Cache for Gaussian filters - only compute each sigma once
        gaussian_cache = {}
        
        # Compute all unique sigmas needed (from both sigmas and dogs)
        all_sigmas = set(self.sigmas)
        for s1, s2 in self.dogs:
            all_sigmas.add(s1)
            all_sigmas.add(s2)
        
        # Pre-compute all Gaussian filters
        for sigma in all_sigmas:
            gaussian_cache[sigma] = ndimage.gaussian_filter(image_3d, sigma)
        
        # Gaussian smoothing - use cached results
        for sigma in self.sigmas:
            features[..., feature_idx] = gaussian_cache[sigma]
            feature_idx += 1
        
        # Difference of Gaussians - use cached results
        for s1, s2 in self.dogs:
            features[..., feature_idx] = gaussian_cache[s1] - gaussian_cache[s2]
            feature_idx += 1
        
        # Gaussian gradient magnitudes for each sigma (vectorized)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            gx = ndimage.sobel(gaussian_img, axis=2, mode='reflect')
            gy = ndimage.sobel(gaussian_img, axis=1, mode='reflect')
            gz = ndimage.sobel(gaussian_img, axis=0, mode='reflect')
            features[..., feature_idx] = np.sqrt(gx**2 + gy**2 + gz**2)
            feature_idx += 1
        
        # Laplacian of Gaussian for each sigma (vectorized)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            features[..., feature_idx] = ndimage.laplace(gaussian_img, mode='reflect')
            feature_idx += 1
        
        # Largest Hessian eigenvalue for each sigma (fully vectorized)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            
            # Compute second derivatives (Hessian components) - all vectorized
            hxx = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 0, 2], mode='reflect')
            hyy = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 2, 0], mode='reflect')
            hzz = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[2, 0, 0], mode='reflect')
            hxy = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 1, 1], mode='reflect')
            hxz = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 0, 1], mode='reflect')
            hyz = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 1, 0], mode='reflect')
            
            # Vectorized eigenvalue computation using numpy broadcasting
            # Create arrays with shape (d0, d1, d2, 3, 3) for all Hessian matrices
            shape = image_3d.shape
            hessian_matrices = np.zeros(shape + (3, 3))
            
            # Fill the symmetric Hessian matrices
            hessian_matrices[..., 0, 0] = hxx
            hessian_matrices[..., 1, 1] = hyy
            hessian_matrices[..., 2, 2] = hzz
            hessian_matrices[..., 0, 1] = hessian_matrices[..., 1, 0] = hxy
            hessian_matrices[..., 0, 2] = hessian_matrices[..., 2, 0] = hxz
            hessian_matrices[..., 1, 2] = hessian_matrices[..., 2, 1] = hyz
            
            # Reshape for batch eigenvalue computation
            original_shape = hessian_matrices.shape[:-2]  # (d0, d1, d2)
            batch_size = np.prod(original_shape)
            hessian_batch = hessian_matrices.reshape(batch_size, 3, 3)
            
            # Compute eigenvalues for all matrices at once
            eigenvalues_batch = np.real(np.linalg.eigvals(hessian_batch))
            
            # Get only the largest eigenvalue for each matrix
            largest_eigenvalues = np.max(eigenvalues_batch, axis=1)
            
            # Reshape back to original spatial dimensions
            largest_eigenvalues = largest_eigenvalues.reshape(original_shape)
            
            # Add the largest eigenvalue as a feature
            features[..., feature_idx] = largest_eigenvalues
            feature_idx += 1
        

        # Normalize only morphological features, keep intensity features raw
        intensity_features = features[..., :num_basic_features]  # original + gaussians + DoGs
        morphology_features = features[..., num_basic_features:]  # gradients + laplacians + eigenvalues

        # Normalize only morphological features
        morph_means = np.mean(morphology_features, axis=(0,1,2), keepdims=True)
        morph_stds = np.std(morphology_features, axis=(0,1,2), keepdims=True)
        morph_stds = np.where(morph_stds == 0, 1, morph_stds)
        morphology_features = (morphology_features - morph_means) / morph_stds

        # Recombine
        features = np.concatenate([intensity_features, morphology_features], axis=-1)
        
        return features

    def compute_deep_feature_maps_cpu_smaller(self, image_3d=None): #smaller
        """Optimized version using determinant instead of full eigenvalue computation. Currently not in use anywhere"""
        if image_3d is None:
            image_3d = self.image_3d
        
        # Calculate total number of features (using determinant instead of 3 eigenvalues)
        num_basic_features = 1 + len(self.sigmas) + len(self.dogs)
        num_gradient_features = len(self.sigmas)
        num_laplacian_features = len(self.sigmas)
        num_hessian_features = len(self.sigmas) * 3  # determinant + trace + frobenius norm
        
        total_features = num_basic_features + num_gradient_features + num_laplacian_features + num_hessian_features
        
        # Pre-allocate result array
        features = np.empty(image_3d.shape + (total_features,), dtype=image_3d.dtype)
        features[..., 0] = image_3d
        
        feature_idx = 1
        
        # Cache for Gaussian filters
        gaussian_cache = {}
        all_sigmas = set(self.sigmas)
        for s1, s2 in self.dogs:
            all_sigmas.add(s1)
            all_sigmas.add(s2)
        
        # Pre-compute all Gaussian filters
        for sigma in all_sigmas:
            gaussian_cache[sigma] = ndimage.gaussian_filter(image_3d, sigma)
        
        # Gaussian smoothing
        for sigma in self.sigmas:
            features[..., feature_idx] = gaussian_cache[sigma]
            feature_idx += 1
        
        # Difference of Gaussians
        for s1, s2 in self.dogs:
            features[..., feature_idx] = gaussian_cache[s1] - gaussian_cache[s2]
            feature_idx += 1
        
        # Gaussian gradient magnitudes
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            gx = ndimage.sobel(gaussian_img, axis=2, mode='reflect')
            gy = ndimage.sobel(gaussian_img, axis=1, mode='reflect')
            gz = ndimage.sobel(gaussian_img, axis=0, mode='reflect')
            features[..., feature_idx] = np.sqrt(gx**2 + gy**2 + gz**2)
            feature_idx += 1
        
        # Laplacian of Gaussian
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            features[..., feature_idx] = ndimage.laplace(gaussian_img, mode='reflect')
            feature_idx += 1
        
        # Hessian-based features (much faster than full eigenvalue computation)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            
            # Compute second derivatives
            hxx = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 0, 2], mode='reflect')
            hyy = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 2, 0], mode='reflect')
            hzz = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[2, 0, 0], mode='reflect')
            hxy = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 1, 1], mode='reflect')
            hxz = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 0, 1], mode='reflect')
            hyz = ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 1, 0], mode='reflect')
            
            # Hessian determinant (captures overall curvature)
            determinant = (hxx * (hyy * hzz - hyz**2) - 
                          hxy * (hxy * hzz - hxz * hyz) + 
                          hxz * (hxy * hyz - hyy * hxz))
            features[..., feature_idx] = determinant
            feature_idx += 1
            
            # Hessian trace (sum of eigenvalues)
            trace = hxx + hyy + hzz
            features[..., feature_idx] = trace
            feature_idx += 1
            
            # Frobenius norm (overall curvature magnitude)
            frobenius_norm = np.sqrt(hxx**2 + hyy**2 + hzz**2 + 2*(hxy**2 + hxz**2 + hyz**2))
            features[..., feature_idx] = frobenius_norm
            feature_idx += 1

        """
        # Normalize features: zero-mean, unit variance per feature band
        # Compute mean and std across spatial dimensions (0,1,2), keeping feature dimension
        feature_means = np.mean(features, axis=(0, 1, 2), keepdims=True)
        feature_stds = np.std(features, axis=(0, 1, 2), keepdims=True)
        
        # Avoid division by zero for constant features
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)
        
        # Normalize in-place for memory efficiency
        features = (features - feature_means) / feature_stds
        """
        # Normalize only morphological features, keep intensity features raw
        intensity_features = features[..., :num_basic_features]  # original + gaussians + DoGs
        morphology_features = features[..., num_basic_features:]  # gradients + laplacians + eigenvalues

        # Normalize only morphological features
        morph_means = np.mean(morphology_features, axis=(0,1,2), keepdims=True)
        morph_stds = np.std(morphology_features, axis=(0,1,2), keepdims=True)
        morph_stds = np.where(morph_stds == 0, 1, morph_stds)
        morphology_features = (morphology_features - morph_means) / morph_stds

        # Recombine
        features = np.concatenate([intensity_features, morphology_features], axis=-1)

        return features


    def compute_feature_maps_cpu(self, image_3d=None): #lil
        """Optimized version that caches Gaussian filters to avoid redundant computation"""
        if image_3d is None:
            image_3d = self.image_3d

        if image_3d.ndim == 4 and image_3d.shape[-1] == 3:
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(3):
                channel_features = self.compute_feature_maps_cpu(image_3d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return np.concatenate(features_per_channel, axis=-1)
        
        # Pre-allocate result array
        num_features = len(self.sigmas) + len(self.dogs) + 2
        features = np.empty(image_3d.shape + (num_features,), dtype=image_3d.dtype)
        features[..., 0] = image_3d
        
        feature_idx = 1
        
        # Cache for Gaussian filters - only compute each sigma once
        gaussian_cache = {}
        
        # Compute all unique sigmas needed (from both sigmas and dogs)
        all_sigmas = set(self.sigmas)
        for s1, s2 in self.dogs:
            all_sigmas.add(s1)
            all_sigmas.add(s2)
        
        # Pre-compute all Gaussian filters
        for sigma in all_sigmas:
            gaussian_cache[sigma] = ndimage.gaussian_filter(image_3d, sigma)
        
        # Gaussian smoothing - use cached results
        for sigma in self.sigmas:
            features[..., feature_idx] = gaussian_cache[sigma]
            feature_idx += 1
        
        # Difference of Gaussians - use cached results
        for s1, s2 in self.dogs:
            features[..., feature_idx] = gaussian_cache[s1] - gaussian_cache[s2]
            feature_idx += 1
        
        # Gradient magnitude
        gx = ndimage.sobel(image_3d, axis=2, mode='reflect')
        gy = ndimage.sobel(image_3d, axis=1, mode='reflect')
        gz = ndimage.sobel(image_3d, axis=0, mode='reflect')
        features[..., feature_idx] = np.sqrt(gx**2 + gy**2 + gz**2)

        return features

    def organize_by_z(self, coordinates):
        """
        Organizes a list of [z, y, x] coordinates into a dictionary of [y, x] coordinates grouped by z-value.
        
        Args:
            coordinates: List of [z, y, x] coordinate lists
            
        Returns:
            Dictionary with z-values as keys and lists of corresponding [y, x] coordinates as values
        """
        z_dict = defaultdict(list)

        for z, y, x in coordinates:
            z_dict[z].append((y, x))

        
        return dict(z_dict)  # Convert back to regular dict

    def process_chunk(self, chunk_coords):
        """Updated process_chunk with chunked 2D feature computation"""
        
        if self.realtimechunks is None:
            # 3D processing (unchanged)
            z_min, z_max = chunk_coords[0], chunk_coords[1]
            y_min, y_max = chunk_coords[2], chunk_coords[3]
            x_min, x_max = chunk_coords[4], chunk_coords[5]

            z_range = np.arange(z_min, z_max)
            y_range = np.arange(y_min, y_max)
            x_range = np.arange(x_min, x_max)
            
            z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, x_range, indexing='ij')
            chunk_coords_array = np.column_stack([
                z_grid.ravel(), y_grid.ravel(), x_grid.ravel()
            ])
            
            # Extract subarray
            subarray = self.image_3d[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
            
            # Compute features for entire subarray
            if self.speed:
                feature_map = self.compute_feature_maps_cpu(subarray)
            else:
                feature_map = self.compute_deep_feature_maps_cpu(subarray)
            
            # Vectorized feature extraction
            local_coords = chunk_coords_array - np.array([z_min, y_min, x_min])
            features = feature_map[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
            
            # Vectorized predictions
            predictions = self.model.predict(features)
            predictions = np.array(predictions, dtype=bool)
            
            # Use boolean indexing to separate coordinates
            foreground_coords = chunk_coords_array[predictions]
            background_coords = chunk_coords_array[~predictions]
            
            # Convert to sets
            foreground = set(map(tuple, foreground_coords))
            background = set(map(tuple, background_coords))
            
            return foreground, background
        
        else:
            # 2D processing - compute features for chunk only
            chunk_coords_array = np.array(chunk_coords)
            z_coords = chunk_coords_array[:, 0]
            y_coords = chunk_coords_array[:, 1]
            x_coords = chunk_coords_array[:, 2]
            
            z = int(np.unique(z_coords)[0])  # All coordinates should have same Z
            
            # Get chunk bounds
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            
            # Expand bounds slightly to ensure we capture the chunk properly
            y_min = max(0, y_min)
            x_min = max(0, x_min)
            y_max = min(self.image_3d.shape[1], y_max + 1)
            x_max = min(self.image_3d.shape[2], x_max + 1)
            
            # Extract 2D subarray for this chunk
            subarray_2d = self.image_3d[z, y_min:y_max, x_min:x_max]
            
            # Compute features for just this chunk
            if self.speed:
                feature_map = self.compute_feature_maps_cpu_2d(image_2d=subarray_2d)
            else:
                feature_map = self.compute_deep_feature_maps_cpu_2d(image_2d=subarray_2d)
            
            # Convert global coordinates to local chunk coordinates
            local_y_coords = y_coords - y_min
            local_x_coords = x_coords - x_min
            
            # Extract features using local coordinates
            features = feature_map[local_y_coords, local_x_coords]
            
            # Vectorized predictions
            predictions = self.model.predict(features)
            predictions = np.array(predictions, dtype=bool)
            
            # Use boolean indexing to separate coordinates
            foreground_coords = chunk_coords_array[predictions]
            background_coords = chunk_coords_array[~predictions]
            
            # Convert to sets
            foreground = set(map(tuple, foreground_coords))
            background = set(map(tuple, background_coords))
            
            return foreground, background

    def twodim_coords(self, y_dim, x_dim, z, chunk_size = None, subrange = None):

        if subrange is None:
            y_coords, x_coords = np.meshgrid(
                np.arange(y_dim),
                np.arange(x_dim),
                indexing='ij'
            )
        
            slice_coords = np.column_stack((
                np.full(chunk_size, z),
                y_coords.ravel(),
                x_coords.ravel()
            ))

        elif subrange[0] == 'y':

            y_subrange = np.arange(subrange[1], subrange[2])

            # Create meshgrid for this subchunk
            y_sub, x_sub = np.meshgrid(
                y_subrange,
                np.arange(x_dim),
                indexing='ij'
            )
            
            # Create coordinates for this subchunk
            subchunk_size = len(y_subrange) * x_dim
            slice_coords = np.column_stack((
                np.full(subchunk_size, z),
                y_sub.ravel(),
                x_sub.ravel()
            ))

        elif subrange[0] == 'x':

            x_subrange = np.arange(subrange[1], subrange[2])
            
            # Create meshgrid for this subchunk
            y_sub, x_sub = np.meshgrid(
                np.arange(y_dim),
                x_subrange,
                indexing='ij'
            )
            
            # Create coordinates for this subchunk
            subchunk_size = y_dim * len(x_subrange)
            slice_coords = np.column_stack((
                np.full(subchunk_size, z),
                y_sub.ravel(),
                x_sub.ravel()
            ))



        return list(map(tuple, slice_coords))
        


    def segment_volume(self, array, chunk_size=None, gpu=False):
        """
        Process chunks in batches equal to CPU cores for optimal GIL handling
        """
        
        self.realtimechunks = None
        chunk_size = self.master_chunk

        def create_2d_chunks():
            """Same as your existing implementation"""
            MAX_CHUNK_SIZE = self.twod_chunk_size
            chunks = []
            
            for z in range(self.image_3d.shape[0]):
                y_dim = self.image_3d.shape[1]
                x_dim = self.image_3d.shape[2]
                total_pixels = y_dim * x_dim
                
                if total_pixels <= MAX_CHUNK_SIZE:
                    chunks.append([y_dim, x_dim, z, total_pixels, None])
                else:
                    largest_dim = 'y' if y_dim >= x_dim else 'x'
                    num_divisions = int(np.ceil(total_pixels / MAX_CHUNK_SIZE))
                    
                    if largest_dim == 'y':
                        div_size = int(np.ceil(y_dim / num_divisions))
                        for i in range(0, y_dim, div_size):
                            end_i = min(i + div_size, y_dim)
                            chunks.append([y_dim, x_dim, z, None, ['y', i, end_i]])
                    else:
                        div_size = int(np.ceil(x_dim / num_divisions))
                        for i in range(0, x_dim, div_size):
                            end_i = min(i + div_size, x_dim)
                            chunks.append([y_dim, x_dim, z, None, ['x', i, end_i]])
            
            return chunks

        print("Chunking data...")
        
        if not self.use_two:
            # Create smaller chunks for better load balancing
            if chunk_size is None:
                total_cores = multiprocessing.cpu_count()
                total_volume = np.prod(self.image_3d.shape)
                target_volume_per_chunk = total_volume / (total_cores * 4)  # 4x more chunks
                
                chunk_size = int(np.cbrt(target_volume_per_chunk))
                chunk_size = max(16, min(chunk_size, min(self.image_3d.shape) // 2))
                chunk_size = ((chunk_size + 7) // 16) * 16
            
            z_chunks = (self.image_3d.shape[0] + chunk_size - 1) // chunk_size
            y_chunks = (self.image_3d.shape[1] + chunk_size - 1) // chunk_size
            x_chunks = (self.image_3d.shape[2] + chunk_size - 1) // chunk_size
            
            chunk_starts = np.array(np.meshgrid(
                np.arange(z_chunks) * chunk_size,
                np.arange(y_chunks) * chunk_size,
                np.arange(x_chunks) * chunk_size,
                indexing='ij'
            )).reshape(3, -1).T
            
            chunks = []
            for z_start, y_start, x_start in chunk_starts:
                z_end = min(z_start + chunk_size, self.image_3d.shape[0])
                y_end = min(y_start + chunk_size, self.image_3d.shape[1])
                x_end = min(x_start + chunk_size, self.image_3d.shape[2])
                coords = [z_start, z_end, y_start, y_end, x_start, x_end]
                chunks.append(coords)
        else:
            chunks = create_2d_chunks()

        print("Processing chunks in batches...")
        
        # Process chunks in batches equal to CPU count
        max_workers = self.batch_amplifier * multiprocessing.cpu_count()
        batch_size = max_workers  # One batch per core
        total_processed = 0
        
        # Configure sklearn for maximum parallelism
        if hasattr(self.model, 'n_jobs'):
            original_n_jobs = self.model.n_jobs
            self.model.n_jobs = -1  # Use all cores for sklearn prediction
        
        try:
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                chunk_batch = chunks[batch_start:batch_end]
                
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # PHASE 1: Extract features in parallel (releases GIL)
                batch_results = []
                
                with ThreadPoolExecutor(max_workers=len(chunk_batch)) as executor:
                    futures = []
                    for chunk in chunk_batch:
                        future = executor.submit(self.extract_chunk_features, chunk)
                        futures.append(future)
                    
                    # Collect feature results
                    for future in futures:
                        features, coords = future.result()
                        if len(features) > 0:
                            batch_results.append((features, coords))
                
                # PHASE 2: Batch predict with sklearn's parallelism (no GIL issues)
                if batch_results:
                    # Combine all features from this batch
                    all_batch_features = np.vstack([result[0] for result in batch_results])
                    all_batch_coords = np.vstack([result[1] for result in batch_results])
                    
                    # Single prediction call using sklearn's internal parallelism
                    predictions = self.model.predict(all_batch_features)
                    predictions = np.array(predictions, dtype=bool)
                    
                    # Apply predictions to array
                    foreground_coords = all_batch_coords[predictions]
                    if len(foreground_coords) > 0:
                        z_coords, y_coords, x_coords = foreground_coords[:, 0], foreground_coords[:, 1], foreground_coords[:, 2]
                        array[z_coords, y_coords, x_coords] = 255
                    
                    # Clean up batch data for memory management
                    del all_batch_features, all_batch_coords, predictions, foreground_coords
                    
                total_processed += len(chunk_batch)
                print(f"Completed {total_processed}/{len(chunks)} chunks")
        
        finally:
            # Restore original sklearn settings
            if hasattr(self.model, 'n_jobs'):
                self.model.n_jobs = original_n_jobs
        
        return array

    def extract_chunk_features(self, chunk_coords):
        """
        Extract features for a single chunk without prediction
        Designed to release GIL effectively
        """
        
        if self.previewing or not self.use_two:
            if self.realtimechunks is None:
                z_min, z_max = chunk_coords[0], chunk_coords[1]
                y_min, y_max = chunk_coords[2], chunk_coords[3]
                x_min, x_max = chunk_coords[4], chunk_coords[5]

                # Vectorized coordinate generation (releases GIL)
                z_range = np.arange(z_min, z_max)
                y_range = np.arange(y_min, y_max)
                x_range = np.arange(x_min, x_max)
                
                z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, x_range, indexing='ij')
                chunk_coords_array = np.column_stack([
                    z_grid.ravel(), y_grid.ravel(), x_grid.ravel()
                ])
            else:
                chunk_coords_array = np.array(chunk_coords)
                z_coords, y_coords, x_coords = chunk_coords_array[:, 0], chunk_coords_array[:, 1], chunk_coords_array[:, 2]
                z_min, z_max = z_coords.min(), z_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()

            # Extract subarray and compute features (releases GIL)
            subarray = self.image_3d[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
            
            if self.speed:
                feature_map = self.compute_feature_maps_cpu(subarray)
            else:
                feature_map = self.compute_deep_feature_maps_cpu(subarray)
            
            # Vectorized feature extraction (releases GIL)
            local_coords = chunk_coords_array - np.array([z_min, y_min, x_min])
            features = feature_map[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
            
            return features, chunk_coords_array
        
        else:
            # Handle 2D case
            chunk_coords_list = self.twodim_coords(chunk_coords[0], chunk_coords[1], 
                                                 chunk_coords[2], chunk_coords[3], chunk_coords[4])
            chunk_coords_by_z = self.organize_by_z(chunk_coords_list)
            
            all_features = []
            all_coords = []
            
            for z, coords in chunk_coords_by_z.items():
                coords_array = np.array(coords)
                
                # Get features for this z-slice
                features_slice = self.get_feature_map_slice(z, self.speed, self.cur_gpu)
                features = features_slice[coords_array[:, 0], coords_array[:, 1]]

                
                # Convert to 3D coordinates
                coords_3d = np.column_stack([
                    np.full(len(coords_array), z),
                    coords_array[:, 0],
                    coords_array[:, 1]
                ])
                
                all_features.append(features)
                all_coords.append(coords_3d)
            
            if all_features:
                return np.vstack(all_features), np.vstack(all_coords)
            else:
                return np.array([]), np.array([])
                
    def update_position(self, z=None, x=None, y=None):
        """Update current position for chunk prioritization with safeguards"""
        
        # Check if we should skip this update
        if hasattr(self, '_skip_next_update') and self._skip_next_update:
            self._skip_next_update = False
            return
        
        # Store the previous z-position if not set
        if not hasattr(self, 'prev_z') or self.prev_z is None:
            self.prev_z = z
        
        # Check if currently processing - if so, only update position but don't trigger map_slice changes
        if hasattr(self, '_currently_processing') and self._currently_processing:
            self.current_z = z
            self.current_x = x
            self.current_y = y
            self.prev_z = z
            return
        
        # Update current positions
        self.current_z = z
        self.current_x = x
        self.current_y = y
        
        # Only clear map_slice if z changes and we're not already generating a new one
        if self.current_z != self.prev_z:

            self._currently_segmenting = None
        
        # Update previous z
        self.prev_z = z


    def get_realtime_chunks(self, chunk_size = 49):

        # Determine if we need to chunk XY planes
        small_dims = (self.image_3d.shape[1] <= chunk_size and 
                     self.image_3d.shape[2] <= chunk_size)
        few_z = self.image_3d.shape[0] <= 100  # arbitrary threshold
        
        # If small enough, each Z is one chunk
        if small_dims and few_z:
            chunk_size_xy = max(self.image_3d.shape[1], self.image_3d.shape[2])
        else:
            chunk_size_xy = chunk_size
        
        # Calculate chunks for XY plane
        y_chunks = (self.image_3d.shape[1] + chunk_size_xy - 1) // chunk_size_xy
        x_chunks = (self.image_3d.shape[2] + chunk_size_xy - 1) // chunk_size_xy
        
        # Populate chunk dictionary
        chunk_dict = {}
        
        # Create chunks for each Z plane
        for z in range(self.image_3d.shape[0]):
            if small_dims:
                
                chunk_dict[(z, 0, 0)] = {
                    'coords': [0, self.image_3d.shape[1], 0, self.image_3d.shape[2]],
                    'processed': False,
                    'z': z
                }
            else:
                # Multiple chunks per Z
                for y_chunk in range(y_chunks):
                    for x_chunk in range(x_chunks):
                        y_start = y_chunk * chunk_size_xy
                        x_start = x_chunk * chunk_size_xy
                        y_end = min(y_start + chunk_size_xy, self.image_3d.shape[1])
                        x_end = min(x_start + chunk_size_xy, self.image_3d.shape[2])
                        
                        chunk_dict[(z, y_start, x_start)] = {
                            'coords': [y_start, y_end, x_start, x_end],
                            'processed': False,
                            'z': z
                        }

            self.realtimechunks = chunk_dict

        print("Ready!")

    def get_realtime_chunks_2d(self, chunk_size=None):
        """
        Updated 2D chunking to match create_2d_chunks logic
        """
        
        MAX_CHUNK_SIZE = self.twod_chunk_size
        
        # Populate chunk dictionary
        chunk_dict = {}
        
        # Create chunks for each Z plane using the same logic as create_2d_chunks
        for z in range(self.image_3d.shape[0]):
            y_dim = self.image_3d.shape[1]
            x_dim = self.image_3d.shape[2]
            total_pixels = y_dim * x_dim
            
            if total_pixels <= MAX_CHUNK_SIZE:
                # Single chunk for entire Z slice
                chunk_dict[(z, 0, 0)] = {
                    'coords': [0, y_dim, 0, x_dim],  # [y_start, y_end, x_start, x_end]
                    'processed': False,
                    'z': z
                }
            else:
                # Multiple chunks per Z plane - divide along largest dimension
                largest_dim = 'y' if y_dim >= x_dim else 'x'
                num_divisions = int(np.ceil(total_pixels / MAX_CHUNK_SIZE))
                
                if largest_dim == 'y':
                    # Divide along Y dimension
                    div_size = int(np.ceil(y_dim / num_divisions))
                    for i in range(0, y_dim, div_size):
                        end_i = min(i + div_size, y_dim)
                        # Use (z, y_start, x_start) as key for consistency
                        chunk_dict[(z, i, 0)] = {
                            'coords': [i, end_i, 0, x_dim],  # [y_start, y_end, x_start, x_end]
                            'processed': False,
                            'z': z
                        }
                else:
                    # Divide along X dimension
                    div_size = int(np.ceil(x_dim / num_divisions))
                    for i in range(0, x_dim, div_size):
                        end_i = min(i + div_size, x_dim)
                        # Use (z, y_start, x_start) as key for consistency
                        chunk_dict[(z, 0, i)] = {
                            'coords': [0, y_dim, i, end_i],  # [y_start, y_end, x_start, x_end]
                            'processed': False,
                            'z': z
                        }
        
        self.realtimechunks = chunk_dict
        print("Ready!")

    def process_slice_features(self, z: int, speed: Any, use_gpu: bool, 
                              z_fores: Dict[int, List[Tuple[int, int]]], 
                              z_backs: Dict[int, List[Tuple[int, int]]]) -> Tuple[List[Any], List[Any]]:
        """
        Helper function to process a single slice and extract features.
        Returns tuple of (foreground_features, background_features) for this slice.
        """
        slice_foreground_features = []
        slice_background_features = []
        
        current_map = self.get_feature_map_slice(z, speed, use_gpu)
        
        if z in z_fores:
            for y, x in z_fores[z]:
                feature_vector = current_map[y, x]
                slice_foreground_features.append(feature_vector)
        
        if z in z_backs:
            for y, x in z_backs[z]:
                feature_vector = current_map[y, x]
                slice_background_features.append(feature_vector)
        
        return slice_foreground_features, slice_background_features

    def extract_features_parallel(self, needed_chunks, speed, use_gpu, z_fores, z_backs):
        """
        Updated version that processes chunks instead of full Z-slices
        """
        max_cores = multiprocessing.cpu_count()
        foreground_features = []
        background_features = []
        
        # Flatten all chunks into a single list for parallel processing
        all_chunk_tasks = []
        for z in needed_chunks:
            for chunk_coords in needed_chunks[z]:
                all_chunk_tasks.append((chunk_coords, z_fores, z_backs, speed))
        
        with ThreadPoolExecutor(max_workers=max_cores) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(self.process_chunk_features_for_training, chunk_task): chunk_task 
                for chunk_task in all_chunk_tasks
            }
            
            # Collect results as they complete
            for future in future_to_chunk:
                chunk_foreground, chunk_background = future.result()
                foreground_features.extend(chunk_foreground)
                background_features.extend(chunk_background)
        
        return foreground_features, background_features

    def segment_volume_realtime(self, gpu=False):
        """Updated realtime segmentation with chunked 2D processing"""

        if self.realtimechunks is None:
            if not self.use_two:
                self.get_realtime_chunks()
            else:
                self.get_realtime_chunks_2d()
        else:
            for chunk_pos in self.realtimechunks:
                self.realtimechunks[chunk_pos]['processed'] = False

        chunk_dict = self.realtimechunks
        
        def get_nearest_unprocessed_chunk(self):
            """Get nearest unprocessed chunk prioritizing current Z"""
            curr_z = self.current_z if self.current_z is not None else self.image_3d.shape[0] // 2
            curr_y = self.current_y if self.current_y is not None else self.image_3d.shape[1] // 2
            curr_x = self.current_x if self.current_x is not None else self.image_3d.shape[2] // 2
            
            # First try to find chunks at current Z
            current_z_chunks = [(pos, info) for pos, info in chunk_dict.items() 
                              if pos[0] == curr_z and not info['processed']]
            
            if current_z_chunks:
                nearest = min(current_z_chunks, 
                            key=lambda x: ((x[0][1] - curr_y) ** 2 + 
                                         (x[0][2] - curr_x) ** 2))
                return nearest[0]
            
            # If no chunks at current Z, find nearest Z with available chunks
            available_z = sorted(
                [(pos[0], pos) for pos, info in chunk_dict.items() 
                 if not info['processed']],
                key=lambda x: abs(x[0] - curr_z)
            )
            
            if available_z:
                target_z = available_z[0][0]
                z_chunks = [(pos, info) for pos, info in chunk_dict.items() 
                           if pos[0] == target_z and not info['processed']]
                nearest = min(z_chunks, 
                            key=lambda x: ((x[0][1] - curr_y) ** 2 + 
                                         (x[0][2] - curr_x) ** 2))
                return nearest[0]
            
            return None
        
        while True:
            chunk_idx = get_nearest_unprocessed_chunk(self)
            if chunk_idx is None:
                break
                
            chunk = chunk_dict[chunk_idx]
            chunk['processed'] = True
            coords = chunk['coords']  # [y_start, y_end, x_start, x_end]
            z = chunk['z']

            # Generate coordinates for this chunk
            coords_array = np.stack(np.meshgrid(
                [z],
                np.arange(coords[0], coords[1]),
                np.arange(coords[2], coords[3]),
                indexing='ij'
            )).reshape(3, -1).T

            coords_list = list(map(tuple, coords_array))
            
            # Process the chunk with updated method
            fore, back = self.process_chunk(coords_list)
            
            yield fore, back

    def cleanup(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()
            except:
                pass

    def process_grid_cell(self, grid_cell_info):
        """
        Process a single grid cell and return foreground and background features.
        
        Args:
            grid_cell_info: tuple of (grid_z, grid_y, grid_x, box_size, depth, height, width, foreground_array)
        
        Returns:
            tuple: (foreground_features, background_features)
        """
        grid_z, grid_y, grid_x, box_size, depth, height, width, foreground_array = grid_cell_info
        
        # Calculate the boundaries of this grid cell
        z_min = grid_z * box_size
        y_min = grid_y * box_size
        x_min = grid_x * box_size
        
        z_max = min(z_min + box_size, depth)
        y_max = min(y_min + box_size, height)
        x_max = min(x_min + box_size, width)
        
        # Extract the subarray
        subarray = self.image_3d[z_min:z_max, y_min:y_max, x_min:x_max]
        subarray2 = foreground_array[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Compute features for this subarray
        if self.speed:
            subarray_features = self.compute_feature_maps_cpu(subarray)
        else:
            subarray_features = self.compute_deep_feature_maps_cpu(subarray)
        
        # Extract foreground features
        local_fore_coords = np.argwhere(subarray2 == 1)
        foreground_features = []
        for local_z, local_y, local_x in local_fore_coords:
            feature = subarray_features[local_z, local_y, local_x]
            foreground_features.append(feature)
        
        # Extract background features
        local_back_coords = np.argwhere(subarray2 == 2)
        background_features = []
        for local_z, local_y, local_x in local_back_coords:
            feature = subarray_features[local_z, local_y, local_x]
            background_features.append(feature)
        
        return foreground_features, background_features

    # Modified main processing code
    def process_grid_cells_parallel(self, grid_cells_with_scribbles, box_size, depth, height, width, foreground_array, max_workers=None):
        """
        Process grid cells in parallel using ThreadPoolExecutor.
        
        Args:
            grid_cells_with_scribbles: List of grid cell coordinates
            box_size: Size of each grid cell
            depth, height, width: Dimensions of the 3D image
            foreground_array: Array marking foreground/background points
            max_workers: Maximum number of threads (None for default)
        
        Returns:
            tuple: (foreground_features, background_features)
        """
        # Prepare data for each grid cell
        grid_cell_data = [
            (grid_z, grid_y, grid_x, box_size, depth, height, width, foreground_array)
            for grid_z, grid_y, grid_x in grid_cells_with_scribbles
        ]
        
        foreground_features = []
        background_features = []
        
        # Process grid cells in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self.process_grid_cell, cell_data) for cell_data in grid_cell_data]
            
            # Collect results as they complete
            for future in futures:
                fore_features, back_features = future.result()
                foreground_features.extend(fore_features)
                background_features.extend(back_features)
        
        return foreground_features, background_features

    def train_batch(self, foreground_array, speed=True, use_gpu=False, use_two=False, mem_lock=False, saving=False):
        """Updated train_batch with chunked 2D processing"""

        if not saving:
            print("Training model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                max_depth=None
            )

        self.speed = speed
        self.cur_gpu = use_gpu

        if use_two != self.use_two:
            self.realtimechunks = None

        if not use_two:
            self.use_two = False

        self.mem_lock = mem_lock

        if use_two:
            if not self.use_two:
                self.use_two = True
            self.two_slices = []

            # Get foreground coordinates and features
            z_fore, y_fore, x_fore = np.where(foreground_array == 1)
            fore_coords = list(zip(z_fore, y_fore, x_fore))
            
            # Get background coordinates and features
            z_back, y_back, x_back = np.where(foreground_array == 2)
            back_coords = list(zip(z_back, y_back, x_back))
            
            foreground_features = []
            background_features = []
            
            # Organize coordinates by Z
            z_fores = self.organize_by_z(fore_coords)
            z_backs = self.organize_by_z(back_coords)
            
            # Combine all Z-slices that have coordinates
            all_z_coords = {}
            for z in z_fores:
                if z not in all_z_coords:
                    all_z_coords[z] = []
                all_z_coords[z].extend(z_fores[z])
            for z in z_backs:
                if z not in all_z_coords:
                    all_z_coords[z] = []
                all_z_coords[z].extend(z_backs[z])
            
            # Get minimal chunks needed to cover all coordinates
            needed_chunks = self.get_minimal_chunks_for_coordinates_cpu(all_z_coords)
            
            # Use existing parallel infrastructure with chunked approach
            foreground_features, background_features = self.extract_features_parallel(
                needed_chunks, speed, use_gpu, z_fores, z_backs
            )

        else:
            # 3D processing (unchanged - your existing code)
            box_size = self.master_chunk
            foreground_features = []
            background_features = []
            
            z_fore = np.argwhere(foreground_array == 1)
            z_back = np.argwhere(foreground_array == 2)
            
            if len(z_fore) == 0 and len(z_back) == 0:
                return foreground_features, background_features
            
            depth, height, width = foreground_array.shape
            z_grid_size = (depth + box_size - 1) // box_size
            y_grid_size = (height + box_size - 1) // box_size
            x_grid_size = (width + box_size - 1) // box_size
            
            grid_cells_with_scribbles = set()
            
            for z, y, x in np.vstack((z_fore, z_back)) if len(z_back) > 0 else z_fore:
                grid_z = z // box_size
                grid_y = y // box_size
                grid_x = x // box_size
                grid_cells_with_scribbles.add((grid_z, grid_y, grid_x))
            
            foreground_features, background_features = self.process_grid_cells_parallel(
                grid_cells_with_scribbles, box_size, depth, height, width, foreground_array)

        # Rest of the method unchanged (combining with previous features, training, etc.)
        if self.previous_foreground is not None:
            failed = True
            try:
                foreground_features = np.vstack([self.previous_foreground, foreground_features])
                failed = False
            except:
                pass
            try:
                background_features = np.vstack([self.previous_background, background_features])
                failed = False
            except:
                pass
            try:
                z_fore = np.concatenate([self.previous_z_fore, z_fore])
            except:
                pass
            try:
                z_back = np.concatenate([self.previous_z_back, z_back])
            except:
                pass
            if failed:
                print("Could not combine new model with old loaded model. Perhaps you are trying to combine a quick model with a deep model? I cannot combine these...")

        if saving:
            return foreground_features, background_features, z_fore, z_back

        # Combine features and labels
        X = np.vstack([foreground_features, background_features])
        y = np.hstack([np.ones(len(z_fore)), np.zeros(len(z_back))])
        
        # Train the model
        try:
            self.model.fit(X, y)
        except:
            print(X)
            print(y)

        self.current_speed = speed
        print("Done")

    def save_model(self, file_name, foreground_array):

        print("Saving model data")

        foreground_features, background_features, z_fore, z_back = self.train_batch(foreground_array, speed = self.speed, use_gpu = self.use_gpu, use_two = self.use_two, mem_lock = self.mem_lock, saving = True)


        np.savez(file_name, 
                 foreground_features=foreground_features,
                 background_features=background_features,
                 z_fore=z_fore,
                 z_back=z_back,
                 speed=self.speed,
                 use_gpu=self.use_gpu,
                 use_two=self.use_two,
                 mem_lock=self.mem_lock)

        print(f"Model data saved to {file_name}.")


    def load_model(self, file_name):

        print("Loading model data")

        data = np.load(file_name)

        # Unpack the arrays
        self.previous_foreground = data['foreground_features']
        self.previous_background = data['background_features']
        self.previous_z_fore = data['z_fore']
        self.previous_z_back = data['z_back']
        self.speed = bool(data['speed'])
        self.use_gpu = bool(data['use_gpu'])
        self.use_two = bool(data['use_two'])
        self.mem_lock = bool(data['mem_lock'])

        X = np.vstack([self.previous_foreground, self.previous_background])
        y = np.hstack([np.ones(len(self.previous_z_fore)), np.zeros(len(self.previous_z_back))])

        try:
            self.model.fit(X, y)
        except:
            print(X)
            print(y)

        print("Done")

    def get_feature_map_slice(self, z, speed, use_gpu):

        if self._currently_segmenting is not None:
            return

        if speed:
            output = self.compute_feature_maps_cpu_2d(z = z)

        elif not speed:
            output = self.compute_deep_feature_maps_cpu_2d(z = z)

        return output


    def process_chunk_features_for_training(self, chunk_task):
        """
        Process a single chunk for training feature extraction
        chunk_task: (chunk_coords, z_fores, z_backs, speed)
        """
        chunk_coords, z_fores, z_backs, speed = chunk_task
        z = chunk_coords[0]
        
        # Compute features for this chunk
        feature_map, (y_offset, x_offset) = self.compute_features_for_chunk_2d_cpu(chunk_coords, speed)
        
        chunk_foreground_features = []
        chunk_background_features = []
        
        # Extract foreground features from this chunk
        if z in z_fores:
            for y, x in z_fores[z]:
                # Check if this coordinate is in the current chunk
                y_start, y_end = chunk_coords[1], chunk_coords[2]
                x_start, x_end = chunk_coords[3], chunk_coords[4]
                
                if y_start <= y < y_end and x_start <= x < x_end:
                    # Convert global coordinates to local chunk coordinates
                    local_y = y - y_offset
                    local_x = x - x_offset
                    feature_vector = feature_map[local_y, local_x]
                    chunk_foreground_features.append(feature_vector)
        
        # Extract background features from this chunk
        if z in z_backs:
            for y, x in z_backs[z]:
                # Check if this coordinate is in the current chunk
                y_start, y_end = chunk_coords[1], chunk_coords[2]
                x_start, x_end = chunk_coords[3], chunk_coords[4]
                
                if y_start <= y < y_end and x_start <= x < x_end:
                    # Convert global coordinates to local chunk coordinates
                    local_y = y - y_offset
                    local_x = x - x_offset
                    feature_vector = feature_map[local_y, local_x]
                    chunk_background_features.append(feature_vector)
        
        return chunk_foreground_features, chunk_background_features