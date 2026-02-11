import numpy as np
#try:
    #import torch
#except:
    #pass
import cupy as cp
import cupyx.scipy.ndimage as cpx
#try:
    #from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
#except:
    #pass
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy import ndimage
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

class InteractiveSegmenter:
    def __init__(self, image_3d):
        image_3d = cp.asarray(image_3d)
        self.image_3d = image_3d
        self.patterns = []

        self.model = RandomForestClassifier(
                        n_estimators=150,           
                        max_depth=20,          
                        min_samples_split=10,       
                        min_samples_leaf=5,
                        max_features='sqrt',        
                        max_samples=0.7,           
                        class_weight='balanced',    
                        n_jobs=-1,
                        random_state=42,
                        bootstrap=True,
                        oob_score=True
                    )

        self.lock = threading.Lock()
        self._currently_segmenting = None
        self.use_gpu = True

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
        self.batch_amplifier = 2 # Can raise this number to make SKLearn batches larger

        #  flags to track state
        self._currently_processing = False
        self._skip_next_update = False
        self._last_processed_slice = None
        self.mem_lock = False

        #Adjustable feature map params:
        self.sigmas = [1,2,4,8]
        self.windows = 10
        self.dogs = [(1, 2), (2, 4), (4, 8)]
        self.master_chunk = 64
        self.twod_chunk_size = 117649

        #Data when loading prev model:
        self.previous_foreground = None
        self.previous_background = None
        self.previous_z_fore = None
        self.previous_z_back = None


    def get_minimal_chunks_for_coordinates(self, coordinates_by_z):
        """
        GPU version - Get minimal set of 2D chunks needed to cover the given coordinates
        Uses same chunking logic as create_2d_chunks()
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
                # Calculate optimal grid dimensions for square-ish chunks
                num_chunks_needed = int(cp.ceil(total_pixels / MAX_CHUNK_SIZE))
                
                # Find factors that give us the most square-like grid
                best_y_chunks = 1
                best_x_chunks = num_chunks_needed
                best_aspect_ratio = float('inf')
                
                for y_chunks in range(1, num_chunks_needed + 1):
                    x_chunks = int(cp.ceil(num_chunks_needed / y_chunks))
                    
                    # Calculate actual chunk dimensions
                    chunk_y_size = int(cp.ceil(y_dim / y_chunks))
                    chunk_x_size = int(cp.ceil(x_dim / x_chunks))
                    
                    # Check if chunk size constraint is satisfied
                    chunk_pixels = chunk_y_size * chunk_x_size
                    if chunk_pixels > MAX_CHUNK_SIZE:
                        continue
                    
                    # Calculate aspect ratio of the chunk
                    aspect_ratio = max(chunk_y_size, chunk_x_size) / min(chunk_y_size, chunk_x_size)
                    
                    # Prefer more square-like chunks (aspect ratio closer to 1)
                    if aspect_ratio < best_aspect_ratio:
                        best_aspect_ratio = aspect_ratio
                        best_y_chunks = y_chunks
                        best_x_chunks = x_chunks
                
                chunks_for_z = []
                
                # If no valid configuration found, fall back to single dimension division
                if best_aspect_ratio == float('inf'):
                    # Fall back to original logic
                    largest_dim = 'y' if y_dim >= x_dim else 'x'
                    num_divisions = int(cp.ceil(total_pixels / MAX_CHUNK_SIZE))
                    
                    if largest_dim == 'y':
                        div_size = int(cp.ceil(y_dim / num_divisions))
                        for i in range(0, y_dim, div_size):
                            end_i = min(i + div_size, y_dim)
                            # Check if this chunk contains any of our coordinates
                            if any(i <= y <= end_i-1 for y in y_coords):
                                chunks_for_z.append([z, i, end_i, 0, x_dim])
                    else:
                        div_size = int(cp.ceil(x_dim / num_divisions))
                        for i in range(0, x_dim, div_size):
                            end_i = min(i + div_size, x_dim)
                            # Check if this chunk contains any of our coordinates
                            if any(i <= x <= end_i-1 for x in x_coords):
                                chunks_for_z.append([z, 0, y_dim, i, end_i])
                else:
                    # Create the 2D grid of chunks and check which ones contain coordinates
                    y_chunk_size = int(cp.ceil(y_dim / best_y_chunks))
                    x_chunk_size = int(cp.ceil(x_dim / best_x_chunks))
                    
                    for y_idx in range(best_y_chunks):
                        for x_idx in range(best_x_chunks):
                            y_start = y_idx * y_chunk_size
                            y_end = min(y_start + y_chunk_size, y_dim)
                            x_start = x_idx * x_chunk_size
                            x_end = min(x_start + x_chunk_size, x_dim)
                            
                            # Skip empty chunks (can happen at edges)
                            if y_start >= y_dim or x_start >= x_dim:
                                continue
                            
                            # Check if this chunk contains any of our coordinates
                            chunk_contains_coords = any(
                                y_start <= y <= y_end-1 and x_start <= x <= x_end-1
                                for y, x in zip(y_coords, x_coords)
                            )
                            
                            if chunk_contains_coords:
                                chunks_for_z.append([z, y_start, y_end, x_start, x_end])
                
                needed_chunks[z] = chunks_for_z
        
        return needed_chunks

    def compute_features_for_chunk_2d(self, chunk_coords, speed):
        """
        Compute features for a 2D chunk
        chunk_coords: [z, y_start, y_end, x_start, x_end]
        """
        z, y_start, y_end, x_start, x_end = chunk_coords
        
        # Extract 2D subarray for this chunk
        subarray_2d = self.image_3d[z, y_start:y_end, x_start:x_end]
        
        # Compute features for this chunk
        if speed:
            feature_map = self.compute_feature_maps_gpu_2d(image_2d=subarray_2d)
        else:
            feature_map = self.compute_deep_feature_maps_gpu_2d(image_2d=subarray_2d)
        
        return feature_map, (y_start, x_start)  # Return offset for coordinate mapping


    def process_chunk(self, chunk_coords):
        """Updated GPU process_chunk with manual coordinate generation (faster)"""
        import cupy as cp
        
        chunk_info = self.realtimechunks[chunk_coords]
        
        if not self.use_two:
            chunk_bounds = chunk_info['bounds']
        else:
            # 2D chunk format: key is (z, y_start, x_start), coords are [y_start, y_end, x_start, x_end]
            z = chunk_info['z']
            y_start, y_end, x_start, x_end = chunk_info['coords']
            chunk_bounds = (z, y_start, y_end, x_start, x_end)  # Convert to tuple for consistency
        
        if not self.use_two:
            # 3D processing - generate coordinates manually
            z_start, z_end, y_start, y_end, x_start, x_end = chunk_bounds
            
            # Generate coordinate array manually using CuPy
            chunk_coords_array = cp.stack(cp.meshgrid(
                cp.arange(z_start, z_end),
                cp.arange(y_start, y_end),
                cp.arange(x_start, x_end),
                indexing='ij'
            )).reshape(3, -1).T
            
            # Extract subarray
            subarray = self.image_3d[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Compute features for entire subarray
            if self.speed:
                feature_map = self.compute_feature_maps_gpu(subarray)
            else:
                feature_map = self.compute_deep_feature_maps_gpu(subarray)
            
            # Vectorized feature extraction using local coordinates
            local_coords = chunk_coords_array - cp.array([z_start, y_start, x_start])
            features_gpu = feature_map[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
            
        else:
            # 2D processing - generate coordinates manually
            z, y_start, y_end, x_start, x_end = chunk_bounds
            
            # Generate coordinate array for this Z slice using CuPy
            chunk_coords_array = cp.stack(cp.meshgrid(
                cp.array([z]),
                cp.arange(y_start, y_end),
                cp.arange(x_start, x_end),
                indexing='ij'
            )).reshape(3, -1).T
            
            # Extract 2D subarray for this chunk
            subarray_2d = self.image_3d[z, y_start:y_end, x_start:x_end]
            
            # Compute 2D features for the subarray
            if self.speed:
                feature_map = self.compute_feature_maps_gpu_2d(image_2d=subarray_2d)
            else:
                feature_map = self.compute_deep_feature_maps_gpu_2d(image_2d=subarray_2d)
            
            # Convert global coordinates to local chunk coordinates
            local_y_coords = chunk_coords_array[:, 1] - y_start
            local_x_coords = chunk_coords_array[:, 2] - x_start
            
            # Extract features using local coordinates
            features_gpu = feature_map[local_y_coords, local_x_coords]
        
        # Common prediction logic - convert to CPU for model prediction
        features_cpu = cp.asnumpy(features_gpu)
        predictions = self.model.predict(features_cpu)
        predictions_gpu = cp.array(predictions, dtype=bool)
        
        # Use boolean indexing to separate coordinates
        foreground_coords = chunk_coords_array[predictions_gpu]
        background_coords = chunk_coords_array[~predictions_gpu]
        
        # Convert to sets (convert back to CPU for set operations)
        foreground = set(map(tuple, cp.asnumpy(foreground_coords)))
        background = set(map(tuple, cp.asnumpy(background_coords)))
        
        return foreground, background

    def twodim_coords(self, z, y_start, y_end, x_start, x_end):
        """
        Generate 2D coordinates for a z-slice using CuPy for GPU acceleration.
        
        Args:
            z (int): Z-slice index
            y_start (int): Start index for y dimension
            y_end (int): End index for y dimension
            x_start (int): Start index for x dimension
            x_end (int): End index for x dimension
        
        Returns:
            CuPy array of coordinates in format (z, y, x)
        """
        import cupy as cp
        
        # Create ranges for y and x dimensions
        y_range = cp.arange(y_start, y_end, dtype=int)
        x_range = cp.arange(x_start, x_end, dtype=int)
        
        # Create meshgrid
        y_coords, x_coords = cp.meshgrid(y_range, x_range, indexing='ij')
        
        # Calculate total size
        total_size = len(y_range) * len(x_range)
        
        # Stack coordinates with z values
        slice_coords = cp.column_stack((
            cp.full(total_size, z, dtype=int),
            y_coords.ravel(),
            x_coords.ravel()
        ))
        
        return slice_coords


    def compute_feature_maps_gpu(self, image_3d=None):
        """Optimized GPU version that caches Gaussian filters to avoid redundant computation"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        if image_3d is None:
            image_3d = self.image_3d  # Assuming this is already a cupy array

        if image_3d.ndim == 4 and (image_3d.shape[-1] == 3 or image_3d.shape[-1] == 4):
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(image_3d.shape[-1]):
                channel_features = self.compute_feature_maps_gpu(image_3d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return cp.concatenate(features_per_channel, axis=-1)
        
        # Pre-allocate result array
        num_features = len(self.sigmas) + len(self.dogs) + 2
        features = cp.empty(image_3d.shape + (num_features,), dtype=image_3d.dtype)
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
            gaussian_cache[sigma] = cupy_ndimage.gaussian_filter(image_3d, sigma)
        
        # Gaussian smoothing - use cached results
        for sigma in self.sigmas:
            features[..., feature_idx] = gaussian_cache[sigma]
            feature_idx += 1
        
        # Difference of Gaussians - use cached results
        for s1, s2 in self.dogs:
            features[..., feature_idx] = gaussian_cache[s1] - gaussian_cache[s2]
            feature_idx += 1
        
        # Gradient magnitude
        gx = cupy_ndimage.sobel(image_3d, axis=2, mode='reflect')
        gy = cupy_ndimage.sobel(image_3d, axis=1, mode='reflect')
        gz = cupy_ndimage.sobel(image_3d, axis=0, mode='reflect')
        features[..., feature_idx] = cp.sqrt(gx**2 + gy**2 + gz**2)
        
        return features

    def compute_deep_feature_maps_gpu(self, image_3d=None):
        """Vectorized detailed GPU version with Gaussian gradient magnitudes, Laplacians, and largest Hessian eigenvalue only"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        if image_3d is None:
            image_3d = self.image_3d  # Assuming this is already a cupy array

        if image_3d.ndim == 4 and (image_3d.shape[-1] == 3 or image_3d.shape[-1] == 4):
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(image_3d.shape[-1]):
                channel_features = self.compute_deep_feature_maps_gpu(image_3d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return cp.concatenate(features_per_channel, axis=-1)
        
        # Calculate total number of features
        num_basic_features = 1 + len(self.sigmas) + len(self.dogs)  # original + gaussians + dogs
        num_gradient_features = len(self.sigmas)  # gradient magnitude for each sigma
        num_laplacian_features = len(self.sigmas)  # laplacian for each sigma
        num_hessian_features = len(self.sigmas) * 1  # 1 eigenvalue (largest) for each sigma
        
        total_features = num_basic_features + num_gradient_features + num_laplacian_features + num_hessian_features
        
        # Pre-allocate result array
        features = cp.empty(image_3d.shape + (total_features,), dtype=image_3d.dtype)
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
            gaussian_cache[sigma] = cupy_ndimage.gaussian_filter(image_3d, sigma)
        
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
            gx = cupy_ndimage.sobel(gaussian_img, axis=2, mode='reflect')
            gy = cupy_ndimage.sobel(gaussian_img, axis=1, mode='reflect')
            gz = cupy_ndimage.sobel(gaussian_img, axis=0, mode='reflect')
            features[..., feature_idx] = cp.sqrt(gx**2 + gy**2 + gz**2)
            feature_idx += 1
        
        # Laplacian of Gaussian for each sigma (vectorized)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            features[..., feature_idx] = cupy_ndimage.laplace(gaussian_img, mode='reflect')
            feature_idx += 1
        
        # Largest Hessian eigenvalue for each sigma (fully vectorized)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            
            # Compute second derivatives (Hessian components) - all vectorized
            hxx = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 0, 2], mode='reflect')
            hyy = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 2, 0], mode='reflect')
            hzz = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[2, 0, 0], mode='reflect')
            hxy = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 1, 1], mode='reflect')
            hxz = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 0, 1], mode='reflect')
            hyz = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 1, 0], mode='reflect')
            
            # Vectorized eigenvalue computation using cupy broadcasting
            # Create arrays with shape (d0, d1, d2, 3, 3) for all Hessian matrices
            shape = image_3d.shape
            hessian_matrices = cp.zeros(shape + (3, 3))
            
            # Fill the symmetric Hessian matrices
            hessian_matrices[..., 0, 0] = hxx
            hessian_matrices[..., 1, 1] = hyy
            hessian_matrices[..., 2, 2] = hzz
            hessian_matrices[..., 0, 1] = hessian_matrices[..., 1, 0] = hxy
            hessian_matrices[..., 0, 2] = hessian_matrices[..., 2, 0] = hxz
            hessian_matrices[..., 1, 2] = hessian_matrices[..., 2, 1] = hyz
            
            # Reshape for batch eigenvalue computation
            original_shape = hessian_matrices.shape[:-2]  # (d0, d1, d2)
            batch_size = int(cp.prod(cp.array(original_shape)))
            hessian_batch = hessian_matrices.reshape(batch_size, 3, 3)
            
            # Compute eigenvalues for all matrices at once using CuPy
            # Since Hessian matrices are symmetric, we can use eigvalsh
            eigenvalues_batch = cp.linalg.eigvalsh(hessian_batch)
            
            # Get only the largest eigenvalue for each matrix
            largest_eigenvalues = cp.max(eigenvalues_batch, axis=1)
            
            # Reshape back to original spatial dimensions
            largest_eigenvalues = largest_eigenvalues.reshape(original_shape)
            
            # Add the largest eigenvalue as a feature
            features[..., feature_idx] = largest_eigenvalues
            feature_idx += 1
        
        # Normalize only morphological features, keep intensity features raw
        intensity_features = features[..., :num_basic_features]  # original + gaussians + DoGs
        morphology_features = features[..., num_basic_features:]  # gradients + laplacians + eigenvalues

        # Normalize only morphological features using CuPy
        morph_means = cp.mean(morphology_features, axis=(0, 1, 2), keepdims=True)
        morph_stds = cp.std(morphology_features, axis=(0, 1, 2), keepdims=True)
        morph_stds = cp.where(morph_stds == 0, 1, morph_stds)
        morphology_features = (morphology_features - morph_means) / morph_stds

        # Recombine
        features = cp.concatenate([intensity_features, morphology_features], axis=-1)
        
        return features


    def compute_feature_maps_gpu_2d(self, z=None, image_2d = None):
        """Compute feature maps for 2D images using GPU with caching optimization"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        # Extract 2D slice - convert to CuPy array if needed
        if image_2d is None:
            image_2d = cp.asarray(self.image_3d[z, :, :])

        if image_2d.ndim == 3 and (image_2d.shape[-1] == 3 or image_2d.shape[-1] == 4):
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(image_2d.shape[-1]):
                channel_features = self.compute_feature_maps_gpu_2d(image_2d = image_2d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return cp.concatenate(features_per_channel, axis=-1)

        # Pre-allocate result array
        num_features = len(self.sigmas) + len(self.dogs) + 2  # +2 for original image + gradient
        features = cp.empty(image_2d.shape + (num_features,), dtype=image_2d.dtype)
        
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
            gaussian_cache[sigma] = cupy_ndimage.gaussian_filter(image_2d, sigma)
        
        # Gaussian smoothing - use cached results
        for sigma in self.sigmas:
            features[..., feature_idx] = gaussian_cache[sigma]
            feature_idx += 1
        
        # Difference of Gaussians - use cached results
        for s1, s2 in self.dogs:
            features[..., feature_idx] = gaussian_cache[s1] - gaussian_cache[s2]
            feature_idx += 1
        
        # Gradient magnitude (2D version)
        gx = cupy_ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
        gy = cupy_ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
        features[..., feature_idx] = cp.sqrt(gx**2 + gy**2)
        
        return features

    def compute_deep_feature_maps_gpu_2d(self, z=None, image_2d = None):
        """Vectorized detailed GPU version with Gaussian gradient magnitudes, Laplacians, and largest Hessian eigenvalue for 2D images"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        if z is None:
            z = self.image_3d.shape[0] // 2  # Use middle slice if not specified
        
        # Extract 2D slice - convert to CuPy array if needed
        if image_2d is None:
            image_2d = cp.asarray(self.image_3d[z, :, :])

        if image_2d.ndim == 3 and (image_2d.shape[-1] == 3 or image_2d.shape[-1] == 4):
            # RGB case - process each channel
            features_per_channel = []
            for channel in range(image_2d.shape[-1]):
                channel_features = self.compute_deep_feature_maps_gpu_2d(image_2d = image_2d[..., channel])
                features_per_channel.append(channel_features)
            
            # Stack all channel features
            return cp.concatenate(features_per_channel, axis=-1)
        
        
        # Calculate total number of features
        num_basic_features = 1 + len(self.sigmas) + len(self.dogs)  # original + gaussians + dogs
        num_gradient_features = len(self.sigmas)  # gradient magnitude for each sigma
        num_laplacian_features = len(self.sigmas)  # laplacian for each sigma
        num_hessian_features = len(self.sigmas) * 1  # 1 eigenvalue (largest) for each sigma
        
        total_features = num_basic_features + num_gradient_features + num_laplacian_features + num_hessian_features
        
        # Pre-allocate result array
        features = cp.empty(image_2d.shape + (total_features,), dtype=image_2d.dtype)
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
            gaussian_cache[sigma] = cupy_ndimage.gaussian_filter(image_2d, sigma)
        
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
            gx = cupy_ndimage.sobel(gaussian_img, axis=1, mode='reflect')  # x direction
            gy = cupy_ndimage.sobel(gaussian_img, axis=0, mode='reflect')  # y direction
            features[..., feature_idx] = cp.sqrt(gx**2 + gy**2)
            feature_idx += 1
        
        # Laplacian of Gaussian for each sigma (vectorized, 2D version)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            features[..., feature_idx] = cupy_ndimage.laplace(gaussian_img, mode='reflect')
            feature_idx += 1
        
        # Largest Hessian eigenvalue for each sigma (fully vectorized, 2D version)
        for sigma in self.sigmas:
            gaussian_img = gaussian_cache[sigma]
            
            # Compute second derivatives (Hessian components) - all vectorized for 2D
            hxx = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[0, 2], mode='reflect')
            hyy = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[2, 0], mode='reflect')
            hxy = cupy_ndimage.gaussian_filter(gaussian_img, sigma=0, order=[1, 1], mode='reflect')
            
            # Analytical eigenvalue computation for 2x2 symmetric matrices
            # For matrix [[hxx, hxy], [hxy, hyy]], eigenvalues are:
            # λ = (trace ± sqrt(trace² - 4*det)) / 2
            
            trace = hxx + hyy
            det = hxx * hyy - hxy * hxy
            
            # Calculate discriminant and ensure it's non-negative
            discriminant = trace * trace - 4 * det
            discriminant = cp.maximum(discriminant, 0)  # Handle numerical errors
            
            sqrt_discriminant = cp.sqrt(discriminant)
            
            # Calculate both eigenvalues
            eigenval1 = (trace + sqrt_discriminant) / 2
            eigenval2 = (trace - sqrt_discriminant) / 2
            
            # Take the larger eigenvalue (most positive/least negative)
            largest_eigenvalues = cp.maximum(eigenval1, eigenval2)
            
            # Add the largest eigenvalue as a feature
            features[..., feature_idx] = largest_eigenvalues
            feature_idx += 1
        
        # Normalize only morphological features, keep intensity features raw
        intensity_features = features[..., :num_basic_features]  # original + gaussians + DoGs
        morphology_features = features[..., num_basic_features:]  # gradients + laplacians + eigenvalues

        # Normalize only morphological features using CuPy
        morph_means = cp.mean(morphology_features, axis=(0, 1), keepdims=True)
        morph_stds = cp.std(morphology_features, axis=(0, 1), keepdims=True)
        morph_stds = cp.where(morph_stds == 0, 1, morph_stds)
        morphology_features = (morphology_features - morph_means) / morph_stds

        # Recombine
        features = cp.concatenate([intensity_features, morphology_features], axis=-1)
        
        return features

    def create_2d_chunks(self):
        """GPU version - Updated 2D chunking to create more square-like chunks"""
        MAX_CHUNK_SIZE = self.twod_chunk_size
        chunks = []
        
        for z in range(self.image_3d.shape[0]):
            y_dim = self.image_3d.shape[1]
            x_dim = self.image_3d.shape[2]
            total_pixels = y_dim * x_dim
            
            if total_pixels <= MAX_CHUNK_SIZE:
                chunks.append([z, 0, y_dim, 0, x_dim])
            else:
                # Calculate optimal grid dimensions for square-ish chunks
                num_chunks_needed = int(cp.ceil(total_pixels / MAX_CHUNK_SIZE))
                
                # Find factors that give us the most square-like grid
                best_y_chunks = 1
                best_x_chunks = num_chunks_needed
                best_aspect_ratio = float('inf')
                
                for y_chunks in range(1, num_chunks_needed + 1):
                    x_chunks = int(cp.ceil(num_chunks_needed / y_chunks))
                    
                    # Calculate actual chunk dimensions
                    chunk_y_size = int(cp.ceil(y_dim / y_chunks))
                    chunk_x_size = int(cp.ceil(x_dim / x_chunks))
                    
                    # Check if chunk size constraint is satisfied
                    chunk_pixels = chunk_y_size * chunk_x_size
                    if chunk_pixels > MAX_CHUNK_SIZE:
                        continue
                    
                    # Calculate aspect ratio of the chunk
                    aspect_ratio = max(chunk_y_size, chunk_x_size) / min(chunk_y_size, chunk_x_size)
                    
                    # Prefer more square-like chunks (aspect ratio closer to 1)
                    if aspect_ratio < best_aspect_ratio:
                        best_aspect_ratio = aspect_ratio
                        best_y_chunks = y_chunks
                        best_x_chunks = x_chunks
                
                # If no valid configuration found, fall back to single dimension division
                if best_aspect_ratio == float('inf'):
                    # Fall back to original logic
                    largest_dim = 'y' if y_dim >= x_dim else 'x'
                    num_divisions = int(cp.ceil(total_pixels / MAX_CHUNK_SIZE))
                    
                    if largest_dim == 'y':
                        div_size = int(cp.ceil(y_dim / num_divisions))
                        for i in range(0, y_dim, div_size):
                            end_i = min(i + div_size, y_dim)
                            chunks.append([z, i, end_i, 0, x_dim])
                    else:
                        div_size = int(cp.ceil(x_dim / num_divisions))
                        for i in range(0, x_dim, div_size):
                            end_i = min(i + div_size, x_dim)
                            chunks.append([z, 0, y_dim, i, end_i])
                else:
                    # Create the 2D grid of chunks
                    y_chunk_size = int(cp.ceil(y_dim / best_y_chunks))
                    x_chunk_size = int(cp.ceil(x_dim / best_x_chunks))
                    
                    for y_idx in range(best_y_chunks):
                        for x_idx in range(best_x_chunks):
                            y_start = y_idx * y_chunk_size
                            y_end = min(y_start + y_chunk_size, y_dim)
                            x_start = x_idx * x_chunk_size
                            x_end = min(x_start + x_chunk_size, x_dim)
                            
                            # Skip empty chunks (can happen at edges)
                            if y_start >= y_dim or x_start >= x_dim:
                                continue
                            
                            chunks.append([z, y_start, y_end, x_start, x_end])
            
        return chunks

 
    def segment_volume(self, array, chunk_size=None, gpu=True):
        """Optimized GPU version with sequential GPU processing and batched sklearn prediction"""
        
        array = cp.asarray(array)
        self.realtimechunks = None
        chunk_size = self.master_chunk
        
        
        print("Chunking data...")
        
        if not self.use_two:
            chunks = self.compute_3d_chunks(chunk_size)

        else:
            chunks = self.create_2d_chunks()
        
        print("Processing chunks with optimized GPU batching...")
        
        # Optimal batch size - balance memory usage vs sklearn efficiency
        max_workers = multiprocessing.cpu_count()
        batch_size = max_workers * self.batch_amplifier  # Process more chunks per batch for better sklearn utilization
        total_processed = 0
        
        # Configure sklearn for maximum parallelism
        if hasattr(self.model, 'n_jobs'):
            original_n_jobs = self.model.n_jobs
            self.model.n_jobs = -1
        
        try:
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                chunk_batch = chunks[batch_start:batch_end]
                
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # PHASE 1: Sequential GPU feature extraction (much faster than threading)
                batch_results = []
                
                for chunk in chunk_batch:
                    features_cpu, coords_gpu = self.extract_chunk_features_gpu(chunk)
                    if len(features_cpu) > 0:
                        batch_results.append((features_cpu, coords_gpu))
                
                # PHASE 2: Batch predict with sklearn's parallelism
                if batch_results:
                    # Combine all CPU features from this batch
                    all_batch_features = cp.vstack([result[0] for result in batch_results])
                    all_batch_coords = cp.vstack([result[1] for result in batch_results])
                    all_batch_features = cp.asnumpy(all_batch_features)
                    
                    # Single prediction call using sklearn's internal parallelism
                    predictions = self.model.predict(all_batch_features)
                    predictions = cp.array(predictions, dtype=bool)
                    
                    # Apply predictions to array
                    foreground_coords = all_batch_coords[predictions]
                    if len(foreground_coords) > 0:
                        try:
                            array[foreground_coords[:, 0], foreground_coords[:, 1], foreground_coords[:, 2]] = 255
                        except IndexError as e:
                            print(f"Index error when updating array: {e}")
                            # Fallback approach
                            for coord in foreground_coords:
                                z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
                                if 0 <= z < array.shape[0] and 0 <= y < array.shape[1] and 0 <= x < array.shape[2]:
                                    array[z, y, x] = 255
                    
                    # Memory cleanup for this batch
                    del all_batch_features, all_batch_coords, predictions, foreground_coords
                    cp.get_default_memory_pool().free_all_blocks()
                
                total_processed += len(chunk_batch)
                print(f"Completed {total_processed}/{len(chunks)} chunks")
        
        finally:
            # Restore sklearn settings
            if hasattr(self.model, 'n_jobs'):
                self.model.n_jobs = original_n_jobs
            
            # Final GPU memory cleanup
            cp.get_default_memory_pool().free_all_blocks()
        
        return cp.asnumpy(array)
            
    def extract_chunk_features_gpu(self, chunk_coords):
        """
        Updated GPU version of feature extraction with chunked 2D processing
        Returns GPU features and GPU coordinates for efficient batch processing
        """
        
        if not self.use_two:
            # 3D processing (unchanged)
            if self.realtimechunks is None:
                z_min, z_max = chunk_coords[0], chunk_coords[1]
                y_min, y_max = chunk_coords[2], chunk_coords[3]
                x_min, x_max = chunk_coords[4], chunk_coords[5]
                
                # Create coordinates using CuPy (GPU operations)
                z_range = cp.arange(z_min, z_max)
                y_range = cp.arange(y_min, y_max)
                x_range = cp.arange(x_min, x_max)
                
                chunk_coords_gpu = cp.stack(cp.meshgrid(
                    z_range, y_range, x_range, indexing='ij'
                )).reshape(3, -1).T
            else:
                chunk_coords_gpu = cp.array(chunk_coords)
                z_coords = chunk_coords_gpu[:, 0]
                y_coords = chunk_coords_gpu[:, 1]
                x_coords = chunk_coords_gpu[:, 2]
                
                z_min, z_max = cp.min(z_coords).item(), cp.max(z_coords).item()
                y_min, y_max = cp.min(y_coords).item(), cp.max(y_coords).item()
                x_min, x_max = cp.min(x_coords).item(), cp.max(x_coords).item()
            
            # Extract subarray and compute features (GPU operations)
            subarray = self.image_3d[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
            
            if self.speed:
                feature_map = self.compute_feature_maps_gpu(subarray)
            else:
                feature_map = self.compute_deep_feature_maps_gpu(subarray)
            
            # Extract features using GPU operations
            local_coords = chunk_coords_gpu.copy()
            local_coords[:, 0] -= z_min
            local_coords[:, 1] -= y_min
            local_coords[:, 2] -= x_min
            
            features_gpu = feature_map[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
            
            return features_gpu, chunk_coords_gpu
        
        else:
            # 2D processing - Updated to use chunked feature computation
            if len(chunk_coords) == 5:
                z = chunk_coords[0]
                y_start = chunk_coords[1]
                y_end = chunk_coords[2]
                x_start = chunk_coords[3]
                x_end = chunk_coords[4]
                
                # Generate coordinates for this chunk
                coords_array = self.twodim_coords(z, y_start, y_end, x_start, x_end)
                
                # NEW: Compute features for just this chunk instead of full Z-slice
                # Extract 2D subarray for this chunk
                subarray_2d = self.image_3d[z, y_start:y_end, x_start:x_end]
                
                # Compute features for this chunk only
                if self.speed:
                    feature_map = self.compute_feature_maps_gpu_2d(image_2d=subarray_2d)
                else:
                    feature_map = self.compute_deep_feature_maps_gpu_2d(image_2d=subarray_2d)
                
                # Convert global coordinates to local chunk coordinates
                y_indices = coords_array[:, 1] - y_start  # Local Y coordinates
                x_indices = coords_array[:, 2] - x_start  # Local X coordinates
                
                # Extract features using local coordinates
                features_gpu = feature_map[y_indices, x_indices]
                
                return features_gpu, coords_array
        
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

    def get_realtime_chunks_2d(self, chunk_size=None):
        """
        GPU version - Updated 2D chunking to match create_2d_chunks logic
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
                # Calculate optimal grid dimensions for square-ish chunks
                num_chunks_needed = int(cp.ceil(total_pixels / MAX_CHUNK_SIZE))
                
                # Find factors that give us the most square-like grid
                best_y_chunks = 1
                best_x_chunks = num_chunks_needed
                best_aspect_ratio = float('inf')
                
                for y_chunks in range(1, num_chunks_needed + 1):
                    x_chunks = int(cp.ceil(num_chunks_needed / y_chunks))
                    
                    # Calculate actual chunk dimensions
                    chunk_y_size = int(cp.ceil(y_dim / y_chunks))
                    chunk_x_size = int(cp.ceil(x_dim / x_chunks))
                    
                    # Check if chunk size constraint is satisfied
                    chunk_pixels = chunk_y_size * chunk_x_size
                    if chunk_pixels > MAX_CHUNK_SIZE:
                        continue
                    
                    # Calculate aspect ratio of the chunk
                    aspect_ratio = max(chunk_y_size, chunk_x_size) / min(chunk_y_size, chunk_x_size)
                    
                    # Prefer more square-like chunks (aspect ratio closer to 1)
                    if aspect_ratio < best_aspect_ratio:
                        best_aspect_ratio = aspect_ratio
                        best_y_chunks = y_chunks
                        best_x_chunks = x_chunks
                
                # If no valid configuration found, fall back to single dimension division
                if best_aspect_ratio == float('inf'):
                    # Fall back to original logic
                    largest_dim = 'y' if y_dim >= x_dim else 'x'
                    num_divisions = int(cp.ceil(total_pixels / MAX_CHUNK_SIZE))
                    
                    if largest_dim == 'y':
                        div_size = int(cp.ceil(y_dim / num_divisions))
                        for i in range(0, y_dim, div_size):
                            end_i = min(i + div_size, y_dim)
                            chunk_dict[(z, i, 0)] = {
                                'coords': [i, end_i, 0, x_dim],
                                'processed': False,
                                'z': z
                            }
                    else:
                        div_size = int(cp.ceil(x_dim / num_divisions))
                        for i in range(0, x_dim, div_size):
                            end_i = min(i + div_size, x_dim)
                            chunk_dict[(z, 0, i)] = {
                                'coords': [0, y_dim, i, end_i],
                                'processed': False,
                                'z': z
                            }
                else:
                    # Create the 2D grid of chunks
                    y_chunk_size = int(cp.ceil(y_dim / best_y_chunks))
                    x_chunk_size = int(cp.ceil(x_dim / best_x_chunks))
                    
                    for y_idx in range(best_y_chunks):
                        for x_idx in range(best_x_chunks):
                            y_start = y_idx * y_chunk_size
                            y_end = min(y_start + y_chunk_size, y_dim)
                            x_start = x_idx * x_chunk_size
                            x_end = min(x_start + x_chunk_size, x_dim)
                            
                            # Skip empty chunks (can happen at edges)
                            if y_start >= y_dim or x_start >= x_dim:
                                continue
                            
                            chunk_dict[(z, y_start, x_start)] = {
                                'coords': [y_start, y_end, x_start, x_end],
                                'processed': False,
                                'z': z
                            }
        
        self.realtimechunks = chunk_dict
        print("Ready!")

    def compute_3d_chunks(self, chunk_size=None, thickness=49):
        """
        Compute 3D chunks as rectangular prisms with consistent logic across all operations.
        Creates chunks that are thin in Z and square-like in X/Y dimensions.
        
        Args:
            chunk_size: Optional chunk size for volume calculation, otherwise uses dynamic calculation
            thickness: Z-dimension thickness of chunks (default: 9)
            
        Returns:
            list: List of chunk coordinates [z_start, z_end, y_start, y_end, x_start, x_end]
        """
        # Use consistent chunk size calculation
        if chunk_size is None:
            if hasattr(self, 'master_chunk') and self.master_chunk is not None:
                chunk_size = self.master_chunk
            else:
                # Dynamic calculation (same as segmentation)
                total_cores = multiprocessing.cpu_count()
                total_volume = np.prod(self.image_3d.shape)
                target_volume_per_chunk = total_volume / (total_cores * 4)
                
                chunk_size = int(np.cbrt(target_volume_per_chunk))
                chunk_size = max(16, min(chunk_size, min(self.image_3d.shape) // 2))
                chunk_size = ((chunk_size + 7) // 16) * 16
        
        try:
            depth, height, width = self.image_3d.shape
        except:
            depth, height, width, rgb = self.image_3d.shape
        
        # Calculate target volume per chunk (same as original cube)
        target_volume = chunk_size ** 3
        
        # Calculate XY side length based on thickness and target volume
        # Volume = thickness * xy_side * xy_side
        # So xy_side = sqrt(volume / thickness)
        xy_side = int(np.sqrt(target_volume / thickness))
        xy_side = max(1, xy_side)  # Ensure minimum size of 1
        
        # Calculate actual chunk dimensions for grid calculation
        z_chunk_size = thickness
        xy_chunk_size = xy_side
        
        # Calculate number of chunks in each dimension
        z_chunks = (depth + z_chunk_size - 1) // z_chunk_size
        y_chunks = (height + xy_chunk_size - 1) // xy_chunk_size
        x_chunks = (width + xy_chunk_size - 1) // xy_chunk_size
        
        # Calculate actual chunk sizes to distribute remainder evenly
        # This ensures all chunks are roughly the same size
        z_sizes = np.full(z_chunks, depth // z_chunks)
        z_remainder = depth % z_chunks
        z_sizes[:z_remainder] += 1
        
        y_sizes = np.full(y_chunks, height // y_chunks)
        y_remainder = height % y_chunks
        y_sizes[:y_remainder] += 1
        
        x_sizes = np.full(x_chunks, width // x_chunks)
        x_remainder = width % x_chunks
        x_sizes[:x_remainder] += 1
        
        # Calculate cumulative positions
        z_positions = np.concatenate([[0], np.cumsum(z_sizes)])
        y_positions = np.concatenate([[0], np.cumsum(y_sizes)])
        x_positions = np.concatenate([[0], np.cumsum(x_sizes)])
        
        # Generate all chunk coordinates
        chunks = []
        for z_idx in range(z_chunks):
            for y_idx in range(y_chunks):
                for x_idx in range(x_chunks):
                    z_start = z_positions[z_idx]
                    z_end = z_positions[z_idx + 1]
                    y_start = y_positions[y_idx]
                    y_end = y_positions[y_idx + 1]
                    x_start = x_positions[x_idx]
                    x_end = x_positions[x_idx + 1]
                    
                    coords = [z_start, z_end, y_start, y_end, x_start, x_end]
                    chunks.append(coords)
        
        return chunks



    def get_realtime_chunks(self, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.master_chunk
        
        all_chunks = self.compute_3d_chunks(chunk_size)
        
        self.realtimechunks = {
            i: {
                'bounds': chunk_coords,  # Only store [z_start, z_end, y_start, y_end, x_start, x_end]
                'processed': False,
                'center': self._get_chunk_center(chunk_coords),  # Small tuple for distance calc
                'is_3d': True  # Flag to indicate this is 3D chunking
            }
            for i, chunk_coords in enumerate(all_chunks)
        }
        print("Ready!")

    def _get_chunk_center(self, chunk_coords):
        """Get center coordinate of chunk for distance calculations"""
        z_start, z_end, y_start, y_end, x_start, x_end = chunk_coords
        return (
            (z_start + z_end) // 2,
            (y_start + y_end) // 2,
            (x_start + x_end) // 2
        )


    def segment_volume_realtime(self, gpu=False):
        if self.realtimechunks is None:
            if not self.use_two:
                self.get_realtime_chunks()  # 3D chunks
            else:
                self.get_realtime_chunks_2d()  # 2D chunks
        else:
            for chunk_key in self.realtimechunks:
                self.realtimechunks[chunk_key]['processed'] = False
        
        def get_nearest_unprocessed_chunk():
            curr_z = self.current_z if self.current_z is not None else self.image_3d.shape[0] // 2
            curr_y = self.current_y if self.current_y is not None else self.image_3d.shape[1] // 2
            curr_x = self.current_x if self.current_x is not None else self.image_3d.shape[2] // 2
            
            unprocessed_chunks = [
                (key, info) for key, info in self.realtimechunks.items() 
                if not info['processed']
            ]
            
            if not unprocessed_chunks:
                return None
            
            if self.use_two:
                # 2D chunks: key format is (z, y_start, x_start)
                # First try to find chunks at current Z
                current_z_chunks = [
                    (key, info) for key, info in unprocessed_chunks 
                    if key[0] == curr_z
                ]
                
                if current_z_chunks:
                    # Find nearest chunk at current Z by y,x distance
                    nearest = min(current_z_chunks, 
                                 key=lambda x: ((x[0][1] - curr_y) ** 2 + 
                                               (x[0][2] - curr_x) ** 2))
                    return nearest[0]
                
                # If no chunks at current Z, find nearest Z with available chunks
                available_z_chunks = sorted(unprocessed_chunks, 
                                           key=lambda x: abs(x[0][0] - curr_z))
                
                if available_z_chunks:
                    # Get the nearest Z that has unprocessed chunks
                    target_z = available_z_chunks[0][0][0]
                    z_chunks = [
                        (key, info) for key, info in unprocessed_chunks 
                        if key[0] == target_z
                    ]
                    # Find nearest chunk in that Z by y,x distance
                    nearest = min(z_chunks, 
                                 key=lambda x: ((x[0][1] - curr_y) ** 2 + 
                                               (x[0][2] - curr_x) ** 2))
                    return nearest[0]
            else:
                # 3D chunks: find chunks on nearest Z-plane, then closest in X/Y
                # First find the nearest Z-plane among available chunks
                nearest_z = min(unprocessed_chunks, 
                               key=lambda x: abs(x[1]['center'][0] - curr_z))[1]['center'][0]
                
                # Get all chunks on that nearest Z-plane
                nearest_z_chunks = [chunk for chunk in unprocessed_chunks 
                                   if chunk[1]['center'][0] == nearest_z]
                
                # From those chunks, find closest in X/Y
                nearest = min(nearest_z_chunks, 
                             key=lambda x: sum((a - b) ** 2 for a, b in 
                                             zip(x[1]['center'][1:], (curr_y, curr_x))))
                
                return nearest[0]
            
            return None
        
        while True:
            chunk_key = get_nearest_unprocessed_chunk()
            if chunk_key is None:
                break
                
            self.realtimechunks[chunk_key]['processed'] = True
            
            # Process the chunk - pass the key, process_chunk will handle the rest
            fore, back = self.process_chunk(chunk_key)
            
            yield fore, back


    def cleanup(self):
        """Clean up GPU memory"""
        import cupy as cp
        
        try:
            # Force garbage collection first
            import gc
            gc.collect()
            
            # Clean up CuPy memory pools
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Print memory usage before cleanup (optional)
            # print(f"Used GPU memory: {mempool.used_bytes() / 1024**2:.2f} MB")
            
            # Free all blocks
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            # Print memory usage after cleanup (optional)
            # print(f"Used GPU memory after cleanup: {mempool.used_bytes() / 1024**2:.2f} MB")
            
        except Exception as e:
            print(f"Warning: Could not clean up GPU memory: {e}")

    def train_batch(self, foreground_array, speed=True, use_gpu=True, use_two=False, mem_lock=False, saving = False):
        """Train directly on foreground and background arrays using GPU acceleration"""
        import cupy as cp

        if not saving:
            print("Training model...")
            self.model = RandomForestClassifier(
                            n_estimators=150,           
                            max_depth=20,          
                            min_samples_split=10,       
                            min_samples_leaf=5,
                            max_features='sqrt',        
                            max_samples=0.7,           
                            class_weight='balanced',    
                            n_jobs=-1,
                            random_state=42,
                            bootstrap=True,
                            oob_score=True
                        )

        self.speed = speed
        self.cur_gpu = use_gpu
        
        self.mem_lock = mem_lock

        if use_two != self.use_two:
            self.realtimechunks = None

        if not use_two:
            self.use_two = False

        if use_two:
            if not self.use_two:
                self.use_two = True
            self.two_slices = []
            foreground_array = cp.asarray(foreground_array)
            
            # Get foreground coordinates and features
            z_fore, y_fore, x_fore = cp.where(foreground_array == 1)
            fore_coords = [(int(z), int(y), int(x)) for z, y, x in zip(z_fore, y_fore, x_fore)]
            
            # Get background coordinates and features
            z_back, y_back, x_back = cp.where(foreground_array == 2)
            back_coords = [(int(z), int(y), int(x)) for z, y, x in zip(z_back, y_back, x_back)]
            
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
            needed_chunks = self.get_minimal_chunks_for_coordinates(all_z_coords)
            
            # Process each chunk and extract features
            for z in needed_chunks:
                for chunk_coords in needed_chunks[z]:
                    # Compute features for this chunk
                    feature_map, (y_offset, x_offset) = self.compute_features_for_chunk_2d(chunk_coords, speed)
                    
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
                                foreground_features.append(cp.asnumpy(feature_vector))
                    
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
                                background_features.append(cp.asnumpy(feature_vector))
                                
        else:
            # 3D processing - match segmentation chunking logic using compute_3d_chunks
            chunk_size = self.master_chunk
            
            # Memory-efficient approach: compute features only for necessary subarrays
            foreground_features = []
            background_features = []
            
            # Convert foreground_array to CuPy array
            foreground_array_gpu = cp.asarray(foreground_array)
            
            # Find coordinates of foreground and background scribbles
            z_fore = cp.argwhere(foreground_array_gpu == 1)
            z_back = cp.argwhere(foreground_array_gpu == 2)
            
            # If no scribbles, return empty lists
            if len(z_fore) == 0 and len(z_back) == 0:
                return foreground_features, background_features
            
            # Get all chunks using consistent method
            all_chunks = self.compute_3d_chunks(self.master_chunk)
            # Convert chunks to cupy array for vectorized operations
            chunks_array = cp.array(all_chunks)  # Shape: (n_chunks, 6)
            # columns: [z_start, z_end, y_start, y_end, x_start, x_end]
            
            # Combine all scribbles
            all_scribbles = cp.vstack((z_fore, z_back)) if len(z_back) > 0 else z_fore
            
            # For each scribble, find which chunk it belongs to using vectorized operations
            chunks_with_scribbles = set()
            all_scribbles_cpu = cp.asnumpy(all_scribbles)  # Convert once for iteration
            
            for z, y, x in all_scribbles_cpu:
                # Vectorized check: find chunks that contain this scribble
                # Check if scribble falls within each chunk's bounds
                z_in_chunk = (chunks_array[:, 0] <= z) & (z < chunks_array[:, 1])
                y_in_chunk = (chunks_array[:, 2] <= y) & (y < chunks_array[:, 3]) 
                x_in_chunk = (chunks_array[:, 4] <= x) & (x < chunks_array[:, 5])
                
                # Find chunks where all conditions are true
                matching_chunks = z_in_chunk & y_in_chunk & x_in_chunk
                
                # Get the chunk indices that match
                chunk_indices = cp.where(matching_chunks)[0]
                
                # Add matching chunks to set (set automatically handles duplicates)
                for idx in cp.asnumpy(chunk_indices):
                    chunk_coords = tuple(cp.asnumpy(chunks_array[idx]))
                    chunks_with_scribbles.add(chunk_coords)
            
            # Convert set to list
            chunks_with_scribbles = list(chunks_with_scribbles)
            
            # Step 2: Process each chunk that contains scribbles (manual coordinate extraction)
            for chunk_coords in chunks_with_scribbles:
                # Extract chunk boundaries
                z_min, z_max, y_min, y_max, x_min, x_max = chunk_coords
                
                # Extract the subarray (assuming image_3d is already a CuPy array)
                subarray = self.image_3d[z_min:z_max, y_min:y_max, x_min:x_max]
                subarray2 = foreground_array_gpu[z_min:z_max, y_min:y_max, x_min:x_max]
                
                # Compute features for this subarray
                if self.speed:
                    subarray_features = self.compute_feature_maps_gpu(subarray)
                else:
                    subarray_features = self.compute_deep_feature_maps_gpu(subarray)
                
                # Extract foreground features using a direct mask comparison
                local_fore_coords = cp.argwhere(subarray2 == 1)
                for local_z, local_y, local_x in cp.asnumpy(local_fore_coords):
                    feature = subarray_features[int(local_z), int(local_y), int(local_x)]
                    foreground_features.append(cp.asnumpy(feature))
                
                # Extract background features using a direct mask comparison
                local_back_coords = cp.argwhere(subarray2 == 2)
                for local_z, local_y, local_x in cp.asnumpy(local_back_coords):
                    feature = subarray_features[int(local_z), int(local_y), int(local_x)]
                    background_features.append(cp.asnumpy(feature))

        if self.previous_foreground is not None:
            failed = True

            try:
                # Handle foreground features
                if isinstance(foreground_features, list):
                    if len(foreground_features) > 0:
                        # Check if first element is CuPy or NumPy
                        if hasattr(foreground_features[0], 'get'):  # CuPy array
                            foreground_features = cp.stack(foreground_features)
                        else:  # NumPy array
                            import numpy as np
                            foreground_features = cp.asarray(np.stack(foreground_features))
                    else:
                        foreground_features = cp.array([])
                
                # Convert CuPy arrays to NumPy if necessary for consistent handling
                if hasattr(foreground_features, 'get'):
                    foreground_features = foreground_features.get()
                
                # Combine with previous foreground features
                if len(foreground_features) > 0:
                    foreground_features = np.vstack([self.previous_foreground, foreground_features])
                else:
                    foreground_features = self.previous_foreground
                
                failed = False
            except Exception as e:
                print(f"Error combining foreground features: {e}")
                # Keep only new features if combination fails
                if isinstance(foreground_features, list):
                    if len(foreground_features) > 0:
                        # Check if first element is CuPy or NumPy
                        if hasattr(foreground_features[0], 'get'):  # CuPy array
                            foreground_features = cp.stack(foreground_features)
                        else:  # NumPy array
                            import numpy as np
                            foreground_features = cp.asarray(np.stack(foreground_features))
                    else:
                        foreground_features = cp.array([])
                if hasattr(foreground_features, 'get'):
                    foreground_features = foreground_features.get()
            
            try:
                # Handle background features
                if isinstance(background_features, list):
                    if len(background_features) > 0:
                        # Check if first element is CuPy or NumPy
                        if hasattr(background_features[0], 'get'):  # CuPy array
                            background_features = cp.stack(background_features)
                        else:  # NumPy array
                            import numpy as np
                            background_features = cp.asarray(np.stack(background_features))
                    else:
                        background_features = cp.array([])
                
                # Convert CuPy arrays to NumPy if necessary for consistent handling
                if hasattr(background_features, 'get'):
                    background_features = background_features.get()
                
                # Combine with previous background features
                if len(background_features) > 0:
                    background_features = np.vstack([self.previous_background, background_features])
                else:
                    background_features = self.previous_background
                
                failed = False
            except Exception as e:
                print(f"Error combining background features: {e}")
                # Keep only new features if combination fails
                if isinstance(background_features, list):
                    if len(background_features) > 0:
                        # Check if first element is CuPy or NumPy
                        if hasattr(background_features[0], 'get'):  # CuPy array
                            background_features = cp.stack(background_features)
                        else:  # NumPy array
                            import numpy as np
                            background_features = cp.asarray(np.stack(background_features))
                    else:
                        background_features = cp.array([])
                if hasattr(background_features, 'get'):
                    background_features = background_features.get()
                        
            try:
                # Handle foreground coordinates - always combine when we have new ones
                if hasattr(z_fore, 'get'):
                    z_fore_numpy = z_fore.get()
                else:
                    z_fore_numpy = z_fore
                    
                if hasattr(self.previous_z_fore, 'get'):
                    prev_z_fore_numpy = self.previous_z_fore.get()
                else:
                    prev_z_fore_numpy = self.previous_z_fore
                
                # Always combine coordinates when we have new ones
                if len(z_fore_numpy) > 0:  # We have new coordinates
                    z_fore = np.concatenate([prev_z_fore_numpy, z_fore_numpy])
                else:  # No new coordinates, keep old ones
                    z_fore = prev_z_fore_numpy
                    
            except Exception as e:
                print(f"Error combining foreground coordinates: {e}")
                # Fallback: keep new coordinates if combination fails
                if hasattr(z_fore, 'get'):
                    z_fore = z_fore.get()
                
            try:
                # Handle background coordinates - always combine when we have new ones
                if hasattr(z_back, 'get'):
                    z_back_numpy = z_back.get()
                else:
                    z_back_numpy = z_back
                    
                if hasattr(self.previous_z_back, 'get'):
                    prev_z_back_numpy = self.previous_z_back.get()
                else:
                    prev_z_back_numpy = self.previous_z_back
                
                # Always combine coordinates when we have new ones
                if len(z_back_numpy) > 0:  # We have new coordinates
                    z_back = np.concatenate([prev_z_back_numpy, z_back_numpy])
                else:  # No new coordinates, keep old ones
                    z_back = prev_z_back_numpy
                    
            except Exception as e:
                print(f"Error combining background coordinates: {e}")
                # Fallback: keep new coordinates if combination fails
                if hasattr(z_back, 'get'):
                    z_back = z_back.get()
                    
            if failed:
                print("Could not combine new model with old loaded model. Perhaps you are trying to combine a quick model with a deep model? I cannot combine these...")

        if saving:
            return foreground_features, background_features, z_fore, z_back

        # Ensure features are proper arrays for training
        if isinstance(foreground_features, list):
            if len(foreground_features) > 0:
                # Check if first element is CuPy or NumPy
                if hasattr(foreground_features[0], 'get'):  # CuPy array
                    foreground_features = cp.stack(foreground_features)
                else:  # NumPy array
                    import numpy as np
                    foreground_features = cp.asarray(np.stack(foreground_features))
            else:
                foreground_features = cp.array([])

        if isinstance(background_features, list):
            if len(background_features) > 0:
                # Check if first element is CuPy or NumPy
                if hasattr(background_features[0], 'get'):  # CuPy array
                    background_features = cp.stack(background_features)
                else:  # NumPy array
                    import numpy as np
                    background_features = cp.asarray(np.stack(background_features))
            else:
                background_features = cp.array([])

        # Convert to NumPy for sklearn
        if hasattr(foreground_features, 'get'):
            foreground_features = foreground_features.get()
        if hasattr(background_features, 'get'):
            background_features = background_features.get()

        # Validate dimensions before training

        # Ensure we have matching numbers of features and coordinates
        if len(foreground_features) != len(z_fore):
            print(f"Warning: Foreground features ({len(foreground_features)}) and coordinates ({len(z_fore)}) don't match!")
            # Trim to the smaller size
            min_len = min(len(foreground_features), len(z_fore))
            foreground_features = foreground_features[:min_len]
            z_fore = z_fore[:min_len]

        if len(background_features) != len(z_back):
            print(f"Warning: Background features ({len(background_features)}) and coordinates ({len(z_back)}) don't match!")
            # Trim to the smaller size
            min_len = min(len(background_features), len(z_back))
            background_features = background_features[:min_len]
            z_back = z_back[:min_len]

        # Combine features and labels for training
        if len(foreground_features) > 0 and len(background_features) > 0:
            X = np.vstack([foreground_features, background_features])
            y = np.hstack([np.ones(len(z_fore)), np.zeros(len(z_back))])
                        
            # Train the model
            try:
                self.model.fit(X, y)
            except Exception as e:
                print(f"Error during model training: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Not enough features to train the model")

        self.current_speed = speed

        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        print("Done")


    def save_model(self, file_name, foreground_array):

        print("Saving model data")

        foreground_features, background_features, z_fore, z_back = self.train_batch(foreground_array, speed = self.speed, use_gpu = self.use_gpu, use_two = self.use_two, mem_lock = self.mem_lock, saving = True)


        cp.savez(file_name, 
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

        data = cp.load(file_name)

        # Unpack the arrays
        self.previous_foreground = data['foreground_features']
        self.previous_background = data['background_features']
        self.previous_z_fore = data['z_fore']
        self.previous_z_back = data['z_back']
        self.speed = bool(data['speed'])
        self.use_gpu = bool(data['use_gpu'])
        self.use_two = bool(data['use_two'])
        self.mem_lock = bool(data['mem_lock'])

        X = cp.vstack([self.previous_foreground, self.previous_background])
        y = cp.hstack([cp.ones(len(self.previous_z_fore)), cp.zeros(len(self.previous_z_back))])
        X = cp.asnumpy(X)
        y = cp.asnumpy(y)

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
            output = self.compute_feature_maps_gpu_2d(z = z)

        elif not speed:
            output = self.compute_deep_feature_maps_gpu_2d(z = z)

        return output



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

