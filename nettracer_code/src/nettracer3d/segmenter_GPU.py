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
            n_estimators=100,
            n_jobs=-1,
            max_depth=None
        )

        self.feature_cache = None
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
        self.map_slice = None
        self.prev_z = None
        self.previewing = False

        #  flags to track state
        self._currently_processing = False
        self._skip_next_update = False
        self._last_processed_slice = None
        self.mem_lock = False

        #Adjustable feature map params:
        self.alphas = [1,2,4,8]
        self.windows = 10
        self.dogs = [(1, 2), (2, 4), (4, 8)]
        self.master_chunk = 49

        #Data when loading prev model:
        self.previous_foreground = None
        self.previous_background = None
        self.previous_z_fore = None
        self.previous_z_back = None

    def segment_slice_chunked(self, slice_z, block_size=49):
        """
        A completely standalone method to segment a single z-slice in chunks
        with improved safeguards.
        """
        # Check if we're already processing this slice
        if self._currently_processing and self._currently_processing == slice_z:
            return
        
        # Set processing flag with the slice we're processing
        self._currently_processing = slice_z
        
        try:
            # First attempt to get the feature map
            feature_map = None
            
            try:
                if slice_z in self.feature_cache:
                    feature_map = self.feature_cache[slice_z]
                elif hasattr(self, 'map_slice') and self.map_slice is not None and slice_z == self.current_z:
                    feature_map = self.map_slice
                else:
                    # Generate new feature map
                    try:
                        feature_map = self.get_feature_map_slice(slice_z, self.current_speed, False)
                        self.map_slice = feature_map
                    except Exception as e:
                        print(f"Error generating feature map: {e}")
                        import traceback
                        traceback.print_exc()
                        return  # Exit if we can't generate the feature map
            except:
                # Generate new feature map
                try:
                    feature_map = self.get_feature_map_slice(slice_z, self.current_speed, False)
                    self.map_slice = feature_map
                except Exception as e:
                    print(f"Error generating feature map: {e}")
                    import traceback
                    traceback.print_exc()
                    return  # Exit if we can't generate the feature map
            
            # Check that we have a valid feature map
            if feature_map is None:
                return
            
            # Get dimensions of the slice
            y_size, x_size = self.image_3d.shape[1], self.image_3d.shape[2]
            chunk_count = 0
            
            # Determine if feature_map is a CuPy array
            is_cupy_array = hasattr(feature_map, 'get')
            
            # Process in blocks for chunked feedback
            for y_start in range(0, y_size, block_size):
                if self._currently_processing != slice_z:
                    return
                    
                for x_start in range(0, x_size, block_size):
                    if self._currently_processing != slice_z:
                        return
                        
                    y_end = min(y_start + block_size, y_size)
                    x_end = min(x_start + block_size, x_size)
                    
                    # Create coordinates and features for this block
                    coords = []
                    features_list = []
                    
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            coords.append((slice_z, y, x))
                            features_list.append(feature_map[y, x])
                    
                    # Convert features to NumPy properly based on type
                    if is_cupy_array:
                        # If feature_map is a CuPy array, we need to extract a CuPy array
                        # from the list and then convert it to NumPy
                        try:
                            # Create a CuPy array from the list of feature vectors
                            features_array = cp.stack(features_list)
                            # Convert to NumPy explicitly using .get()
                            features = features_array.get()
                        except Exception as e:
                            print(f"Error converting features to NumPy: {e}")
                            # Fallback: convert each feature individually
                            features = []
                            for feat in features_list:
                                if hasattr(feat, 'get'):
                                    features.append(feat.get())
                                else:
                                    features.append(feat)
                    else:
                        # If it's already a NumPy array, we can use it directly
                        features = features_list
                    
                    # Skip empty blocks
                    if not coords:
                        continue
                    
                    # Predict
                    try:
                        try:
                            predictions = self.model.predict(features)
                        except ValueError:
                            self.feature_cache = None
                            self.map_slice = None
                            return None, None
                        
                        # Split results
                        foreground = set()
                        background = set()
                        
                        for coord, pred in zip(coords, predictions):
                            if pred:
                                foreground.add(coord)
                            else:
                                background.add(coord)
                        
                        # Yield this chunk
                        chunk_count += 1
                        yield foreground, background
                        
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
                        import traceback
                        traceback.print_exc()
        
        finally:
            # Only clear if we're still processing the same slice
            # (otherwise, another slice might have taken over)
            if self._currently_processing == slice_z:
                self._currently_processing = None

    def process_chunk(self, chunk_coords):
        """Process a chunk staying in CuPy as much as possible"""
        
        foreground_coords = []  # Keep as list of CuPy coordinates
        background_coords = []

        if self.previewing or not self.use_two:
        
            if self.realtimechunks is None:
                z_min, z_max = chunk_coords[0], chunk_coords[1]
                y_min, y_max = chunk_coords[2], chunk_coords[3]
                x_min, x_max = chunk_coords[4], chunk_coords[5]
                
                # Create meshgrid using CuPy - already good
                z_range = cp.arange(z_min, z_max)
                y_range = cp.arange(y_min, y_max)
                x_range = cp.arange(x_min, x_max)
                
                # More efficient way to create coordinates
                chunk_coords_array = cp.stack(cp.meshgrid(
                    z_range, y_range, x_range, indexing='ij'
                )).reshape(3, -1).T
                
                # Keep as CuPy array instead of converting to list
                chunk_coords_gpu = chunk_coords_array
            else:
                # Convert list to CuPy array once
                chunk_coords_gpu = cp.array(chunk_coords)
                z_coords = chunk_coords_gpu[:, 0]
                y_coords = chunk_coords_gpu[:, 1]
                x_coords = chunk_coords_gpu[:, 2]
                
                z_min, z_max = cp.min(z_coords).item(), cp.max(z_coords).item()
                y_min, y_max = cp.min(y_coords).item(), cp.max(y_coords).item()
                x_min, x_max = cp.min(x_coords).item(), cp.max(x_coords).item()
            
            # Extract subarray - already good
            subarray = self.image_3d[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
            
            # Compute features
            if self.speed:
                feature_map = self.compute_feature_maps_gpu(subarray)
            else:
                feature_map = self.compute_deep_feature_maps_gpu(subarray)
            
            # Extract features more efficiently
            local_coords = chunk_coords_gpu.copy()
            local_coords[:, 0] -= z_min
            local_coords[:, 1] -= y_min
            local_coords[:, 2] -= x_min
            
            # Vectorized feature extraction
            features_gpu = feature_map[local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]]
            
            features_cpu = cp.asnumpy(features_gpu)
            predictions = self.model.predict(features_cpu)
            
            # Keep coordinates as CuPy arrays
            pred_mask = cp.array(predictions, dtype=bool)
            foreground_coords = chunk_coords_gpu[pred_mask]
            background_coords = chunk_coords_gpu[~pred_mask]

        else:
            # 2D implementation for GPU
            foreground_coords = []
            background_coords = []
            
            # Check if chunk_coords is in the 2D format [z, y_start, y_end, x_start, x_end]
            if len(chunk_coords) == 5:
                z = chunk_coords[0]
                y_start = chunk_coords[1]
                y_end = chunk_coords[2]
                x_start = chunk_coords[3]
                x_end = chunk_coords[4]
                
                # Generate coordinates for this slice or subchunk using the new function
                coords_array = self.twodim_coords(z, y_start, y_end, x_start, x_end)
                
                # Get the feature map for this z-slice
                if self.feature_cache is None:
                    feature_map = self.get_feature_map_slice(z, self.speed, True)  # Use GPU
                elif z not in self.feature_cache and not self.previewing:
                    feature_map = self.get_feature_map_slice(z, self.speed, True)  # Use GPU
                elif (z not in self.feature_cache or self.feature_cache is None) and self.previewing:
                    feature_map = self.map_slice
                    if feature_map is None:
                        return [], []
                else:
                    feature_map = self.feature_cache[z]
                
                # Check if we have a valid feature map
                if feature_map is None:
                    return [], []
                    
                # Extract y and x coordinates from the array
                y_indices = coords_array[:, 1]
                x_indices = coords_array[:, 2]
                
                # Extract features using CuPy indexing
                features_gpu = feature_map[y_indices, x_indices]
                
                # Convert to NumPy for the model
                features_cpu = features_gpu.get()
                
                # Make predictions
                predictions = self.model.predict(features_cpu)
                
                # Create CuPy boolean mask from predictions
                pred_mask = cp.array(predictions, dtype=bool)
                
                # Split into foreground and background using the mask
                fore_coords = coords_array[pred_mask]
                back_coords = coords_array[~pred_mask]
                
                return fore_coords, back_coords
            
        return foreground_coords, background_coords

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
        """Compute feature maps using GPU with CuPy"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        features = []
        if image_3d is None:
            image_3d = self.image_3d  # Assuming this is already a cupy array
        
        original_shape = image_3d.shape
        
        # Gaussian smoothing at different scales
        for sigma in self.alphas:
            smooth = cupy_ndimage.gaussian_filter(image_3d, sigma)
            features.append(smooth)
        
        # Difference of Gaussians
        for (s1, s2) in self.dogs:
            g1 = cupy_ndimage.gaussian_filter(image_3d, s1)
            g2 = cupy_ndimage.gaussian_filter(image_3d, s2)
            dog = g1 - g2
            features.append(dog)
        
        # Gradient computations using cupyx
        gx = cupy_ndimage.sobel(image_3d, axis=2, mode='reflect')  # x direction
        gy = cupy_ndimage.sobel(image_3d, axis=1, mode='reflect')  # y direction
        gz = cupy_ndimage.sobel(image_3d, axis=0, mode='reflect')  # z direction
        
        # Gradient magnitude
        gradient_magnitude = cp.sqrt(gx**2 + gy**2 + gz**2)
        features.append(gradient_magnitude)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                feat_adjusted = cp.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                features[i] = feat_adjusted
        
        return cp.stack(features, axis=-1)

    def compute_deep_feature_maps_gpu(self, image_3d=None):
        """Compute feature maps using GPU"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        features = []
        if image_3d is None:
            image_3d = self.image_3d  # Assuming this is already a cupy array
        original_shape = image_3d.shape
        
        # Gaussian and DoG using cupyx
        for sigma in self.alphas:
            smooth = cupy_ndimage.gaussian_filter(image_3d, sigma)
            features.append(smooth)
        
        # Difference of Gaussians
        for (s1, s2) in self.dogs:
            g1 = cupy_ndimage.gaussian_filter(image_3d, s1)
            g2 = cupy_ndimage.gaussian_filter(image_3d, s2)
            dog = g1 - g2
            features.append(dog)
        
        # Local statistics using cupyx's convolve
        window_size = self.windows
        kernel = cp.ones((window_size, window_size, window_size)) / (window_size**3)
        
        # Local mean
        local_mean = cupy_ndimage.convolve(image_3d, kernel, mode='reflect')
        features.append(local_mean)
        
        # Local variance
        mean = cp.mean(image_3d)
        local_var = cupy_ndimage.convolve((image_3d - mean)**2, kernel, mode='reflect')
        features.append(local_var)
        
        # Gradient computations using cupyx
        gx = cupy_ndimage.sobel(image_3d, axis=2, mode='reflect')
        gy = cupy_ndimage.sobel(image_3d, axis=1, mode='reflect')
        gz = cupy_ndimage.sobel(image_3d, axis=0, mode='reflect')
        
        # Gradient magnitude
        gradient_magnitude = cp.sqrt(gx**2 + gy**2 + gz**2)
        features.append(gradient_magnitude)
        
        # Second-order gradients
        gxx = cupy_ndimage.sobel(gx, axis=2, mode='reflect')
        gyy = cupy_ndimage.sobel(gy, axis=1, mode='reflect')
        gzz = cupy_ndimage.sobel(gz, axis=0, mode='reflect')
        
        # Laplacian (sum of second derivatives)
        laplacian = gxx + gyy + gzz
        features.append(laplacian)
        
        # Hessian determinant
        hessian_det = gxx * gyy * gzz
        features.append(hessian_det)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                feat_adjusted = cp.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                features[i] = feat_adjusted
        
        return cp.stack(features, axis=-1)


    def compute_feature_maps_gpu_2d(self, z=None):
        """Compute feature maps for 2D images using GPU with CuPy"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        # Extract 2D slice if z is provided, otherwise use the image directly
        if z is not None:
            image_2d = cp.asarray(self.image_3d[z, :, :])
        else:
            # Assuming image_2d is already available or passed
            image_2d = cp.asarray(self.image_2d)
        
        original_shape = image_2d.shape
        features = []
        
        # Gaussian smoothing at different scales
        for sigma in self.alphas:
            smooth = cupy_ndimage.gaussian_filter(image_2d, sigma)
            features.append(smooth)
        
        # Difference of Gaussians
        for (s1, s2) in self.dogs:
            g1 = cupy_ndimage.gaussian_filter(image_2d, s1)
            g2 = cupy_ndimage.gaussian_filter(image_2d, s2)
            dog = g1 - g2
            features.append(dog)
        
        # Gradient computations for 2D
        gx = cupy_ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
        gy = cupy_ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
        
        # Gradient magnitude (2D version - no z component)
        gradient_magnitude = cp.sqrt(gx**2 + gy**2)
        features.append(gradient_magnitude)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Check dimensionality and expand if needed
                if len(feat.shape) < len(original_shape):
                    feat_adjusted = feat
                    missing_dims = len(original_shape) - len(feat.shape)
                    for _ in range(missing_dims):
                        feat_adjusted = cp.expand_dims(feat_adjusted, axis=0)
                    
                    if feat_adjusted.shape != original_shape:
                        raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                    
                    features[i] = feat_adjusted
        
        # Stack features along a new dimension
        result = cp.stack(features, axis=-1)
        
        # Optional: Return as numpy array if needed
        # result = cp.asnumpy(result)
        
        return result

    def compute_deep_feature_maps_gpu_2d(self, z=None):
        """Compute 2D feature maps using GPU with CuPy"""
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_ndimage
        
        # Extract 2D slice if z is provided, otherwise use the image directly
        if z is not None:
            image_2d = cp.asarray(self.image_3d[z, :, :])
        else:
            # Assuming image_2d is already available or passed
            image_2d = cp.asarray(self.image_2d)
        
        original_shape = image_2d.shape
        features = []
        
        # Stage 1: Compute all base features
        
        # Gaussian smoothing
        gaussian_results = {}
        for sigma in self.alphas:
            smooth = cupy_ndimage.gaussian_filter(image_2d, sigma)
            gaussian_results[sigma] = smooth
            features.append(smooth)
        
        # Difference of Gaussians
        for (s1, s2) in self.dogs:
            g1 = cupy_ndimage.gaussian_filter(image_2d, s1)
            g2 = cupy_ndimage.gaussian_filter(image_2d, s2)
            dog = g1 - g2
            features.append(dog)
        
        # Local statistics using 2D kernel
        window_size = self.windows
        kernel = cp.ones((window_size, window_size)) / (window_size**2)
        
        # Local mean
        local_mean = cupy_ndimage.convolve(image_2d, kernel, mode='reflect')
        features.append(local_mean)
        
        # Local variance
        mean = cp.mean(image_2d)
        local_var = cupy_ndimage.convolve((image_2d - mean)**2, kernel, mode='reflect')
        features.append(local_var)
        
        # First-order gradients
        gx = cupy_ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
        gy = cupy_ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
        
        # Gradient magnitude
        gradient_magnitude = cp.sqrt(gx**2 + gy**2)
        features.append(gradient_magnitude)
        
        # Stage 2: Compute derived features
        
        # Second-order gradients
        gxx = cupy_ndimage.sobel(gx, axis=1, mode='reflect')
        gyy = cupy_ndimage.sobel(gy, axis=0, mode='reflect')
        
        # Cross derivatives for Hessian determinant
        gxy = cupy_ndimage.sobel(gx, axis=0, mode='reflect')
        gyx = cupy_ndimage.sobel(gy, axis=1, mode='reflect')
        
        # Laplacian (sum of second derivatives)
        laplacian = gxx + gyy
        features.append(laplacian)
        
        # Hessian determinant
        hessian_det = gxx * gyy - gxy * gyx
        features.append(hessian_det)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Check dimensionality and expand if needed
                if len(feat.shape) < len(original_shape):
                    feat_adjusted = feat
                    missing_dims = len(original_shape) - len(feat.shape)
                    for _ in range(missing_dims):
                        feat_adjusted = cp.expand_dims(feat_adjusted, axis=0)
                    
                    if feat_adjusted.shape != original_shape:
                        raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                    
                    features[i] = feat_adjusted
        
        # Stack all features along a new dimension
        result = cp.stack(features, axis=-1)
        
        # Optional: Return as numpy array if needed
        # result = cp.asnumpy(result)
        
        return result

    def segment_volume(self, array, chunk_size=None, gpu=True):
        """Segment volume using parallel processing of chunks with vectorized chunk creation"""
        
        array = cp.asarray(array)  # Ensure CuPy array
        
        self.realtimechunks = None
        self.map_slice = None
        chunk_size = self.master_chunk
        
        def create_2d_chunks():
            """
            Create chunks by z-slices for 2D processing.
            Each chunk is a complete z-slice with all y,x coordinates,
            unless the slice exceeds 262144 pixels, in which case it's divided into subchunks.
            
            Returns:
                List of chunks where each chunk contains the parameters needed for processing.
                Format depends on subchunking: 
                - No subchunking: [y_dim, x_dim, z, total_pixels, None]
                - Y subchunking: [y_dim, x_dim, z, None, ['y', start_y, end_y]]
                - X subchunking: [y_dim, x_dim, z, None, ['x', start_x, end_x]]
            """
            MAX_CHUNK_SIZE = 262144
            chunks = []
            
            for z in range(self.image_3d.shape[0]):
                # Get the dimensions of this z-slice
                y_dim = self.image_3d.shape[1]
                x_dim = self.image_3d.shape[2]
                total_pixels = y_dim * x_dim
                
                # If the slice is small enough, do not subchunk
                if total_pixels <= MAX_CHUNK_SIZE:
                    chunks.append([z, 0, y_dim, 0, x_dim])  # [z_start, y_start, y_end, x_start, x_end]
                else:
                    # Determine which dimension to divide (the largest one)
                    largest_dim = 'y' if y_dim >= x_dim else 'x'
                    
                    # Calculate how many divisions we need
                    num_divisions = int(cp.ceil(total_pixels / MAX_CHUNK_SIZE))
                    
                    # Calculate the approx size of each division along the largest dimension
                    if largest_dim == 'y':
                        div_size = int(cp.ceil(y_dim / num_divisions))
                        # Create subchunks by dividing the y-dimension
                        for i in range(0, y_dim, div_size):
                            end_i = min(i + div_size, y_dim)
                            chunks.append([z, i, end_i, 0, x_dim])  # [z, y_start, y_end, x_start, x_end]
                    else:  # largest_dim == 'x'
                        div_size = int(cp.ceil(x_dim / num_divisions))
                        # Create subchunks by dividing the x-dimension
                        for i in range(0, x_dim, div_size):
                            end_i = min(i + div_size, x_dim)
                            chunks.append([z, 0, y_dim, i, end_i])  # [z, y_start, y_end, x_start, x_end]
            
            return chunks
        
        print("Chunking data...")
        
        if not self.use_two:
            # 3D Processing - Create chunks for 3D volume
            # Round to nearest multiple of 32 for better memory alignment
            chunk_size = ((chunk_size + 15) // 32) * 32
            
            # Calculate number of chunks in each dimension
            z_chunks = (self.image_3d.shape[0] + chunk_size - 1) // chunk_size
            y_chunks = (self.image_3d.shape[1] + chunk_size - 1) // chunk_size
            x_chunks = (self.image_3d.shape[2] + chunk_size - 1) // chunk_size
            
            # Create start indices for all chunks at once using CuPy
            chunk_starts = cp.array(cp.meshgrid(
                cp.arange(z_chunks) * chunk_size,
                cp.arange(y_chunks) * chunk_size,
                cp.arange(x_chunks) * chunk_size,
                indexing='ij'
            )).reshape(3, -1).T
            
            chunks = []
            for chunk_start_gpu in chunk_starts:
                # Extract values from CuPy array
                z_start = int(chunk_start_gpu[0])  # Convert to regular Python int
                y_start = int(chunk_start_gpu[1])
                x_start = int(chunk_start_gpu[2])
                
                z_end = min(z_start + chunk_size, self.image_3d.shape[0])
                y_end = min(y_start + chunk_size, self.image_3d.shape[1])
                x_end = min(x_start + chunk_size, self.image_3d.shape[2])
                
                coords = [z_start, z_end, y_start, y_end, x_start, x_end]
                chunks.append(coords)
        else:
            # 2D Processing - Create chunks by z-slices
            chunks = create_2d_chunks()
            self.feature_cache = None  # Reset feature cache for 2D processing
        
        # Process chunks
        print("Segmenting chunks...")
        
        for i, chunk in enumerate(chunks):
            # Process chunk - returns CuPy arrays of coordinates
            fore_coords, _ = self.process_chunk(chunk)
            
            if isinstance(fore_coords, list) and len(fore_coords) == 0:
                # Skip empty results
                pass
            elif hasattr(fore_coords, 'shape') and len(fore_coords) > 0:
                # Direct indexing with CuPy arrays
                try:
                    array[fore_coords[:, 0], fore_coords[:, 1], fore_coords[:, 2]] = 255
                except IndexError as e:
                    print(f"Index error when updating array: {e}")
                    # Fallback to a safer but slower approach
                    for coord in fore_coords:
                        try:
                            z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
                            if 0 <= z < array.shape[0] and 0 <= y < array.shape[1] and 0 <= x < array.shape[2]:
                                array[z, y, x] = 255
                        except Exception as inner_e:
                            print(f"Error updating coordinate {coord}: {inner_e}")
            
            # Memory management - release reference to chunk data
            if i % 10 == 0:  # Do periodic memory cleanup
                cp.get_default_memory_pool().free_all_blocks()
            
            print(f"Processed {i+1}/{len(chunks)} chunks")
        
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
        # Convert to NumPy at the very end for return
        return cp.asnumpy(array)

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
            # Instead of setting to None, check if we already have it in the cache
            if hasattr(self, 'feature_cache') and self.feature_cache is not None:
                if self.current_z not in self.feature_cache:
                    self.map_slice = None
            self._currently_segmenting = None
        
        # Update previous z
        self.prev_z = z


    def get_realtime_chunks(self, chunk_size=49):
        
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


    def segment_volume_realtime(self, gpu=True):
        """Segment volume in realtime using CuPy for GPU acceleration"""
        import cupy as cp
        
        #try:
            #from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
            #gpu_ml_available = True
        #except:
            #print("Cannot find cuML, using CPU to segment instead...")
            #gpu_ml_available = False
            #gpu = False

        if self.realtimechunks is None:
            self.get_realtime_chunks()
        else:
            for chunk_pos in self.realtimechunks:  # chunk_pos is the (z, y_start, x_start) tuple
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
                # Find nearest chunk in current Z plane using the chunk positions from the key
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
                # Find nearest chunk in target Z plane
                z_chunks = [(pos, info) for pos, info in chunk_dict.items() 
                           if pos[0] == target_z and not info['processed']]
                nearest = min(z_chunks, 
                            key=lambda x: ((x[0][1] - curr_y) ** 2 + 
                                         (x[0][2] - curr_x) ** 2))
                return nearest[0]
            
            return None
        
        while True:
            # Find nearest unprocessed chunk using class attributes
            chunk_idx = get_nearest_unprocessed_chunk(self)
            if chunk_idx is None:
                break
                
            # Process the chunk directly
            chunk = chunk_dict[chunk_idx]
            chunk['processed'] = True
            coords = chunk['coords']

            # Use CuPy for meshgrid
            coords_array = cp.stack(cp.meshgrid(
                cp.array([chunk['z']]),
                cp.arange(coords[0], coords[1]),
                cp.arange(coords[2], coords[3]),
                indexing='ij'
            )).reshape(3, -1).T

            # Convert to CPU for further processing - add cp.asnumpy() here
            coords = list(map(tuple, cp.asnumpy(coords_array)))
            
            # Process the chunk directly based on whether GPU is available
            fore, back = self.process_chunk(coords)
            
            # Yield the results
            yield cp.asnumpy(fore), cp.asnumpy(back)


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

        self.speed = speed
        self.cur_gpu = use_gpu
        self.realtimechunks = None  # dump ram
        
        self.mem_lock = mem_lock
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            max_depth=None
        )

        if use_two:

            #changed = [] #Track which slices need feature maps

            if not self.use_two: #Clarifies if we need to redo feature cache for 2D
                self.feature_cache = None
                self.use_two = True

            self.feature_cache = None #Decided this should reset, can remove this line to have it retain prev feature maps
            self.two_slices = []

            if self.feature_cache == None:
                self.feature_cache = {}

            foreground_array = cp.asarray(foreground_array)

            # Get foreground coordinates and features
            z_fore, y_fore, x_fore = cp.where(foreground_array == 1)

            z_fore_cpu = cp.asnumpy(z_fore)
            y_fore_cpu = cp.asnumpy(y_fore)
            x_fore_cpu = cp.asnumpy(x_fore)

            fore_coords = list(zip(z_fore_cpu, y_fore_cpu, x_fore_cpu))
            
            # Get background coordinates and features
            z_back, y_back, x_back = cp.where(foreground_array == 2)

            z_back_cpu = cp.asnumpy(z_back)
            y_back_cpu = cp.asnumpy(y_back)
            x_back_cpu = cp.asnumpy(x_back)

            back_coords = list(zip(z_back_cpu, y_back_cpu, x_back_cpu))

            foreground_features = []
            background_features = []

            z_fores = self.organize_by_z(fore_coords)
            z_backs = self.organize_by_z(back_coords)
            slices = set(list(z_fores.keys()) + list(z_backs.keys()))

            for z in slices:


                current_map = self.get_feature_map_slice(z, speed, use_gpu)

                if z in z_fores:
                
                    for y, x in z_fores[z]:
                        # Get the feature vector for this foreground point
                        feature_vector = current_map[y, x]
                        
                        # Add to our collection
                        foreground_features.append(cp.asnumpy(feature_vector))

                if z in z_backs:
                
                    for y, x in z_backs[z]:
                        # Get the feature vector for this foreground point
                        feature_vector = current_map[y, x]
                    
                        # Add to our collection
                        background_features.append(cp.asnumpy(feature_vector))

        else:
        
            box_size = self.master_chunk
            
            # Memory-efficient approach: compute features only for necessary subarrays
            foreground_features = []
            background_features = []
            
            # Convert foreground_array to CuPy array
            foreground_array_gpu = cp.asarray(foreground_array)
            
            # Find coordinates of foreground and background scribbles
            z_fore = cp.argwhere(foreground_array_gpu == 1)
            z_back = cp.argwhere(foreground_array_gpu == 2)
            
            # Convert back to NumPy for compatibility with the rest of the code
            z_fore_cpu = cp.asnumpy(z_fore)
            z_back_cpu = cp.asnumpy(z_back)
            
            # If no scribbles, return empty lists
            if len(z_fore_cpu) == 0 and len(z_back_cpu) == 0:
                return foreground_features, background_features
            
            # Get dimensions of the input array
            depth, height, width = foreground_array.shape
            
            # Determine the minimum number of boxes needed to cover all scribbles
            half_box = box_size // 2
            
            # Step 1: Find the minimum set of boxes that cover all scribbles
            # We'll divide the volume into a grid of boxes of size box_size
            
            # Calculate how many boxes are needed in each dimension
            z_grid_size = (depth + box_size - 1) // box_size
            y_grid_size = (height + box_size - 1) // box_size
            x_grid_size = (width + box_size - 1) // box_size
            
            # Track which grid cells contain scribbles
            grid_cells_with_scribbles = set()
            
            # Map original coordinates to grid cells
            for z, y, x in cp.vstack((z_fore_cpu, z_back_cpu)) if len(z_back_cpu) > 0 else z_fore_cpu:
                grid_z = int(z // box_size)
                grid_y = int(y // box_size)
                grid_x = int(x // box_size)
                grid_cells_with_scribbles.add((grid_z, grid_y, grid_x))
            
            # Step 2: Process each grid cell that contains scribbles
            for grid_z, grid_y, grid_x in grid_cells_with_scribbles:
                # Calculate the boundaries of this grid cell
                z_min = grid_z * box_size
                y_min = grid_y * box_size
                x_min = grid_x * box_size
                
                z_max = min(z_min + box_size, depth)
                y_max = min(y_min + box_size, height)
                x_max = min(x_min + box_size, width)
                
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
                # Make sure foreground_features is a NumPy array before vstack
                if isinstance(foreground_features, list):
                    foreground_features = np.array(foreground_features)
                
                # Convert CuPy arrays to NumPy if necessary
                if hasattr(foreground_features, 'get'):
                    foreground_features = foreground_features.get()
                
                foreground_features = np.vstack([self.previous_foreground, foreground_features])
                failed = False
            except Exception as e:
                pass
            
            try:
                # Make sure background_features is a NumPy array before vstack
                if isinstance(background_features, list):
                    background_features = np.array(background_features)
                
                # Convert CuPy arrays to NumPy if necessary
                if hasattr(background_features, 'get'):
                    background_features = background_features.get()
                
                background_features = np.vstack([self.previous_background, background_features])
                failed = False
            except Exception as e:
                pass            
            try:
                # Ensure coordinate arrays are NumPy arrays
                if hasattr(z_fore_cpu, 'get'):
                    z_fore_cpu = z_fore_cpu.get()
                if hasattr(self.previous_z_fore, 'get'):
                    self.previous_z_fore = self.previous_z_fore.get()
                
                z_fore_cpu = np.concatenate([self.previous_z_fore, z_fore_cpu])
            except Exception as e:
                pass            
            try:
                # Ensure coordinate arrays are NumPy arrays
                if hasattr(z_back_cpu, 'get'):
                    z_back_cpu = z_back_cpu.get()
                if hasattr(self.previous_z_back, 'get'):
                    self.previous_z_back = self.previous_z_back.get()
                
                z_back_cpu = np.concatenate([self.previous_z_back, z_back_cpu])
            except Exception as e:
                pass            
            if failed:
                print("Could not combine new model with old loaded model. Perhaps you are trying to combine a quick model with a deep model? I cannot combine these...")

        if saving:
            # Make sure to return NumPy arrays, not CuPy arrays
            if hasattr(foreground_features, 'get'):
                foreground_features = foreground_features.get()
            if hasattr(background_features, 'get'):
                background_features = background_features.get()
            if hasattr(z_fore_cpu, 'get'):
                z_fore_cpu = z_fore_cpu.get()
            if hasattr(z_back_cpu, 'get'):
                z_back_cpu = z_back_cpu.get()
            
            return foreground_features, background_features, z_fore_cpu, z_back_cpu
        
        # Make sure foreground_features and background_features are NumPy arrays
        if isinstance(foreground_features, list):
            foreground_features = np.array(foreground_features)
        elif hasattr(foreground_features, 'get'):
            foreground_features = foreground_features.get()
        
        if isinstance(background_features, list):
            background_features = np.array(background_features)
        elif hasattr(background_features, 'get'):
            background_features = background_features.get()
        
        # Combine features and labels for training
        X = np.vstack([foreground_features, background_features])
        y = np.hstack([np.ones(len(z_fore_cpu)), np.zeros(len(z_back_cpu))])
        
        # Train the model
        try:
            self.model.fit(X, y)
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
        
        self.current_speed = speed
        
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
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

        print(f"Model data saved to {file_name}. Please retrain current model prior to segmentation.")


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

