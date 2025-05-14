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

    def process_chunk(self, chunk_coords):
        """Process a chunk staying in CuPy as much as possible"""
        
        foreground_coords = []  # Keep as list of CuPy coordinates
        background_coords = []
        
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
        
        return foreground_coords, background_coords

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

    def segment_volume(self, array, chunk_size=None, gpu=True):
        """Segment volume using parallel processing of chunks with vectorized chunk creation"""
        
        array = cp.asarray(array)  # Ensure CuPy array
        
        self.realtimechunks = None
        self.map_slice = None
        chunk_size = self.master_chunk
        
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
        
        # Process chunks
        print("Segmenting chunks...")
        
        for i, chunk_start_gpu in enumerate(chunk_starts):
            # Extract values from CuPy array
            z_start = int(chunk_start_gpu[0])  # Convert to regular Python int
            y_start = int(chunk_start_gpu[1])
            x_start = int(chunk_start_gpu[2])
            
            z_end = min(z_start + chunk_size, self.image_3d.shape[0])
            y_end = min(y_start + chunk_size, self.image_3d.shape[1])
            x_end = min(x_start + chunk_size, self.image_3d.shape[2])
            
            coords = [z_start, z_end, y_start, y_end, x_start, x_end]
            
            # Process chunk - returns CuPy arrays
            fore_coords, _ = self.process_chunk(coords)
            
            if len(fore_coords) > 0:
                # Direct indexing with CuPy arrays
                array[fore_coords[:, 0], fore_coords[:, 1], fore_coords[:, 2]] = 255
            
            print(f"Processed {i}/{len(chunk_starts)} chunks")
        
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
        # Only convert to NumPy at the very end for return
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
        
        try:
            from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
            gpu_ml_available = True
        except:
            print("Cannot find cuML, using CPU to segment instead...")
            gpu_ml_available = False
            gpu = False

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

    def train_batch(self, foreground_array, speed=True, use_gpu=True, use_two=False, mem_lock=False):
        """Train directly on foreground and background arrays using GPU acceleration"""
        import cupy as cp
        
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
        
        # Combine features and labels - convert to NumPy for sklearn compatibility
        if foreground_features and background_features:
            X = np.vstack([np.array(foreground_features), np.array(background_features)])
            y = np.hstack([np.ones(len(z_fore_cpu)), np.zeros(len(z_back_cpu))])
        elif foreground_features:
            X = np.array(foreground_features)
            y = np.ones(len(z_fore_cpu))
        elif background_features:
            X = np.array(background_features)
            y = np.zeros(len(z_back_cpu))
        else:
            X = np.array([])
            y = np.array([])
        
        # Train the model
        try:
            self.model.fit(X, y)
        except Exception as e:
            print(f"Error during model training: {e}")
            print(X)
            print(y)
        
        self.current_speed = speed
        
        # Clean up GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        
        print("Done")