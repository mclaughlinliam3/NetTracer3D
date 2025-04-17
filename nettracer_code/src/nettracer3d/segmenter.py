from sklearn.ensemble import RandomForestClassifier
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy import ndimage
import multiprocessing
from collections import defaultdict

class InteractiveSegmenter:
    def __init__(self, image_3d, use_gpu=False):
        self.image_3d = image_3d
        self.patterns = []

        try:
            self.use_gpu = use_gpu and cp.cuda.is_available()
        except:
            self.use_gpu = False
        if self.use_gpu:
            try:
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            except:
                pass
            self.image_gpu = cp.asarray(image_3d)
            try:
                self.model = cuRandomForestClassifier(
                    n_estimators=100,
                    max_depth=None
                )
            except:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    n_jobs=-1,
                    max_depth=None
                )

        else:

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

    def segment_slice_chunked(self, slice_z, block_size = 49):
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
                        # Cache the feature map for future use
                        #if not hasattr(self, 'feature_cache'):
                            #self.feature_cache = {}
                        #self.feature_cache[slice_z] = feature_map
                    except Exception as e:
                        print(f"Error generating feature map: {e}")
                        import traceback
                        traceback.print_exc()
                        return  # Exit if we can't generate the feature map
            except:
                # Generate new feature map
                #self.feature_cache = {}
                try:
                    feature_map = self.get_feature_map_slice(slice_z, self.current_speed, False)
                    self.map_slice = feature_map
                    # Cache the feature map for future use
                    #if not hasattr(self, 'feature_cache'):
                        #self.feature_cache = {}
                    #self.feature_cache[slice_z] = feature_map
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
                    features = []
                    
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            coords.append((slice_z, y, x))
                            features.append(feature_map[y, x])
                    
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

    def compute_deep_feature_maps_cpu(self, image_3d = None):
        """Compute feature maps using CPU"""
        features = []
        if image_3d is None:
            image_3d = self.image_3d
        original_shape = image_3d.shape
        
        # Gaussian and DoG using scipy
        #print("Obtaining gaussians")
        for sigma in self.alphas:
            smooth = ndimage.gaussian_filter(image_3d, sigma)
            features.append(smooth)

        # Difference of Gaussians
        for (s1, s2) in self.dogs:
            g1 = ndimage.gaussian_filter(image_3d, s1)
            g2 = ndimage.gaussian_filter(image_3d, s2)
            dog = g1 - g2
            features.append(dog)
        
        #print("Computing local statistics")
        # Local statistics using scipy's convolve
        window_size = self.windows
        kernel = np.ones((window_size, window_size, window_size)) / (window_size**3)
        
        # Local mean
        local_mean = ndimage.convolve(image_3d, kernel, mode='reflect')
        features.append(local_mean)
        
        # Local variance
        mean = np.mean(image_3d)
        local_var = ndimage.convolve((image_3d - mean)**2, kernel, mode='reflect')
        features.append(local_var)
        
        #print("Computing sobel and gradients")
        # Gradient computations using scipy
        gx = ndimage.sobel(image_3d, axis=2, mode='reflect')
        gy = ndimage.sobel(image_3d, axis=1, mode='reflect')
        gz = ndimage.sobel(image_3d, axis=0, mode='reflect')
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        features.append(gradient_magnitude)
        
        #print("Computing second-order features")
        # Second-order gradients
        gxx = ndimage.sobel(gx, axis=2, mode='reflect')
        gyy = ndimage.sobel(gy, axis=1, mode='reflect')
        gzz = ndimage.sobel(gz, axis=0, mode='reflect')
        
        # Laplacian (sum of second derivatives)
        laplacian = gxx + gyy + gzz
        features.append(laplacian)
        
        # Hessian determinant
        hessian_det = gxx * gyy * gzz
        features.append(hessian_det)
        
        #print("Verifying shapes")
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                feat_adjusted = np.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_deep_feature_maps_cpu_parallel(self, image_3d=None):
        """Compute deep feature maps using CPU with thread-based parallelism"""
        if image_3d is None:
            image_3d = self.image_3d
        
        original_shape = image_3d.shape
        
        # Use ThreadPoolExecutor for parallelization
        with ThreadPoolExecutor(max_workers=min(7, multiprocessing.cpu_count())) as executor:
            # Stage 1: Independent computations that can be parallelized
            futures = []
            
            # Gaussian smoothing
            def compute_gaussian(sigma):
                return ndimage.gaussian_filter(image_3d, sigma)
            
            for sigma in self.alphas:
                future = executor.submit(compute_gaussian, sigma)
                futures.append(('gaussian', sigma, future))

            def compute_dog_local(img, s1, s2):
                g1 = ndimage.gaussian_filter(img, s1)
                g2 = ndimage.gaussian_filter(img, s2)
                return g1 - g2

            # Difference of Gaussians
            for (s1, s2) in self.dogs:
                
                future = executor.submit(compute_dog_local, image_3d, s1, s2)
                futures.append(('dog', s1, future))
            
            # Local statistics computation
            def compute_local_mean():
                window_size = self.windows
                kernel = np.ones((window_size, window_size, window_size)) / (window_size**3)
                return ndimage.convolve(image_3d, kernel, mode='reflect')
            
            future = executor.submit(compute_local_mean)
            futures.append(('local_mean', None, future))
            
            def compute_local_variance():
                window_size = self.windows
                kernel = np.ones((window_size, window_size, window_size)) / (window_size**3)
                mean = np.mean(image_3d)
                return ndimage.convolve((image_3d - mean)**2, kernel, mode='reflect')
            
            future = executor.submit(compute_local_variance)
            futures.append(('local_var', None, future))
            
            # Gradient computation
            def compute_gradients():
                gx = ndimage.sobel(image_3d, axis=2, mode='reflect')
                gy = ndimage.sobel(image_3d, axis=1, mode='reflect')
                gz = ndimage.sobel(image_3d, axis=0, mode='reflect')
                return gx, gy, gz
            
            future = executor.submit(compute_gradients)
            futures.append(('gradients', None, future))
            
            # Collect results for the independent computations
            results = {}
            for task_type, params, future in futures:
                try:
                    result = future.result()
                    if task_type == 'gradients':
                        # Store the gradient components separately
                        gx, gy, gz = result
                        results['gx'] = gx
                        results['gy'] = gy
                        results['gz'] = gz
                    else:
                        results[f"{task_type}_{params}" if params is not None else task_type] = result
                except Exception as e:
                    raise RuntimeError(f"Error in task {task_type}: {str(e)}")
            
            # Stage 2: Dependent computations that need results from Stage 1
            futures = []
            
            # Gradient magnitude (depends on gradients)
            def compute_gradient_magnitude(gx, gy, gz):
                return np.sqrt(gx**2 + gy**2 + gz**2)
            
            future = executor.submit(compute_gradient_magnitude, 
                                   results['gx'], results['gy'], results['gz'])
            futures.append(('gradient_magnitude', None, future))
            
            # Second-order gradients (depend on first gradients)
            def compute_second_derivatives(gx, gy, gz):
                gxx = ndimage.sobel(gx, axis=2, mode='reflect')
                gyy = ndimage.sobel(gy, axis=1, mode='reflect')
                gzz = ndimage.sobel(gz, axis=0, mode='reflect')
                return gxx, gyy, gzz
            
            future = executor.submit(compute_second_derivatives, 
                                   results['gx'], results['gy'], results['gz'])
            futures.append(('second_derivatives', None, future))
            
            # Collect results for the dependent computations
            for task_type, params, future in futures:
                try:
                    result = future.result()
                    if task_type == 'second_derivatives':
                        # Store the second derivative components separately
                        gxx, gyy, gzz = result
                        results['gxx'] = gxx
                        results['gyy'] = gyy
                        results['gzz'] = gzz
                    else:
                        results[task_type] = result
                except Exception as e:
                    raise RuntimeError(f"Error in task {task_type}: {str(e)}")
            
            # Stage 3: Final computations that depend on Stage 2 results
            futures = []
            
            # Laplacian and Hessian determinant (depend on second derivatives)
            def compute_laplacian(gxx, gyy, gzz):
                return gxx + gyy + gzz
            
            future = executor.submit(compute_laplacian, 
                                   results['gxx'], results['gyy'], results['gzz'])
            futures.append(('laplacian', None, future))
            
            def compute_hessian_det(gxx, gyy, gzz):
                return gxx * gyy * gzz
            
            future = executor.submit(compute_hessian_det, 
                                   results['gxx'], results['gyy'], results['gzz'])
            futures.append(('hessian_det', None, future))
            
            # Collect final results
            for task_type, params, future in futures:
                try:
                    result = future.result()
                    results[task_type] = result
                except Exception as e:
                    raise RuntimeError(f"Error in task {task_type}: {str(e)}")
        
        # Organize results in the expected order
        features = []
        
        # Add Gaussian features
        for sigma in self.alphas:
            features.append(results[f'gaussian_{sigma}'])

        for sigma in self.dogs:
            features.append(results[f'dog_{sigma[0]}'])
        
        # Add local statistics
        features.append(results['local_mean'])
        features.append(results['local_var'])
        
        # Add gradient magnitude
        features.append(results['gradient_magnitude'])
        
        # Add Laplacian and Hessian determinant
        features.append(results['laplacian'])
        features.append(results['hessian_det'])
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                feat_adjusted = np.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_deep_feature_maps_cpu_2d(self, z = None):
        """Compute 2D feature maps using CPU"""
        features = []

        image_2d = self.image_3d[z, :, :]
        original_shape = image_2d.shape
        
        # Gaussian using scipy
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            smooth = ndimage.gaussian_filter(image_2d, sigma)
            features.append(smooth)
        
        # Local statistics using scipy's convolve - adjusted for 2D
        window_size = 5
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        
        # Local mean
        local_mean = ndimage.convolve(image_2d, kernel, mode='reflect')
        features.append(local_mean)
        
        # Local variance
        mean = np.mean(image_2d)
        local_var = ndimage.convolve((image_2d - mean)**2, kernel, mode='reflect')
        features.append(local_var)
        
        # Gradient computations using scipy - adjusted axes for 2D
        gx = ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
        gy = ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
        
        # Gradient magnitude (2D version)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        features.append(gradient_magnitude)
        
        # Second-order gradients
        gxx = ndimage.sobel(gx, axis=1, mode='reflect')
        gyy = ndimage.sobel(gy, axis=0, mode='reflect')
        
        # Laplacian (sum of second derivatives) - 2D version
        laplacian = gxx + gyy
        features.append(laplacian)
        
        # Hessian determinant - 2D version
        hessian_det = gxx * gyy - ndimage.sobel(gx, axis=0, mode='reflect') * ndimage.sobel(gy, axis=1, mode='reflect')
        features.append(hessian_det)
        
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Check dimensionality and expand if needed
                if len(feat.shape) < len(original_shape):
                    feat_adjusted = feat
                    missing_dims = len(original_shape) - len(feat.shape)
                    for _ in range(missing_dims):
                        feat_adjusted = np.expand_dims(feat_adjusted, axis=0)
                    
                    if feat_adjusted.shape != original_shape:
                        raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                    
                    features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_deep_feature_maps_cpu_2d_parallel(self, z=None):
        """Compute 2D feature maps using CPU with thread-based parallelism"""
        image_2d = self.image_3d[z, :, :]
        original_shape = image_2d.shape
        
        # Use ThreadPoolExecutor for parallelization
        with ThreadPoolExecutor(max_workers=min(7, multiprocessing.cpu_count())) as executor:
            # Stage 1: Independent computations that can be parallelized
            futures = []
            
            # Gaussian smoothing
            def compute_gaussian(sigma):
                return ndimage.gaussian_filter(image_2d, sigma)
            
            for sigma in self.alphas:
                future = executor.submit(compute_gaussian, sigma)
                futures.append(('gaussian', sigma, future))

            # Difference of Gaussians
            def compute_dog(s1, s2):
                g1 = ndimage.gaussian_filter(image_2d, s1)
                g2 = ndimage.gaussian_filter(image_2d, s2)
                return g1 - g2
            
            dog_pairs = self.dogs
            for (s1, s2) in dog_pairs:
                future = executor.submit(compute_dog, s1, s2)
                futures.append(('dog', s1, future))
            
            # Local statistics computation
            def compute_local_mean():
                window_size = self.windows
                kernel = np.ones((window_size, window_size)) / (window_size**2)
                return ndimage.convolve(image_2d, kernel, mode='reflect')
            
            future = executor.submit(compute_local_mean)
            futures.append(('local_mean', None, future))
            
            def compute_local_variance():
                window_size = self.windows
                kernel = np.ones((window_size, window_size)) / (window_size**2)
                mean = np.mean(image_2d)
                return ndimage.convolve((image_2d - mean)**2, kernel, mode='reflect')
            
            future = executor.submit(compute_local_variance)
            futures.append(('local_var', None, future))
            
            # Gradient computation
            def compute_gradients():
                gx = ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
                gy = ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
                return gx, gy
            
            future = executor.submit(compute_gradients)
            futures.append(('gradients', None, future))
            
            # Collect results for the independent computations
            results = {}
            for task_type, params, future in futures:
                try:
                    result = future.result()
                    if task_type == 'gradients':
                        # Store the gradient components separately
                        gx, gy = result
                        results['gx'] = gx
                        results['gy'] = gy
                    else:
                        results[f"{task_type}_{params}" if params is not None else task_type] = result
                except Exception as e:
                    raise RuntimeError(f"Error in task {task_type}: {str(e)}")
            
            # Stage 2: Dependent computations that need results from Stage 1
            futures = []
            
            # Gradient magnitude (depends on gradients)
            def compute_gradient_magnitude(gx, gy):
                return np.sqrt(gx**2 + gy**2)
            
            future = executor.submit(compute_gradient_magnitude, results['gx'], results['gy'])
            futures.append(('gradient_magnitude', None, future))
            
            # Second-order gradients (depend on first gradients)
            def compute_second_derivatives(gx, gy):
                gxx = ndimage.sobel(gx, axis=1, mode='reflect')
                gyy = ndimage.sobel(gy, axis=0, mode='reflect')
                # Cross derivatives for Hessian determinant
                gxy = ndimage.sobel(gx, axis=0, mode='reflect')
                gyx = ndimage.sobel(gy, axis=1, mode='reflect')
                return gxx, gyy, gxy, gyx
            
            future = executor.submit(compute_second_derivatives, results['gx'], results['gy'])
            futures.append(('second_derivatives', None, future))
            
            # Collect results for the dependent computations
            for task_type, params, future in futures:
                try:
                    result = future.result()
                    if task_type == 'second_derivatives':
                        # Store the second derivative components separately
                        gxx, gyy, gxy, gyx = result
                        results['gxx'] = gxx
                        results['gyy'] = gyy
                        results['gxy'] = gxy
                        results['gyx'] = gyx
                    else:
                        results[task_type] = result
                except Exception as e:
                    raise RuntimeError(f"Error in task {task_type}: {str(e)}")
            
            # Stage 3: Final computations that depend on Stage 2 results
            futures = []
            
            # Laplacian and Hessian determinant (depend on second derivatives)
            def compute_laplacian(gxx, gyy):
                return gxx + gyy
            
            future = executor.submit(compute_laplacian, results['gxx'], results['gyy'])
            futures.append(('laplacian', None, future))
            
            def compute_hessian_det(gxx, gyy, gxy, gyx):
                return gxx * gyy - gxy * gyx
            
            future = executor.submit(compute_hessian_det, 
                                  results['gxx'], results['gyy'], 
                                  results['gxy'], results['gyx'])
            futures.append(('hessian_det', None, future))
            
            # Collect final results
            for task_type, params, future in futures:
                try:
                    result = future.result()
                    results[task_type] = result
                except Exception as e:
                    raise RuntimeError(f"Error in task {task_type}: {str(e)}")
        
        # Organize results in the expected order
        features = []
        
        # Add Gaussian features
        for sigma in self.alphas:
            features.append(results[f'gaussian_{sigma}'])

        for sigma in self.dogs:
            features.append(results[f'dog_{sigma[0]}'])
        
        # Add local statistics
        features.append(results['local_mean'])
        features.append(results['local_var'])
        
        # Add gradient magnitude
        features.append(results['gradient_magnitude'])
        
        # Add Laplacian and Hessian determinant
        features.append(results['laplacian'])
        features.append(results['hessian_det'])
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Check dimensionality and expand if needed
                if len(feat.shape) < len(original_shape):
                    feat_adjusted = feat
                    missing_dims = len(original_shape) - len(feat.shape)
                    for _ in range(missing_dims):
                        feat_adjusted = np.expand_dims(feat_adjusted, axis=0)
                    
                    if feat_adjusted.shape != original_shape:
                        raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                    
                    features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_feature_maps(self):
        """Compute all feature maps using GPU acceleration"""
        #if not self.use_gpu:
            #return super().compute_feature_maps()
        
        features = []
        image = self.image_gpu
        image_3d = self.image_3d
        original_shape = self.image_3d.shape


        
        # Gaussian smoothing at different scales
        print("Obtaining gaussians")
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            smooth = cp.asnumpy(self.gaussian_filter_gpu(image, sigma))
            features.append(smooth)
        
        print("Obtaining dif of gaussians")

        # Difference of Gaussians
        for (s1, s2) in [(1, 2), (2, 4)]:
            g1 = self.gaussian_filter_gpu(image, s1)
            g2 = self.gaussian_filter_gpu(image, s2)
            dog = cp.asnumpy(g1 - g2)
            features.append(dog)
        
        # Convert image to PyTorch tensor for gradient operations
        image_torch = torch.from_numpy(image_3d).cuda()
        image_torch = image_torch.float().unsqueeze(0).unsqueeze(0)
        
        # Calculate required padding
        kernel_size = 3
        padding = kernel_size // 2
        
        # Create a single padded version with same padding
        pad = torch.nn.functional.pad(image_torch, (padding, padding, padding, padding, padding, padding), mode='replicate')

        print("Computing sobel kernels")

        # Create sobel kernels
        sobel_x = torch.tensor([-1, 0, 1], device='cuda').float().view(1,1,1,1,3)
        sobel_y = torch.tensor([-1, 0, 1], device='cuda').float().view(1,1,1,3,1)
        sobel_z = torch.tensor([-1, 0, 1], device='cuda').float().view(1,1,3,1,1)
        
        # Compute gradients
        print("Computing gradiants")

        gx = torch.nn.functional.conv3d(pad, sobel_x, padding=0)[:,:,:original_shape[0],:original_shape[1],:original_shape[2]]
        gy = torch.nn.functional.conv3d(pad, sobel_y, padding=0)[:,:,:original_shape[0],:original_shape[1],:original_shape[2]]
        gz = torch.nn.functional.conv3d(pad, sobel_z, padding=0)[:,:,:original_shape[0],:original_shape[1],:original_shape[2]]
        
        # Compute gradient magnitude
        print("Computing gradiant mags")

        gradient_magnitude = torch.sqrt(gx**2 + gy**2 + gz**2)
        gradient_feature = gradient_magnitude.cpu().numpy().squeeze()
        
        features.append(gradient_feature)

        print(features.shape)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Create a copy of the feature to modify
                feat_adjusted = np.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                # Important: Update the original features list with the expanded version
                features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_feature_maps_2d(self, z=None):
        """Compute all feature maps for 2D images using GPU acceleration"""
        
        features = []

        image = self.image_gpu[z, :, :]
        image_2d = self.image_3d[z, :, :]
        original_shape = image_2d.shape

        # Gaussian smoothing at different scales
        print("Obtaining gaussians")
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            smooth = cp.asnumpy(self.gaussian_filter_gpu(image, sigma))
            features.append(smooth)
        
        print("Obtaining diff of gaussians")
        # Difference of Gaussians
        for (s1, s2) in [(1, 2), (2, 4)]:
            g1 = self.gaussian_filter_gpu(image, s1)
            g2 = self.gaussian_filter_gpu(image, s2)
            dog = cp.asnumpy(g1 - g2)
            features.append(dog)
        
        # Convert image to PyTorch tensor for gradient operations
        image_torch = torch.from_numpy(image_2d).cuda()
        image_torch = image_torch.float().unsqueeze(0).unsqueeze(0)
        
        # Calculate required padding
        kernel_size = 3
        padding = kernel_size // 2
        
        # Create a single padded version with same padding
        pad = torch.nn.functional.pad(image_torch, (padding, padding, padding, padding), mode='replicate')
        
        print("Computing sobel kernels")
        # Create 2D sobel kernels
        sobel_x = torch.tensor([-1, 0, 1], device='cuda').float().view(1, 1, 1, 3)
        sobel_y = torch.tensor([-1, 0, 1], device='cuda').float().view(1, 1, 3, 1)
        
        # Compute gradients
        print("Computing gradients")
        gx = torch.nn.functional.conv2d(pad, sobel_x, padding=0)[:, :, :original_shape[0], :original_shape[1]]
        gy = torch.nn.functional.conv2d(pad, sobel_y, padding=0)[:, :, :original_shape[0], :original_shape[1]]
        
        # Compute gradient magnitude (no z component in 2D)
        print("Computing gradient mags")
        gradient_magnitude = torch.sqrt(gx**2 + gy**2)
        gradient_feature = gradient_magnitude.cpu().numpy().squeeze()
        
        features.append(gradient_feature)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Create a copy of the feature to modify
                feat_adjusted = feat
                # Check dimensionality and expand if needed
                if len(feat.shape) < len(original_shape):
                    missing_dims = len(original_shape) - len(feat.shape)
                    for _ in range(missing_dims):
                        feat_adjusted = np.expand_dims(feat_adjusted, axis=0)
                
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                
                # Update the original features list with the adjusted version
                features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_feature_maps_cpu_2d(self, z = None):
        """Compute feature maps for 2D images using CPU"""


        features = []

        image_2d = self.image_3d[z, :, :]
        original_shape = image_2d.shape
        
        # Gaussian smoothing at different scales
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            smooth = ndimage.gaussian_filter(image_2d, sigma)
            features.append(smooth)
        
        # Difference of Gaussians
        for (s1, s2) in [(1, 2), (2, 4)]:
            g1 = ndimage.gaussian_filter(image_2d, s1)
            g2 = ndimage.gaussian_filter(image_2d, s2)
            dog = g1 - g2
            features.append(dog)
        
        # Gradient computations using scipy - note axis changes for 2D
        gx = ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
        gy = ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
        
        # Gradient magnitude (no z component in 2D)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        features.append(gradient_magnitude)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Check dimensionality and expand if needed
                if len(feat.shape) < len(original_shape):
                    feat_adjusted = feat
                    missing_dims = len(original_shape) - len(feat.shape)
                    for _ in range(missing_dims):
                        feat_adjusted = np.expand_dims(feat_adjusted, axis=0)
                    
                    if feat_adjusted.shape != original_shape:
                        raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                    
                    features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_feature_maps_cpu_2d_parallel(self, z=None):
        """Compute feature maps for 2D images using CPU with thread-based parallelism"""
        image_2d = self.image_3d[z, :, :]
        original_shape = image_2d.shape
        
        # Use ThreadPoolExecutor for parallelization
        with ThreadPoolExecutor(max_workers=min(7, multiprocessing.cpu_count())) as executor:
            # Submit tasks for independent computations
            futures = []
            
            # Gaussian smoothing at different scales
            def compute_gaussian(sigma):
                return ndimage.gaussian_filter(image_2d, sigma)
            
            gaussian_sigmas = self.alphas
            for sigma in gaussian_sigmas:
                future = executor.submit(compute_gaussian, sigma)
                futures.append(('gaussian', sigma, future))
            
            # Difference of Gaussians
            def compute_dog(s1, s2):
                g1 = ndimage.gaussian_filter(image_2d, s1)
                g2 = ndimage.gaussian_filter(image_2d, s2)
                return g1 - g2
            
            dog_pairs = self.dogs
            for (s1, s2) in dog_pairs:
                future = executor.submit(compute_dog, s1, s2)
                futures.append(('dog', (s1, s2), future))
            
            # Gradient computation
            def compute_gradient_magnitude():
                gx = ndimage.sobel(image_2d, axis=1, mode='reflect')  # x direction
                gy = ndimage.sobel(image_2d, axis=0, mode='reflect')  # y direction
                return np.sqrt(gx**2 + gy**2)
            
            future = executor.submit(compute_gradient_magnitude)
            futures.append(('gradient_magnitude', None, future))
            
            # Collect results
            results = {}
            for task_type, params, future in futures:
                try:
                    result = future.result()
                    if params is not None:
                        if task_type == 'dog':
                            s1, s2 = params
                            results[f"{task_type}_{s1}_{s2}"] = result
                        else:
                            results[f"{task_type}_{params}"] = result
                    else:
                        results[task_type] = result
                except Exception as e:
                    raise RuntimeError(f"Error in task {task_type} with params {params}: {str(e)}")
        
        # Organize results in the expected order
        features = []
        
        # Add Gaussian features
        for sigma in gaussian_sigmas:
            features.append(results[f'gaussian_{sigma}'])
        
        # Add Difference of Gaussians features
        for (s1, s2) in dog_pairs:
            features.append(results[f'dog_{s1}_{s2}'])
        
        # Add gradient magnitude
        features.append(results['gradient_magnitude'])
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                # Check dimensionality and expand if needed
                if len(feat.shape) < len(original_shape):
                    feat_adjusted = feat
                    missing_dims = len(original_shape) - len(feat.shape)
                    for _ in range(missing_dims):
                        feat_adjusted = np.expand_dims(feat_adjusted, axis=0)
                    
                    if feat_adjusted.shape != original_shape:
                        raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                    
                    features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_feature_maps_cpu(self, image_3d = None):
        """Compute feature maps using CPU"""
        features = []
        if image_3d is None:
            image_3d = self.image_3d

        original_shape = image_3d.shape

        
        # Gaussian smoothing at different scales
        #print("Obtaining gaussians")
        for sigma in self.alphas:
            smooth = ndimage.gaussian_filter(image_3d, sigma)
            features.append(smooth)
        
        #print("Obtaining dif of gaussians")
        # Difference of Gaussians
        for (s1, s2) in self.dogs:
            g1 = ndimage.gaussian_filter(image_3d, s1)
            g2 = ndimage.gaussian_filter(image_3d, s2)
            dog = g1 - g2
            features.append(dog)
        
        #print("Computing sobel and gradients")
        # Gradient computations using scipy
        gx = ndimage.sobel(image_3d, axis=2, mode='reflect')  # x direction
        gy = ndimage.sobel(image_3d, axis=1, mode='reflect')  # y direction
        gz = ndimage.sobel(image_3d, axis=0, mode='reflect')  # z direction
        
        # Gradient magnitude
        #print("Computing gradient magnitude")
        gradient_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        features.append(gradient_magnitude)
        
        # Verify shapes
        #print("Verifying shapes")
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                feat_adjusted = np.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_feature_maps_cpu_parallel(self, image_3d=None):
        """Use ThreadPoolExecutor 
        
        While threads don't give true parallelism for CPU-bound tasks due to the GIL,
        numpy/scipy release the GIL during computation, so this can still be effective.
        """
        if image_3d is None:
            image_3d = self.image_3d
        original_shape = image_3d.shape
        
        features = []
        
        # Using ThreadPoolExecutor which is more compatible with GUI applications
        with ThreadPoolExecutor(max_workers=min(7, multiprocessing.cpu_count())) as executor:
            # Submit all tasks to the executor
            futures = []
            
            # Gaussian smoothing at different scales
            for sigma in self.alphas:
                future = executor.submit(ndimage.gaussian_filter, image_3d, sigma)
                futures.append(future)
            
            def compute_dog_local(img, s1, s2):
                g1 = ndimage.gaussian_filter(img, s1) # Consider just having this return the gaussians to
                g2 = ndimage.gaussian_filter(img, s2)
                return g1 - g2

            # Difference of Gaussians
            for (s1, s2) in self.dogs:
                
                future = executor.submit(compute_dog_local, image_3d, s1, s2)
                futures.append(future)
            
            # Gradient magnitude
            def compute_gradient_local(img):
                gx = ndimage.sobel(img, axis=2, mode='reflect')
                gy = ndimage.sobel(img, axis=1, mode='reflect')
                gz = ndimage.sobel(img, axis=0, mode='reflect')
                return np.sqrt(gx**2 + gy**2 + gz**2)
            
            future = executor.submit(compute_gradient_local, image_3d)
            futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                features.append(result)
        
        # Verify shapes
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                feat_adjusted = np.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def compute_deep_feature_maps(self):
        """Compute all feature maps using GPU acceleration"""
        #if not self.use_gpu:
            #return super().compute_feature_maps()
        
        features = []
        image = self.image_gpu
        original_shape = self.image_3d.shape
        
        # Original features (Gaussians and DoG)
        print("Obtaining gaussians")
        for sigma in [0.5, 1.0, 2.0, 4.0]:
            smooth = cp.asnumpy(self.gaussian_filter_gpu(image, sigma))
            features.append(smooth)
        
            print("Computing local statistics")
            image_torch = torch.from_numpy(self.image_3d).cuda()
            image_torch = image_torch.float().unsqueeze(0).unsqueeze(1)  # [1, 1, 1, 512, 384]

            # Create kernel
            window_size = 5
            pad = window_size // 2

            if image_torch.shape[2] == 1:  # Single slice case
                # Squeeze out the z dimension for 2D operations
                image_2d = image_torch.squeeze(2)  # Now [1, 1, 512, 384]
                kernel_2d = torch.ones((1, 1, window_size, window_size), device='cuda')
                kernel_2d = kernel_2d / (window_size**2)
                
                # 2D padding and convolution
                padded = torch.nn.functional.pad(image_2d, 
                                               (pad, pad,     # x dimension
                                                pad, pad),    # y dimension
                                               mode='reflect')
                
                local_mean = torch.nn.functional.conv2d(padded, kernel_2d)
                local_mean = local_mean.unsqueeze(2)  # Add z dimension back
                features.append(local_mean.cpu().numpy().squeeze())
                
                # Local variance
                mean = torch.mean(image_2d)
                padded_sq = torch.nn.functional.pad((image_2d - mean)**2, 
                                                  (pad, pad, pad, pad),
                                                  mode='reflect')
                local_var = torch.nn.functional.conv2d(padded_sq, kernel_2d)
                local_var = local_var.unsqueeze(2)  # Add z dimension back
                features.append(local_var.cpu().numpy().squeeze())
            else:
                # Original 3D operations for multi-slice case
                kernel = torch.ones((1, 1, window_size, window_size, window_size), device='cuda')
                kernel = kernel / (window_size**3)
                
                padded = torch.nn.functional.pad(image_torch, 
                                               (pad, pad,     # x dimension
                                                pad, pad,     # y dimension
                                                pad, pad),    # z dimension
                                               mode='reflect')
                local_mean = torch.nn.functional.conv3d(padded, kernel)
                features.append(local_mean.cpu().numpy().squeeze())
                
                mean = torch.mean(image_torch)
                padded_sq = torch.nn.functional.pad((image_torch - mean)**2, 
                                                  (pad, pad, pad, pad, pad, pad),
                                                  mode='reflect')
                local_var = torch.nn.functional.conv3d(padded_sq, kernel)
                features.append(local_var.cpu().numpy().squeeze())

        # Original gradient computations
        print("Computing sobel and gradients")
        kernel_size = 3
        padding = kernel_size // 2
        pad = torch.nn.functional.pad(image_torch, (padding,)*6, mode='replicate')
        
        sobel_x = torch.tensor([-1, 0, 1], device='cuda').float().view(1,1,1,1,3)
        sobel_y = torch.tensor([-1, 0, 1], device='cuda').float().view(1,1,1,3,1)
        sobel_z = torch.tensor([-1, 0, 1], device='cuda').float().view(1,1,3,1,1)
        
        gx = torch.nn.functional.conv3d(pad, sobel_x, padding=0)[:,:,:original_shape[0],:original_shape[1],:original_shape[2]]
        gy = torch.nn.functional.conv3d(pad, sobel_y, padding=0)[:,:,:original_shape[0],:original_shape[1],:original_shape[2]]
        gz = torch.nn.functional.conv3d(pad, sobel_z, padding=0)[:,:,:original_shape[0],:original_shape[1],:original_shape[2]]
        
        gradient_magnitude = torch.sqrt(gx**2 + gy**2 + gz**2)
        features.append(gradient_magnitude.cpu().numpy().squeeze())
        
        # Second-order gradients
        print("Computing second-order features")
        gxx = torch.nn.functional.conv3d(gx, sobel_x, padding=padding)
        gyy = torch.nn.functional.conv3d(gy, sobel_y, padding=padding)
        gzz = torch.nn.functional.conv3d(gz, sobel_z, padding=padding)

        # Get minimum size in each dimension
        min_size_0 = min(gxx.size(2), gyy.size(2), gzz.size(2))
        min_size_1 = min(gxx.size(3), gyy.size(3), gzz.size(3))
        min_size_2 = min(gxx.size(4), gyy.size(4), gzz.size(4))

        # Crop to smallest common size
        gxx = gxx[:, :, :min_size_0, :min_size_1, :min_size_2]
        gyy = gyy[:, :, :min_size_0, :min_size_1, :min_size_2]
        gzz = gzz[:, :, :min_size_0, :min_size_1, :min_size_2]

        laplacian = gxx + gyy + gzz  # Second derivatives in each direction
        features.append(laplacian.cpu().numpy().squeeze())

        # Now they should have matching dimensions for multiplication
        hessian_det = gxx * gyy * gzz
        features.append(hessian_det.cpu().numpy().squeeze())

        print("Verifying shapes")
        for i, feat in enumerate(features):
            if feat.shape != original_shape:
                feat_adjusted = np.expand_dims(feat, axis=0)
                if feat_adjusted.shape != original_shape:
                    raise ValueError(f"Feature {i} has shape {feat.shape}, expected {original_shape}")
                features[i] = feat_adjusted
        
        return np.stack(features, axis=-1)

    def gaussian_filter_gpu(self, image, sigma):
        """GPU-accelerated Gaussian filter"""
        # Create Gaussian kernel
        result = cpx.gaussian_filter(image, sigma=sigma)

        return result

    def process_chunk_GPU(self, chunk_coords):
        """Process a chunk of coordinates using GPU acceleration"""
        coords = np.array(chunk_coords)
        z, y, x = coords.T
        
        # Extract features
        features = self.feature_cache[z, y, x]
        
        if self.use_gpu:
            # Move to GPU
            features_gpu = cp.array(features)
            
            # Predict on GPU
            predictions = self.model.predict(features_gpu)
            predictions = cp.asnumpy(predictions)
        else:
            predictions = self.model.predict(features)
        
        # Split results
        foreground_mask = predictions == 1
        background_mask = ~foreground_mask
        
        foreground = set(map(tuple, coords[foreground_mask]))
        background = set(map(tuple, coords[background_mask]))
        
        return foreground, background

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
        """
        Process a chunk of coordinates, handling both mem_lock and non-mem_lock cases.
        Uses a consistent approach based on coordinates.
        
        Parameters:
        -----------
        chunk_coords : list of tuples
            List of (z,y,x) coordinate tuples to process
        
        Returns:
        --------
        tuple : (foreground, background)
            Sets of coordinates classified as foreground or background
        """
        foreground = set()
        background = set()
        
        if self.previewing or not self.use_two:
            if self.mem_lock:
                # For mem_lock, we need to extract a subarray and compute features

                if self.realtimechunks is None: #Presuming we're segmenting all
                    z_min, z_max = chunk_coords[0], chunk_coords[1]
                    y_min, y_max = chunk_coords[2], chunk_coords[3]
                    x_min, x_max = chunk_coords[4], chunk_coords[5]

                    # Consider moving this to process chunk ??
                    chunk_coords = np.stack(np.meshgrid(
                        np.arange(z_min, z_max),
                        np.arange(y_min, y_max),
                        np.arange(x_min, x_max),
                        indexing='ij'
                    )).reshape(3, -1).T
                    
                    chunk_coords = (list(map(tuple, chunk_coords)))
                else: #Presumes we're not segmenting all
                    # Find min/max bounds of the coordinates to get the smallest containing subarray
                    z_coords = [z for z, y, x in chunk_coords]
                    y_coords = [y for z, y, x in chunk_coords]
                    x_coords = [x for z, y, x in chunk_coords]
                    
                    z_min, z_max = min(z_coords), max(z_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    x_min, x_max = min(x_coords), max(x_coords)

                
                # Extract the subarray
                subarray = self.image_3d[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
                
                # Compute features for this subarray
                if self.speed:
                    feature_map = self.compute_feature_maps_cpu_parallel(subarray) #If the interactive segmenter is slow
                else:                                                              #Due to the parallel, consider singleton implementation for it specifically
                    feature_map = self.compute_deep_feature_maps_cpu_parallel(subarray)
                
                # Extract features for each coordinate, adjusting for subarray offset
                features = []
                for z, y, x in chunk_coords:
                    # Transform global coordinates to local subarray coordinates
                    local_z = z - z_min
                    local_y = y - y_min
                    local_x = x - x_min
                    
                    # Get feature at this position
                    feature = feature_map[local_z, local_y, local_x]
                    features.append(feature)
                    
            else:
                # For non-mem_lock, simply use the feature cache
                features = [self.feature_cache[z, y, x] for z, y, x in chunk_coords]
            
            # Make predictions
            predictions = self.model.predict(features)
            
            # Assign coordinates based on predictions
            for coord, pred in zip(chunk_coords, predictions):
                if pred:
                    foreground.add(coord)
                else:
                    background.add(coord)

        else:

            if self.mem_lock:
                chunk_coords = self.twodim_coords(chunk_coords[0], chunk_coords[1], chunk_coords[2], chunk_coords[3], chunk_coords[4])

            chunk_coords = self.organize_by_z(chunk_coords)

            for z, coords in chunk_coords.items():

                if self.feature_cache is None:
                    features = self.get_feature_map_slice(z, self.speed, self.cur_gpu)
                    features = [features[y, x] for y, x in coords]
                elif z not in self.feature_cache and not self.previewing:
                    features = self.get_feature_map_slice(z, self.speed, self.cur_gpu)
                    features = [features[y, x] for y, x in coords]
                elif z not in self.feature_cache or self.feature_cache is None and self.previewing:
                    features = self.map_slice
                    try:
                        features = [features[y, x] for y, x in coords]
                    except:
                        return [], []
                else:  
                    features = [self.feature_cache[z][y, x] for y, x in coords]

                predictions = self.model.predict(features)
                
                for (y, x), pred in zip(coords, predictions):
                    coord = (z, y, x)  # Reconstruct the 3D coordinate as a tuple
                    if pred:
                        foreground.add(coord)
                    else:
                        background.add(coord)
        
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
        


    def segment_volume(self, chunk_size=None, gpu=False):
        """Segment volume using parallel processing of chunks with vectorized chunk creation"""
        #Change the above chunk size to None to have it auto-compute largest chunks (not sure which is faster, 64 seems reasonable in test cases)

        self.realtimechunks = None # Presumably no longer need this.
        self.map_slice = None

        if self.mem_lock:
            chunk_size = self.master_chunk #memory efficient chunk


        def create_2d_chunks():
            """
            Create chunks by z-slices for 2D processing.
            Each chunk is a complete z-slice with all y,x coordinates,
            unless the slice exceeds 262144 pixels, in which case it's divided into subchunks.
            
            Returns:
                List of chunks, where each chunk contains the coordinates for one z-slice or subchunk
            """
            MAX_CHUNK_SIZE = 262144
            if not self.mem_lock:
                MAX_CHUNK_SIZE = 10000000000000000000000000 #unlimited i guess
            chunks = []
            
            for z in range(self.image_3d.shape[0]):
                # Get the dimensions of this z-slice
                y_dim = self.image_3d.shape[1]
                x_dim = self.image_3d.shape[2]
                total_pixels = y_dim * x_dim
                
                # If the slice is small enough, do not subchunk
                if total_pixels <= MAX_CHUNK_SIZE:


                    if not self.mem_lock:
                        chunks.append(self.twodim_coords(y_dim, x_dim, z, total_pixels))
                    else:
                        chunks.append([y_dim, x_dim, z, total_pixels, None])



                else:
                    # Determine which dimension to divide (the largest one)
                    largest_dim = 'y' if y_dim >= x_dim else 'x'
                    
                    # Calculate how many divisions we need
                    num_divisions = int(np.ceil(total_pixels / MAX_CHUNK_SIZE))
                    
                    # Calculate the approx size of each division along the largest dimension
                    if largest_dim == 'y':
                        div_size = int(np.ceil(y_dim / num_divisions))
                        # Create subchunks by dividing the y-dimension
                        for i in range(0, y_dim, div_size):
                            end_i = min(i + div_size, y_dim)

                            if not self.mem_lock:
                                chunks.append(self.twodim_coords(y_dim, x_dim, z, None, ['y', i, end_i]))
                            else:
                                chunks.append([y_dim, x_dim, z, None, ['y', i, end_i]])

                    else:  # largest_dim == 'x'
                        div_size = int(np.ceil(x_dim / num_divisions))
                        # Create subchunks by dividing the x-dimension
                        for i in range(0, x_dim, div_size):
                            end_i = min(i + div_size, x_dim)

                            if not self.mem_lock:
                                chunks.append(self.twodim_coords(y_dim, x_dim, z, None, ['x', i, end_i]))
                            else:
                                chunks.append([y_dim, x_dim, z, None, ['x', i, end_i]])
            
            return chunks

        #try:
            #from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
        #except:
            #print("Cannot find cuML, using CPU to segment instead...")
            #gpu = False

        if self.feature_cache is None and not self.mem_lock and not self.use_two:
            with self.lock:
                if self.feature_cache is None:
                    self.feature_cache = self.compute_feature_maps()

        print("Chunking data...")
        
        if not self.use_two:
            # Determine optimal chunk size based on number of cores if not specified
            if chunk_size is None:
                total_cores = multiprocessing.cpu_count()
                
                # Calculate total volume and target volume per core
                total_volume = np.prod(self.image_3d.shape)
                target_volume_per_chunk = total_volume / total_cores
                
                # Calculate chunk size that would give us roughly one chunk per core
                # Using cube root since we want roughly equal sizes in all dimensions
                chunk_size = int(np.cbrt(target_volume_per_chunk))
                
                # Ensure chunk size is at least 32 (minimum reasonable size) and not larger than smallest dimension
                chunk_size = max(32, min(chunk_size, min(self.image_3d.shape)))
                
                # Round to nearest multiple of 32 for better memory alignment
                chunk_size = ((chunk_size + 15) // 32) * 32
            
            # Calculate number of chunks in each dimension
            z_chunks = (self.image_3d.shape[0] + chunk_size - 1) // chunk_size
            y_chunks = (self.image_3d.shape[1] + chunk_size - 1) // chunk_size
            x_chunks = (self.image_3d.shape[2] + chunk_size - 1) // chunk_size
            
            # Create start indices for all chunks at once
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
                
                if self.mem_lock:
                    # Create coordinates for this chunk efficiently
                    coords = [z_start, z_end, y_start, y_end, x_start, x_end]
                    chunks.append(coords)

                else:
                    # Consider moving this to process chunk ??
                    coords = np.stack(np.meshgrid(
                        np.arange(z_start, z_end),
                        np.arange(y_start, y_end),
                        np.arange(x_start, x_end),
                        indexing='ij'
                    )).reshape(3, -1).T
                
                    chunks.append(list(map(tuple, coords)))



        else:
            chunks = create_2d_chunks()
            self.feature_cache = None #Decided this should not maintain training data for segmenting 2D
        
        foreground_coords = set()
        background_coords = set()

        print("Segmenting chunks...")

        if not self.mem_lock:
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                if gpu:
                    try:
                        futures = [executor.submit(self.process_chunk_GPU, chunk) for chunk in chunks]
                    except:
                        futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]

                else:
                    futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
                
                for i, future in enumerate(futures):
                    fore, back = future.result()
                    foreground_coords.update(fore)
                    background_coords.update(back)
                    print(f"Processed {i}/{len(chunks)} chunks")
        else: #Prioritize RAM
            for i, chunk in enumerate(chunks):
                fore, back = self.process_chunk(chunk)
                foreground_coords.update(fore)
                background_coords.update(back)
                try:
                    chunk[i] = None #Help garbage collection
                except:
                    pass
                print(f"Processed {i}/{len(chunks)} chunks")
        
        return foreground_coords, background_coords

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


    def segment_volume_realtime(self, gpu = False):

        try:
            from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
        except:
            print("Cannot find cuML, using CPU to segment instead...")
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

            coords = np.stack(np.meshgrid(
                [chunk['z']],
                np.arange(coords[0], coords[1]),
                np.arange(coords[2], coords[3]),
                indexing='ij'
            )).reshape(3, -1).T

            coords = list(map(tuple, coords))

            
            # Process the chunk directly based on whether GPU is available
            if gpu:
                try:
                    fore, back = self.process_chunk_GPU(coords)
                except:
                    fore, back = self.process_chunk(coords)
            else:
                fore, back = self.process_chunk(coords)
            
            # Yield the results
            yield fore, back


    def cleanup(self):
        """Clean up GPU memory"""
        if self.use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()
            except:
                pass

    def train_batch(self, foreground_array, speed = True, use_gpu = False, use_two = False, mem_lock = False):
        """Train directly on foreground and background arrays"""

        print("Training model...")
        self.speed = speed
        self.cur_gpu = use_gpu
        if mem_lock != self.mem_lock:
            self.realtimechunks = None #dump ram
            self.feature_cache = None

        if not use_two:
            self.use_two = False

        self.mem_lock = mem_lock

        if self.current_speed != speed:
            self.feature_cache = None
            if use_gpu:
                try:
                    self.model = cuRandomForestClassifier(
                        n_estimators=100,
                        max_depth=None
                    )
                except:
                    self.model = RandomForestClassifier(
                        n_estimators=100,
                        n_jobs=-1,
                        max_depth=None
                    )
            else:
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

            # Get foreground coordinates and features
            z_fore, y_fore, x_fore = np.where(foreground_array == 1)


            fore_coords = list(zip(z_fore, y_fore, x_fore))
            
            # Get background coordinates and features
            z_back, y_back, x_back = np.where(foreground_array == 2)

            back_coords = list(zip(z_back, y_back, x_back))


            #slices = set(list(z_back) + list(z_fore))

            #for z in slices:
                #if z not in self.two_slices:
                    #changed.append(z)
                    #self.two_slices.append(z) #Tracks assigning coords to feature map slices

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
                        foreground_features.append(feature_vector)

                if z in z_backs:
                
                    for y, x in z_backs[z]:
                        # Get the feature vector for this foreground point
                        feature_vector = current_map[y, x]
                    
                        # Add to our collection
                        background_features.append(feature_vector)


        elif mem_lock: #Forces ram efficiency

            box_size = self.master_chunk

            # Memory-efficient approach: compute features only for necessary subarrays
            foreground_features = []
            background_features = []
            
            # Find coordinates of foreground and background scribbles
            z_fore = np.argwhere(foreground_array == 1)
            z_back = np.argwhere(foreground_array == 2)
            
            # If no scribbles, return empty lists
            if len(z_fore) == 0 and len(z_back) == 0:
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
            for z, y, x in np.vstack((z_fore, z_back)) if len(z_back) > 0 else z_fore:
                grid_z = z // box_size
                grid_y = y // box_size
                grid_x = x // box_size
                grid_cells_with_scribbles.add((grid_z, grid_y, grid_x))
            
            # Create a mapping from original coordinates to their corresponding subarray and local coordinates
            coord_mapping = {}
            
            # Step 2: Process each grid cell that contains scribbles
            for grid_z, grid_y, grid_x in grid_cells_with_scribbles:
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
                    subarray_features = self.compute_feature_maps_cpu_parallel(subarray)
                else:
                    subarray_features = self.compute_deep_feature_maps_cpu_parallel(subarray)
                
                # For each foreground point in this grid cell, extract its feature
                # Extract foreground features using a direct mask comparison
                local_fore_coords = np.argwhere(subarray2 == 1)
                for local_z, local_y, local_x in local_fore_coords:
                    feature = subarray_features[local_z, local_y, local_x]
                    foreground_features.append(feature)
                
                # Extract background features using a direct mask comparison
                local_back_coords = np.argwhere(subarray2 == 2)
                for local_z, local_y, local_x in local_back_coords:
                    feature = subarray_features[local_z, local_y, local_x]
                    background_features.append(feature)

        else:

            self.two_slices = []

            if self.use_two: #Clarifies if we need to redo feature cache for 3D

                self.feature_cache = None
                self.use_two = False

            if self.feature_cache is None:
                with self.lock:
                    if self.feature_cache is None and speed:
                        if use_gpu:
                            self.feature_cache = self.compute_feature_maps()
                        else:
                            self.feature_cache = self.compute_feature_maps_cpu()

                    elif self.feature_cache is None and not speed:
                        if use_gpu:

                            self.feature_cache = self.compute_deep_feature_maps()
                        else:
                            self.feature_cache = self.compute_deep_feature_maps_cpu()


            try:
                # Get foreground coordinates and features
                z_fore, y_fore, x_fore = np.where(foreground_array == 1)
                foreground_features = self.feature_cache[z_fore, y_fore, x_fore]

                # Get background coordinates and features
                z_back, y_back, x_back = np.where(foreground_array == 2)
                background_features = self.feature_cache[z_back, y_back, x_back]
            except:
                print("Features maps computed, but no segmentation examples were provided so the model was not trained")


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

    def get_feature_map_slice(self, z, speed, use_gpu):

        if self._currently_segmenting is not None:
            return

        #with self.lock <- cant remember why this was here
        if speed:

            if self.mem_lock:
                output = self.compute_feature_maps_cpu_2d_parallel(z = z)
            else:
                output = self.compute_feature_maps_cpu_2d(z = z)

        elif not speed:

            if self.mem_lock:
                output = self.compute_deep_feature_maps_cpu_2d_parallel(z = z)
            else:
                output = self.compute_deep_feature_maps_cpu_2d(z = z)

        return output

