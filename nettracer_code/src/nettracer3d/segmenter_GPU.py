"""
Interactive GPU Segmenter — CuPy acceleration

Architecture:
  - Feature maps match the CPU segmenter exactly (Gaussians, consecutive DoGs,
    per-scale gradient magnitude, per-scale Laplacian, and in deep mode:
    Hessian eigenvalues, structure-tensor eigenvalues, min/max/mean/var filters).
  - All heavy array work runs on GPU via CuPy; results are pulled back to NumPy
    only for LightGBM classifier interaction.
  - Chunk padding: features are computed on a padded region from the full source
    image and then cropped to the requested window 
  - _ChunkCache: LRU cache of computed feature chunks for interactive reuse.
  - 3D eigenvalue computation uses cp.linalg.eigvalsh (batch-reshaped) because
    CuPy's element-wise Cardano is unreliable on some GPU drivers.
  - 2D eigenvalue computation uses the fast analytical trace/det formula on GPU.
  - Speed mode: sigmas [1,2,4,8]. Deep mode: sigmas [1,2,4,8,16].
"""

import warnings
warnings.filterwarnings("ignore",
                        message="X does not have valid feature names",
                        category=UserWarning)
import lightgbm as lgb
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cpx
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy import ndimage
import multiprocessing
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Any
import math


# ============================================================
# GPU derivative helper
# ============================================================
_K1_gpu = cp.array([-0.5, 0.0, 0.5], dtype=cp.float32)
_K2_gpu = cp.array([1.0, -2.0, 1.0], dtype=cp.float32)


def _d_gpu(image, axis, order=1):
    """First or second derivative along *axis* using a 3-tap kernel (GPU)."""
    return cpx.convolve1d(image, _K1_gpu if order == 1 else _K2_gpu,
                          axis=axis, mode='reflect')


# ============================================================
# GPU eigenvalue helpers
# ============================================================

def _eigen2x2_gpu(hxx, hyy, hxy):
    """Eigenvalues of 2×2 symmetric matrices on GPU → (small, large).

    Uses the analytical trace / determinant formula which is perfectly
    suited to element-wise GPU work.
    """
    tr = (hxx + hyy).astype(cp.float64)
    det = (hxx * hyy - hxy * hxy).astype(cp.float64)
    disc = cp.maximum(tr * tr - 4.0 * det, 0.0)
    s = cp.sqrt(disc)
    return ((tr - s) * 0.5).astype(cp.float32), ((tr + s) * 0.5).astype(cp.float32)


def _eigen3x3_gpu(h11, h22, h33, h12, h13, h23):
    """Eigenvalues of 3×3 symmetric matrices on GPU → (e0, e1, e2) ascending.

    Uses cp.linalg.eigvalsh with a batch reshape.  This is the safest
    path on CUDA — some drivers choke on element-wise Cardano, but
    batch eigvalsh is rock-solid across GPU generations.
    """
    shape = h11.shape
    n = int(cp.prod(cp.array(shape)))
    mat = cp.zeros((n, 3, 3), dtype=cp.float64)
    mat[:, 0, 0] = h11.ravel().astype(cp.float64)
    mat[:, 1, 1] = h22.ravel().astype(cp.float64)
    mat[:, 2, 2] = h33.ravel().astype(cp.float64)
    mat[:, 0, 1] = mat[:, 1, 0] = h12.ravel().astype(cp.float64)
    mat[:, 0, 2] = mat[:, 2, 0] = h13.ravel().astype(cp.float64)
    mat[:, 1, 2] = mat[:, 2, 1] = h23.ravel().astype(cp.float64)
    eigs = cp.linalg.eigvalsh(mat)  # (n, 3)  already sorted ascending
    return (eigs[:, 0].reshape(shape).astype(cp.float32),
            eigs[:, 1].reshape(shape).astype(cp.float32),
            eigs[:, 2].reshape(shape).astype(cp.float32))


# ============================================================
# Feature assembly functions (GPU)
# ============================================================

def _assemble_2d_gpu(G, sigmas, st_scales, deep):
    """Build the full feature stack for a 2-D image on GPU.

    *G* is a list of CuPy arrays: [original, gauss_σ1, gauss_σ2, …].
    The feature order exactly mirrors the CPU ``_assemble_2d``.
    """
    N = len(sigmas)
    feats = []

    # --- original ---
    feats.append(G[0])

    # --- Gaussians ---
    for i in range(1, N + 1):
        feats.append(G[i])

    # --- Consecutive DoG pairs ---
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            feats.append(G[i] - G[j])

    # --- Gradient magnitude per scale ---
    for i in range(N + 1):
        gx = _d_gpu(G[i], 1)
        gy = _d_gpu(G[i], 0)
        feats.append(cp.sqrt(gx * gx + gy * gy))

    # --- Laplacian per scale ---
    for i in range(N + 1):
        feats.append(_d_gpu(G[i], 1, 2) + _d_gpu(G[i], 0, 2))

    if deep:
        # --- Hessian eigenvalues per scale ---
        for i in range(N + 1):
            hxx = _d_gpu(G[i], 1, 2)
            hyy = _d_gpu(G[i], 0, 2)
            hxy = _d_gpu(_d_gpu(G[i], 1), 0)
            s, l = _eigen2x2_gpu(hxx, hyy, hxy)
            feats.append(s)
            feats.append(l)

        # --- Structure-tensor eigenvalues per scale ---
        for i in range(N + 1):
            gx = _d_gpu(G[i], 1)
            gy = _d_gpu(G[i], 0)
            Pxx = gx * gx; Pxy = gx * gy; Pyy = gy * gy
            for gamma in st_scales:
                Qxx = cpx.gaussian_filter(Pxx, gamma)
                Qxy = cpx.gaussian_filter(Pxy, gamma)
                Qyy = cpx.gaussian_filter(Pyy, gamma)
                s, l = _eigen2x2_gpu(Qxx, Qyy, Qxy)
                feats.append(s)
                feats.append(l)

        # --- Min / Max / Mean / Variance filters ---
        orig = G[0]
        orig64 = orig.astype(cp.float64)
        for sigma in sigmas:
            win = int(1 + 2 * sigma)
            feats.append(cpx.minimum_filter(orig, size=win, mode='reflect'))
            feats.append(cpx.maximum_filter(orig, size=win, mode='reflect'))
            mn = cpx.uniform_filter(orig64, size=win, mode='reflect').astype(cp.float32)
            feats.append(mn)
            var = cpx.uniform_filter(orig64 ** 2, size=win, mode='reflect').astype(cp.float32) - mn * mn
            feats.append(cp.maximum(var, 0))

    return cp.stack(feats, axis=-1)


def _assemble_3d_gpu(G, sigmas, st_scales, deep):
    """Build the full feature stack for a 3-D volume on GPU.

    Mirrors CPU ``_assemble_3d`` exactly.  3-D eigenvalue paths use
    the batch-eigvalsh approach for driver compatibility.
    """
    N = len(sigmas)
    feats = []

    # --- original ---
    feats.append(G[0])

    # --- Gaussians ---
    for i in range(1, N + 1):
        feats.append(G[i])

    # --- Consecutive DoG pairs ---
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            feats.append(G[i] - G[j])

    # --- Gradient magnitude per scale ---
    for i in range(N + 1):
        gx = _d_gpu(G[i], 2)
        gy = _d_gpu(G[i], 1)
        gz = _d_gpu(G[i], 0)
        feats.append(cp.sqrt(gx * gx + gy * gy + gz * gz))

    # --- Laplacian per scale ---
    for i in range(N + 1):
        feats.append(_d_gpu(G[i], 2, 2) + _d_gpu(G[i], 1, 2) + _d_gpu(G[i], 0, 2))

    if deep:
        # --- Hessian eigenvalues per scale (batch eigvalsh) ---
        for i in range(N + 1):
            hxx = _d_gpu(G[i], 2, 2)
            hyy = _d_gpu(G[i], 1, 2)
            hzz = _d_gpu(G[i], 0, 2)
            hxy = _d_gpu(_d_gpu(G[i], 2), 1)
            hxz = _d_gpu(_d_gpu(G[i], 2), 0)
            hyz = _d_gpu(_d_gpu(G[i], 1), 0)
            e0, e1, e2 = _eigen3x3_gpu(hxx, hyy, hzz, hxy, hxz, hyz)
            feats.append(e0)
            feats.append(e1)
            feats.append(e2)

        # --- Structure-tensor eigenvalues per scale (batch eigvalsh) ---
        for i in range(N + 1):
            gx = _d_gpu(G[i], 2)
            gy = _d_gpu(G[i], 1)
            gz = _d_gpu(G[i], 0)
            Pxx = gx * gx; Pxy = gx * gy; Pxz = gx * gz
            Pyy = gy * gy; Pyz = gy * gz; Pzz = gz * gz
            for gamma in st_scales:
                e0, e1, e2 = _eigen3x3_gpu(
                    cpx.gaussian_filter(Pxx, gamma),
                    cpx.gaussian_filter(Pyy, gamma),
                    cpx.gaussian_filter(Pzz, gamma),
                    cpx.gaussian_filter(Pxy, gamma),
                    cpx.gaussian_filter(Pxz, gamma),
                    cpx.gaussian_filter(Pyz, gamma))
                feats.append(e0)
                feats.append(e1)
                feats.append(e2)

        # --- Min / Max / Mean / Variance filters ---
        orig = G[0]
        orig64 = orig.astype(cp.float64)
        for sigma in sigmas:
            win = int(1 + 2 * sigma)
            feats.append(cpx.minimum_filter(orig, size=win, mode='reflect'))
            feats.append(cpx.maximum_filter(orig, size=win, mode='reflect'))
            mn = cpx.uniform_filter(orig64, size=win, mode='reflect').astype(cp.float32)
            feats.append(mn)
            var = cpx.uniform_filter(orig64 ** 2, size=win, mode='reflect').astype(cp.float32) - mn * mn
            feats.append(cp.maximum(var, 0))

    return cp.stack(feats, axis=-1)


# ============================================================
# Chunk Cache — LRU for computed feature chunks
# ============================================================
class _ChunkCache:
    """Bounded in-memory LRU cache for assembled feature chunks."""

    def __init__(self, max_bytes=1 * 1024**3):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._current_bytes = 0

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key, value):
        nbytes = value.nbytes
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return
            while self._current_bytes + nbytes > self._max_bytes and self._cache:
                _, old = self._cache.popitem(last=False)
                self._current_bytes -= old.nbytes
            self._cache[key] = value
            self._current_bytes += nbytes

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    @property
    def memory_used_mb(self):
        return self._current_bytes / (1024**2)


# ============================================================
# Interactive Segmenter (GPU)
# ============================================================
class InteractiveSegmenter:
    def __init__(self, image_3d, use_gpu=True):
        self.image_3d = cp.asarray(image_3d)
        self.patterns = []
        self.use_gpu = True

        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=63,
            max_depth=-1,

            min_child_samples=10,
            min_split_gain=0.01,

            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,

            reg_alpha=0.1,
            reg_lambda=1.0,

            max_bin=255,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )

        self.feature_cache = None
        self.lock = threading.Lock()
        self._currently_segmenting = None

        self.current_z = None
        self.current_x = None
        self.current_y = None

        self.realtimechunks = None
        self.current_speed = False

        self.use_two = False
        self.two_slices = []
        self.speed = True
        self.cur_gpu = True
        self.prev_z = None
        self.previewing = False

        self._currently_processing = False
        self._skip_next_update = False
        self._last_processed_slice = None
        self.mem_lock = False

        # Feature parameters — identical to CPU segmenter
        self.sigmas = [1, 2, 4, 8]
        self.sigmas_deep = [1, 2, 4, 8, 16]
        self.structure_tensor_scales = [1, 3]
        self.windows = 10
        self.master_chunk = 49
        self.twod_chunk_size = 117649
        self.batch_amplifier = 1

        # Chunk-level LRU cache for interactive preview reuse (1 GB)
        self._chunk_cache = _ChunkCache(max_bytes=1 * 1024**3)

        # Data when loading prev model
        self.previous_foreground = None
        self.previous_background = None
        self.previous_z_fore = None
        self.previous_z_back = None

    # ================================================================
    # Helpers
    # ================================================================

    def _is_rgb(self):
        return self.image_3d.ndim == 4 and self.image_3d.shape[-1] in (3, 4)

    def _spatial_shape(self):
        return self.image_3d.shape[:3]

    def _get_sigmas(self, deep):
        return self.sigmas_deep if deep else self.sigmas

    def _get_pad(self, deep):
        max_sig = max(self._get_sigmas(deep))
        return int(4 * max_sig) + max(self.structure_tensor_scales) + 2

    # ================================================================
    # Feature computation — direct from source image with padding (GPU)
    # ================================================================

    def _compute_features_for_region_2d(self, z, y_s, y_e, x_s, x_e, deep=False):
        """Compute features for a 2D chunk from the source image on GPU,
        using padded extraction and cropping."""
        sigmas = self._get_sigmas(deep)
        pad = self._get_pad(deep)
        _, H, W = self._spatial_shape()

        py_s = max(0, y_s - pad); py_e = min(H, y_e + pad)
        px_s = max(0, x_s - pad); px_e = min(W, x_e + pad)

        if self._is_rgb():
            parts = []
            for ch in range(self.image_3d.shape[-1]):
                src = cp.ascontiguousarray(
                    self.image_3d[z, py_s:py_e, px_s:px_e, ch]).astype(cp.float32)
                G = [src] + [cpx.gaussian_filter(src, s) for s in sigmas]
                parts.append(_assemble_2d_gpu(G, sigmas, self.structure_tensor_scales, deep))
            features = cp.concatenate(parts, axis=-1)
        else:
            src = cp.ascontiguousarray(
                self.image_3d[z, py_s:py_e, px_s:px_e]).astype(cp.float32)
            G = [src] + [cpx.gaussian_filter(src, s) for s in sigmas]
            features = _assemble_2d_gpu(G, sigmas, self.structure_tensor_scales, deep)

        cy = y_s - py_s; cx = x_s - px_s
        return features[cy:cy + (y_e - y_s), cx:cx + (x_e - x_s)]

    def _compute_features_for_region_2d_cached(self, z, y_s, y_e, x_s, x_e, deep=False):
        """Compute 2D region features with chunk-level LRU caching."""
        key = (z, y_s, y_e, x_s, x_e, deep)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            return cached
        result = self._compute_features_for_region_2d(z, y_s, y_e, x_s, x_e, deep)
        self._chunk_cache.put(key, result)
        return result

    def _compute_features_for_region_3d(self, z_s, z_e, y_s, y_e, x_s, x_e, deep=False):
        """Compute features for a 3D chunk from the source image on GPU,
        using padded extraction and cropping."""
        sigmas = self._get_sigmas(deep)
        pad = self._get_pad(deep)
        D, H, W = self._spatial_shape()

        pz_s = max(0, z_s - pad); pz_e = min(D, z_e + pad)
        py_s = max(0, y_s - pad); py_e = min(H, y_e + pad)
        px_s = max(0, x_s - pad); px_e = min(W, x_e + pad)

        if self._is_rgb():
            parts = []
            for ch in range(self.image_3d.shape[-1]):
                src = cp.ascontiguousarray(
                    self.image_3d[pz_s:pz_e, py_s:py_e, px_s:px_e, ch]).astype(cp.float32)
                G = [src] + [cpx.gaussian_filter(src, s) for s in sigmas]
                parts.append(_assemble_3d_gpu(G, sigmas, self.structure_tensor_scales, deep))
            features = cp.concatenate(parts, axis=-1)
        else:
            src = cp.ascontiguousarray(
                self.image_3d[pz_s:pz_e, py_s:py_e, px_s:px_e]).astype(cp.float32)
            G = [src] + [cpx.gaussian_filter(src, s) for s in sigmas]
            features = _assemble_3d_gpu(G, sigmas, self.structure_tensor_scales, deep)

        cz = z_s - pz_s; cy = y_s - py_s; cx = x_s - px_s
        return features[cz:cz + (z_e - z_s), cy:cy + (y_e - y_s), cx:cx + (x_e - x_s)]

    # ================================================================
    # Backward-compatible direct computation (for arbitrary images)
    # ================================================================

    def compute_feature_maps_gpu_2d(self, z=None, image_2d=None):
        if image_2d is None:
            if z is None:
                z = self._spatial_shape()[0] // 2
            _, H, W = self._spatial_shape()
            return self._compute_features_for_region_2d(z, 0, H, 0, W, deep=False)
        image_2d = cp.asarray(image_2d)
        if image_2d.ndim == 3 and image_2d.shape[-1] in (3, 4):
            return cp.concatenate([
                self.compute_feature_maps_gpu_2d(image_2d=image_2d[..., c])
                for c in range(image_2d.shape[-1])], axis=-1)
        src = image_2d.astype(cp.float32)
        G = [src] + [cpx.gaussian_filter(src, s) for s in self.sigmas]
        return _assemble_2d_gpu(G, self.sigmas, self.structure_tensor_scales, deep=False)

    def compute_deep_feature_maps_gpu_2d(self, z=None, image_2d=None):
        if image_2d is None:
            if z is None:
                z = self._spatial_shape()[0] // 2
            _, H, W = self._spatial_shape()
            return self._compute_features_for_region_2d(z, 0, H, 0, W, deep=True)
        image_2d = cp.asarray(image_2d)
        if image_2d.ndim == 3 and image_2d.shape[-1] in (3, 4):
            return cp.concatenate([
                self.compute_deep_feature_maps_gpu_2d(image_2d=image_2d[..., c])
                for c in range(image_2d.shape[-1])], axis=-1)
        src = image_2d.astype(cp.float32)
        G = [src] + [cpx.gaussian_filter(src, s) for s in self.sigmas_deep]
        return _assemble_2d_gpu(G, self.sigmas_deep, self.structure_tensor_scales, deep=True)

    def compute_feature_maps_gpu(self, image_3d=None):
        if image_3d is None:
            image_3d = self.image_3d
        image_3d = cp.asarray(image_3d)
        if image_3d.ndim == 4 and image_3d.shape[-1] in (3, 4):
            return cp.concatenate([
                self.compute_feature_maps_gpu(image_3d[..., c])
                for c in range(image_3d.shape[-1])], axis=-1)
        src = image_3d.astype(cp.float32)
        G = [src] + [cpx.gaussian_filter(src, s) for s in self.sigmas]
        return _assemble_3d_gpu(G, self.sigmas, self.structure_tensor_scales, deep=False)

    def compute_deep_feature_maps_gpu(self, image_3d=None):
        if image_3d is None:
            image_3d = self.image_3d
        image_3d = cp.asarray(image_3d)
        if image_3d.ndim == 4 and image_3d.shape[-1] in (3, 4):
            return cp.concatenate([
                self.compute_deep_feature_maps_gpu(image_3d[..., c])
                for c in range(image_3d.shape[-1])], axis=-1)
        src = image_3d.astype(cp.float32)
        G = [src] + [cpx.gaussian_filter(src, s) for s in self.sigmas_deep]
        return _assemble_3d_gpu(G, self.sigmas_deep, self.structure_tensor_scales, deep=True)

    # ================================================================
    # get_feature_map_slice (preview — uses chunk cache)
    # ================================================================

    def get_feature_map_slice(self, z, speed, use_gpu):
        if self._currently_segmenting is not None:
            return
        _, H, W = self._spatial_shape()
        return self._compute_features_for_region_2d_cached(z, 0, H, 0, W, deep=not speed)

    # ================================================================
    # Chunk-based feature helpers
    # ================================================================

    def compute_features_for_chunk_2d(self, chunk_coords, speed):
        z, y_s, y_e, x_s, x_e = chunk_coords
        fmap = self._compute_features_for_region_2d(z, y_s, y_e, x_s, x_e, deep=not speed)
        return fmap, (y_s, x_s)

    # ================================================================
    # 2D chunking helpers
    # ================================================================

    def get_minimal_chunks_for_coordinates(self, coordinates_by_z):
        MAX = self.twod_chunk_size
        needed = {}
        for z in coordinates_by_z:
            yc = [c[0] for c in coordinates_by_z[z]]
            xc = [c[1] for c in coordinates_by_z[z]]
            yd = self.image_3d.shape[1]; xd = self.image_3d.shape[2]
            tp = yd * xd
            if tp <= MAX:
                needed[z] = [[z, 0, yd, 0, xd]]
            else:
                nn = int(np.ceil(tp / MAX))
                byc = 1; bxc = nn; bar = float('inf')
                for ycc in range(1, nn + 1):
                    xcc = int(np.ceil(nn / ycc))
                    cy = int(np.ceil(yd / ycc)); cx = int(np.ceil(xd / xcc))
                    if cy * cx > MAX:
                        continue
                    ar = max(cy, cx) / max(min(cy, cx), 1)
                    if ar < bar:
                        bar = ar; byc = ycc; bxc = xcc
                cz = []
                if bar == float('inf'):
                    ld = 'y' if yd >= xd else 'x'
                    nd = int(np.ceil(tp / MAX))
                    if ld == 'y':
                        ds = int(np.ceil(yd / nd))
                        for i in range(0, yd, ds):
                            ei = min(i + ds, yd)
                            if any(i <= y <= ei - 1 for y in yc):
                                cz.append([z, i, ei, 0, xd])
                    else:
                        ds = int(np.ceil(xd / nd))
                        for i in range(0, xd, ds):
                            ei = min(i + ds, xd)
                            if any(i <= x <= ei - 1 for x in xc):
                                cz.append([z, 0, yd, i, ei])
                else:
                    ycs = int(np.ceil(yd / byc)); xcs = int(np.ceil(xd / bxc))
                    for yi in range(byc):
                        for xi in range(bxc):
                            ys = yi * ycs; ye = min(ys + ycs, yd)
                            xs = xi * xcs; xe = min(xs + xcs, xd)
                            if ys >= yd or xs >= xd:
                                continue
                            if any(ys <= y <= ye - 1 and xs <= x <= xe - 1
                                   for y, x in zip(yc, xc)):
                                cz.append([z, ys, ye, xs, xe])
                needed[z] = cz
        return needed

    def organize_by_z(self, coordinates):
        z_dict = defaultdict(list)
        for z, y, x in coordinates:
            z_dict[z].append((y, x))
        return dict(z_dict)

    # ================================================================
    # Chunk processing (segmentation prediction)
    # ================================================================

    def process_chunk(self, chunk_coords):
        info = self.realtimechunks[chunk_coords]
        deep = not self.speed
        if not self.use_two:
            zs, ze, ys, ye, xs, xe = info['bounds']
            ca = cp.stack(cp.meshgrid(
                cp.arange(zs, ze), cp.arange(ys, ye), cp.arange(xs, xe),
                indexing='ij')).reshape(3, -1).T
            fm = self._compute_features_for_region_3d(zs, ze, ys, ye, xs, xe, deep=deep)
            loc = ca - cp.array([zs, ys, xs])
            features_gpu = fm[loc[:, 0], loc[:, 1], loc[:, 2]]
        else:
            z = info['z']
            ys, ye, xs, xe = info['coords']
            ca = cp.stack(cp.meshgrid(
                cp.array([z]), cp.arange(ys, ye), cp.arange(xs, xe),
                indexing='ij')).reshape(3, -1).T
            fm = self._compute_features_for_region_2d_cached(z, ys, ye, xs, xe, deep=deep)
            features_gpu = fm[ca[:, 1] - ys, ca[:, 2] - xs]

        # Convert to numpy for LightGBM prediction
        features_cpu = cp.asnumpy(features_gpu)
        preds = self.model.predict(features_cpu).astype(bool)
        preds_gpu = cp.array(preds)

        fg_coords = cp.asnumpy(ca[preds_gpu])
        bg_coords = cp.asnumpy(ca[~preds_gpu])
        return set(map(tuple, fg_coords)), set(map(tuple, bg_coords))

    def twodim_coords(self, z, y_start, y_end, x_start, x_end):
        yr = cp.arange(y_start, y_end, dtype=int)
        xr = cp.arange(x_start, x_end, dtype=int)
        yg, xg = cp.meshgrid(yr, xr, indexing='ij')
        n = len(yr) * len(xr)
        return cp.column_stack((cp.full(n, z, dtype=int), yg.ravel(), xg.ravel()))

    # ================================================================
    # segment_volume
    # ================================================================

    def segment_volume(self, array, chunk_size=None, gpu=True):
        array = cp.asarray(array)
        self.realtimechunks = None
        chunk_size = self.master_chunk

        def create_2d_chunks():
            MAX = self.twod_chunk_size; chunks = []
            for z in range(self.image_3d.shape[0]):
                yd = self.image_3d.shape[1]; xd = self.image_3d.shape[2]; tp = yd * xd
                if tp <= MAX:
                    chunks.append([yd, xd, z, tp, None])
                else:
                    nn = int(np.ceil(tp / MAX))
                    byc = 1; bxc = nn; bar = float('inf')
                    for ycc in range(1, nn + 1):
                        xcc = int(np.ceil(nn / ycc))
                        cy = int(np.ceil(yd / ycc)); cx = int(np.ceil(xd / xcc))
                        if cy * cx > MAX:
                            continue
                        ar = max(cy, cx) / max(min(cy, cx), 1)
                        if ar < bar:
                            bar = ar; byc = ycc; bxc = xcc
                    if bar == float('inf'):
                        ld = 'y' if yd >= xd else 'x'; nd = int(np.ceil(tp / MAX))
                        if ld == 'y':
                            ds = int(np.ceil(yd / nd))
                            for i in range(0, yd, ds):
                                chunks.append([yd, xd, z, None, ['y', i, min(i + ds, yd)]])
                        else:
                            ds = int(np.ceil(xd / nd))
                            for i in range(0, xd, ds):
                                chunks.append([yd, xd, z, None, ['x', i, min(i + ds, xd)]])
                    else:
                        ycs = int(np.ceil(yd / byc)); xcs = int(np.ceil(xd / bxc))
                        for yi in range(byc):
                            for xi in range(bxc):
                                yss = yi * ycs; yee = min(yss + ycs, yd)
                                xss = xi * xcs; xee = min(xss + xcs, xd)
                                if yss >= yd or xss >= xd:
                                    continue
                                chunks.append([yd, xd, z, None, ['2d', yss, yee, xss, xee]])
            return chunks

        print("Chunking data...")
        chunks = self.compute_3d_chunks(chunk_size) if not self.use_two else create_2d_chunks()
        print("Processing chunks in batches...")

        max_workers = self.batch_amplifier * multiprocessing.cpu_count()
        batch_size = max_workers; total_processed = 0

        try:
            for bs in range(0, len(chunks), batch_size):
                be = min(bs + batch_size, len(chunks))
                batch = chunks[bs:be]
                print(f"Processing batch {bs // batch_size + 1}/"
                      f"{(len(chunks) + batch_size - 1) // batch_size}")

                # Sequential GPU feature extraction (GPU serialisation is faster
                # than contention from threaded kernel launches)
                results = []
                for c in batch:
                    feat, coord = self.extract_chunk_features(c)
                    if len(feat) > 0:
                        results.append((feat, coord))

                if results:
                    af = cp.vstack([r[0] for r in results])
                    ac = cp.vstack([r[1] for r in results])
                    af_cpu = cp.asnumpy(af)
                    preds = self.model.predict(af_cpu).astype(bool)
                    preds_gpu = cp.array(preds)
                    fg = ac[preds_gpu]
                    if len(fg) > 0:
                        array[fg[:, 0], fg[:, 1], fg[:, 2]] = 255
                    del af, ac, af_cpu, preds, preds_gpu, fg

                total_processed += len(batch)
                print(f"Completed {total_processed}/{len(chunks)} chunks")
                cp.get_default_memory_pool().free_all_blocks()
        finally:
            cp.get_default_memory_pool().free_all_blocks()
        return cp.asnumpy(array)

    # ================================================================
    # extract_chunk_features
    # ================================================================

    def extract_chunk_features(self, chunk_coords):
        deep = not self.speed
        if self.previewing or not self.use_two:
            if self.realtimechunks is None:
                z_min, z_max = chunk_coords[0], chunk_coords[1]
                y_min, y_max = chunk_coords[2], chunk_coords[3]
                x_min, x_max = chunk_coords[4], chunk_coords[5]
                zr = cp.arange(z_min, z_max); yr = cp.arange(y_min, y_max)
                xr = cp.arange(x_min, x_max)
                zg, yg, xg = cp.meshgrid(zr, yr, xr, indexing='ij')
                ca = cp.column_stack([zg.ravel(), yg.ravel(), xg.ravel()])
                fm = self._compute_features_for_region_3d(
                    z_min, z_max, y_min, y_max, x_min, x_max, deep=deep)
                loc = ca - cp.array([z_min, y_min, x_min])
            else:
                ca = cp.array(chunk_coords)
                z_min = int(ca[:, 0].min()); z_max = int(ca[:, 0].max())
                y_min = int(ca[:, 1].min()); y_max = int(ca[:, 1].max())
                x_min = int(ca[:, 2].min()); x_max = int(ca[:, 2].max())
                fm = self._compute_features_for_region_3d(
                    z_min, z_max + 1, y_min, y_max + 1, x_min, x_max + 1, deep=deep)
                loc = ca - cp.array([z_min, y_min, x_min])
            return fm[loc[:, 0], loc[:, 1], loc[:, 2]], ca
        else:
            yd, xd, z, _, subrange = chunk_coords
            if subrange is None:
                ys, ye = 0, yd; xs, xe = 0, xd
            elif subrange[0] == 'y':
                ys, ye = subrange[1], subrange[2]; xs, xe = 0, xd
            elif subrange[0] == 'x':
                ys, ye = 0, yd; xs, xe = subrange[1], subrange[2]
            elif subrange[0] == '2d':
                ys, ye = subrange[1], subrange[2]; xs, xe = subrange[3], subrange[4]
            else:
                raise ValueError(f"Unknown subrange: {subrange}")
            ca = self.twodim_coords(z, ys, ye, xs, xe)
            fm = self._compute_features_for_region_2d(z, ys, ye, xs, xe, deep=deep)
            return fm[ca[:, 1] - ys, ca[:, 2] - xs], ca

    # ================================================================
    # Position tracking
    # ================================================================

    def update_position(self, z=None, x=None, y=None):
        if hasattr(self, '_skip_next_update') and self._skip_next_update:
            self._skip_next_update = False; return
        if not hasattr(self, 'prev_z') or self.prev_z is None:
            self.prev_z = z
        if hasattr(self, '_currently_processing') and self._currently_processing:
            self.current_z = z; self.current_x = x; self.current_y = y
            self.prev_z = z; return
        self.current_z = z; self.current_x = x; self.current_y = y
        if self.current_z != self.prev_z:
            self._currently_segmenting = None
        self.prev_z = z

    # ================================================================
    # Realtime chunk management
    # ================================================================

    def get_realtime_chunks(self, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.master_chunk
        all_chunks = self.compute_3d_chunks(chunk_size)
        self.realtimechunks = {
            i: {'bounds': cc, 'processed': False,
                'center': self._get_chunk_center(cc), 'is_3d': True}
            for i, cc in enumerate(all_chunks)
        }

    def _get_chunk_center(self, cc):
        zs, ze, ys, ye, xs, xe = cc
        return ((zs + ze) // 2, (ys + ye) // 2, (xs + xe) // 2)

    def get_realtime_chunks_2d(self, chunk_size=None):
        MAX = self.twod_chunk_size; cd = {}
        for z in range(self.image_3d.shape[0]):
            yd = self.image_3d.shape[1]; xd = self.image_3d.shape[2]; tp = yd * xd
            if tp <= MAX:
                cd[(z, 0, 0)] = {'coords': [0, yd, 0, xd], 'processed': False, 'z': z}
            else:
                nn = int(np.ceil(tp / MAX))
                byc = 1; bxc = nn; bar = float('inf')
                for ycc in range(1, nn + 1):
                    xcc = int(np.ceil(nn / ycc))
                    cy = int(np.ceil(yd / ycc)); cx = int(np.ceil(xd / xcc))
                    if cy * cx > MAX:
                        continue
                    ar = max(cy, cx) / max(min(cy, cx), 1)
                    if ar < bar:
                        bar = ar; byc = ycc; bxc = xcc
                if bar == float('inf'):
                    ld = 'y' if yd >= xd else 'x'; nd = int(np.ceil(tp / MAX))
                    if ld == 'y':
                        ds = int(np.ceil(yd / nd))
                        for i in range(0, yd, ds):
                            ei = min(i + ds, yd)
                            cd[(z, i, 0)] = {'coords': [i, ei, 0, xd], 'processed': False, 'z': z}
                    else:
                        ds = int(np.ceil(xd / nd))
                        for i in range(0, xd, ds):
                            ei = min(i + ds, xd)
                            cd[(z, 0, i)] = {'coords': [0, yd, i, ei], 'processed': False, 'z': z}
                else:
                    ycs = int(np.ceil(yd / byc)); xcs = int(np.ceil(xd / bxc))
                    for yi in range(byc):
                        for xi in range(bxc):
                            ys = yi * ycs; ye = min(ys + ycs, yd)
                            xs = xi * xcs; xe = min(xs + xcs, xd)
                            if ys >= yd or xs >= xd:
                                continue
                            cd[(z, ys, xs)] = {'coords': [ys, ye, xs, xe], 'processed': False, 'z': z}
        self.realtimechunks = cd
        print("Ready!")

    def process_slice_features(self, z, speed, use_gpu, z_fores, z_backs):
        sfg = []; sbg = []
        current_map = self.get_feature_map_slice(z, speed, use_gpu)
        if z in z_fores:
            for y, x in z_fores[z]:
                sfg.append(current_map[y, x])
        if z in z_backs:
            for y, x in z_backs[z]:
                sbg.append(current_map[y, x])
        return sfg, sbg

    def extract_features_parallel(self, needed_chunks, speed, use_gpu, z_fores, z_backs):
        mc = multiprocessing.cpu_count(); fg = []; bg = []
        tasks = []
        for z in needed_chunks:
            for cc in needed_chunks[z]:
                tasks.append((cc, z_fores, z_backs, speed))
        with ThreadPoolExecutor(max_workers=mc) as ex:
            futs = {ex.submit(self.process_chunk_features_for_training, t): t for t in tasks}
            for f in futs:
                cf, cb = f.result()
                fg.extend(cf); bg.extend(cb)
        return fg, bg

    # ================================================================
    # Realtime segmentation
    # ================================================================

    def segment_volume_realtime(self, gpu=True):
        if self.realtimechunks is None:
            (self.get_realtime_chunks() if not self.use_two else self.get_realtime_chunks_2d())
        else:
            for ck in self.realtimechunks:
                self.realtimechunks[ck]['processed'] = False

        def get_nearest():
            D, H, W = self._spatial_shape()
            cz = self.current_z if self.current_z is not None else D // 2
            cy = self.current_y if self.current_y is not None else H // 2
            cx = self.current_x if self.current_x is not None else W // 2
            unp = [(k, v) for k, v in self.realtimechunks.items() if not v['processed']]
            if not unp:
                return None
            if self.use_two:
                czc = [(k, v) for k, v in unp if k[0] == cz]
                if czc:
                    return min(czc, key=lambda x: (x[0][1] - cy)**2 + (x[0][2] - cx)**2)[0]
                nz = sorted(unp, key=lambda x: abs(x[0][0] - cz))
                if nz:
                    tz = nz[0][0][0]
                    zcc = [(k, v) for k, v in unp if k[0] == tz]
                    return min(zcc, key=lambda x: (x[0][1] - cy)**2 + (x[0][2] - cx)**2)[0]
            else:
                nzv = min(unp, key=lambda x: abs(x[1]['center'][0] - cz))[1]['center'][0]
                nzc = [c for c in unp if c[1]['center'][0] == nzv]
                return min(nzc, key=lambda x: sum((a - b)**2 for a, b in
                           zip(x[1]['center'][1:], (cy, cx))))[0]
            return None

        while True:
            ck = get_nearest()
            if ck is None:
                break
            self.realtimechunks[ck]['processed'] = True
            yield self.process_chunk(ck)

    # ================================================================
    # Cleanup
    # ================================================================

    def cleanup(self):
        self._chunk_cache.clear()
        try:
            import gc; gc.collect()
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()
        except Exception as e:
            print(f"Warning: Could not clean up GPU memory: {e}")

    # ================================================================
    # 3D training helpers (sparse z-slice optimization for 2D mode)
    # ================================================================

    def process_grid_cell(self, chunk_info):
        cc, fa = chunk_info
        # fa is expected as numpy; convert chunk region to cupy for feature computation
        zs, ze, ys, ye, xs, xe = [int(v) for v in cc]
        deep = not self.speed
        fa_np = fa if isinstance(fa, np.ndarray) else cp.asnumpy(fa)
        sl = fa_np[zs:ze, ys:ye, xs:xe]
        fore_coords = np.argwhere(sl == 1)
        back_coords = np.argwhere(sl == 2)
        if len(fore_coords) == 0 and len(back_coords) == 0:
            return [], []
        if self.use_two:
            needed_local_z = set()
            if len(fore_coords) > 0:
                needed_local_z.update(fore_coords[:, 0].tolist())
            if len(back_coords) > 0:
                needed_local_z.update(back_coords[:, 0].tolist())
            z_features = {}
            for lz in needed_local_z:
                gz = zs + lz
                fm_gpu = self._compute_features_for_region_2d(gz, ys, ye, xs, xe, deep=deep)
                z_features[lz] = cp.asnumpy(fm_gpu)
            fg = [z_features[lz][ly, lx] for lz, ly, lx in fore_coords]
            bg = [z_features[lz][ly, lx] for lz, ly, lx in back_coords]
        else:
            fm_gpu = self._compute_features_for_region_3d(zs, ze, ys, ye, xs, xe, deep=deep)
            fm_cpu = cp.asnumpy(fm_gpu)
            fg = [fm_cpu[lz, ly, lx] for lz, ly, lx in fore_coords]
            bg = [fm_cpu[lz, ly, lx] for lz, ly, lx in back_coords]
        return fg, bg

    def process_grid_cells_parallel(self, chunks_with_scribbles, foreground_array, max_workers=None):
        data = [(cc, foreground_array) for cc in chunks_with_scribbles]
        fg = []; bg = []
        # Sequential processing — GPU kernel serialisation is faster than contention
        for d in data:
            cf, cb = self.process_grid_cell(d)
            fg.extend(cf); bg.extend(cb)
        return fg, bg

    # ================================================================
    # 3D chunk computation
    # ================================================================

    def compute_3d_chunks(self, chunk_size=None, thickness=49):
        if chunk_size is None:
            if hasattr(self, 'master_chunk') and self.master_chunk is not None:
                chunk_size = self.master_chunk
            else:
                tc = multiprocessing.cpu_count()
                tv = int(np.prod(np.array(self.image_3d.shape[:3])))
                tgt = tv / (tc * 4)
                chunk_size = int(np.cbrt(tgt))
                chunk_size = max(16, min(chunk_size, min(self.image_3d.shape[:3]) // 2))
                chunk_size = ((chunk_size + 7) // 16) * 16
        try:
            depth, height, width = self.image_3d.shape
        except:
            depth, height, width, _ = self.image_3d.shape
        target_volume = chunk_size ** 3
        xy_side = max(1, int(np.sqrt(target_volume / thickness)))
        zc = (depth + thickness - 1) // thickness
        yc = (height + xy_side - 1) // xy_side
        xc = (width + xy_side - 1) // xy_side
        zs = np.full(zc, depth // zc); zs[:depth % zc] += 1
        ys = np.full(yc, height // yc); ys[:height % yc] += 1
        xs = np.full(xc, width // xc); xs[:width % xc] += 1
        zp = np.concatenate([[0], np.cumsum(zs)])
        yp = np.concatenate([[0], np.cumsum(ys)])
        xp = np.concatenate([[0], np.cumsum(xs)])
        chunks = []
        for zi in range(zc):
            for yi in range(yc):
                for xi in range(xc):
                    chunks.append([zp[zi], zp[zi + 1], yp[yi], yp[yi + 1], xp[xi], xp[xi + 1]])
        return chunks

    # ================================================================
    # Training
    # ================================================================

    def train_batch(self, foreground_array, speed=True, use_gpu=True,
                    use_two=False, mem_lock=False, saving=False):
        if not saving:
            print("Training model...")
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=63,
                max_depth=-1,

                min_child_samples=10,
                min_split_gain=0.01,

                colsample_bytree=0.8,
                subsample=0.8,
                subsample_freq=1,

                reg_alpha=0.1,
                reg_lambda=1.0,

                max_bin=255,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )

        self.speed = speed; self.cur_gpu = use_gpu
        if use_two != self.use_two:
            self.realtimechunks = None
        if not use_two:
            self.use_two = False
        self.mem_lock = mem_lock

        # Ensure foreground_array is numpy for coordinate lookup
        fa_np = foreground_array if isinstance(foreground_array, np.ndarray) else cp.asnumpy(cp.asarray(foreground_array))

        if use_two:
            if not self.use_two:
                self.use_two = True
            self.two_slices = []
            zf, yf, xf = np.where(fa_np == 1)
            zb, yb, xb = np.where(fa_np == 2)
            fore_coords = list(zip(zf, yf, xf)); back_coords = list(zip(zb, yb, xb))
            z_fores = self.organize_by_z(fore_coords)
            z_backs = self.organize_by_z(back_coords)
            azc = {}
            for z in z_fores:
                azc.setdefault(z, []).extend(z_fores[z])
            for z in z_backs:
                azc.setdefault(z, []).extend(z_backs[z])
            nc = self.get_minimal_chunks_for_coordinates(azc)
            foreground_features, background_features = self.extract_features_parallel(
                nc, speed, use_gpu, z_fores, z_backs)
            z_fore = np.argwhere(fa_np == 1)
            z_back = np.argwhere(fa_np == 2)
        else:
            foreground_features = []; background_features = []
            z_fore = np.argwhere(fa_np == 1)
            z_back = np.argwhere(fa_np == 2)
            if len(z_fore) == 0 and len(z_back) == 0:
                return foreground_features, background_features
            ac = self.compute_3d_chunks(self.master_chunk)
            ca = np.array(ac)
            asc = np.vstack((z_fore, z_back)) if len(z_back) > 0 else z_fore
            cws = set()
            for z, y, x in asc:
                zm = (ca[:, 0] <= z) & (z < ca[:, 1])
                ym = (ca[:, 2] <= y) & (y < ca[:, 3])
                xm = (ca[:, 4] <= x) & (x < ca[:, 5])
                for idx in np.where(zm & ym & xm)[0]:
                    cws.add(tuple(ca[idx]))
            foreground_features, background_features = self.process_grid_cells_parallel(
                list(cws), fa_np)

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
                print("Could not combine new model with old loaded model.")

        if saving:
            return foreground_features, background_features, z_fore, z_back

        X = np.vstack([foreground_features, background_features])
        y = np.hstack([np.ones(len(z_fore)), np.zeros(len(z_back))])
        try:
            self.model.fit(X, y)
        except:
            print(X); print(y)
        self.current_speed = speed
        cp.get_default_memory_pool().free_all_blocks()
        print("Done")

    # ================================================================
    # Save / Load
    # ================================================================

    def save_model(self, file_name, foreground_array):
        print("Saving model data")
        fg, bg, zf, zb = self.train_batch(foreground_array, speed=self.speed,
            use_gpu=self.use_gpu, use_two=self.use_two, mem_lock=self.mem_lock, saving=True)
        np.savez(file_name, foreground_features=fg, background_features=bg,
                 z_fore=zf, z_back=zb, speed=self.speed, use_gpu=self.use_gpu,
                 use_two=self.use_two, mem_lock=self.mem_lock)
        print(f"Model data saved to {file_name}.")

    def load_model(self, file_name):
        print("Loading model data")
        d = np.load(file_name)
        self.previous_foreground = d['foreground_features']
        self.previous_background = d['background_features']
        self.previous_z_fore = d['z_fore']; self.previous_z_back = d['z_back']
        self.speed = bool(d['speed']); self.use_gpu = bool(d['use_gpu'])
        self.use_two = bool(d['use_two']); self.mem_lock = bool(d['mem_lock'])
        X = np.vstack([self.previous_foreground, self.previous_background])
        y = np.hstack([np.ones(len(self.previous_z_fore)), np.zeros(len(self.previous_z_back))])
        try:
            self.model.fit(X, y)
        except:
            print(X); print(y)
        print("Done")

    # ================================================================
    # Training chunk feature extraction (2D)
    # ================================================================

    def process_chunk_features_for_training(self, chunk_task):
        cc, z_fores, z_backs, speed = chunk_task
        z = cc[0]; ys, ye = cc[1], cc[2]; xs, xe = cc[3], cc[4]
        fm_gpu = self._compute_features_for_region_2d(z, ys, ye, xs, xe, deep=not speed)
        fm = cp.asnumpy(fm_gpu)
        fg = []; bg = []
        if z in z_fores:
            for y, x in z_fores[z]:
                if ys <= y < ye and xs <= x < xe:
                    fg.append(fm[y - ys, x - xs])
        if z in z_backs:
            for y, x in z_backs[z]:
                if ys <= y < ye and xs <= x < xe:
                    bg.append(fm[y - ys, x - xs])
        return fg, bg