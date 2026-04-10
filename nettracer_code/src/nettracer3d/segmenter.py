"""
Interactive Segmenter.

Architecture:
  - Features computed directly on padded chunk regions from the full source
    image
  - Numba: optional acceleration for eigenvalue hot loops (2x2 and 3x3).
  - Threading: all parallelism uses ThreadPoolExecutor (no multiprocessing pools).
  - Speed mode: sigmas [1,2,4,8]. Deep mode: sigmas [1,2,4,8,16].
"""

import warnings
warnings.filterwarnings("ignore",
                        message="X does not have valid feature names",
                        category=UserWarning)
import lightgbm as lgb
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy import ndimage
import multiprocessing
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Any
import math

# ============================================================
# Optional Numba acceleration
# ============================================================
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _eigen2x2_numba(hxx, hyy, hxy, out_s, out_l):
        """Eigenvalues of 2x2 symmetric matrices, parallel over pixels."""
        rows, cols = hxx.shape
        for i in prange(rows):
            for j in range(cols):
                a = float(hxx[i, j]); d = float(hyy[i, j]); b = float(hxy[i, j])
                tr = a + d
                det = a * d - b * b
                disc = tr * tr - 4.0 * det
                if disc < 0.0:
                    disc = 0.0
                s = math.sqrt(disc)
                out_s[i, j] = (tr - s) * 0.5
                out_l[i, j] = (tr + s) * 0.5

    @njit(cache=True, parallel=True)
    def _eigen3x3_numba(h11, h22, h33, h12, h13, h23, out0, out1, out2):
        """Eigenvalues of 3x3 symmetric matrices via Cardano's method, sorted ascending."""
        sz0, sz1, sz2 = h11.shape
        TWO_PI_3 = 2.0 * math.pi / 3.0
        for i in prange(sz0):
            for j in range(sz1):
                for k in range(sz2):
                    a = float(h11[i,j,k]); b = float(h22[i,j,k]); c = float(h33[i,j,k])
                    d = float(h12[i,j,k]); f = float(h13[i,j,k]); e = float(h23[i,j,k])
                    p1 = d*d + f*f + e*e
                    q = (a + b + c) / 3.0
                    if p1 < 1e-30:
                        e1 = a; e2 = b; e3 = c
                    else:
                        p2 = (a-q)*(a-q) + (b-q)*(b-q) + (c-q)*(c-q) + 2.0*p1
                        p = math.sqrt(p2 / 6.0)
                        b11=(a-q)/p; b22=(b-q)/p; b33=(c-q)/p
                        b12=d/p; b13=f/p; b23=e/p
                        det_b = (b11*(b22*b33-b23*b23)
                                 - b12*(b12*b33-b23*b13)
                                 + b13*(b12*b23-b22*b13))
                        r = det_b * 0.5
                        if r > 1.0: r = 1.0
                        if r < -1.0: r = -1.0
                        phi = math.acos(r) / 3.0
                        e1 = q + 2.0*p*math.cos(phi)
                        e3 = q + 2.0*p*math.cos(phi + TWO_PI_3)
                        e2 = 3.0*q - e1 - e3
                    if e1 > e2: e1, e2 = e2, e1
                    if e2 > e3: e2, e3 = e3, e2
                    if e1 > e2: e1, e2 = e2, e1
                    out0[i,j,k] = e1; out1[i,j,k] = e2; out2[i,j,k] = e3


def _eigen2x2(hxx, hyy, hxy):
    """Dispatch: eigenvalues of 2x2 symmetric matrices -> (small, large)."""
    if _HAS_NUMBA and hxx.ndim == 2:
        out_s = np.empty(hxx.shape, dtype=np.float64)
        out_l = np.empty(hxx.shape, dtype=np.float64)
        _eigen2x2_numba(hxx.astype(np.float64), hyy.astype(np.float64),
                        hxy.astype(np.float64), out_s, out_l)
        return out_s.astype(np.float32), out_l.astype(np.float32)
    tr = (hxx + hyy).astype(np.float64)
    det = (hxx * hyy - hxy * hxy).astype(np.float64)
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    s = np.sqrt(disc)
    return ((tr - s) * 0.5).astype(np.float32), ((tr + s) * 0.5).astype(np.float32)


def _eigen3x3(h11, h22, h33, h12, h13, h23):
    """Dispatch: eigenvalues of 3x3 symmetric matrices -> (e0, e1, e2) ascending."""
    if _HAS_NUMBA and h11.ndim == 3:
        out0 = np.empty(h11.shape, dtype=np.float64)
        out1 = np.empty(h11.shape, dtype=np.float64)
        out2 = np.empty(h11.shape, dtype=np.float64)
        _eigen3x3_numba(h11.astype(np.float64), h22.astype(np.float64),
                        h33.astype(np.float64), h12.astype(np.float64),
                        h13.astype(np.float64), h23.astype(np.float64),
                        out0, out1, out2)
        return out0.astype(np.float32), out1.astype(np.float32), out2.astype(np.float32)
    shape = h11.shape
    n = int(np.prod(shape))
    mat = np.zeros((n, 3, 3), dtype=np.float64)
    mat[:,0,0] = h11.ravel(); mat[:,1,1] = h22.ravel(); mat[:,2,2] = h33.ravel()
    mat[:,0,1] = mat[:,1,0] = h12.ravel()
    mat[:,0,2] = mat[:,2,0] = h13.ravel()
    mat[:,1,2] = mat[:,2,1] = h23.ravel()
    eigs = np.linalg.eigvalsh(mat)
    return (eigs[:,0].reshape(shape).astype(np.float32),
            eigs[:,1].reshape(shape).astype(np.float32),
            eigs[:,2].reshape(shape).astype(np.float32))


# ============================================================
# Derivative helper
# ============================================================
_K1 = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
_K2 = np.array([1.0, -2.0, 1.0], dtype=np.float32)

def _d(image, axis, order=1):
    return ndimage.convolve1d(image, _K1 if order == 1 else _K2, axis=axis, mode='reflect')


# ============================================================
# Feature assembly functions
# ============================================================
def _assemble_2d(G, sigmas, st_scales, deep):
    N = len(sigmas); feats = []
    feats.append(G[0])
    for i in range(1, N+1): feats.append(G[i])
    for i in range(1, N+1):
        for j in range(i+1, N+1): feats.append(G[i] - G[j])
    for i in range(N+1):
        gx = _d(G[i], 1); gy = _d(G[i], 0)
        feats.append(np.sqrt(gx*gx + gy*gy))
    for i in range(N+1):
        feats.append(_d(G[i], 1, 2) + _d(G[i], 0, 2))
    if deep:
        for i in range(N+1):
            hxx = _d(G[i], 1, 2); hyy = _d(G[i], 0, 2)
            hxy = _d(_d(G[i], 1), 0)
            s, l = _eigen2x2(hxx, hyy, hxy)
            feats.append(s); feats.append(l)
        for i in range(N+1):
            gx = _d(G[i], 1); gy = _d(G[i], 0)
            Pxx = gx*gx; Pxy = gx*gy; Pyy = gy*gy
            for gamma in st_scales:
                Qxx = ndimage.gaussian_filter(Pxx, gamma)
                Qxy = ndimage.gaussian_filter(Pxy, gamma)
                Qyy = ndimage.gaussian_filter(Pyy, gamma)
                s, l = _eigen2x2(Qxx, Qyy, Qxy)
                feats.append(s); feats.append(l)
        orig = G[0]; orig64 = orig.astype(np.float64)
        for sigma in sigmas:
            win = int(1 + 2 * sigma)
            feats.append(ndimage.minimum_filter(orig, size=win, mode='reflect'))
            feats.append(ndimage.maximum_filter(orig, size=win, mode='reflect'))
            mn = ndimage.uniform_filter(orig64, size=win, mode='reflect').astype(np.float32)
            feats.append(mn)
            var = ndimage.uniform_filter(orig64**2, size=win, mode='reflect').astype(np.float32) - mn*mn
            feats.append(np.maximum(var, 0))
    return np.stack(feats, axis=-1)


def _assemble_3d(G, sigmas, st_scales, deep):
    N = len(sigmas); feats = []
    feats.append(G[0])
    for i in range(1, N+1): feats.append(G[i])
    for i in range(1, N+1):
        for j in range(i+1, N+1): feats.append(G[i] - G[j])
    for i in range(N+1):
        gx=_d(G[i],2); gy=_d(G[i],1); gz=_d(G[i],0)
        feats.append(np.sqrt(gx*gx+gy*gy+gz*gz))
    for i in range(N+1):
        feats.append(_d(G[i],2,2)+_d(G[i],1,2)+_d(G[i],0,2))
    if deep:
        for i in range(N+1):
            hxx=_d(G[i],2,2); hyy=_d(G[i],1,2); hzz=_d(G[i],0,2)
            hxy=_d(_d(G[i],2),1); hxz=_d(_d(G[i],2),0); hyz=_d(_d(G[i],1),0)
            e0,e1,e2=_eigen3x3(hxx,hyy,hzz,hxy,hxz,hyz)
            feats.append(e0); feats.append(e1); feats.append(e2)
        for i in range(N+1):
            gx=_d(G[i],2); gy=_d(G[i],1); gz=_d(G[i],0)
            Pxx=gx*gx; Pxy=gx*gy; Pxz=gx*gz; Pyy=gy*gy; Pyz=gy*gz; Pzz=gz*gz
            for gamma in st_scales:
                e0,e1,e2=_eigen3x3(
                    ndimage.gaussian_filter(Pxx,gamma), ndimage.gaussian_filter(Pyy,gamma),
                    ndimage.gaussian_filter(Pzz,gamma), ndimage.gaussian_filter(Pxy,gamma),
                    ndimage.gaussian_filter(Pxz,gamma), ndimage.gaussian_filter(Pyz,gamma))
                feats.append(e0); feats.append(e1); feats.append(e2)
        orig=G[0]; orig64=orig.astype(np.float64)
        for sigma in sigmas:
            win=int(1+2*sigma)
            feats.append(ndimage.minimum_filter(orig,size=win,mode='reflect'))
            feats.append(ndimage.maximum_filter(orig,size=win,mode='reflect'))
            mn=ndimage.uniform_filter(orig64,size=win,mode='reflect').astype(np.float32)
            feats.append(mn)
            var=ndimage.uniform_filter(orig64**2,size=win,mode='reflect').astype(np.float32)-mn*mn
            feats.append(np.maximum(var,0))
    return np.stack(feats, axis=-1)


# ============================================================
# Chunk Cache - LRU for computed feature chunks
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
# Interactive Segmenter
# ============================================================
class InteractiveSegmenter:
    def __init__(self, image_3d, use_gpu=False):
        self.image_3d = image_3d
        self.patterns = []
        self.use_gpu = False

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
        self.cur_gpu = False
        self.prev_z = None
        self.previewing = False

        self._currently_processing = False
        self._skip_next_update = False
        self._last_processed_slice = None
        self.mem_lock = False

        # Feature parameters: speed uses 1-8, deep adds 16
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
    # Feature computation — direct from source image with padding
    # ================================================================

    def _compute_features_for_region_2d(self, z, y_s, y_e, x_s, x_e, deep=False):
        """Compute features for a 2D chunk directly from the source image."""
        sigmas = self._get_sigmas(deep)
        pad = self._get_pad(deep)
        _, H, W = self._spatial_shape()

        py_s = max(0, y_s - pad); py_e = min(H, y_e + pad)
        px_s = max(0, x_s - pad); px_e = min(W, x_e + pad)

        if self._is_rgb():
            parts = []
            for ch in range(self.image_3d.shape[-1]):
                src = np.ascontiguousarray(self.image_3d[z, py_s:py_e, px_s:px_e, ch], dtype=np.float32)
                G = [src] + [ndimage.gaussian_filter(src, s) for s in sigmas]
                parts.append(_assemble_2d(G, sigmas, self.structure_tensor_scales, deep))
            features = np.concatenate(parts, axis=-1)
        else:
            src = np.ascontiguousarray(self.image_3d[z, py_s:py_e, px_s:px_e], dtype=np.float32)
            G = [src] + [ndimage.gaussian_filter(src, s) for s in sigmas]
            features = _assemble_2d(G, sigmas, self.structure_tensor_scales, deep)

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
        """Compute features for a 3D chunk directly from the source image."""
        sigmas = self._get_sigmas(deep)
        pad = self._get_pad(deep)
        D, H, W = self._spatial_shape()

        pz_s = max(0, z_s - pad); pz_e = min(D, z_e + pad)
        py_s = max(0, y_s - pad); py_e = min(H, y_e + pad)
        px_s = max(0, x_s - pad); px_e = min(W, x_e + pad)

        if self._is_rgb():
            parts = []
            for ch in range(self.image_3d.shape[-1]):
                src = np.ascontiguousarray(self.image_3d[pz_s:pz_e, py_s:py_e, px_s:px_e, ch], dtype=np.float32)
                G = [src] + [ndimage.gaussian_filter(src, s) for s in sigmas]
                parts.append(_assemble_3d(G, sigmas, self.structure_tensor_scales, deep))
            features = np.concatenate(parts, axis=-1)
        else:
            src = np.ascontiguousarray(self.image_3d[pz_s:pz_e, py_s:py_e, px_s:px_e], dtype=np.float32)
            G = [src] + [ndimage.gaussian_filter(src, s) for s in sigmas]
            features = _assemble_3d(G, sigmas, self.structure_tensor_scales, deep)

        cz = z_s - pz_s; cy = y_s - py_s; cx = x_s - px_s
        return features[cz:cz+(z_e-z_s), cy:cy+(y_e-y_s), cx:cx+(x_e-x_s)]

    # ================================================================
    # Backward-compatible direct computation (for arbitrary images)
    # ================================================================

    def compute_feature_maps_cpu_2d(self, z=None, image_2d=None):
        if image_2d is None:
            if z is None: z = self._spatial_shape()[0] // 2
            _, H, W = self._spatial_shape()
            return self._compute_features_for_region_2d(z, 0, H, 0, W, deep=False)
        if image_2d.ndim == 3 and image_2d.shape[-1] in (3, 4):
            return np.concatenate([self.compute_feature_maps_cpu_2d(image_2d=image_2d[..., c])
                                   for c in range(image_2d.shape[-1])], axis=-1)
        src = image_2d.astype(np.float32)
        G = [src] + [ndimage.gaussian_filter(src, s) for s in self.sigmas]
        return _assemble_2d(G, self.sigmas, self.structure_tensor_scales, deep=False)

    def compute_deep_feature_maps_cpu_2d(self, z=None, image_2d=None):
        if image_2d is None:
            if z is None: z = self._spatial_shape()[0] // 2
            _, H, W = self._spatial_shape()
            return self._compute_features_for_region_2d(z, 0, H, 0, W, deep=True)
        if image_2d.ndim == 3 and image_2d.shape[-1] in (3, 4):
            return np.concatenate([self.compute_deep_feature_maps_cpu_2d(image_2d=image_2d[..., c])
                                   for c in range(image_2d.shape[-1])], axis=-1)
        src = image_2d.astype(np.float32)
        G = [src] + [ndimage.gaussian_filter(src, s) for s in self.sigmas_deep]
        return _assemble_2d(G, self.sigmas_deep, self.structure_tensor_scales, deep=True)

    def compute_feature_maps_cpu(self, image_3d=None):
        if image_3d is None: image_3d = self.image_3d
        if image_3d.ndim == 4 and image_3d.shape[-1] in (3, 4):
            return np.concatenate([self.compute_feature_maps_cpu(image_3d[..., c])
                                   for c in range(image_3d.shape[-1])], axis=-1)
        src = image_3d.astype(np.float32)
        G = [src] + [ndimage.gaussian_filter(src, s) for s in self.sigmas]
        return _assemble_3d(G, self.sigmas, self.structure_tensor_scales, deep=False)

    def compute_deep_feature_maps_cpu(self, image_3d=None):
        if image_3d is None: image_3d = self.image_3d
        if image_3d.ndim == 4 and image_3d.shape[-1] in (3, 4):
            return np.concatenate([self.compute_deep_feature_maps_cpu(image_3d[..., c])
                                   for c in range(image_3d.shape[-1])], axis=-1)
        src = image_3d.astype(np.float32)
        G = [src] + [ndimage.gaussian_filter(src, s) for s in self.sigmas_deep]
        return _assemble_3d(G, self.sigmas_deep, self.structure_tensor_scales, deep=True)

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

    def compute_features_for_chunk_2d_cpu(self, chunk_coords, speed):
        z, y_s, y_e, x_s, x_e = chunk_coords
        fmap = self._compute_features_for_region_2d(z, y_s, y_e, x_s, x_e, deep=not speed)
        return fmap, (y_s, x_s)

    # ================================================================
    # 2D chunking helpers
    # ================================================================

    def get_minimal_chunks_for_coordinates_cpu(self, coordinates_by_z):
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
                    if cy * cx > MAX: continue
                    ar = max(cy, cx) / max(min(cy, cx), 1)
                    if ar < bar: bar = ar; byc = ycc; bxc = xcc
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
                            ys = yi*ycs; ye = min(ys+ycs, yd)
                            xs = xi*xcs; xe = min(xs+xcs, xd)
                            if ys >= yd or xs >= xd: continue
                            if any(ys <= y <= ye-1 and xs <= x <= xe-1
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
            ca = np.stack(np.meshgrid(
                np.arange(zs, ze), np.arange(ys, ye), np.arange(xs, xe),
                indexing='ij')).reshape(3, -1).T
            fm = self._compute_features_for_region_3d(zs, ze, ys, ye, xs, xe, deep=deep)
            loc = ca - np.array([zs, ys, xs])
            features = fm[loc[:,0], loc[:,1], loc[:,2]]
        else:
            z = info['z']
            ys, ye, xs, xe = info['coords']
            ca = np.stack(np.meshgrid(
                [z], np.arange(ys, ye), np.arange(xs, xe), indexing='ij'
            )).reshape(3, -1).T
            fm = self._compute_features_for_region_2d_cached(z, ys, ye, xs, xe, deep=deep)
            features = fm[ca[:,1]-ys, ca[:,2]-xs]
        preds = self.model.predict(features).astype(bool)
        return set(map(tuple, ca[preds])), set(map(tuple, ca[~preds]))

    def twodim_coords(self, z, y_start, y_end, x_start, x_end):
        yr = np.arange(y_start, y_end, dtype=int)
        xr = np.arange(x_start, x_end, dtype=int)
        yg, xg = np.meshgrid(yr, xr, indexing='ij')
        n = len(yr) * len(xr)
        return np.column_stack((np.full(n, z, dtype=int), yg.ravel(), xg.ravel()))

    # ================================================================
    # segment_volume
    # ================================================================

    def segment_volume(self, array, chunk_size=None, gpu=False):
        self.realtimechunks = None
        chunk_size = self.master_chunk

        def create_2d_chunks():
            MAX = self.twod_chunk_size; chunks = []
            for z in range(self.image_3d.shape[0]):
                yd = self.image_3d.shape[1]; xd = self.image_3d.shape[2]; tp = yd*xd
                if tp <= MAX:
                    chunks.append([yd, xd, z, tp, None])
                else:
                    nn = int(np.ceil(tp/MAX))
                    byc=1; bxc=nn; bar=float('inf')
                    for ycc in range(1, nn+1):
                        xcc = int(np.ceil(nn/ycc))
                        cy=int(np.ceil(yd/ycc)); cx=int(np.ceil(xd/xcc))
                        if cy*cx > MAX: continue
                        ar = max(cy,cx)/max(min(cy,cx),1)
                        if ar < bar: bar=ar; byc=ycc; bxc=xcc
                    if bar == float('inf'):
                        ld = 'y' if yd >= xd else 'x'; nd = int(np.ceil(tp/MAX))
                        if ld == 'y':
                            ds = int(np.ceil(yd/nd))
                            for i in range(0, yd, ds):
                                chunks.append([yd,xd,z,None,['y',i,min(i+ds,yd)]])
                        else:
                            ds = int(np.ceil(xd/nd))
                            for i in range(0, xd, ds):
                                chunks.append([yd,xd,z,None,['x',i,min(i+ds,xd)]])
                    else:
                        ycs=int(np.ceil(yd/byc)); xcs=int(np.ceil(xd/bxc))
                        for yi in range(byc):
                            for xi in range(bxc):
                                yss=yi*ycs; yee=min(yss+ycs,yd)
                                xss=xi*xcs; xee=min(xss+xcs,xd)
                                if yss>=yd or xss>=xd: continue
                                chunks.append([yd,xd,z,None,['2d',yss,yee,xss,xee]])
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
                print(f"Processing batch {bs//batch_size+1}/{(len(chunks)+batch_size-1)//batch_size}")

                results = []
                with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                    futs = [ex.submit(self.extract_chunk_features, c) for c in batch]
                    for f in futs:
                        feat, coord = f.result()
                        if len(feat) > 0: results.append((feat, coord))

                if results:
                    af = np.vstack([r[0] for r in results])
                    ac = np.vstack([r[1] for r in results])
                    preds = self.model.predict(af).astype(bool)
                    fg = ac[preds]
                    if len(fg) > 0: array[fg[:,0], fg[:,1], fg[:,2]] = 255
                    del af, ac, preds, fg

                total_processed += len(batch)
                print(f"Completed {total_processed}/{len(chunks)} chunks")
        finally:
            pass
        return array

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
                zr = np.arange(z_min, z_max); yr = np.arange(y_min, y_max); xr = np.arange(x_min, x_max)
                zg, yg, xg = np.meshgrid(zr, yr, xr, indexing='ij')
                ca = np.column_stack([zg.ravel(), yg.ravel(), xg.ravel()])
                fm = self._compute_features_for_region_3d(z_min, z_max, y_min, y_max, x_min, x_max, deep=deep)
                loc = ca - np.array([z_min, y_min, x_min])
            else:
                ca = np.array(chunk_coords)
                z_min = ca[:,0].min(); z_max = ca[:,0].max()
                y_min = ca[:,1].min(); y_max = ca[:,1].max()
                x_min = ca[:,2].min(); x_max = ca[:,2].max()
                fm = self._compute_features_for_region_3d(z_min, z_max+1, y_min, y_max+1, x_min, x_max+1, deep=deep)
                loc = ca - np.array([z_min, y_min, x_min])
            return fm[loc[:,0], loc[:,1], loc[:,2]], ca
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
            return fm[ca[:,1]-ys, ca[:,2]-xs], ca

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
        if chunk_size is None: chunk_size = self.master_chunk
        all_chunks = self.compute_3d_chunks(chunk_size)
        self.realtimechunks = {
            i: {'bounds': cc, 'processed': False,
                'center': self._get_chunk_center(cc), 'is_3d': True}
            for i, cc in enumerate(all_chunks)
        }

    def _get_chunk_center(self, cc):
        zs, ze, ys, ye, xs, xe = cc
        return ((zs+ze)//2, (ys+ye)//2, (xs+xe)//2)

    def get_realtime_chunks_2d(self, chunk_size=None):
        MAX = self.twod_chunk_size; cd = {}
        for z in range(self.image_3d.shape[0]):
            yd = self.image_3d.shape[1]; xd = self.image_3d.shape[2]; tp = yd*xd
            if tp <= MAX:
                cd[(z,0,0)] = {'coords':[0,yd,0,xd], 'processed':False, 'z':z}
            else:
                nn = int(np.ceil(tp/MAX))
                byc=1; bxc=nn; bar=float('inf')
                for ycc in range(1, nn+1):
                    xcc=int(np.ceil(nn/ycc))
                    cy=int(np.ceil(yd/ycc)); cx=int(np.ceil(xd/xcc))
                    if cy*cx > MAX: continue
                    ar = max(cy,cx)/max(min(cy,cx),1)
                    if ar < bar: bar=ar; byc=ycc; bxc=xcc
                if bar == float('inf'):
                    ld = 'y' if yd >= xd else 'x'; nd = int(np.ceil(tp/MAX))
                    if ld == 'y':
                        ds = int(np.ceil(yd/nd))
                        for i in range(0, yd, ds):
                            ei=min(i+ds,yd)
                            cd[(z,i,0)] = {'coords':[i,ei,0,xd], 'processed':False, 'z':z}
                    else:
                        ds = int(np.ceil(xd/nd))
                        for i in range(0, xd, ds):
                            ei=min(i+ds,xd)
                            cd[(z,0,i)] = {'coords':[0,yd,i,ei], 'processed':False, 'z':z}
                else:
                    ycs=int(np.ceil(yd/byc)); xcs=int(np.ceil(xd/bxc))
                    for yi in range(byc):
                        for xi in range(bxc):
                            ys=yi*ycs; ye=min(ys+ycs,yd); xs=xi*xcs; xe=min(xs+xcs,xd)
                            if ys>=yd or xs>=xd: continue
                            cd[(z,ys,xs)] = {'coords':[ys,ye,xs,xe], 'processed':False, 'z':z}
        self.realtimechunks = cd
        print("Ready!")

    def process_slice_features(self, z, speed, use_gpu, z_fores, z_backs):
        sfg = []; sbg = []
        current_map = self.get_feature_map_slice(z, speed, use_gpu)
        if z in z_fores:
            for y, x in z_fores[z]: sfg.append(current_map[y, x])
        if z in z_backs:
            for y, x in z_backs[z]: sbg.append(current_map[y, x])
        return sfg, sbg

    def extract_features_parallel(self, needed_chunks, speed, use_gpu, z_fores, z_backs):
        mc = multiprocessing.cpu_count(); fg=[]; bg=[]
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

    def segment_volume_realtime(self, gpu=False):
        if self.realtimechunks is None:
            (self.get_realtime_chunks() if not self.use_two else self.get_realtime_chunks_2d())
        else:
            for ck in self.realtimechunks: self.realtimechunks[ck]['processed'] = False

        def get_nearest():
            D, H, W = self._spatial_shape()
            cz = self.current_z if self.current_z is not None else D//2
            cy = self.current_y if self.current_y is not None else H//2
            cx = self.current_x if self.current_x is not None else W//2
            unp = [(k,v) for k,v in self.realtimechunks.items() if not v['processed']]
            if not unp: return None
            if self.use_two:
                czc = [(k,v) for k,v in unp if k[0]==cz]
                if czc:
                    return min(czc, key=lambda x: (x[0][1]-cy)**2+(x[0][2]-cx)**2)[0]
                nz = sorted(unp, key=lambda x: abs(x[0][0]-cz))
                if nz:
                    tz = nz[0][0][0]
                    zcc = [(k,v) for k,v in unp if k[0]==tz]
                    return min(zcc, key=lambda x: (x[0][1]-cy)**2+(x[0][2]-cx)**2)[0]
            else:
                nzv = min(unp, key=lambda x: abs(x[1]['center'][0]-cz))[1]['center'][0]
                nzc = [c for c in unp if c[1]['center'][0]==nzv]
                return min(nzc, key=lambda x: sum((a-b)**2 for a,b in
                           zip(x[1]['center'][1:], (cy,cx))))[0]
            return None

        while True:
            ck = get_nearest()
            if ck is None: break
            self.realtimechunks[ck]['processed'] = True
            yield self.process_chunk(ck)

    # ================================================================
    # Cleanup
    # ================================================================

    def cleanup(self):
        self._chunk_cache.clear()
        if self.use_gpu:
            try:
                import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
                import torch; torch.cuda.empty_cache()
            except: pass

    # ================================================================
    # 3D training helpers (sparse z-slice optimization for 2D mode)
    # ================================================================

    def process_grid_cell(self, chunk_info):
        cc, fa = chunk_info
        zs, ze, ys, ye, xs, xe = [int(v) for v in cc]
        deep = not self.speed
        sl = fa[zs:ze, ys:ye, xs:xe]
        fore_coords = np.argwhere(sl == 1)
        back_coords = np.argwhere(sl == 2)
        if len(fore_coords) == 0 and len(back_coords) == 0:
            return [], []
        if self.use_two:
            needed_local_z = set()
            if len(fore_coords) > 0: needed_local_z.update(fore_coords[:, 0].tolist())
            if len(back_coords) > 0: needed_local_z.update(back_coords[:, 0].tolist())
            z_features = {}
            for lz in needed_local_z:
                gz = zs + lz
                z_features[lz] = self._compute_features_for_region_2d(gz, ys, ye, xs, xe, deep=deep)
            fg = [z_features[lz][ly, lx] for lz, ly, lx in fore_coords]
            bg = [z_features[lz][ly, lx] for lz, ly, lx in back_coords]
        else:
            fm = self._compute_features_for_region_3d(zs, ze, ys, ye, xs, xe, deep=deep)
            fg = [fm[lz, ly, lx] for lz, ly, lx in fore_coords]
            bg = [fm[lz, ly, lx] for lz, ly, lx in back_coords]
        return fg, bg

    def process_grid_cells_parallel(self, chunks_with_scribbles, foreground_array, max_workers=None):
        data = [(cc, foreground_array) for cc in chunks_with_scribbles]
        fg=[]; bg=[]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.process_grid_cell, d) for d in data]
            for f in futs:
                cf, cb = f.result(); fg.extend(cf); bg.extend(cb)
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
                tv = np.prod(self.image_3d.shape[:3])
                tgt = tv / (tc * 4)
                chunk_size = int(np.cbrt(tgt))
                chunk_size = max(16, min(chunk_size, min(self.image_3d.shape[:3])//2))
                chunk_size = ((chunk_size+7)//16)*16
        try: depth, height, width = self.image_3d.shape
        except: depth, height, width, _ = self.image_3d.shape
        target_volume = chunk_size**3
        xy_side = max(1, int(np.sqrt(target_volume / thickness)))
        zc = (depth+thickness-1)//thickness
        yc = (height+xy_side-1)//xy_side
        xc = (width+xy_side-1)//xy_side
        zs = np.full(zc, depth//zc); zs[:depth%zc] += 1
        ys = np.full(yc, height//yc); ys[:height%yc] += 1
        xs = np.full(xc, width//xc); xs[:width%xc] += 1
        zp = np.concatenate([[0], np.cumsum(zs)])
        yp = np.concatenate([[0], np.cumsum(ys)])
        xp = np.concatenate([[0], np.cumsum(xs)])
        chunks = []
        for zi in range(zc):
            for yi in range(yc):
                for xi in range(xc):
                    chunks.append([zp[zi],zp[zi+1],yp[yi],yp[yi+1],xp[xi],xp[xi+1]])
        return chunks

    # ================================================================
    # Training
    # ================================================================

    def train_batch(self, foreground_array, speed=True, use_gpu=False,
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
        if use_two != self.use_two: self.realtimechunks = None
        if not use_two: self.use_two = False
        self.mem_lock = mem_lock

        if use_two:
            if not self.use_two: self.use_two = True
            self.two_slices = []
            zf,yf,xf = np.where(foreground_array==1)
            zb,yb,xb = np.where(foreground_array==2)
            fore_coords = list(zip(zf,yf,xf)); back_coords = list(zip(zb,yb,xb))
            z_fores = self.organize_by_z(fore_coords)
            z_backs = self.organize_by_z(back_coords)
            azc = {}
            for z in z_fores: azc.setdefault(z,[]).extend(z_fores[z])
            for z in z_backs: azc.setdefault(z,[]).extend(z_backs[z])
            nc = self.get_minimal_chunks_for_coordinates_cpu(azc)
            foreground_features, background_features = self.extract_features_parallel(
                nc, speed, use_gpu, z_fores, z_backs)
            z_fore = np.argwhere(foreground_array==1)
            z_back = np.argwhere(foreground_array==2)
        else:
            foreground_features=[]; background_features=[]
            z_fore = np.argwhere(foreground_array==1)
            z_back = np.argwhere(foreground_array==2)
            if len(z_fore)==0 and len(z_back)==0:
                return foreground_features, background_features
            ac = self.compute_3d_chunks(self.master_chunk)
            ca = np.array(ac)
            asc = np.vstack((z_fore, z_back)) if len(z_back)>0 else z_fore
            cws = set()
            for z,y,x in asc:
                zm=(ca[:,0]<=z)&(z<ca[:,1]); ym=(ca[:,2]<=y)&(y<ca[:,3]); xm=(ca[:,4]<=x)&(x<ca[:,5])
                for idx in np.where(zm&ym&xm)[0]: cws.add(tuple(ca[idx]))
            foreground_features, background_features = self.process_grid_cells_parallel(list(cws), foreground_array)

        if self.previous_foreground is not None:
            failed = True
            try: foreground_features = np.vstack([self.previous_foreground, foreground_features]); failed=False
            except: pass
            try: background_features = np.vstack([self.previous_background, background_features]); failed=False
            except: pass
            try: z_fore = np.concatenate([self.previous_z_fore, z_fore])
            except: pass
            try: z_back = np.concatenate([self.previous_z_back, z_back])
            except: pass
            if failed: print("Could not combine new model with old loaded model.")

        if saving: return foreground_features, background_features, z_fore, z_back

        X = np.vstack([foreground_features, background_features])
        y = np.hstack([np.ones(len(z_fore)), np.zeros(len(z_back))])
        try: self.model.fit(X, y)
        except: print(X); print(y)
        self.current_speed = speed; print("Done")

    # ================================================================
    # Save / Load
    # ================================================================

    def save_model(self, file_name, foreground_array):
        print("Saving model data")
        fg,bg,zf,zb = self.train_batch(foreground_array, speed=self.speed,
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
        try: self.model.fit(X, y)
        except: print(X); print(y)
        print("Done")

    # ================================================================
    # Training chunk feature extraction (2D)
    # ================================================================

    def process_chunk_features_for_training(self, chunk_task):
        cc, z_fores, z_backs, speed = chunk_task
        z = cc[0]; ys, ye = cc[1], cc[2]; xs, xe = cc[3], cc[4]
        fm = self._compute_features_for_region_2d(z, ys, ye, xs, xe, deep=not speed)
        fg=[]; bg=[]
        if z in z_fores:
            for y,x in z_fores[z]:
                if ys<=y<ye and xs<=x<xe: fg.append(fm[y-ys, x-xs])
        if z in z_backs:
            for y,x in z_backs[z]:
                if ys<=y<ye and xs<=x<xe: bg.append(fm[y-ys, x-xs])
        return fg, bg