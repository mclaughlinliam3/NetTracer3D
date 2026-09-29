from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Optional
import umap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as mcolors
from collections import Counter
from . import community_extractor
import random
import re
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Patch
import os
import warnings
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg as la
try:
    from pynndescent import NNDescent
    HAVE_PYNND = True
except ImportError:                                    # optional dependency
    HAVE_PYNND = False

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
APPROX_KNN_ABOVE = 50_000
DROP_JACCARD_ABOVE = 1_000_000
BISECT_WARN_ABOVE = 200_000

def cluster_arrays_dbscan(data_input, seed=42):
    """
    Simple DBSCAN clustering of 1D arrays with sensible defaults.
    
    Parameters:
    -----------
    data_input : dict or List[List[float]]
        Dictionary {key: array} or list of arrays to cluster
    seed : int
        Random seed for reproducibility (used for parameter estimation)
        
    Returns:
    --------
    list: [[key1, key2], [key3, key4, key5]] - List of clusters, each containing keys/indices
          Note: Outliers are excluded from the output
    """
    
    # Handle both dict and list inputs
    if isinstance(data_input, dict):
        keys = list(data_input.keys())
        array_values = list(data_input.values())
    else:
        keys = list(range(len(data_input)))
        array_values = data_input
    
    # Convert to numpy
    data = np.array(array_values)
    n_samples = len(data)
    
    # Simple heuristics for DBSCAN parameters
    min_samples = max(3, int(np.sqrt(n_samples) * 0.2))  # Roughly sqrt(n)/5, minimum 3
    
    # Estimate eps using 4th nearest neighbor distance (common heuristic)
    k = min(4, n_samples - 1)
    if k > 0:
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(data)
        distances, _ = nbrs.kneighbors(data)
        # Use 80th percentile of k-nearest distances as eps
        eps = np.percentile(distances[:, k], 80)
    else:
        eps = 0.1  # fallback
    
    print(f"Using DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Organize results into clusters (excluding outliers)
    clusters = []
    
    # Add only the main clusters (non-noise points)
    unique_labels = np.unique(labels)
    main_clusters = [label for label in unique_labels if label != -1]
    
    for label in main_clusters:
        cluster_indices = np.where(labels == label)[0]
        cluster_keys = [keys[i] for i in cluster_indices]
        clusters.append(cluster_keys)
    
    n_main_clusters = len(main_clusters)
    n_outliers = np.sum(labels == -1)
    
    print(f"Found {n_main_clusters} main clusters and {n_outliers} outliers (go in neighborhood 0)")
    
    return clusters

# KMeans related:

def _apply_weighting(data, weighting=None, weights=None, eps=1e-12, verbose=True):
    """
    Transform columns before clustering so each contributes as intended to
    Euclidean distance.
 
    weighting :
        None / "none"   - raw proportions. Distance dominated by abundant types.
        "sqrt"          - variance-stabilising for proportion data. Partially
                          lifts rare columns without fully equalising them.
        "zscore"        - standardise each column to mean 0, SD 1. Every cell
                          type contributes equally regardless of abundance.
        "sqrt_zscore"   - sqrt then standardise. Equalises contribution while
                          damping the spikiness of very rare columns.
        "custom"        - use `weights` only, no transform.
 
    weights : optional 1D array of per-column multipliers, applied AFTER the
        transform. Works with any mode, so you can standardise and then
        upweight specific columns.
 
    Returns (transformed_data, kept_column_mask).
    Zero-variance columns are zeroed rather than dropped, so column indices
    still line up with your original label order.
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected 2D data, got shape {X.shape}")
 
    mode = (weighting or "none").lower()
    valid = {"none", "sqrt", "zscore", "sqrt_zscore", "custom"}
    if mode not in valid:
        raise ValueError(f"weighting must be one of {sorted(valid)}, got {weighting!r}")
 
    if np.any(X < -eps) and mode in ("sqrt", "sqrt_zscore"):
        raise ValueError("sqrt weighting expects non-negative data (proportions)")
 
    Xw = X.copy()
    if mode in ("sqrt", "sqrt_zscore"):
        Xw = np.sqrt(np.clip(Xw, 0.0, None))
 
    keep = np.ones(X.shape[1], dtype=bool)
    if mode in ("zscore", "sqrt_zscore"):
        sd = Xw.std(axis=0)
        keep = sd > eps
        mu = Xw.mean(axis=0)
        out = np.zeros_like(Xw)
        out[:, keep] = (Xw[:, keep] - mu[keep]) / sd[keep]
        Xw = out
        n_dropped = int((~keep).sum())
        if n_dropped and verbose:
            print(f"  [weighting] {n_dropped} zero-variance column(s) zeroed: "
                  f"{np.where(~keep)[0].tolist()}")
 
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != X.shape[1]:
            raise ValueError(f"weights has length {w.size}, expected {X.shape[1]}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        Xw = Xw * w
        keep = keep & (w > eps)
 
    if verbose:
        v = Xw.var(axis=0)
        nz = v[v > eps]
        if nz.size:
            print(f"  [weighting] mode={mode}  column variance "
                  f"min={nz.min():.4g} max={nz.max():.4g} ratio={nz.max()/nz.min():.1f}x")
    return Xw, keep
 
 
def cluster_arrays(data_input, n_clusters=None, seed=42, max_k=None,
                   weighting=None, weights=None, return_details=False,
                   verbose=True):
    """
    Cluster 1D arrays with key tracking, automatic cluster count detection,
    and optional column weighting.
 
    Parameters
    ----------
    data_input : dict {key: array} or list of arrays
    n_clusters : int or None. If None, chosen by Calinski-Harabasz.
    seed : int
    max_k : int or None. Cap for auto-detection. Defaults to len(keys)//3.
    weighting : see _apply_weighting.
    weights : optional per-column multipliers.
    return_details : if True, also return a dict with labels, centroids in
        both weighted and original space, inertia and the chosen k.
 
    Returns
    -------
    list of lists of keys, or (clusters, details) if return_details.
    """
    if isinstance(data_input, dict):
        keys = list(data_input.keys())
        array_values = list(data_input.values())
    else:
        keys = list(range(len(data_input)))
        array_values = data_input
 
    data = np.asarray(array_values, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
 
    # transform once, then use the SAME space for k selection and the final fit
    data_w, keep = _apply_weighting(data, weighting, weights, verbose=verbose)
 
    if n_clusters is None:
        if max_k is None:
            max_k = max(2, len(keys) // 3)
        n_clusters = _find_optimal_clusters(data_w, seed, max_k=max_k, verbose=verbose)
        if verbose:
            print(f"Auto-detected optimal number of clusters: {n_clusters}")
 
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(data_w)
 
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(keys[i])
 
    if not return_details:
        return clusters
 
    # centroids in the original (untransformed) space are what you profile on
    centroids_orig = np.vstack([
        data[labels == k].mean(axis=0) if np.any(labels == k) else np.full(data.shape[1], np.nan)
        for k in range(n_clusters)
    ])
    details = {
        "labels": labels,
        "keys": keys,
        "n_clusters": n_clusters,
        "weighting": weighting,
        "centroids_weighted": kmeans.cluster_centers_,
        "centroids_original": centroids_orig,
        "inertia": kmeans.inertia_,
        "kept_columns": keep,
        "sizes": np.bincount(labels, minlength=n_clusters),
    }
    return clusters
 
 
def _find_optimal_clusters(data, seed, max_k, verbose=True):
    """Find optimal k using the Calinski-Harabasz index, on already-weighted data."""
    n_samples = len(data)
    if n_samples < 2:
        return 1
 
    max_k = min(max_k, n_samples - 1, 20)
    if verbose:
        print(f"Max_k: {max_k}, n_samples: {n_samples}")
    if max_k < 2:
        return 1
 
    ch_scores = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        try:
            if verbose:
                print(f"Testing {k} clusters")
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(data)
            if len(np.unique(labels)) == k:
                ch_scores.append(calinski_harabasz_score(data, labels))
            else:
                ch_scores.append(0)
        except Exception:
            ch_scores.append(0)
 
    if ch_scores and max(ch_scores) > 0:
        optimal_k = k_range[int(np.argmax(ch_scores))]
        if verbose:
            print(f"Using {optimal_k} neighborhoods")
        return optimal_k
    return 2
 
 
def cluster_stability(data_input, n_clusters, seeds=(0, 1, 2, 3, 4),
                      weighting=None, weights=None):
    """
    Rerun clustering across seeds and report how consistently pairs of samples
    land together. Worth running whenever weighting='zscore', since rare
    columns can produce clusters driven by single cells.
 
    Returns mean pairwise co-assignment agreement across seed pairs (0-1).
    """
    from itertools import combinations
 
    if isinstance(data_input, dict):
        arrays = list(data_input.values())
    else:
        arrays = data_input
    data = np.asarray(arrays, dtype=float)
    data_w, _ = _apply_weighting(data, weighting, weights, verbose=False)
 
    all_labels = []
    for s in seeds:
        km = KMeans(n_clusters=n_clusters, random_state=s, n_init=10)
        all_labels.append(km.fit_predict(data_w))
 
    agreements = []
    for a, b in combinations(all_labels, 2):
        same_a = a[:, None] == a[None, :]
        same_b = b[:, None] == b[None, :]
        iu = np.triu_indices(len(a), k=1)
        agreements.append(float((same_a[iu] == same_b[iu]).mean()))
    return float(np.mean(agreements))


# Leiden array clustering:

def _knn_indices(data, k, approx, metric, n_jobs, seed, log):
    n = data.shape[0]
    k = min(k, n - 1)
 
    if approx == "auto":
        approx = n > APPROX_KNN_ABOVE
        if approx and not HAVE_PYNND:
            log(f"n={n:,} would benefit from approximate kNN, but pynndescent "
                f"is not installed. Falling back to exact search (slower). "
                f"`pip install pynndescent` is recommended at this scale.")
            approx = False
        elif approx:
            log(f"n={n:,} > {APPROX_KNN_ABOVE:,}: using approximate kNN "
                f"(recall ~0.998). Pass approx=False to force exact.")
 
    if approx:
        if not HAVE_PYNND:
            raise ImportError("approx=True requires pynndescent")
        idx = NNDescent(data, n_neighbors=k + 1, metric=metric,
                        random_state=seed, n_jobs=n_jobs).neighbor_graph[0]
        return idx[:, 1:]
 
    # 'brute' explicitly, never 'auto': sklearn's auto can select a kd_tree,
    # which above ~20 dimensions is roughly 10x SLOWER than brute force because
    # axis-aligned splits stop pruning. Brute force here is a BLAS matmul.
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                          metric=metric, n_jobs=n_jobs).fit(data)
    return nn.kneighbors(data, return_distance=False)[:, 1:]
 
 
# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------
 
def _jaccard_on_edges(A, chunk, log):
    """Jaccard weights for the edges of A only. Values identical to the full
    (A @ A.T) route; memory is O(chunk * k^2) instead of O(n * k^2)."""
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    rs, cs, ws = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = A[start:stop]
        shared = (block @ A.T).multiply(block).tocoo()
        if shared.nnz:
            r = (shared.row + start).astype(np.int32)
            c = shared.col.astype(np.int32)
            s = shared.data.astype(np.float32)
            rs.append(r); cs.append(c)
            ws.append(s / np.maximum(deg[r] + deg[c] - s, 1e-12))
    if not rs:
        return sp.csr_matrix(A.shape, dtype=np.float32)
    return sp.csr_matrix((np.concatenate(ws),
                          (np.concatenate(rs), np.concatenate(cs))),
                         shape=A.shape, dtype=np.float32)
 
 
def build_knn_graph(data, k=30, use_jaccard=True, metric="euclidean",
                    prune=0.0, n_jobs=-1, approx="auto", seed=42,
                    chunk=20_000, log=print):
    """kNN graph in FEATURE space (not physical space), as an igraph.Graph."""
    n = data.shape[0]
    idx = _knn_indices(data, k, approx, metric, n_jobs, seed, log)
    kk = idx.shape[1]
 
    rows = np.repeat(np.arange(n, dtype=np.int32), kk)
    A = sp.csr_matrix((np.ones(n * kk, dtype=np.float32),
                       (rows, idx.ravel().astype(np.int32))), shape=(n, n))
    del rows, idx
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
 
    if use_jaccard:
        A = _jaccard_on_edges(A, chunk, log)
        A = A.maximum(A.T)
 
    if prune > 0:
        before = A.nnz
        A.data[A.data <= prune] = 0
        A.eliminate_zeros()
        log(f"pruned {(before - A.nnz) // 2:,} edges at weight<={prune}. "
            f"Note: pruning isolates nodes, and every isolated node becomes a "
            f"singleton community. Pair with min_cluster_size.")
 
    A = sp.triu(A, k=1).tocsr()
 
    # Hand igraph the sparse matrix directly. Never build a Python edge list.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ig.Graph.Weighted_Adjacency(A, mode="undirected",
                                           attr="weight", loops=False)
 
 
# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
 
def cluster_arrays_leiden(data_input, n_clusters=None, seed=42,
                          k=30, resolution=1.0,
                          use_jaccard="auto", approx="auto", n_pca=None,
                          prune=0.0, min_cluster_size=1, n_iterations=2,
                          metric="euclidean", n_jobs=-1, chunk=20_000,
                          tol=0, max_bisect=25, verbose=True,
                          return_info=False):
    """
    Cluster feature vectors by Leiden community detection on a kNN graph.
 
    Same I/O contract as a k-means helper: dict {key: vector} or a sequence of
    vectors in, list-of-lists of keys out, largest cluster first.
 
    Parameters that affect RESULTS
    ------------------------------
    k : int
        Neighbours per node. The primary granularity control -- small k (~15)
        resolves fine subsets, large k (~50) merges into broad groups. Never
        adjusted automatically.
    resolution : float
        Higher -> more, smaller communities. Ignored when n_clusters is set.
    n_clusters : int or None
        None (recommended) lets the graph determine the count. An int bisects
        `resolution` to hit that target, which re-runs Leiden up to max_bisect
        times -- expensive at scale, and the target may be unreachable if the
        graph's structure does not support it.
    n_pca : int or None
        Reduce to this many components before the kNN search. OFF by default:
        it changes the partition and would silently transform every user's
        feature space. Worth enabling manually for high-dimensional data where
        the kNN search dominates runtime.
    prune : float
        Leave at 0. Pruning isolates nodes into singleton communities.
 
    Parameters that trade speed for a different (not worse) partition
    ----------------------------------------------------------------
    approx : "auto" | True | False
        Approximate kNN. "auto" enables above 50,000 rows if pynndescent is
        installed. Recall ~0.998.
    use_jaccard : "auto" | True | False
        Shared-neighbour edge reweighting. "auto" keeps it on except above
        1,000,000 rows, where it roughly doubles runtime for no measured gain
        in agreement with curated labels.
 
    Returns
    -------
    list of lists of keys, or (clusters, info) when return_info=True.
    """
    def log(msg):
        if verbose:
            print(f"[leiden] {msg}")
 
    if isinstance(data_input, dict):
        keys = list(data_input.keys())
        values = list(data_input.values())
    else:
        keys = list(range(len(data_input)))
        values = data_input
 
    data = np.asarray(values, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] < 3:
        return ([[key] for key in keys], {}) if return_info else \
               [[key] for key in keys]
    if not np.isfinite(data).all():
        raise ValueError("data contains NaN/inf; impute or drop those rows first")
 
    n, d = data.shape
    log(f"{n:,} vectors x {d} features, k={k}")
 
    # ---- resolve the automatic decisions, and say what they were ----------
    if use_jaccard == "auto":
        use_jaccard = n <= DROP_JACCARD_ABOVE
        if not use_jaccard:
            log(f"n={n:,} > {DROP_JACCARD_ABOVE:,}: skipping the Jaccard "
                f"reweight for speed. Pass use_jaccard=True to force it.")
 
    if n_clusters is not None and n > BISECT_WARN_ABOVE:
        warnings.warn(
            f"n_clusters={n_clusters} at n={n:,} re-runs Leiden up to "
            f"{max_bisect} times to bisect resolution. Prefer n_clusters=None "
            f"and tune `resolution`.", RuntimeWarning)
        max_bisect = min(max_bisect, 6)
 
    if n_pca and n_pca < d:
        log(f"PCA {d} -> {n_pca} components (user-requested)")
        data = np.ascontiguousarray(
            PCA(n_components=n_pca, random_state=seed,
                svd_solver="randomized").fit_transform(data), dtype=np.float32)
 
    # ---- graph ------------------------------------------------------------
    g = build_knn_graph(data, k=k, use_jaccard=use_jaccard, metric=metric,
                        prune=prune, n_jobs=n_jobs, approx=approx, seed=seed,
                        chunk=chunk, log=log)
    del data
    log(f"graph: {g.vcount():,} nodes, {g.ecount():,} edges")
 
    weights = g.es["weight"] if "weight" in g.es.attributes() else None
 
    def partition_at(res):
        return la.find_partition(g, la.RBConfigurationVertexPartition,
                                 weights=weights, resolution_parameter=res,
                                 n_iterations=n_iterations, seed=seed)
 
    # ---- partition --------------------------------------------------------
    used_res = resolution
    if n_clusters is None:
        part = partition_at(resolution)
        log(f"resolution={resolution} -> {len(part)} communities")
    else:
        lo, hi = 1e-3, 10.0
        part, best = None, None
        for _ in range(max_bisect):
            used_res = (lo + hi) / 2
            p = partition_at(used_res)
            c = len(p)
            if best is None or abs(c - n_clusters) < abs(best - n_clusters):
                part, best = p, c
            if abs(c - n_clusters) <= tol:
                break
            lo, hi = (used_res, hi) if c < n_clusters else (lo, used_res)
        log(f"targeted {n_clusters}, resolved {best} at resolution~{used_res:.4f}")
        if best != n_clusters:
            log("exact count not reachable; the graph's community structure "
                "does not support it")
 
    labels = np.asarray(part.membership)
    del g
 
    # group without a Python loop over every node
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(labels.max() + 2))
    keys_arr = np.asarray(keys, dtype=object)
    groups = [list(keys_arr[order[bounds[i]:bounds[i + 1]]])
              for i in range(labels.max() + 1)]
    groups.sort(key=len, reverse=True)
 
    if min_cluster_size > 1:
        big = [c for c in groups if len(c) >= min_cluster_size]
        small = [key for c in groups if len(c) < min_cluster_size for key in c]
        if small:
            big.append(small)
            log(f"{len(small):,} items in communities below "
                f"min_cluster_size={min_cluster_size} -> trailing bucket")
        groups = big
 
    sizes = [len(c) for c in groups]
    log(f"{len(groups)} clusters, sizes {sizes[:8]}"
        f"{' ...' if len(sizes) > 8 else ''}")
 
    if return_info:
        return groups, dict(n=n, n_features=d, k=k, resolution=used_res,
                            use_jaccard=use_jaccard, approx=approx,
                            n_pca=n_pca, prune=prune, seed=seed,
                            n_clusters_found=len(groups))
    return groups



# Graphing methods        
    
def plot_dict_heatmap(unsorted_data_dict, id_set, figsize=(12, 8), title="Neighborhood Heatmap",
                     center_at_one=False, center_at_zero=False, sublabel="Community",
                     bar_label="Representation Ratio", x_label=None):
    """
    Create a heatmap from a dictionary of numpy arrays.
    (docstring unchanged)
 
    Non-finite handling:
      +inf -> a modest tint above neutral, striped with gray ('/' hatch)
      -inf -> a modest tint below neutral, striped with gray ('\' hatch)
      nan  -> solid gray via cmap.set_bad
    The tint is deliberately mild, so the hatch (not the color) is what marks a
    cell as infinite; a legend explaining the stripes is added whenever any
    appear. Striping keeps infinities readable at any matrix size, since the
    numeric annotations only render when there are <= 20 columns.
    Display-only; the returned dict preserves the raw inf/nan values.
    """
 
    # Infinity rendering, all on the 0-1 transformed color scale.
    # Diverging scales step this far off neutral (0.5); the sequential scale has no
    # negative direction, so both infinities take one mild tint and the hatch
    # direction carries the sign.
    INF_TINT_DIVERGING = 0.18   # -> 0.68 / 0.32
    INF_TINT_SEQUENTIAL = 0.30
    STRIPE_COLOR = 'dimgray'    # must stay dark: the fills are now pale
 
    def _truncate(label, max_len=18):
        s = str(label)
        return s if len(s) <= max_len else s[:max_len - 1] + '\u2026'
 
    data_dict = {k: unsorted_data_dict[k] for k in sorted(unsorted_data_dict.keys())}
    keys = list(data_dict.keys())
 
    # Build the matrix as float so None -> nan (numpy coerces None to nan under dtype=float).
    try:
        data_matrix = np.array([data_dict[key] for key in keys], dtype=float)
    except (TypeError, ValueError):
        # Ragged rows or values that don't cast cleanly -> coerce element by element.
        data_matrix = np.array([
            [0 if v is None else float(v) for v in np.atleast_1d(data_dict[key])]
            for key in keys
        ], dtype=float)
 
    # Convert id_set to sorted list if it's a set or unsorted list
    if isinstance(id_set, set):
        sorted_id_set = sorted(list(id_set))
    else:
        sorted_id_set = sorted(id_set)
 
    original_id_list = list(id_set) if isinstance(id_set, set) else id_set
    sorted_indices = [original_id_list.index(id_val) for id_val in sorted_id_set]
    data_matrix = data_matrix[:, sorted_indices]
 
    # Move key 0 to the bottom if it exists as the first key
    if keys and keys[0] == 0:
        keys.append(keys.pop(0))
        data_matrix = np.vstack([data_matrix[1:], data_matrix[0:1]])
 
    # Range from FINITE values only, so nan/inf can't break min/max or tick filtering.
    finite_mask = np.isfinite(data_matrix)
    if np.any(finite_mask):
        data_min = float(np.min(data_matrix[finite_mask]))
        data_max = float(np.max(data_matrix[finite_mask]))
    else:
        data_min, data_max = 0.0, 1.0
 
    fig, ax = plt.subplots(figsize=figsize)
 
    if center_at_zero:
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
                  '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
        cmap.set_bad(color='lightgray')  # nan cells only
 
        max_abs = max(abs(data_min), abs(data_max))
 
        inf_pos_level = 0.5 + INF_TINT_DIVERGING
        inf_neg_level = 0.5 - INF_TINT_DIVERGING
 
        def transform_data(data):
            data = np.asarray(data, dtype=float)
            transformed = np.full(data.shape, np.nan)
            transformed[np.isposinf(data)] = inf_pos_level
            transformed[np.isneginf(data)] = inf_neg_level
            finite = np.isfinite(data)
            if max_abs == 0:
                transformed[finite] = 0.5
                return transformed
            neg = finite & (data < 0)
            transformed[neg] = 0.5 * (1 - np.sqrt(np.abs(data[neg])) / np.sqrt(max_abs))
            pos = finite & (data > 0)
            transformed[pos] = 0.5 + 0.5 * np.sqrt(data[pos]) / np.sqrt(max_abs)
            transformed[finite & (data == 0)] = 0.5
            return transformed
 
        transformed_matrix = transform_data(data_matrix)
        im = ax.imshow(transformed_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax)
 
        if max_abs <= 1:
            step_values = [0, 0.25, 0.5, 0.75, 1.0]
        elif max_abs <= 2:
            step_values = [0, 0.5, 1.0, 1.5, 2.0]
        else:
            step_values = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
 
        tick_values = []
        for val in step_values:
            if -val >= data_min and val != 0:
                tick_values.append(-val)
        tick_values.append(0)
        for val in step_values:
            if val <= data_max and val != 0:
                tick_values.append(val)
        tick_values = sorted(set(tick_values))
 
        cbar.set_ticks(transform_data(np.array(tick_values, dtype=float)))
        cbar.set_ticklabels([f'{v:.2f}' for v in tick_values])
        cbar.ax.set_ylabel('Value (centered at 0)', rotation=-90, va="bottom")
 
    elif center_at_one:
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
                  '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
        cmap.set_bad(color='lightgray')  # nan cells only
 
        # Reference excess from the finite matrix max, shared by matrix and ticks
        # (previously each call recomputed its own max, misaligning the colorbar).
        ref_max_excess = max(data_max - 1, 0.0)
 
        inf_pos_level = 0.5 + INF_TINT_DIVERGING
        inf_neg_level = 0.5 - INF_TINT_DIVERGING
 
        def transform_data(data):
            data = np.asarray(data, dtype=float)
            transformed = np.full(data.shape, np.nan)
            transformed[np.isposinf(data)] = inf_pos_level
            transformed[np.isneginf(data)] = inf_neg_level
            finite = np.isfinite(data)
            low = finite & (data <= 1)
            transformed[low] = 0.5 * np.sqrt(np.clip(data[low], 0, None))
            high = finite & (data > 1)
            if np.any(high):
                if ref_max_excess > 0:
                    excess_norm = np.log1p(data[high] - 1) / np.log1p(ref_max_excess)
                    transformed[high] = 0.5 + 0.5 * np.clip(excess_norm, 0, 1)
                else:
                    transformed[high] = 0.5
            return transformed
 
        transformed_matrix = transform_data(data_matrix)
        im = ax.imshow(transformed_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax)
 
        if data_max > 1:
            tick_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        else:
            tick_values = [0, 0.25, 0.5, 0.75, 1.0]
        tick_values = [v for v in tick_values if data_min <= v <= data_max]
 
        cbar.set_ticks(transform_data(np.array(tick_values, dtype=float)))
        cbar.set_ticklabels([f'{v:.2f}' for v in tick_values])
        cbar.ax.set_ylabel(bar_label, rotation=-90, va="bottom")
 
    else:
        cmap = plt.cm.Reds.copy()
        cmap.set_bad(color='lightgray')  # nan cells only
        # No negative direction on a sequential ramp, so both infinities take the
        # same mild tint and the hatch direction distinguishes them.
        inf_pos_level = inf_neg_level = INF_TINT_SEQUENTIAL
        display_matrix = np.where(np.isinf(data_matrix), INF_TINT_SEQUENTIAL, data_matrix)
        im = ax.imshow(display_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Intensity', rotation=-90, va="bottom")
 
    # Stripe the infinities: the tint is already painted by imshow, so a
    # transparent-faced hatch just lays dark lines over it. Runs for every branch
    # and at every matrix size, unlike the numeric annotations below.
    has_pos_inf = bool(np.any(np.isposinf(data_matrix)))
    has_neg_inf = bool(np.any(np.isneginf(data_matrix)))
    for i, j in zip(*np.nonzero(np.isinf(data_matrix))):
        hatch = '///' if data_matrix[i, j] > 0 else '\\\\\\'
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='none',
                               edgecolor=STRIPE_COLOR, hatch=hatch, linewidth=0))
 
    # Legend, only when there is actually a stripe on the plot to explain. The
    # tint alone no longer reads as "extreme", so this is what carries the meaning.
    legend_handles = []
    if has_pos_inf:
        legend_handles.append(Patch(facecolor=cmap(inf_pos_level), edgecolor=STRIPE_COLOR,
                                    hatch='///', label='\u221e  (divide by zero)'))
    if has_neg_inf:
        legend_handles.append(Patch(facecolor=cmap(inf_neg_level), edgecolor=STRIPE_COLOR,
                                    hatch='\\\\\\', label='-\u221e  (divide by zero)'))
 
    # Ticks and labels (truncated so long names don't squish the plot)
    ax.set_xticks(np.arange(len(sorted_id_set)))
    ax.set_yticks(np.arange(len(keys)))
    ax.set_xticklabels([_truncate(s) for s in sorted_id_set], fontsize=8)
 
    labels = list(keys)
    if labels and labels[-1] == 0:
        labels[-1] = 'Excluded (0)'
    ax.set_yticklabels([_truncate(s) for s in labels], fontsize=8)
 
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
 
    # Annotations: color carries magnitude, glyph carries provenance.
    if len(sorted_id_set) <= 20:
        for i in range(len(keys)):
            for j in range(len(sorted_id_set)):
                val = data_matrix[i, j]
                if np.isfinite(val):
                    txt = f'{val:.3f}'
                elif np.isposinf(val):
                    txt = '\u221e'
                elif np.isneginf(val):
                    txt = '-\u221e'
                else:
                    txt = 'n/a'
                ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
 
    ret_dict = {keys[i]: row for i, row in enumerate(data_matrix)}
 
    if x_label:
        ax.set_xlabel(x_label)
    elif center_at_zero:
        ax.set_xlabel('Value Relative to Zero')
    elif center_at_one:
        ax.set_xlabel('Representation Factor of Node Type')
    else:
        ax.set_xlabel('Proportion of Node Type')
 
    ax.set_ylabel(f'{sublabel}')
    ax.set_title(title)
 
    # Placed after set_xlabel so the legend anchors clear of the rotated tick labels.
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, -0.16),
                  ncol=len(legend_handles), frameon=False, fontsize=8,
                  handlelength=2.2, handleheight=1.4)
    plt.tight_layout()
    plt.show()
 
    return ret_dict, sorted_id_set
    


def generate_distinct_colors(n_colors: int) -> List[str]:
    """
    Generate visually distinct colors using matplotlib's tab and Set colormaps.
    Falls back to HSV generation for large numbers of colors.
    
    Args:
        n_colors: Number of distinct colors needed
    
    Returns:
        List of color strings compatible with matplotlib
    """
    if n_colors <= 10:
        # Use tab10 colormap for up to 10 colors
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_colors, 10)))
        return [mcolors.rgb2hex(color) for color in colors[:n_colors]]
    elif n_colors <= 20:
        # Use tab20 for up to 20 colors
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
        return [mcolors.rgb2hex(color) for color in colors[:n_colors]]
    else:
        # For larger numbers, use HSV space
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            color = mcolors.hsv_to_rgb([hue, 0.8, 0.9])  # High saturation and value
            colors.append(mcolors.rgb2hex(color))
        return colors


def visualize_cluster_composition_umap(cluster_data: Dict[int, np.ndarray], 
                                     class_names: Set[str],
                                     label: bool = False,
                                     n_components: int = 2,
                                     random_state: int = 42,
                                     id_dictionary: Optional[Dict[int, str]] = None, 
                                     graph_label = "Community ID",
                                     title = 'UMAP Visualization of Community Compositions',
                                     neighborhoods: Optional[Dict[int, int]] = None,
                                     original_communities = None,
                                     subname = 'Supercommunity'):
    """
    Convert cluster composition data to UMAP visualization.
    
    Parameters:
    -----------
    cluster_data : dict
        Dictionary where keys are cluster IDs (int) and values are 1D numpy arrays
        representing the composition of each cluster
    class_names : set
        Set of strings representing the class names (order corresponds to array indices)
    label : bool
        Whether to show cluster ID labels on the plot
    n_components : int
        Number of UMAP components (default: 2 for 2D visualization)
    random_state : int
        Random state for reproducibility
    id_dictionary : dict, optional
        Dictionary mapping cluster IDs to identity names. If provided, colors will be
        assigned by identity rather than cluster ID, and a legend will be shown.
    neighborhoods : dict, optional
        Dictionary mapping node IDs to neighborhood IDs {node_id: neighborhood_id}.
        If provided, points will be colored by neighborhood using community coloration methods.
    
    Returns:
    --------
    embedding : numpy.ndarray
        UMAP embedding of the cluster compositions
    """
    
    # Convert set to sorted list for consistent ordering


    try:
        class_labels = sorted(list(class_names))
    except:
        class_labels = None
    
    # Extract cluster IDs and compositions
    cluster_ids = list(cluster_data.keys())
    import math
    n = len(cluster_ids)
    point_size = max(10, min(100, 100 / math.log2(n + 1)))
    compositions = np.array([cluster_data[cluster_id] for cluster_id in cluster_ids])
    
    # Create UMAP reducer
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    
    # Fit and transform the composition data
    embedding = reducer.fit_transform(compositions)
    
    # Determine coloring scheme based on parameters
    if neighborhoods is not None and original_communities is not None:
        # Use neighborhood coloring - import the community extractor methods
        from . import community_extractor
        from collections import Counter
        
        # Use original_communities (which is {node: neighborhood}) for color generation
        # This ensures we use the proper node counts for sorting
        
        # Separate outliers (neighborhood 0) from regular neighborhoods in ORIGINAL structure
        outlier_neighborhoods = {node: neighborhood for node, neighborhood in original_communities.items() if neighborhood == 0}
        non_outlier_neighborhoods = {node: neighborhood for node, neighborhood in original_communities.items() if neighborhood != 0}
        
        # Get neighborhoods excluding outliers
        unique_neighborhoods = sorted(set(non_outlier_neighborhoods.values())) if non_outlier_neighborhoods else list()
        
        # Generate colors for non-outlier neighborhoods only (same as assign_community_colors)
        colors = community_extractor.generate_distinct_colors(len(unique_neighborhoods)) if unique_neighborhoods else []
        
        # Sort neighborhoods by size for consistent color assignment (same logic as assign_community_colors)
        # Use the ORIGINAL node counts from original_communities
        if non_outlier_neighborhoods:
            neighborhood_sizes = Counter(non_outlier_neighborhoods.values())
            sorted_neighborhoods = random.Random(42).sample(list(unique_neighborhoods), len(unique_neighborhoods))
            neighborhood_to_color = {neighborhood: colors[i] for i, neighborhood in enumerate(sorted_neighborhoods)}
        else:
            neighborhood_to_color = {}
        
        # Add brown color for outliers (neighborhood 0) - same as assign_community_colors
        if outlier_neighborhoods:
            neighborhood_to_color[0] = (139, 69, 19)  # Brown color (RGB, not RGBA here)
        
        # Map each cluster to its neighborhood color using 'neighborhoods' ({community: neighborhood}) for assignment
        point_colors = []
        neighborhood_labels = []
        for cluster_id in cluster_ids:
            if cluster_id in neighborhoods:
                neighborhood_id = neighborhoods[cluster_id]  # This is {community: neighborhood}
                if neighborhood_id in neighborhood_to_color:
                    point_colors.append(neighborhood_to_color[neighborhood_id])
                    neighborhood_labels.append(neighborhood_id)
                else:
                    # Default color for neighborhoods not found
                    point_colors.append((128, 128, 128))  # Gray
                    neighborhood_labels.append("Unknown")
            else:
                # Default color for nodes not in any neighborhood
                point_colors.append((128, 128, 128))  # Gray
                neighborhood_labels.append("Unknown")
        
        # Normalize RGB values for matplotlib (0-1 range)
        point_colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in point_colors]
        
        # Get unique neighborhoods for legend
        unique_neighborhoods_for_legend = sorted(list(set(neighborhood_to_color.keys())))
        
        use_neighborhood_coloring = True
        
    elif id_dictionary is not None:
        # Use identity coloring (existing logic)
        identities = [id_dictionary.get(cluster_id, "Unknown") for cluster_id in cluster_ids]
        unique_identities = sorted(list(set(identities)))
        colors = generate_distinct_colors(len(unique_identities))
        identity_to_color = {identity: colors[i] for i, identity in enumerate(unique_identities)}
        point_colors = [identity_to_color[identity] for identity in identities]
        use_identity_coloring = True
        use_neighborhood_coloring = False
    else:
        # Use default cluster ID coloring
        point_colors = cluster_ids
        use_identity_coloring = False
        use_neighborhood_coloring = False
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    if n_components == 2:
        if use_neighborhood_coloring:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=point_colors, s=point_size, alpha=0.7)
        elif use_identity_coloring:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=point_colors, s=point_size, alpha=0.7)
        else:
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=point_colors, cmap='viridis', s=point_size, alpha=0.7)
        
        if label:
            # Add cluster ID labels
            for i, cluster_id in enumerate(cluster_ids):
                display_label = f'{cluster_id}'
                if use_neighborhood_coloring and cluster_id in neighborhoods:
                    neighborhood_id = neighborhoods[cluster_id]
                    display_label = f'{cluster_id}\n(N{neighborhood_id})'
                elif id_dictionary is not None:
                    identity = id_dictionary.get(cluster_id, "Unknown")
                    display_label = f'{cluster_id}\n({identity})'
                
                plt.annotate(display_label, 
                            (embedding[i, 0], embedding[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, alpha=0.8, ha='left')
        
        # Add appropriate legend/colorbar
        if use_neighborhood_coloring:
            # Create custom legend for neighborhoods
            legend_elements = []
            for neighborhood_id in unique_neighborhoods_for_legend:
                color = neighborhood_to_color[neighborhood_id]
                norm_color = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=norm_color, 
                              markersize=10, label=f'{subname} {neighborhood_id}')
                )
            plt.legend(handles=legend_elements, title=f'{subname}', 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        elif use_identity_coloring:
            # Create custom legend for identities
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=identity_to_color[identity], 
                                        markersize=10, label=identity)
                             for identity in unique_identities]
            plt.legend(handles=legend_elements, title='Identity', 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.colorbar(scatter, label=graph_label)
        
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        if use_neighborhood_coloring:
            title += f' (Colored by {subname})'
        elif use_identity_coloring:
            title += ' (Colored by Identity)'
        plt.title(title)
        
    elif n_components == 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if use_neighborhood_coloring:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=point_colors, s=point_size, alpha=0.7)
        elif use_identity_coloring:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=point_colors, s=point_size, alpha=0.7)
        else:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=point_colors, cmap='viridis', s=point_size, alpha=0.7)
        
        if label:
            # Add cluster ID labels
            for i, cluster_id in enumerate(cluster_ids):
                display_label = f'C{cluster_id}'
                if use_neighborhood_coloring and cluster_id in neighborhoods:
                    neighborhood_id = neighborhoods[cluster_id]
                    display_label = f'C{cluster_id}\n(N{neighborhood_id})'
                elif id_dictionary is not None:
                    identity = id_dictionary.get(cluster_id, "Unknown")
                    display_label = f'C{cluster_id}\n({identity})'
                
                ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2],
                       display_label, fontsize=8)
        
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        title = '3D UMAP Visualization of Cluster Compositions'
        if use_neighborhood_coloring:
            title += ' (Colored by Neighborhood)'
        elif use_identity_coloring:
            title += ' (Colored by Identity)'
        ax.set_title(title)
        
        # Add appropriate legend/colorbar
        if use_neighborhood_coloring:
            # Create custom legend for neighborhoods
            legend_elements = []
            for neighborhood_id in unique_neighborhoods_for_legend:
                color = neighborhood_to_color[neighborhood_id]
                norm_color = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=norm_color, 
                              markersize=10, label=f'Neighborhood {neighborhood_id}')
                )
            ax.legend(handles=legend_elements, title='Neighborhoods', 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
        elif use_identity_coloring:
            # Create custom legend for identities
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=identity_to_color[identity], 
                                        markersize=10, label=identity)
                             for identity in unique_identities]
            ax.legend(handles=legend_elements, title='Identity', 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.colorbar(scatter, label='Cluster ID')
    
    plt.tight_layout()
    plt.show()
    
    return embedding

def create_community_heatmap(community_intensity, node_community, node_centroids, shape=None, is_3d=True, 
                           labeled_array=None, figsize=(12, 8), point_size=50, alpha=0.7, 
                           colorbar_label="Community Intensity", title="Community Intensity Heatmap"):
    """
    Create a 2D or 3D heatmap showing nodes colored by their community intensities.
    Can return either matplotlib plot or numpy RGB array for overlay purposes.
    
    Parameters:
    -----------
    community_intensity : dict
        Dictionary mapping community IDs to intensity values
        Keys can be np.int64 or regular ints
        
    node_community : dict
        Dictionary mapping node IDs to community IDs
        
    node_centroids : dict
        Dictionary mapping node IDs to centroids
        Centroids should be [Z, Y, X] for 3D or [1, Y, X] for pseudo-3D
        
    shape : tuple, optional
        Shape of the output array in [Z, Y, X] format
        If None, will be inferred from node_centroids
        
    is_3d : bool, default=True
        If True, create 3D plot/array. If False, create 2D plot/array.
        
    labeled_array : np.ndarray, optional
        If provided, returns numpy RGB array overlay using this labeled array template
        instead of matplotlib plot. Uses lookup table approach for efficiency.
        
    figsize : tuple, default=(12, 8)
        Figure size (width, height) - only used for matplotlib
        
    point_size : int, default=50
        Size of scatter plot points - only used for matplotlib
        
    alpha : float, default=0.7
        Transparency of points (0-1) - only used for matplotlib
        
    colorbar_label : str, default="Community Intensity"
        Label for the colorbar - only used for matplotlib
        
    title : str, default="Community Intensity Heatmap"
        Title for the plot
        
    Returns:
    --------
    If labeled_array is None: fig, ax (matplotlib figure and axis objects)
    If labeled_array is provided: np.ndarray (RGB heatmap array with community intensity colors)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert numpy int64 keys to regular ints for consistency
    community_intensity_clean = {}
    for k, v in community_intensity.items():
        if hasattr(k, 'item'):  # numpy scalar
            community_intensity_clean[k.item()] = v
        else:
            community_intensity_clean[k] = v
    
    # Prepare data for plotting
    node_positions = []
    node_intensities = []
    
    for node_id, centroid in node_centroids.items():
        try:
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
                
            # Get community for this node
            community_id = node_community[node_id]
            
            # Convert community_id to regular int if it's numpy
            if hasattr(community_id, 'item'):
                community_id = community_id.item()
                
            # Get intensity for this community
            intensity = community_intensity_clean[community_id]
            
            node_positions.append(centroid)
            node_intensities.append(intensity)
        except KeyError:
            # Skip nodes that don't have community assignments or community intensities
            pass
    
    # Convert to numpy arrays
    positions = np.array(node_positions)
    intensities = np.array(node_intensities)
    
    # Determine shape if not provided
    if shape is None:
        if len(positions) > 0:
            max_coords = np.max(positions, axis=0).astype(int)
            shape = tuple(max_coords + 1)
        else:
            shape = (100, 100, 100) if is_3d else (1, 100, 100)
    
    # Determine min and max intensities for scaling
    if len(intensities) > 0:
        min_intensity = np.min(intensities)
        max_intensity = np.max(intensities)
    else:
        min_intensity, max_intensity = 0, 1
    
    if labeled_array is not None:
        # Create numpy RGB array output using labeled array and lookup table approach
        
        # Create mapping from node ID to community intensity value
        node_to_community_intensity = {}
        for node_id, centroid in node_centroids.items():
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
            
            try:
                # Get community for this node
                community_id = node_community[node_id]
                
                # Convert community_id to regular int if it's numpy
                if hasattr(community_id, 'item'):
                    community_id = community_id.item()
                    
                # Get intensity for this community
                if community_id in community_intensity_clean:
                    node_to_community_intensity[node_id] = community_intensity_clean[community_id]
            except KeyError:
                # Skip nodes that don't have community assignments
                pass
        
        # Create colormap function (RdBu_r - red for high, blue for low, yellow/white for middle)
        def intensity_to_rgb(intensity, min_val, max_val):
            """Convert intensity value to RGB using RdBu_r colormap logic, centered at 0"""
            
            # Handle edge case where all values are the same
            if max_val == min_val:
                if intensity == 0:
                    return np.array([255, 255, 255], dtype=np.uint8)  # White for 0
                elif intensity > 0:
                    return np.array([255, 200, 200], dtype=np.uint8)  # Light red for positive
                else:
                    return np.array([200, 200, 255], dtype=np.uint8)  # Light blue for negative
            
            # Find the maximum absolute value for symmetric scaling around 0
            max_abs = max(abs(min_val), abs(max_val))
            
            # If max_abs is 0, everything is 0, so return white
            if max_abs == 0:
                return np.array([255, 255, 255], dtype=np.uint8)  # White
            
            # Normalize intensity to -1 to 1 range, centered at 0
            normalized = intensity / max_abs
            normalized = np.clip(normalized, -1, 1)
            
            if normalized > 0:
                # Positive values: white to red (intensity 0 = white, max positive = red)
                r = 255
                g = int(255 * (1 - normalized))
                b = int(255 * (1 - normalized))
            elif normalized < 0:
                # Negative values: white to blue (intensity 0 = white, max negative = blue)  
                r = int(255 * (1 + normalized))
                g = int(255 * (1 + normalized))
                b = 255
            else:
                # Exactly 0: white
                r, g, b = 255, 255, 255
            
            return np.array([r, g, b], dtype=np.uint8)
        
        # Create lookup table for RGB colors
        max_label = max(max(labeled_array.flat), max(node_to_community_intensity.keys()) if node_to_community_intensity else 0)
        color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)  # Default to black (0,0,0)
        
        # Fill lookup table with RGB colors based on community intensity
        for node_id, intensity in node_to_community_intensity.items():
            rgb_color = intensity_to_rgb(intensity, min_intensity, max_intensity)
            color_lut[int(node_id)] = rgb_color
        
        # Apply lookup table to labeled array - single vectorized operation
        if is_3d:
            # Return full 3D RGB array [Z, Y, X, 3]
            heatmap_array = color_lut[labeled_array]
        else:
            # Return 2D RGB array
            if labeled_array.ndim == 3:
                # Take middle slice for 2D representation
                middle_slice = labeled_array.shape[0] // 2
                heatmap_array = color_lut[labeled_array[middle_slice]]
            else:
                # Already 2D
                heatmap_array = color_lut[labeled_array]
        
        return heatmap_array
    
    else:
        # Create matplotlib plot
        fig = plt.figure(figsize=figsize)
        
        if is_3d:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract coordinates (assuming [Z, Y, X] format)
            z_coords = positions[:, 0]
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title}')
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[0])
            
        else:
            # 2D plot (using Y, X coordinates, ignoring Z/first dimension)
            ax = fig.add_subplot(111)
            
            # Extract Y, X coordinates
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'{title}')
            ax.grid(True, alpha=0.3)
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            
            # Set origin to top-left (invert Y-axis)
            ax.invert_yaxis()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label)
        
        # Add text annotations for min/max values
        cbar.ax.text(1.05, 0, f'Min: {min_intensity:.3f}\n(Blue)', 
                    transform=cbar.ax.transAxes, va='bottom')
        cbar.ax.text(1.05, 1, f'Max: {max_intensity:.3f}\n(Red)', 
                    transform=cbar.ax.transAxes, va='top')
        
        plt.tight_layout()
        plt.show()


def create_node_heatmap(node_intensity, node_centroids, shape=None, is_3d=True, 
                        labeled_array=None, figsize=(12, 8), point_size=50, alpha=0.7, 
                        colorbar_label="Node Intensity", title="Node Clustering Intensity Heatmap"):
    """
    Create a 2D or 3D heatmap showing nodes colored by their individual intensities.
    Can return either matplotlib plot or numpy array for overlay purposes.
    
    Parameters:
    -----------
    node_intensity : dict
        Dictionary mapping node IDs to intensity values
        Keys can be np.int64 or regular ints
        
    node_centroids : dict
        Dictionary mapping node IDs to centroids
        Centroids should be [Z, Y, X] for 3D or [1, Y, X] for pseudo-3D
        
    shape : tuple, optional
        Shape of the output array in [Z, Y, X] format
        If None, will be inferred from node_centroids
        
    is_3d : bool, default=True
        If True, create 3D plot/array. If False, create 2D plot/array.
        
    labeled_array : np.ndarray, optional
        If provided, returns numpy array overlay using this labeled array template
        instead of matplotlib plot. Uses lookup table approach for efficiency.
        
    figsize : tuple, default=(12, 8)
        Figure size (width, height) - only used for matplotlib
        
    point_size : int, default=50
        Size of scatter plot points - only used for matplotlib
        
    alpha : float, default=0.7
        Transparency of points (0-1) - only used for matplotlib
        
    colorbar_label : str, default="Node Intensity"
        Label for the colorbar - only used for matplotlib
        
    Returns:
    --------
    If labeled_array is None: fig, ax (matplotlib figure and axis objects)
    If labeled_array is provided: np.ndarray (heatmap array with intensity values)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert numpy int64 keys to regular ints for consistency
    node_intensity_clean = {}
    for k, v in node_intensity.items():
        if hasattr(k, 'item'):  # numpy scalar
            node_intensity_clean[k.item()] = v
        else:
            node_intensity_clean[k] = v
    
    # Prepare data for plotting/array creation
    node_positions = []
    node_intensities = []
    
    for node_id, centroid in node_centroids.items():
        try:
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
                
            # Get intensity for this node
            intensity = node_intensity_clean[node_id]
            
            node_positions.append(centroid)
            node_intensities.append(intensity)
        except KeyError:
            # Skip nodes that don't have intensity values
            pass
    
    # Convert to numpy arrays
    positions = np.array(node_positions)
    intensities = np.array(node_intensities)
    
    # Determine shape if not provided
    if shape is None:
        if len(positions) > 0:
            max_coords = np.max(positions, axis=0).astype(int)
            shape = tuple(max_coords + 1)
        else:
            shape = (100, 100, 100) if is_3d else (1, 100, 100)
    
    # Determine min and max intensities for scaling
    if len(intensities) > 0:
        min_intensity = np.min(intensities)
        max_intensity = np.max(intensities)
    else:
        min_intensity, max_intensity = 0, 1
    
    if labeled_array is not None:
        # Create numpy RGB array output using labeled array and lookup table approach
        
        # Create mapping from node ID to intensity value (keep original float values)
        node_to_intensity = {}
        for node_id, centroid in node_centroids.items():
            # Convert node_id to regular int if it's numpy
            if hasattr(node_id, 'item'):
                node_id = node_id.item()
            
            # Only include nodes that have intensity values
            if node_id in node_intensity_clean:
                node_to_intensity[node_id] = node_intensity_clean[node_id]
        
        # Create colormap function (RdBu_r - red for high, blue for low, yellow/white for middle)
        def intensity_to_rgba(intensity, min_val, max_val):
            """Convert intensity value to RGBA using RdBu_r colormap logic, centered at 0"""
            
            # Handle edge case where all values are the same
            if max_val == min_val:
                if intensity == 0:
                    return np.array([255, 255, 255, 0], dtype=np.uint8)  # Transparent white for 0
                elif intensity > 0:
                    return np.array([255, 200, 200, 255], dtype=np.uint8)  # Opaque light red for positive
                else:
                    return np.array([200, 200, 255, 255], dtype=np.uint8)  # Opaque light blue for negative
            
            # Find the maximum absolute value for symmetric scaling around 0
            max_abs = max(abs(min_val), abs(max_val))
            
            # If max_abs is 0, everything is 0, so return transparent
            if max_abs == 0:
                return np.array([255, 255, 255, 0], dtype=np.uint8)  # Transparent white
            
            # Normalize intensity to -1 to 1 range, centered at 0
            normalized = intensity / max_abs
            normalized = np.clip(normalized, -1, 1)
            
            if normalized > 0:
                # Positive values: white to red (intensity 0 = transparent, max positive = red)
                r = 255
                g = int(255 * (1 - normalized))
                b = int(255 * (1 - normalized))
                alpha = 255  # Fully opaque for all non-zero values
            elif normalized < 0:
                # Negative values: white to blue (intensity 0 = transparent, max negative = blue)  
                r = int(255 * (1 + normalized))
                g = int(255 * (1 + normalized))
                b = 255
                alpha = 255  # Fully opaque for all non-zero values
            else:
                # Exactly 0: transparent
                r, g, b, alpha = 255, 255, 255, 0
            
            return np.array([r, g, b, alpha], dtype=np.uint8)

        # Modified usage in your main function:
        # Create lookup table for RGBA colors (note the 4 channels now)
        max_label = max(max(labeled_array.flat), max(node_to_intensity.keys()) if node_to_intensity else 0)
        color_lut = np.zeros((max_label + 1, 4), dtype=np.uint8)  # Default to transparent (0,0,0,0)

        # Fill lookup table with RGBA colors based on intensity
        for node_id, intensity in node_to_intensity.items():
            rgba_color = intensity_to_rgba(intensity, min_intensity, max_intensity)
            color_lut[int(node_id)] = rgba_color

        # Apply lookup table to labeled array - single vectorized operation
        if is_3d:
            # Return full 3D RGBA array [Z, Y, X, 4]
            heatmap_array = color_lut[labeled_array]
        else:
            # Return 2D RGBA array
            if labeled_array.ndim == 3:
                # Take middle slice for 2D representation
                middle_slice = labeled_array.shape[0] // 2
                heatmap_array = color_lut[labeled_array[middle_slice]]
            else:
                # Already 2D
                heatmap_array = color_lut[labeled_array]

        return heatmap_array
    
    else:
        # Create matplotlib plot
        fig = plt.figure(figsize=figsize)
        
        if is_3d:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract coordinates (assuming [Z, Y, X] format)
            z_coords = positions[:, 0]
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title}')
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[0])
            
        else:
            # 2D plot (using Y, X coordinates, ignoring Z/first dimension)
            ax = fig.add_subplot(111)
            
            # Extract Y, X coordinates
            y_coords = positions[:, 1]
            x_coords = positions[:, 2]
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, 
                               c=intensities, s=point_size, alpha=alpha,
                               cmap='RdBu_r', vmin=min_intensity, vmax=max_intensity)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'{title}')
            ax.grid(True, alpha=0.3)
            
            # Set axis limits based on shape
            ax.set_xlim(0, shape[2])
            ax.set_ylim(0, shape[1])
            
            # Set origin to top-left (invert Y-axis)
            ax.invert_yaxis()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label)
        
        # Add text annotations for min/max values
        cbar.ax.text(1.05, 0, f'Min: {min_intensity:.3f}\n(Blue)', 
                    transform=cbar.ax.transAxes, va='bottom')
        cbar.ax.text(1.05, 1, f'Max: {max_intensity:.3f}\n(Red)', 
                    transform=cbar.ax.transAxes, va='top')
        
        plt.tight_layout()
        plt.show()

def create_violin_plots(data_dict, graph_title="Violin Plots", idens=None, valid_idens = None):
    """
    Create violin plots from dictionary data with distinct colors and IQR lines.
    
    Parameters:
    data_dict (dict): Dictionary where keys are column headers (strings) and 
                     values are lists of floats
    graph_title (str): Title for the overall plot
    idens (list, optional): Full list of identity/community labels used across
                           the application. If provided, colours are assigned
                           from the same palette so they stay consistent with
                           the network / UMAP graphs even when only a subset
                           of labels appears in data_dict.
    valid_idens (list, optional): If provided, will not show the included identities
    """

    if not data_dict:
        print("No data to plot")
        return
    
    # Prepare data
    data_dict = dict(sorted(data_dict.items()))

    if valid_idens:
        new_labels = {}
        for label, vals in data_dict.items():
            if label in valid_idens:
                new_labels[label] = vals
        data_dict = new_labels

    labels = list(data_dict.keys())
    data_lists = list(data_dict.values())
    
    # --- colour schema matching UMAPGraphWidget / NetworkGraphWidget ---
    final_colors = _generate_graph_consistent_colors(labels, idens)
    
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    
    # Create violin plots
    violin_parts = ax.violinplot(
        data_lists, positions=range(len(labels)),
        showmeans=False, showmedians=True, showextrema=True
    )
    
    # Color violins
    for i, pc in enumerate(violin_parts['bodies']):
        if i < len(final_colors):
            pc.set_facecolor(final_colors[i])
            pc.set_alpha(0.7)
    
    # Color other violin parts
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in violin_parts:
            violin_parts[partname].set_edgecolor('black')
            violin_parts[partname].set_linewidth(1)
    
    # Set y-limits using percentiles to reduce extreme outlier influence
    all_data = [val for sublist in data_lists for val in sublist]
    if all_data:
        y_min = np.percentile(all_data, 5)
        y_max = np.percentile(all_data, 95)
        y_range = y_max - y_min
        y_padding = y_range * 0.15
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add IQR and median text annotations and dotted IQR lines
    for i, data in enumerate(data_lists):
        if len(data) > 0:
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1

            # Add dotted green lines for IQR
            ax.hlines(
                [q1, q3],
                i - 0.25, i + 0.25,
                colors='green',
                linestyles='dotted',
                linewidth=1.5,
                zorder=3,
                label='IQR (25th–75th)' if i == 0 else None
            )
            
            # Text annotation below the violins
            y_min_current = ax.get_ylim()[0]
            y_text = y_min_current - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15
            ax.text(
                i, y_text, f'Median: {median:.2f}\nIQR: {iqr:.2f}', 
                ha='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
    
    # Customize appearance
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(graph_title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Values (Z-score-like)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add baseline
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Identity Basepoint')
    ax.legend(loc='upper right')
    
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.show()

    # --- Outlier Detection ---
    outliers_info = []
    non_outlier_data = []

    for i, data in enumerate(data_lists):
        if len(data) > 0:
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [val for val in data if val < lower_bound or val > upper_bound]
            non_outliers = [val for val in data if lower_bound <= val <= upper_bound]

            outliers_info.append({
                'label': labels[i],
                'outliers': outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'total_count': len(data)
            })
            non_outlier_data.append(non_outliers)
        else:
            outliers_info.append({
                'label': labels[i],
                'outliers': [],
                'lower_bound': None,
                'upper_bound': None,
                'total_count': 0
            })
            non_outlier_data.append([])

    print("\n" + "="*60)
    print("OUTLIER DETECTION SUMMARY")
    print("="*60)
    total_outliers = 0
    for info in outliers_info:
        n_outliers = len(info['outliers'])
        total_outliers += n_outliers
        if n_outliers > 0:
            print(f"{info['label']}: {n_outliers} outliers out of {info['total_count']} points "
                  f"({n_outliers/info['total_count']*100:.1f}%)")
            print(f" Outlier Removed Range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
    if total_outliers == 0:
        print("No outliers detected in any dataset.")
    else:
        print(f"\nTotal outliers across all datasets: {total_outliers}")
    print("="*60 + "\n")


def _generate_graph_consistent_colors(labels, idens=None):
    """
    Generate colours for `labels` using the same deterministic scheme as
    UMAPGraphWidget._generate_community_colors / NetworkGraphWidget:
      1. Gather all unique labels (from `idens` if given, else from `labels`).
      2. Sort them.
      3. Deterministically shuffle with Random(42).
      4. Assign evenly-spaced HSV hues in that shuffled order.
      5. Override label 0 → brown (#8B4513).
      6. Return a list of hex colours parallel to `labels`.

    Parameters
    ----------
    labels : list
        The labels that actually appear in the violin plot (in plot order).
    idens : list, optional
        The FULL set of labels used across the app.  Passing this ensures
        a label gets the same colour here as it does on the graph, even if
        only a subset is plotted.

    Returns
    -------
    list of str
        Hex colour strings, one per entry in `labels`.
    """
    import colorsys
    import random as _random

    # Build the global label universe
    if idens is not None:
        all_labels = set(idens)
    else:
        all_labels = set()
    all_labels.update(labels)

    try:
        unique_sorted = sorted(all_labels)
    except TypeError:
        unique_sorted = sorted(all_labels, key=str)

    n = len(unique_sorted)
    if n == 0:
        return ['#808080'] * len(labels)

    # Deterministic shuffle — same seed as the graph widgets
    shuffled = _random.Random(42).sample(unique_sorted, n)

    # HSV hues evenly spaced
    hex_palette = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        hex_palette.append('#{:02x}{:02x}{:02x}'.format(
            int(r * 255), int(g * 255), int(b * 255)))

    color_map = {lbl: hex_palette[i] for i, lbl in enumerate(shuffled)}

    # Label 0 → brown, consistent with graph widgets
    if 0 in color_map:
        color_map[0] = '#8B4513'

    # Map back to the plot-order labels
    return [color_map.get(lbl, '#808080') for lbl in labels]


def create_neighbor_heatmap(distance_dict, identities, dpi=300, title="Nearest Neighbor Distance Matrix", subtitle="Average spatial distances between node populations", y_label="Distance", color_swap=False, max_label_length=20):
    """
    Create a professional heatmap from nearest neighbor distance data.

    Parameters
    ----------
    distance_dict : dict
        Keys are root identity names, values are lists of distances
        where each index corresponds to the identity at that index
        in the identities list. Values may contain None.
    identities : list of str
        Ordered list of identity names corresponding to the list
        indices in distance_dict values.
    dpi : int
        Resolution of the output image.
    color_swap : bool
        If True, uses blue-to-red colormap with log1p normalization
        to better handle left-skewed distributions.
    max_label_length : int or None
        Maximum displayed length for axis tick labels. Longer labels are
        cropped with a trailing ellipsis. Set to None to disable truncation.
    """

    def _truncate(label, max_len):
        s = str(label)
        if max_len is None or len(s) <= max_len:
            return s
        return s[:max_len - 1] + '…'

    class Log1pNorm(mcolors.Normalize):
        """Normalize using log1p to handle zero values and left-skewed distributions."""
        def __call__(self, value, clip=None):
            arr = np.ma.array(value, dtype=float)
            vmin = np.log1p(self.vmin)
            vmax = np.log1p(self.vmax)
            result = np.ma.array(np.log1p(np.ma.filled(arr, 0)), mask=np.ma.getmask(arr))
            return np.ma.array((result - vmin) / (vmax - vmin))

        def inverse(self, value):
            vmin = np.log1p(self.vmin)
            vmax = np.log1p(self.vmax)
            return np.expm1(value * (vmax - vmin) + vmin)

    def natural_sort_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    # Build a sorted index order for display
    sorted_indices = sorted(range(len(identities)), key=lambda i: natural_sort_key(identities[i]))
    sorted_identities = [identities[i] for i in sorted_indices]
    n = len(identities)

    # Build matrix in original order, treating None as NaN
    matrix = np.full((n, n), np.nan)
    for root, distances in distance_dict.items():
        i = identities.index(root)
        for j, val in enumerate(distances):
            if val is not None:
                matrix[i, j] = float(val)

    # Reorder rows and columns by natural sort
    matrix = matrix[np.ix_(sorted_indices, sorted_indices)]

    # Create figure with specific styling
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')

    if not color_swap:
        colors = ['#d32f2f', '#e57373', '#ff9800', '#ffc107', '#64b5f6', '#2196f3', '#1565c0']
        norm = None
    else:
        colors = ['#1565c0', '#2196f3', '#64b5f6', '#ffc107', '#ff9800', '#e57373', '#d32f2f']
        valid_vals = matrix[~np.isnan(matrix)]
        norm = Log1pNorm(vmin=np.min(valid_vals), vmax=np.max(valid_vals)) if valid_vals.size > 0 else None

    cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
    cmap.set_bad(color='#1a1a1a')

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    # Set ticks and labels using sorted identities (truncated for display)
    display_labels = [_truncate(lbl, max_label_length) for lbl in sorted_identities]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(display_labels, color='#cccccc', fontsize=11, fontweight='500')
    ax.set_yticklabels(display_labels, color='#cccccc', fontsize=11, fontweight='500')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    ax.set_xlabel('Neighbor Node Type (searching to)',
                  fontsize=13, fontweight='600', color='#64c8ff', labelpad=15)
    ax.set_ylabel('Root Node Type (searching from)',
                  fontsize=13, fontweight='600', color='#64c8ff', labelpad=15)

    ax.text(0.5, 1.08, title,
            transform=ax.transAxes, fontsize=18, fontweight='700',
            color='#f5f5f5', ha='center')
    ax.text(0.5, 1.03, subtitle,
            transform=ax.transAxes, fontsize=11,
            color='#888888', ha='center')

    # Add cell values if matrix isn't too large (always show raw values)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.1f}',
                            ha='center', va='center', color='#1a1a1a',
                            fontsize=12, fontweight='700', fontfamily='monospace')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(y_label, rotation=270, labelpad=20,
                   fontsize=11, color='#aaaaaa', fontweight='500')
    cbar.ax.yaxis.set_tick_params(color='#666666', labelcolor='#aaaaaa')
    cbar.outline.set_edgecolor('#333333')
    cbar.ax.set_facecolor('#1a1a1a')

    # Grid styling
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which='minor', color='#333333', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)
    ax.tick_params(which='major', size=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def natural_sort_key(s):

    try:

        for i, iden in enumerate(s):
            if iden.endswith('+') or iden.endswith('-'):
                s[i] = iden[:-1]

        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
    except:
        return sorted(s)