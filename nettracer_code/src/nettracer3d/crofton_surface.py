"""
Surface area estimation for labeled 3D images via the discretized Crofton formula.

Estimates the surface area of each label in a 3D labeled array by counting
label transitions along 13 lattice directions and combining them with
solid-angle weights:

    S_label = sum_d  2 * lambda_d * T_d(label) * V_voxel / |v_d|

where for each direction d:
    v_d      = d * spacing            direction vector in physical space
    lambda_d = normalized solid angle of v_d (fraction of the direction sphere)
    T_d      = number of transitions into/out of the label along d
    V_voxel  = product of spacing

This is the same estimator used by ITK's LabelShapeStatisticsImageFilter
(GetPerimeter) and by MorphoLibJ's "Crofton, 13 directions" option; results
agree to floating-point precision.

Accuracy on analytically known shapes (isotropic spacing):
    sphere  r=8    -1.3 %        naive face counting:  +47 %
    sphere  r=18   -0.1 %                              +49 %
    cylinder r=5   -3.5 to +5.9 %                      +39 to +41 %

Naive face counting carries a shape- and orientation-dependent bias of
roughly +20 % to +52 %, so it should not be used for comparisons across
objects of differing shape or orientation.

AXIS CONVENTION
    Arrays are assumed to be indexed (z, y, x), the standard for TIFF stacks
    and most bio-imaging pipelines. `spacing` is given in the same order.
    For xy_scale / z_scale style parameters, pass spacing=(z_scale, xy_scale,
    xy_scale).
"""

from functools import lru_cache

import numpy as np
from scipy.spatial import SphericalVoronoi

# One representative per antipodal pair of the 13 lattice directions:
# 3 axis-aligned, 6 face-diagonal, 4 body-diagonal.
_DIRECTIONS = (
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
)


@lru_cache(maxsize=32)
def _direction_weights(spacing):
    """Solid-angle weights and physical lengths for the 13 directions.

    Weights are the areas of the spherical Voronoi cells of the (physical,
    normalized) directions, so anisotropic spacing is handled correctly:
    the direction set is no longer symmetric on the sphere and the weights
    adjust accordingly. Cached because this is pure geometry.
    """
    sp = np.asarray(spacing, dtype=float)
    if sp.shape != (3,) or np.any(sp <= 0):
        raise ValueError("spacing must be 3 positive values in (z, y, x) order")

    vecs = np.asarray(_DIRECTIONS, dtype=float) * sp
    lengths = np.linalg.norm(vecs, axis=1)

    # SphericalVoronoi needs the full antipodal set to tile the sphere.
    pts = np.empty((2 * len(vecs), 3))
    pts[0::2] = vecs / lengths[:, None]
    pts[1::2] = -pts[0::2]

    sv = SphericalVoronoi(pts)
    sv.sort_vertices_of_regions()
    areas = np.asarray(sv.calculate_areas())

    # Each lattice direction owns its cell plus its antipode's.
    weights = (areas[0::2] + areas[1::2]) / (4.0 * np.pi)
    return weights, lengths


def _neighbor_slices(offset):
    """Slice pairs giving every adjacent voxel pair along `offset`."""
    a, b = [], []
    for o in offset:
        if o == 0:
            a.append(slice(None))
            b.append(slice(None))
        elif o > 0:
            a.append(slice(0, -o))
            b.append(slice(o, None))
        else:
            a.append(slice(-o, None))
            b.append(slice(0, o))
    return tuple(a), tuple(b)


def crofton_surface_areas(labeled, spacing=(1.0, 1.0, 1.0), as_array=False,
                          interface="all"):
    """Estimate the surface area of every label in a 3D labeled image.
 
    Parameters
    ----------
    labeled : ndarray, 3D, integer or boolean
        Label image indexed (z, y, x). Background must be 0. Boolean input is
        treated as a single label. Labels touching the volume border are
        treated as bounded there (the border acts as background), matching the
        behaviour of face-counting implementations.
    spacing : tuple of 3 floats
        Physical voxel size in (z, y, x) order. For the common isotropic-xy
        case pass ``(z_scale, xy_scale, xy_scale)``.
    as_array : bool
        If True return a float array indexed by label, of length
        ``labeled.max() + 1``. Index 0 is background and always 0.0. Note that
        an entry of 0.0 is ambiguous in this form: it means either that the
        label is absent from the image or that it is present with zero area
        (see below). Use the dict return, or ``np.unique``, if you need to
        tell those apart.
    interface : {'all', 'background'}
        Which interfaces to count. 'all' (default) counts the full boundary of
        each region, so two touching labels each receive a contribution at
        their shared interface. 'background' counts only the portion of each
        region's boundary that faces background (label 0), i.e. the surface
        exposed to the outside. For an isolated region the two agree; the
        difference is exactly the contact area with neighbouring labels.
 
        A region fully enclosed by other labels has a background area of
        exactly 0.0. That is a real measurement, not a missing one, so such a
        label still appears in the output with a value of 0.0. Only labels
        that do not occur in ``labeled`` at all are omitted.
 
    Returns
    -------
    dict[int, float] or ndarray
        Surface area per label, in squared physical units. The dict contains
        one entry for every label present in the image, including labels whose
        area is 0.0.
    """
    if interface not in ("all", "background"):
        raise ValueError("interface must be 'all' or 'background'")
 
    labeled = np.asarray(labeled)
    if labeled.ndim != 3:
        raise ValueError(f"expected a 3D array, got {labeled.ndim}D")
    if labeled.dtype == bool:
        labeled = labeled.view(np.uint8)
    if not np.issubdtype(labeled.dtype, np.integer):
        raise TypeError("labeled must be an integer or boolean array")
    if labeled.min() < 0:
        raise ValueError("labels must be non-negative")
 
    weights, lengths = _direction_weights(tuple(float(s) for s in spacing))
    voxel_volume = float(np.prod(spacing))
 
    n_labels = int(labeled.max())
    areas = np.zeros(n_labels + 1, dtype=np.float64)
    if n_labels == 0:
        return areas if as_array else {}
 
    # Which labels actually occur. This, not the computed area, decides which
    # keys appear in the output: a label may legitimately measure 0.0 (an
    # inclusion with no background contact under interface='background'), and
    # that must be reported rather than silently dropped.
    counts = np.bincount(np.ascontiguousarray(labeled).ravel(),
                         minlength=n_labels + 1)
    present = np.nonzero(counts)[0]
    present = present[present > 0]
 
    # Pad by one voxel of background so surfaces at the volume border are
    # counted, then walk each direction once.
    padded = np.pad(labeled, 1, mode="constant", constant_values=0)
 
    for offset, weight, length in zip(_DIRECTIONS, weights, lengths):
        sl_a, sl_b = _neighbor_slices(offset)
        a = padded[sl_a]
        b = padded[sl_b]
        differs = a != b
        coeff = 2.0 * weight * voxel_volume / length
        for side, other in ((a, b), (b, a)):
            if interface == "background":
                hit = (side > 0) & (other == 0)
            else:
                hit = differs & (side > 0)
            if hit.any():
                areas += np.bincount(side[hit], minlength=n_labels + 1) * coeff
 
    areas[0] = 0.0
 
    if as_array:
        return areas
    return {int(i): float(areas[i]) for i in present}