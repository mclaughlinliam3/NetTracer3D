"""
memmap_support.py
=================

Helpers for optional memory-mapped ("virtual") image channels.

Design contract
---------------
* Nothing in here changes behaviour when a channel is a normal in-RAM ndarray.
  Every entry point either short-circuits on ``isinstance(arr, np.memmap)`` or
  is only called from the virtual code paths.
* Virtual channels are opened **read-only** (``mode='r'``).  Any accidental
  in-place write raises ``ValueError: assignment destination is read-only``
  instead of silently corrupting the user's source file on disk.
* Supported operations on a virtual channel: browsing, saving, and
  nearest-neighbour (order=0) downsampling.  Everything else must refuse.
"""

import os
import shutil
import sys
import tempfile
import uuid

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


# ── Verified against tifffile 2026.3.3 ─────────────────────────────────────
#
# imagej=True is the ONLY writer option that preserves z-spacing.  XResolution
# is a standard TIFF tag so xy survives everywhere, but z spacing lives only in
# the ImageJ description string.  Plain and BigTIFF writes lose it:
#
#     imagej            spacing=2.5   xres=3.08
#     plain+resolution  spacing=None  xres=3.08
#     bigtiff           spacing=None  xres=3.08
#
# imagej=True above 4 GB does NOT raise.  tifffile emits "truncating ImageJ
# file" and writes a classic TIFF whose IFD chain holds one page, with the
# remaining planes following contiguously.  ImageJ and tifffile both
# reconstruct the full volume and the file stays memmappable; only a reader
# that walks the IFD chain is fooled.
#
# So tier 1 is NOT size-gated.  Skipping ImageJ above 4 GB would trade a
# cosmetic nonconformance for silent loss of z_scale on every large save.
#
# Plain imwrite (bigtiff=None) does auto-promote above 4 GB less 32 MB.  We
# still pass bigtiff= explicitly, since that is version dependent.
IMAGEJ_MAX_BYTES = (2 ** 32) - (2 ** 25)   # informational: truncation point
BIGTIFF_THRESHOLD = (2 ** 32) - (2 ** 25)

# Big-endian TIFFs memmap fine and every supported operation (browse, zoom
# order=0, min/max, re-save) was verified correct on them.  The catch is that
# the dtype comes back non-native, e.g. '>u2', and:
#
#     arr.dtype == np.uint16        -> False
#     arr.dtype.type is np.uint16   -> True
#
# so any `dtype == np.something` comparison silently takes the wrong branch.
# Refused by default because that failure is invisible; set True to allow it
# if you have audited your dtype comparisons (or use .dtype.type instead).
#
# NOTE: series.dtype and page.dtype report the NATIVE dtype even for a
# big-endian file.  The only signal is TiffFile.byteorder.
ALLOW_NONNATIVE_BYTEORDER = False

# Offer a virtual load when the decoded array would exceed this fraction of
# currently-available RAM.
MEMMAP_PROMPT_FRACTION = 0.5

# Used only when available-memory cannot be determined at all.
MEMMAP_PROMPT_FALLBACK_BYTES = 4 * (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Predicates
# ─────────────────────────────────────────────────────────────────────────────

def is_virtual(arr):
    """True if `arr` is memory-mapped (or a view onto a memmap)."""
    return isinstance(arr, np.memmap)


def available_memory():
    """Best-effort available RAM in bytes, or None if it can't be determined."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
    except Exception:
        pass
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        stat = _MS()
        stat.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullAvailPhys)
    except Exception:
        return None


def should_offer_memmap(nbytes):
    """Would loading `nbytes` into RAM be risky enough to warrant asking?"""
    if not nbytes:
        return False
    avail = available_memory()
    if avail is None:
        return nbytes > MEMMAP_PROMPT_FALLBACK_BYTES
    return nbytes > MEMMAP_PROMPT_FRACTION * avail


def fits_in_ram(nbytes, headroom=1.5):
    """Rough check for whether an in-RAM copy of `nbytes` is safe."""
    avail = available_memory()
    if avail is None:
        return nbytes < MEMMAP_PROMPT_FALLBACK_BYTES
    return nbytes * headroom < avail


def human_bytes(n):
    n = float(n or 0)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f"{n:.1f} {unit}" if unit != 'B' else f"{int(n)} B"
        n /= 1024


# ─────────────────────────────────────────────────────────────────────────────
#  TIFF probing (metadata only — never touches pixel data)
# ─────────────────────────────────────────────────────────────────────────────

def probe_tiff(filename):
    """
    Read shape/dtype/compression/memmappability from TIFF headers only.

    Returns a dict, or None if the file could not be probed.  A None return
    simply means "fall back to the ordinary load path".
    """
    if tifffile is None:
        return None
    try:
        with tifffile.TiffFile(filename) as tf:
            series = tf.series[0]
            shape = tuple(int(s) for s in series.shape)
            dtype = np.dtype(series.dtype)
            nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize

            page = tf.pages[0]

            # Byte order. NOTE: series.dtype and page.dtype both report the
            # NATIVE dtype even for a big-endian file -- the only signal is
            # tf.byteorder. tifffile.memmap() will hand back '>u2'. Irrelevant
            # for single-byte dtypes.
            native_order = '<' if sys.byteorder == 'little' else '>'
            native = (dtype.itemsize == 1
                      or getattr(tf, 'byteorder', native_order) == native_order)

            # Compression: enum value 1 == NONE across tifffile versions.
            compressed = True
            try:
                compressed = int(page.compression) != 1
            except Exception:
                try:
                    compressed = 'NONE' not in str(page.compression).upper()
                except Exception:
                    pass

            # is_memmappable folds together contiguity, tiling, byte order and
            # compression.  Its home has moved between tifffile versions, so
            # check the series first and fall back to the page.
            memmappable = getattr(series, 'is_memmappable', None)
            if memmappable is None:
                memmappable = getattr(page, 'is_memmappable', False)

            return {
                'ok': True,
                'shape': shape,
                'dtype': dtype,
                'nbytes': nbytes,
                'compressed': bool(compressed),
                'memmappable': bool(memmappable) and (
                    native or ALLOW_NONNATIVE_BYTEORDER),
                'native_byteorder': bool(native),
                'ndim': len(shape),
            }
    except Exception:
        return None


def minmax_from_metadata(filename):
    """
    Try to recover the intensity range without reading pixels.

    Returns (min, max) or None.  Note the ImageJ path reports the *display*
    range, which is not guaranteed to equal the data range — it is only used
    to seed contrast, never for quantification.
    """
    if tifffile is None:
        return None
    try:
        with tifffile.TiffFile(filename) as tf:
            page = tf.pages[0]

            smin = page.tags.get('SMinSampleValue')   # tag 340
            smax = page.tags.get('SMaxSampleValue')   # tag 341
            if smin is not None and smax is not None:
                a, b = smin.value, smax.value
                if isinstance(a, (tuple, list)):
                    a = min(a)
                if isinstance(b, (tuple, list)):
                    b = max(b)
                return float(a), float(b)

            if getattr(tf, 'is_imagej', False):
                md = tf.imagej_metadata or {}
                if 'min' in md and 'max' in md:
                    return float(md['min']), float(md['max'])
    except Exception:
        pass
    return None


def sampled_minmax(arr, n_slices=100):
    """
    Estimate min/max by reading ~n_slices whole planes along axis 0.

    Each plane is contiguous for a C-ordered (Z, Y, X) array, so this is
    sequential I/O and bounded RAM regardless of volume size.  May
    underestimate the true max on sparse label data; that only affects
    display contrast.
    """
    try:
        if arr.ndim >= 3 and arr.shape[0] > n_slices:
            step = max(1, arr.shape[0] // n_slices)
            sample = arr[::step]
        else:
            sample = arr
        return float(np.min(sample)), float(np.max(sample))
    except Exception:
        return 0.0, 1.0


def resolve_minmax(arr, source_path=None):
    """
    min/max for a channel.

    In-RAM arrays get an exact np.min/np.max (unchanged behaviour).
    Virtual arrays try file metadata first, then fall back to sampling —
    an exact scan would mean reading the entire file at load time.
    """
    if not is_virtual(arr):
        return float(np.min(arr)), float(np.max(arr))

    if source_path:
        mm = minmax_from_metadata(source_path)
        if mm is not None and mm[1] > mm[0]:
            return mm
    return sampled_minmax(arr)


# ─────────────────────────────────────────────────────────────────────────────
#  Acquiring a virtual array
# ─────────────────────────────────────────────────────────────────────────────

def open_tiff_memmap(filename):
    """
    Direct read-only memmap of an uncompressed, contiguous TIFF.
    Returns the array, or None if the file isn't memmappable.
    """
    if tifffile is None:
        return None
    try:
        arr = tifffile.memmap(filename, mode='r')
    except Exception:
        return None
    if arr is not None and not arr.dtype.isnative and not ALLOW_NONNATIVE_BYTEORDER:
        # See ALLOW_NONNATIVE_BYTEORDER above. Fall back to an eager read,
        # which byteswaps into native order.
        release_memmap(arr)
        return None
    return arr


def decompress_tiff_to_scratch(filename, info, scratch_dir=None,
                               progress_cb=None):
    """
    Decode a compressed TIFF page-by-page into a scratch file on disk and
    return a read-only memmap of it.  RAM stays bounded to roughly one page.

    Returns (array, scratch_path).  Raises on failure.
    """
    if tifffile is None:
        raise RuntimeError("tifffile is not available")

    scratch_dir = scratch_dir or tempfile.gettempdir()
    os.makedirs(scratch_dir, exist_ok=True)

    need = info['nbytes']
    free = shutil.disk_usage(scratch_dir).free
    if free < need * 1.05:
        raise OSError(
            f"Not enough free space in {scratch_dir}: need "
            f"{human_bytes(need)}, have {human_bytes(free)}.")

    path = os.path.join(scratch_dir, f"n3d_scratch_{uuid.uuid4().hex}.dat")

    if progress_cb:
        progress_cb(0, 1)
    # `out=<path>` makes tifffile stream the decode into a disk-backed array.
    tifffile.imread(filename, out=path)
    if progress_cb:
        progress_cb(1, 1)

    # Reopen read-only so stray writes can't quietly mutate the working copy.
    arr = np.memmap(path, dtype=info['dtype'], mode='r', shape=info['shape'])
    return arr, path


def release_memmap(arr):
    """
    Drop a memmap's underlying file handle.

    Windows refuses to delete or overwrite a mapped file, so this must be
    called before any code re-touches the source path (reload, delete, save).
    """
    if not is_virtual(arr):
        return
    try:
        base = arr
        while isinstance(getattr(base, 'base', None), np.memmap):
            base = base.base
        mm = getattr(base, '_mmap', None)
        if mm is not None:
            mm.close()
    except Exception:
        pass


def cleanup_scratch(paths):
    """Delete scratch files created by decompress_tiff_to_scratch."""
    for p in list(paths or []):
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Writing
# ─────────────────────────────────────────────────────────────────────────────

def _plane_iter(array, progress_cb=None):
    """Yield planes one at a time, reporting progress as it goes."""
    n = array.shape[0]
    for z in range(n):
        yield np.asarray(array[z])
        if progress_cb and (z % 16 == 0 or z == n - 1):
            progress_cb(z + 1, n)


def _write_once(array, path, imagej, compression, z_scale, xy_scale,
                stream, progress_cb):
    """
    One write attempt.  `stream` uses tifffile's iterator API, which holds a
    single plane at a time -- required for a memmap, and it also gives the
    progress callback something to hook.
    """
    resolution_value = 1.0 / xy_scale if xy_scale else 1.0
    kwargs = {
        'photometric': 'minisblack',
        'resolution': (resolution_value, resolution_value),
    }
    if compression:
        kwargs['compression'] = compression
    if imagej:
        # The metadata that only this format preserves.
        kwargs['metadata'] = {
            'spacing': z_scale,
            'slices': int(array.shape[0]),
            'channels': 1,
            'axes': 'ZYX',
        }

    if not stream:
        extra = {'imagej': True} if imagej else {
            'bigtiff': int(array.nbytes) > BIGTIFF_THRESHOLD}
        tifffile.imwrite(path, array, **extra, **kwargs)
        if progress_cb:
            progress_cb(1, 1)
        return

    opts = {'imagej': True} if imagej else {
        'bigtiff': int(array.nbytes) > BIGTIFF_THRESHOLD}
    with tifffile.TiffWriter(path, **opts) as tw:
        tw.write(_plane_iter(array, progress_cb),
                 shape=tuple(array.shape), dtype=array.dtype, **kwargs)


def write_tiff_streaming(array, path, z_scale=1.0, xy_scale=1.0,
                         compression=None, progress_cb=None):
    """
    Write plane-by-plane so peak RAM stays at one slice regardless of volume
    size.  Prefers the ImageJ container, because it is the only one that
    carries z-spacing; falls back to BigTIFF if ImageJ rejects the data.
    """
    if tifffile is None:
        raise RuntimeError("tifffile is not available")

    arr = array if array.ndim >= 3 else array[np.newaxis]

    if arr.ndim == 3:
        try:
            _write_once(arr, path, True, compression, z_scale, xy_scale,
                        True, progress_cb)
            return
        except Exception:
            pass
    _write_once(arr, path, False, compression, z_scale, xy_scale,
                True, progress_cb)


def write_image(array, path, z_scale=1.0, xy_scale=1.0, compression=None,
                binarize_fn=None, progress_cb=None):
    """
    Unified TIFF writer replacing the duplicated cascades in save_nodes /
    save_edges / save_network_overlay / save_id_overlay.

    Tier 1 : ImageJ hyperstack, 3D only.  NOT size-gated -- see the note at the
             top of this module.  This is the only tier that preserves z_scale.
    Tier 2 : plain TIFF, BigTIFF flag set explicitly from size.
    Tier 3 : binarize and retry (in-RAM only).

    Same cascade as the original code.  The differences: bool is cast up front
    rather than relying on tier 3, a virtual array streams plane-by-plane
    instead of being handed to imwrite whole, and tier 3 is unreachable for a
    virtual array (it mutates its input).
    """
    if array is None:
        return False

    if str(path).endswith('compressed.tif'):   # sneaky compression option
        compression = compression or 'zlib'

    virtual = is_virtual(array)

    # bool -> uint8 up front.  This is what the old binarize tier was actually
    # compensating for, and doing it here makes tier 3 nearly unreachable.
    if array.dtype == np.bool_:
        if virtual:
            raise TypeError("Cannot cast a virtual boolean stack; "
                            "load it into RAM first.")
        array = array.astype(np.uint8)

    nbytes = int(array.nbytes)
    stream = virtual or progress_cb is not None

    if nbytes > IMAGEJ_MAX_BYTES and array.ndim == 3:
        print(f"Note: {nbytes / 2**30:.1f} GiB exceeds the 4 GB ImageJ limit. "
              "Writing a truncated-IFD ImageJ file to preserve z_scale; "
              "ImageJ and tifffile read it correctly, other readers may see "
              "only the first plane.")

    # Tier 1 — ImageJ hyperstack.
    if array.ndim == 3:
        try:
            _write_once(array, path, True, compression, z_scale, xy_scale,
                        stream, progress_cb)
            return True
        except Exception:
            pass

    # Tier 2 — plain TIFF / BigTIFF.  Loses z_scale; xy_scale survives via
    # the standard XResolution tag.
    try:
        _write_once(array, path, False, compression, z_scale, xy_scale,
                    stream, progress_cb)
        return True
    except Exception:
        pass

    # Tier 3 — last resort.  Mutating, so never reached for a virtual array
    # (which would have succeeded above or raised).
    if binarize_fn is None or virtual:
        raise
    reduced = binarize_fn(array)
    tifffile.imwrite(path, reduced, compression=compression,
                     bigtiff=(int(reduced.nbytes) > BIGTIFF_THRESHOLD))
    return True


def same_file(path_a, path_b):
    """True if two paths point at the same file (guards save-over-source)."""
    if not path_a or not path_b:
        return False
    try:
        if os.path.exists(path_a) and os.path.exists(path_b):
            return os.path.samefile(path_a, path_b)
    except Exception:
        pass
    try:
        return os.path.abspath(path_a) == os.path.abspath(path_b)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-channel splitting (used by MultiChanDialog)
# ─────────────────────────────────────────────────────────────────────────────

def multichan_channel_view(data, axis, index):
    """
    Lazy view of one channel.  On a memmap this materializes nothing — the
    caller decides when (and whether) to pay.
    """
    slicer = [slice(None)] * data.ndim
    slicer[axis] = index
    return data[tuple(slicer)]


def multichan_channel_nbytes(data, axis):
    """Decoded size of a single channel from `data`."""
    per = int(np.prod([s for i, s in enumerate(data.shape) if i != axis],
                      dtype=np.int64))
    return per * data.dtype.itemsize


def multichan_contiguous_run(data, axis):
    """
    Bytes of contiguous memory produced by slicing a single index off `axis`.

    This is what determines whether extracting one channel is cheap.  The
    extracted view's innermost run is everything to the right of `axis`:

        (Z, C, Y, X), axis=1  -> run = Y*X*itemsize   (a whole plane; fine)
        (C, Z, Y, X), axis=0  -> run = Z*Y*X*itemsize (fully contiguous; ideal)
        (Z, Y, X, C), axis=3  -> run = 1*itemsize     (one element; disastrous)

    Note the *file* is still read with a stride between runs, so total I/O is
    roughly the whole file either way — but plane-sized runs let readahead do
    its job, while element-sized runs mean a read-modify-write per page.
    """
    trailing = int(np.prod(data.shape[axis + 1:], dtype=np.int64)) if axis + 1 < data.ndim else 1
    return trailing * data.dtype.itemsize


def multichan_extraction_is_cheap(data, axis, min_run=65536):
    """True if per-channel extraction reads in usefully large contiguous runs."""
    return multichan_contiguous_run(data, axis) >= min_run


def materialize_multichan_channel(data, axis, index, force=False):
    """
    Copy one channel into RAM.

    Raises MemoryError up front rather than letting the OOM killer decide.
    This is the operation that makes splitting a virtual multiplexed stack
    useful: the whole stack doesn't fit, but one channel usually does.
    """
    need = multichan_channel_nbytes(data, axis)
    if not force and not fits_in_ram(need):
        raise MemoryError(
            f"Channel needs {human_bytes(need)}; "
            f"available RAM is {human_bytes(available_memory() or 0)}.")
    return np.array(multichan_channel_view(data, axis, index), copy=True)


def save_multichan_channel(data, axis, index, path, z_scale=1.0, xy_scale=1.0,
                           compression=None, progress_cb=None):
    """
    Write one channel straight to disk without ever holding it in RAM.

    Lets a user split a virtual multiplexed stack to a directory even when a
    single channel is itself too large to materialize.
    """
    view = multichan_channel_view(data, axis, index)
    if is_virtual(data):
        write_tiff_streaming(view, path, z_scale, xy_scale, compression,
                             progress_cb)
    else:
        write_image(view, path, z_scale, xy_scale, compression)
    return True
