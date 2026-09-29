"""
Virtual Loader — NetTracer3D plugin
===================================

Inspect and process TIFFs that are too large to open normally.

Scope is deliberately narrow.  When a file will not fit in memory, there are
only two things worth doing to it before real work starts:

  1. Split it into per-channel files, so each channel can be opened normally.
  2. Downsample it, so the whole thing can be opened normally.

Both are streaming operations.  Neither ever holds the full volume.

HOW THE FILE IS OPENED
----------------------
"Always memmap" is not achievable — tiled, compressed, big-endian and
non-contiguous-IFD files all refuse `tifffile.memmap`, and a QPTIFF is
typically all four at once.  So there are two access modes:

  MEMMAP  np.memmap over the file.  Zero cost, random access, but only works
          for uncompressed contiguous native-order data.
  PAGES   one `page.asarray()` at a time.  Works on EVERY tiff regardless of
          encoding, needs no scratch space, holds one plane in RAM.

For this tool's operations — extract channel N (a subset of pages) and
downsample (planes in order) — PAGES is not a fallback so much as the
general case.  MEMMAP is the fast path when it happens to be available.

PYRAMIDS
--------
QPTIFF and other whole-slide formats store several images in one file: the
full-resolution set, thumbnails, label/macro images, and reduced-resolution
pyramid levels.  Each is a separate tifffile *series*, and pyramidal series
additionally expose *levels*.  Trying to read such a file as one array is
what makes it fail to open.  Here the user picks series and level explicitly.

CHANNEL AXIS
------------
Auto-detected where possible, but always overridable.  Detection is genuinely
ambiguous: a separate-IFD multichannel file reports axes 'IYX', and so does a
plain Z-stack written without ImageJ metadata.  'I' means "tifffile could not
name this axis", not "channels".  So the guess is a starting point and the
user can set any axis, or none.

Install: drop this file in  ~/.nettracer3d/plugins/
"""

import math
import os
import re
import sys
import traceback

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QMessageBox, QProgressBar,
    QTextEdit, QLineEdit, QWidget, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread


PLUGIN_INFO = {
    "name": "Virtual Loader",
    "version": "1.0.0",
    "author": "NetTracer3D Contributors",
    "description": (
        "Inspect TIFFs too large to open normally. Reads any TIFF without "
        "loading it into memory, including multi-series and pyramidal files "
        "such as QPTIFF. Supports splitting channels into separate files and "
        "writing a downsampled copy, both streamed."
    ),
    "api_version": (1, 0),
    "requires": [],
    "category": "io",
}

_api = None

TIFF_FILTER = ("TIFF Files (*.tif *.tiff *.qptiff *.svs *.ome.tif "
               "*.ome.tiff *.btf *.ndpi);;All Files (*)")


# ══════════════════════════════════════════════════════════════════════
#  register / unregister
# ══════════════════════════════════════════════════════════════════════

def register(api):
    global _api
    _api = api
    api.register_menu_action(
        "Extensions/Virtual Loader/Open Virtual Loader...",
        _open_dialog,
        tooltip="Inspect, split, or downsample a TIFF without loading it",
    )
    api.register_menu_action(
        "Extensions/Virtual Loader/Crop Current Image...",
        _open_crop_menu,
        tooltip="Set crop bounds by hand, or by dragging on the viewer",
    )
    api.print("Virtual Loader plugin loaded.")


def unregister(api):
    api.print("Virtual Loader plugin unloaded.")


_open_instances = []


def _open_dialog():
    win = _api.get_unsafe_window()
    dlg = VirtualLoaderDialog(_api, parent=win)
    _open_instances.append(dlg)
    dlg.show()
    dlg.raise_()
    return dlg


def _open_crop_menu():
    """Crop the image already open in the most recent Virtual Loader."""
    for dlg in reversed(_open_instances):
        try:
            if dlg.isVisible() and dlg.vt is not None:
                dlg.raise_()
                dlg._open_crop()
                return
        except RuntimeError:          # C++ side already deleted
            continue
    _api.show_message(
        "No Image Open",
        "Open a TIFF in the Virtual Loader and pick a series first — crop "
        "bounds apply to that image.", level="info")


# ══════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════

def human_bytes(n):
    n = float(n or 0)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f"{int(n)} B" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1024


def available_memory():
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

        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullAvailPhys)
    except Exception:
        return None


def fits_in_ram(nbytes, headroom=1.6):
    avail = available_memory()
    if avail is None:
        return nbytes < 2 * 1024 ** 3
    return nbytes * headroom < avail


# ══════════════════════════════════════════════════════════════════════
#  VirtualTiff — the reader
# ══════════════════════════════════════════════════════════════════════

class VirtualTiff:
    """
    Lazy view onto one series (and one pyramid level) of a TIFF.

    Exposes shape / dtype / axes plus two access primitives:

        plane(flat_index)         -> one 2D (or 2D+samples) array
        iter_planes(indices)      -> generator over a subset of planes

    Everything above is built from those.  Nothing here materializes more
    than a single plane unless you explicitly ask for `as_array()`.
    """

    def __init__(self, path, series_index=0, level=0):
        import tifffile
        self.path = path
        self.tf = tifffile.TiffFile(path)
        self.series_index = series_index
        self.level = level

        base = self.tf.series[series_index]
        self.series = base.levels[level] if level else base

        self.shape = tuple(int(s) for s in self.series.shape)
        self.dtype = np.dtype(self.series.dtype)
        self.axes = str(self.series.axes)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

        page0 = self.series.pages[0]
        self.samples = int(getattr(page0, 'samplesperpixel', 1) or 1)

        # Leading axes enumerate pages; trailing 2 (or 3 with samples) are the
        # plane itself.
        planar_ndim = 3 if self.samples > 1 else 2
        planar_ndim = min(planar_ndim, len(self.shape))
        self.leading_shape = self.shape[:len(self.shape) - planar_ndim]
        self.plane_shape = self.shape[len(self.shape) - planar_ndim:]
        self.n_pages = int(np.prod(self.leading_shape)) if self.leading_shape else 1

        # ── Access mode ──
        self.memmap = None
        self.mode = 'PAGES'
        if series_index == 0 and level == 0:
            try:
                mm = tifffile.memmap(path, mode='r')
                if tuple(mm.shape) == self.shape and mm.dtype.isnative:
                    self.memmap = mm
                    self.mode = 'MEMMAP'
                else:
                    del mm
            except Exception:
                pass

        self.byteorder = getattr(self.tf, 'byteorder', '<')
        self.compression = str(getattr(page0, 'compression', '?'))
        self.is_tiled = bool(getattr(page0, 'is_tiled', False))
        self.description = page0.description or ''

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self):
        if self.memmap is not None:
            try:
                base = self.memmap
                while isinstance(getattr(base, 'base', None), np.memmap):
                    base = base.base
                if getattr(base, '_mmap', None) is not None:
                    base._mmap.close()
            except Exception:
                pass
            self.memmap = None
        try:
            self.tf.close()
        except Exception:
            pass

    # ── plane access ──────────────────────────────────────────────────

    def plane(self, flat_index):
        """One plane by flat page index. The only primitive that reads data."""
        if self.memmap is not None:
            idx = np.unravel_index(flat_index, self.leading_shape) \
                if self.leading_shape else ()
            return np.asarray(self.memmap[idx])
        return self.series.pages[flat_index].asarray()

    def flat_index(self, leading_idx):
        """Multi-index over the leading axes -> flat page index."""
        if not self.leading_shape:
            return 0
        return int(np.ravel_multi_index(tuple(leading_idx), self.leading_shape))

    def plane_indices(self, axis=None, value=None):
        """
        Flat page indices, optionally restricted to one value along one of
        the leading axes.  This is how a channel is extracted: the channel
        axis is a leading axis, so "channel 2" is just a subset of pages.
        """
        if not self.leading_shape:
            return [0]
        grids = np.indices(self.leading_shape).reshape(
            len(self.leading_shape), -1)
        keep = np.ones(grids.shape[1], bool)
        if axis is not None and value is not None:
            keep = grids[axis] == value
        flat = np.ravel_multi_index(grids[:, keep], self.leading_shape)
        return [int(i) for i in np.atleast_1d(flat)]

    def sub_shape(self, drop_axis=None):
        """Shape after removing one leading axis (i.e. one channel's shape)."""
        if drop_axis is None:
            return self.shape
        return tuple(s for i, s in enumerate(self.shape) if i != drop_axis)

    def as_array(self):
        """Materialize everything. Only for small series (thumbnails)."""
        return self.series.asarray()

    # ── metadata ──────────────────────────────────────────────────────

    def scales(self):
        """(xy_scale, z_scale) in microns, best effort. 1.0 when unknown."""
        xy = z = 1.0
        try:
            page = self.series.pages[0]
            tags = page.tags
            if 'XResolution' in tags:
                num, den = tags['XResolution'].value
                if num:
                    unit = str(tags['ResolutionUnit'].value) if 'ResolutionUnit' in tags else ''
                    per_unit = den / num
                    # CENTIMETER -> microns; INCH -> microns; else assume um
                    if 'CENTIMETER' in unit.upper():
                        xy = per_unit * 10000.0
                    elif 'INCH' in unit.upper():
                        xy = per_unit * 25400.0
                    else:
                        xy = per_unit
            if self.tf.is_imagej and self.tf.imagej_metadata:
                z = float(self.tf.imagej_metadata.get('spacing', 1.0))
        except Exception:
            pass
        return xy, z

    def channel_names(self):
        """
        Per-channel names, if the file says so.  Akoya QPTIFF puts a
        <Name>/<Biomarker> in each channel's ImageDescription; OME-TIFF puts
        Channel Name attributes in the OME XML.
        """
        names = []
        try:
            for pg in self.series.pages[:64]:
                d = pg.description or ''
                m = re.search(r'<Biomarker>(.*?)</Biomarker>', d) or \
                    re.search(r'<Name>(.*?)</Name>', d)
                names.append(m.group(1).strip() if m else None)
        except Exception:
            return []
        if not any(names):
            try:
                ome = self.tf.ome_metadata or ''
                names = re.findall(r'<Channel[^>]*\bName="([^"]*)"', ome)
            except Exception:
                names = []
        return [n for n in names if n]

    def info_text(self):
        xy, z = self.scales()
        avail = available_memory()
        lines = [
            f"File          : {os.path.basename(self.path)}",
            f"Size on disk  : {human_bytes(os.path.getsize(self.path))}",
            "",
            f"Series        : {self.series_index}   Level: {self.level}",
            f"Shape         : {self.shape}",
            f"Axes          : {self.axes}" + (
                "    ('I' = unnamed axis, not necessarily channels)"
                if 'I' in self.axes else ""),
            f"dtype         : {self.dtype}",
            f"Decoded size  : {human_bytes(self.nbytes)}"
            + (f"   (available RAM {human_bytes(avail)})" if avail else ""),
            f"Fits in RAM   : {'yes' if fits_in_ram(self.nbytes) else 'NO'}",
            "",
            f"Pages         : {len(self.series.pages)}  "
            f"(leading {self.leading_shape or '()'} -> {self.n_pages})",
            f"Plane shape   : {self.plane_shape}",
            f"Samples/pixel : {self.samples}",
            f"Compression   : {self.compression}",
            f"Tiled         : {self.is_tiled}",
            f"Byte order    : {self.byteorder}"
            + ("   (big-endian)" if self.byteorder == '>' else ""),
            f"BigTIFF       : {self.tf.is_bigtiff}",
            "",
            f"ACCESS MODE   : {self.mode}",
        ]
        if self.mode == 'MEMMAP':
            lines.append("                Direct memory-map; random access is free.")
        else:
            reasons = []
            if self.is_tiled:
                reasons.append("tiled")
            if 'NONE' not in self.compression.upper():
                reasons.append("compressed")
            if self.byteorder == '>':
                reasons.append("big-endian")
            if self.series_index or self.level:
                reasons.append("not the base series/level")
            if not reasons:
                reasons.append("planes not contiguous in the file")
            lines.append("                Per-plane decode ("
                         + ", ".join(reasons) + ").")
            lines.append("                Works on any TIFF; one plane in RAM "
                         "at a time.")

        names = self.channel_names()
        if names:
            lines += ["", "Channel names : " + ", ".join(names)]

        lines += ["", f"xy_scale      : {xy:.4g} um/px",
                  f"z_scale       : {z:.4g} um"]

        if self.description:
            lines += ["", "── ImageDescription (first 1500 chars) ──",
                      self.description[:1500]]
        return "\n".join(lines)


def survey_file(path):
    """
    Enumerate every series and pyramid level without reading pixels.

    This is what makes a QPTIFF openable: instead of failing to read the
    file as one array, list what is actually in it and let the user choose.
    """
    import tifffile
    out = []
    with tifffile.TiffFile(path) as tf:
        for si, s in enumerate(tf.series):
            levels = getattr(s, 'levels', [s]) or [s]
            for li, lv in enumerate(levels):
                shape = tuple(int(x) for x in lv.shape)
                dt = np.dtype(lv.dtype)
                nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
                kind = str(getattr(s, 'kind', '') or '')
                out.append({
                    'series': si, 'level': li, 'shape': shape,
                    'axes': str(lv.axes), 'dtype': dt, 'nbytes': nbytes,
                    'kind': kind, 'name': str(getattr(s, 'name', '') or ''),
                })
    return out


# ══════════════════════════════════════════════════════════════════════
#  Channel-axis detection
# ══════════════════════════════════════════════════════════════════════

def detect_channel_axis(vt):
    """
    Guess which leading axis holds channels.  Returns (axis or None, reason).

    Deliberately conservative: a wrong guess that splits a 2000-slice Z-stack
    into 2000 "channels" is worse than no guess, and the user can always set
    the axis by hand.
    """
    axes = vt.axes

    if 'C' in axes:
        i = axes.index('C')
        if i < len(vt.leading_shape):
            return i, f"axes string names axis {i} as 'C'"

    if vt.channel_names():
        if vt.leading_shape:
            n = len(vt.channel_names())
            for i, s in enumerate(vt.leading_shape):
                if s == n:
                    return i, f"{n} channel names in metadata match axis {i}"
            return 0, "channel names found in metadata"

    if 'S' in axes:
        i = axes.index('S')
        if i < len(vt.leading_shape):
            return i, f"axes string names axis {i} as 'S' (samples)"

    # Unnamed leading axis with a small extent. Ambiguous -- 'I' means
    # tifffile could not name it, which is equally consistent with Z.
    if vt.leading_shape and len(vt.leading_shape) == 1:
        n = vt.leading_shape[0]
        if 2 <= n <= 12 and 'Z' not in axes and 'T' not in axes:
            return 0, (f"single unnamed axis of size {n} — plausible channels, "
                       f"but could be a {n}-slice Z-stack. Verify.")

    if len(vt.leading_shape) >= 2:
        smallest = int(np.argmin(vt.leading_shape))
        if vt.leading_shape[smallest] <= 12:
            return smallest, (f"axis {smallest} is the smallest leading axis "
                              f"({vt.leading_shape[smallest]}) — a guess")

    return None, "no channel axis detected"


# ══════════════════════════════════════════════════════════════════════
#  Writing
# ══════════════════════════════════════════════════════════════════════

def write_planes(path, plane_iter, shape, dtype, xy_scale=1.0, z_scale=1.0,
                 compression=None):
    """
    Stream planes to a TIFF.  One plane in RAM at a time.

    Prefers the ImageJ container for 3D output because it is the only writer
    option that records z-spacing; falls back to BigTIFF if ImageJ refuses
    the data (wrong ndim, wrong dtype).
    """
    import tifffile
    res = 1.0 / xy_scale if xy_scale else 1.0
    kwargs = {'photometric': 'minisblack',
              'resolution': (res, res)}
    if compression:
        kwargs['compression'] = compression

    planes = list(plane_iter) if not hasattr(plane_iter, '__next__') else plane_iter

    def _attempt(imagej):
        it = iter(planes) if isinstance(planes, list) else planes
        opts = {'imagej': True} if imagej else {'bigtiff': True}
        kw = dict(kwargs)
        if imagej:
            kw['metadata'] = {'spacing': z_scale, 'axes': 'ZYX',
                              'slices': int(shape[0])}
        else:
            kw['metadata'] = None
        with tifffile.TiffWriter(path, **opts) as tw:
            tw.write(it, shape=tuple(shape), dtype=dtype, **kw)

    if len(shape) == 3:
        try:
            _attempt(True)
            return
        except Exception:
            if isinstance(planes, list):
                _attempt(False)
                return
            raise
    _attempt(False)


# ══════════════════════════════════════════════════════════════════════
#  Workers
# ══════════════════════════════════════════════════════════════════════

class _SplitWorker(QObject):
    """Write one file per channel, streaming."""

    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            vt = VirtualTiff(self.p['path'], self.p['series'], self.p['level'])
            try:
                axis = self.p['axis']
                outdir = self.p['outdir']
                compression = self.p['compression']
                names = self.p['names']
                xy, z = vt.scales()

                n_chan = vt.shape[axis]
                crop = self.p.get('crop')
                probe_idx = crop_z(vt.plane_indices(axis=axis, value=0), crop)
                probe = crop_yx(vt.plane(probe_idx[0]), crop)
                out_shape = (len(probe_idx),) + tuple(probe.shape[:2])

                total = n_chan * (vt.n_pages // max(1, n_chan))
                done = 0
                written = []

                for c in range(n_chan):
                    if self._cancel:
                        self.status.emit("Cancelled.")
                        self.finished.emit(written)
                        return
                    label = names[c] if c < len(names) and names[c] else f"C{c}"
                    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', label)
                    out = os.path.join(outdir, f"{safe}.tif")
                    self.status.emit(f"Writing {os.path.basename(out)} "
                                     f"({c + 1}/{n_chan})")

                    idxs = crop_z(vt.plane_indices(axis=axis, value=c),
                                  self.p.get('crop'))

                    def gen(idxs=idxs):
                        nonlocal done
                        for i in idxs:
                            if self._cancel:
                                return
                            yield np.ascontiguousarray(
                                crop_yx(vt.plane(i), self.p.get('crop')))
                            done += 1
                            self.progress.emit(done, max(1, total))

                    write_planes(out, gen(), out_shape, vt.dtype,
                                 xy_scale=xy, z_scale=z,
                                 compression=compression)
                    written.append(out)

                self.status.emit(f"Wrote {len(written)} channels.")
                self.finished.emit(written)
            finally:
                vt.close()
        except Exception:
            self.error.emit(traceback.format_exc())


class _DownsampleWorker(QObject):
    """
    Downsample and write, in two bounded passes.

    Pass 1 zooms Y and X one plane at a time into a scratch memmap of shape
    (n_planes, Y', X').  Pass 2 zooms Z on that scratch.

    Splitting it this way avoids halo arithmetic entirely and keeps peak RAM
    at one input plane during pass 1.  After pass 1 the array is already
    reduced by the XY factor, so pass 2 operates on something far smaller
    than the original.
    """

    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        scratch = None
        vt = None
        try:
            from scipy.ndimage import zoom

            vt = VirtualTiff(self.p['path'], self.p['series'], self.p['level'])
            fz, fy, fx = self.p['factors']
            order = self.p['order']
            outpath = self.p['outpath']
            axis = self.p['axis']
            channel = self.p['channel']
            compression = self.p['compression']
            xy, z = vt.scales()

            # Which planes, and what the volume looks like once the channel
            # axis is resolved.
            crop = self.p.get('crop')
            if axis is not None and channel is not None:
                idxs = vt.plane_indices(axis=axis, value=channel)
            else:
                idxs = vt.plane_indices()
            idxs = crop_z(idxs, crop)
            n_in = len(idxs)

            probe = crop_yx(vt.plane(idxs[0]), crop)
            if probe.ndim == 3:                       # RGB plane -> luminance
                probe = probe[..., :3].mean(axis=-1)
            out_yx = (max(1, int(round(probe.shape[0] * fy))),
                      max(1, int(round(probe.shape[1] * fx))))
            n_out_z = max(1, int(round(n_in * fz)))

            need_z = abs(fz - 1.0) > 1e-9
            scratch_bytes = n_in * out_yx[0] * out_yx[1] * vt.dtype.itemsize

            # Pass 2 can run in RAM when the XY-reduced stack fits; otherwise
            # scipy reads it straight off the scratch memmap.
            if need_z:
                import tempfile
                scratch_dir = self.p.get('scratch') or tempfile.gettempdir()
                os.makedirs(scratch_dir, exist_ok=True)
                import uuid
                scratch = os.path.join(
                    scratch_dir, f"vl_scratch_{uuid.uuid4().hex}.dat")
                buf = np.memmap(scratch, dtype=vt.dtype, mode='w+',
                                shape=(n_in,) + out_yx)
                self.status.emit(
                    f"Pass 1/2: XY downsample -> scratch "
                    f"({human_bytes(scratch_bytes)})")
            else:
                buf = None
                self.status.emit("Downsampling XY, streaming to output")

            # ── Pass 1: XY, one plane at a time ──
            planes_out = []
            for k, i in enumerate(idxs):
                if self._cancel:
                    self.status.emit("Cancelled.")
                    self.finished.emit(None)
                    return
                pl = crop_yx(vt.plane(i), crop)
                if pl.ndim == 3:
                    pl = pl[..., :3].mean(axis=-1).astype(vt.dtype)
                if (fy, fx) != (1.0, 1.0):
                    pl = zoom(pl, (fy, fx), order=order)
                    if pl.shape != out_yx:            # rounding drift
                        pl = pl[:out_yx[0], :out_yx[1]]
                pl = pl.astype(vt.dtype, copy=False)
                if buf is not None:
                    buf[k] = pl
                else:
                    planes_out.append(pl)
                self.progress.emit(k + 1, n_in + (n_out_z if need_z else 0))

            out_xy_scale = xy / fx if fx else xy
            out_z_scale = z / fz if fz else z

            if buf is None:
                self.status.emit("Writing output")
                write_planes(outpath, planes_out, (n_in,) + out_yx, vt.dtype,
                             xy_scale=out_xy_scale, z_scale=out_z_scale,
                             compression=compression)
            else:
                buf.flush()
                self.status.emit(f"Pass 2/2: Z downsample {n_in} -> {n_out_z}")
                ro = np.memmap(scratch, dtype=vt.dtype, mode='r',
                               shape=(n_in,) + out_yx)
                result = zoom(ro, (n_out_z / n_in, 1.0, 1.0), order=order)
                result = result.astype(vt.dtype, copy=False)
                self.progress.emit(n_in + n_out_z, n_in + n_out_z)
                self.status.emit("Writing output")
                write_planes(outpath, [result[i] for i in range(result.shape[0])],
                             result.shape, vt.dtype,
                             xy_scale=out_xy_scale, z_scale=out_z_scale,
                             compression=compression)
                del ro, result

            self.status.emit(f"Wrote {os.path.basename(outpath)}")
            self.finished.emit(outpath)

        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            try:
                if vt is not None:
                    vt.close()
            except Exception:
                pass
            if scratch:
                try:
                    del buf
                except Exception:
                    pass
                try:
                    os.remove(scratch)
                except Exception:
                    pass




# ══════════════════════════════════════════════════════════════════════
#  Cropping
# ══════════════════════════════════════════════════════════════════════
#
#  A crop is (z0, z1, y0, y1, x0, x1), half-open, ALWAYS absolute against
#  the uncropped plane stack of the current source.  Keeping it absolute
#  means re-cropping never compounds rounding or off-by-ones -- the dialog
#  just edits six numbers against a fixed origin.
#
#  Crops are LAZY.  Nothing is copied or rewritten; the bounds are applied
#  when a plane is read.  That is why "apply to current image" is instant
#  even on a 200 GB source: there is no new array, only a narrower window
#  onto the same one.

def crop_z(idxs, crop):
    """Restrict a plane-index list to the crop's Z range."""
    if crop is None:
        return idxs
    return idxs[crop[0]:crop[1]]


def crop_yx(plane, crop):
    """Restrict a plane to the crop's Y/X range. A view, not a copy."""
    if crop is None:
        return plane
    return plane[crop[2]:crop[3], crop[4]:crop[5]]


def crop_shape(full_depth, plane_shape, crop):
    if crop is None:
        return (full_depth,) + tuple(plane_shape[:2])
    return (max(0, crop[1] - crop[0]),
            max(0, crop[3] - crop[2]),
            max(0, crop[5] - crop[4]))


def clamp_crop(crop, depth, h, w):
    z0, z1, y0, y1, x0, x1 = crop
    z0 = max(0, min(int(z0), depth)); z1 = max(z0 + 1, min(int(z1), depth))
    y0 = max(0, min(int(y0), h));     y1 = max(y0 + 1, min(int(y1), h))
    x0 = max(0, min(int(x0), w));     x1 = max(x0 + 1, min(int(x1), w))
    return (z0, z1, y0, y1, x0, x1)


class _CropSaveWorker(QObject):
    """Stream a cropped region to a new TIFF. One plane in RAM at a time."""

    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        vt = None
        try:
            vt = VirtualTiff(self.p['path'], self.p['series'], self.p['level'])
            crop = self.p['crop']
            axis, chan = self.p['axis'], self.p['channel']
            xy, z = vt.scales()

            if axis is not None and chan is not None:
                idxs = vt.plane_indices(axis=axis, value=chan)
            else:
                idxs = vt.plane_indices()
            idxs = crop_z(idxs, crop)
            if not idxs:
                raise ValueError("Crop selects no planes.")

            probe = crop_yx(vt.plane(idxs[0]), crop)
            if probe.ndim == 3:
                probe = probe[..., :3].mean(axis=-1).astype(vt.dtype)
            out_shape = (len(idxs),) + tuple(probe.shape[:2])
            self.status.emit(f"Writing crop {out_shape}")

            def gen():
                for k, i in enumerate(idxs):
                    if self._cancel:
                        return
                    pl = crop_yx(vt.plane(i), crop)
                    if pl.ndim == 3:
                        pl = pl[..., :3].mean(axis=-1).astype(vt.dtype)
                    yield np.ascontiguousarray(pl, dtype=vt.dtype)
                    self.progress.emit(k + 1, len(idxs))

            write_planes(self.p['outpath'], gen(), out_shape, vt.dtype,
                         xy_scale=xy, z_scale=z,
                         compression=self.p['compression'])
            self.status.emit(f"Wrote {os.path.basename(self.p['outpath'])}")
            self.finished.emit(self.p['outpath'])
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if vt is not None:
                try:
                    vt.close()
                except Exception:
                    pass


class RegionSelector:
    """
    Click-drag rectangle selection on the host viewer, handled entirely here.

    WHY IT HOOKS WHERE IT DOES
    --------------------------
    The host's WheelViewBox overrides mousePressEvent / mouseMoveEvent /
    mouseReleaseEvent and calls ev.accept() without chaining to super().  That
    makes the ViewBox the Qt mouse grabber, and pyqtgraph's GraphicsScene only
    dispatches its own mouseDragEvent when mouseGrabberItem() is None.  So
    neither an ROI's handles nor a mouseDragEvent override can ever fire here
    -- the three Qt methods are the only live layer.

    They are shadowed on the ViewBox INSTANCE while selection is armed, so the
    host module is untouched and `del` restores its overrides exactly.
    Anything that is not a selection drag is delegated straight back, so pan
    mode, Ctrl+drag and the right button keep working.

    The rectangle is drawn, not manipulated: with the ViewBox grabbing the
    mouse there is no way to give it working handles.  Drag again to redraw,
    or fine-tune in the numeric fields.
    """

    def __init__(self, api, on_region):
        import pyqtgraph as pg
        self.pg = pg
        self.api = api
        self.on_region = on_region
        self.rect = None
        self._orig = {}
        self._anchor = None
        self._dragging = False
        self._bounds = None
        self.active = False

    def _vb(self):
        return self.api.get_unsafe_window().view

    # ── arm / disarm ──────────────────────────────────────────────────

    def start(self, initial=None):
        if self.active:
            return
        vb = self._vb()
        win = self.api.get_unsafe_window()
        sel = self

        for name in ('mousePressEvent', 'mouseMoveEvent', 'mouseReleaseEvent'):
            self._orig[name] = getattr(vb, name)

        def _pt(ev):
            return vb.mapSceneToView(ev.scenePos())

        def _press(ev, _o=self._orig['mousePressEvent']):
            try:
                if (getattr(win, 'pan_mode', False)
                        or ev.button() != Qt.MouseButton.LeftButton
                        or (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)):
                    return _o(ev)
                p = _pt(ev)
                sel._anchor = (p.x(), p.y())
                sel._dragging = True
                ev.accept()
            except Exception:
                traceback.print_exc()
                return _o(ev)

        def _move(ev, _o=self._orig['mouseMoveEvent']):
            try:
                if not sel._dragging or sel._anchor is None:
                    return _o(ev)
                p = _pt(ev)
                sel._update(sel._anchor, (p.x(), p.y()))
                ev.accept()
            except Exception:
                traceback.print_exc()
                return _o(ev)

        def _release(ev, _o=self._orig['mouseReleaseEvent']):
            try:
                if not sel._dragging:
                    return _o(ev)
                p = _pt(ev)
                sel._update(sel._anchor, (p.x(), p.y()))
                sel._dragging = False
                sel._anchor = None
                sel._emit()
                ev.accept()
            except Exception:
                traceback.print_exc()
                sel._dragging = False
                return _o(ev)

        vb.mousePressEvent = _press
        vb.mouseMoveEvent = _move
        vb.mouseReleaseEvent = _release

        if initial is None:
            try:
                (vx0, vx1), (vy0, vy1) = vb.viewRange()
                initial = (vx0 + (vx1 - vx0) * 0.25, vy0 + (vy1 - vy0) * 0.25,
                           (vx1 - vx0) * 0.5, (vy1 - vy0) * 0.5)
            except Exception:
                initial = (0, 0, 100, 100)

        # Visual only -- movable=False and handleSize=0, matching the host's
        # own selection rectangle, because handles could not be dragged here.
        self.rect = self.pg.ROI(
            [initial[0], initial[1]], [initial[2], initial[3]],
            pen=self.pg.mkPen((255, 220, 0), width=2,
                              style=Qt.PenStyle.DashLine),
            movable=False, removable=False)
        try:
            self.rect.handleSize = 0
            self.rect.handlePen = self.pg.mkPen(None)
        except Exception:
            pass
        vb.addItem(self.rect)
        self._bounds = (initial[0], initial[0] + initial[2],
                        initial[1], initial[1] + initial[3])
        self.active = True
        self._emit()

    def stop(self):
        if not self.active:
            return
        vb = self._vb()
        for name in ('mousePressEvent', 'mouseMoveEvent', 'mouseReleaseEvent'):
            try:
                if name in vb.__dict__:
                    del vb.__dict__[name]
                elif self._orig.get(name) is not None:
                    setattr(vb, name, self._orig[name])
            except Exception:
                traceback.print_exc()
        self._orig = {}
        if self.rect is not None:
            try:
                vb.removeItem(self.rect)
            except Exception:
                pass
        self.rect = None
        self._dragging = False
        self._anchor = None
        self.active = False

    # ── geometry ──────────────────────────────────────────────────────

    def _update(self, a, b):
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        self._bounds = (x0, x1, y0, y1)
        if self.rect is not None:
            try:
                self.rect.setPos([x0, y0])
                self.rect.setSize([max(x1 - x0, 0.0), max(y1 - y0, 0.0)])
            except Exception:
                pass
        self._emit()

    def bounds(self):
        """(x0, x1, y0, y1) in the coordinates the viewer is displaying."""
        return self._bounds

    def _emit(self):
        if self._bounds is not None and self.on_region:
            try:
                self.on_region(self._bounds)
            except Exception:
                traceback.print_exc()


class CropOptionsDialog(QDialog):
    """
    Crop bounds plus what to do with them.

    Preview is deliberately separated from the resolving buttons: the user
    normally wants to look at a crop before committing to it, so Preview
    leaves this dialog open and only changes what the viewer is showing.
    """

    def __init__(self, owner, parent=None, prefill=None):
        super().__init__(parent)
        self.owner = owner
        self.setWindowTitle("Crop")
        self.setMinimumWidth(500)
        self._entry_crop = owner.crop          # restored by Cancel
        self._build()
        self._populate(prefill)

    def _build(self):
        lay = QVBoxLayout(self)

        self.header = QLabel("")
        self.header.setWordWrap(True)
        lay.addWidget(self.header)

        box = QGroupBox("Bounds (absolute, half-open: min included, max excluded)")
        g = QGridLayout(box)
        self.fields = {}
        for r, (lo, hi, label) in enumerate([
                ('z0', 'z1', 'Z'), ('y0', 'y1', 'Y'), ('x0', 'x1', 'X')]):
            g.addWidget(QLabel(f"{label} min:"), r, 0)
            e0 = QLineEdit(); self.fields[lo] = e0; g.addWidget(e0, r, 1)
            g.addWidget(QLabel(f"{label} max:"), r, 2)
            e1 = QLineEdit(); self.fields[hi] = e1; g.addWidget(e1, r, 3)
            for e in (e0, e1):
                e.textChanged.connect(self._update_header)
        lay.addWidget(box)

        pick = QGroupBox("Pick X and Y on the viewer")
        pkl = QVBoxLayout(pick)
        self.select_btn = QPushButton("Select Region by Dragging")
        self.select_btn.setCheckable(True)
        self.select_btn.clicked.connect(lambda: self._toggle_select())
        pkl.addWidget(self.select_btn)
        self.select_note = QLabel(
            "Drag on the viewer to draw a box; drag again to redraw it, or "
            "fine-tune the numbers above. Ctrl+drag, right-drag and the Pan "
            "button all still work. Z is unaffected \u2014 set it above.")
        self.select_note.setWordWrap(True)
        self.select_note.setStyleSheet("color: #666; font-style: italic;")
        pkl.addWidget(self.select_note)
        lay.addWidget(pick)

        # Preview sits apart from the resolving actions, and says so.
        pv = QGroupBox("Look before committing")
        pl = QVBoxLayout(pv)
        self.preview_btn = QPushButton("Preview in Viewer  (keeps this open)")
        self.preview_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.preview_btn.clicked.connect(self._preview)
        pl.addWidget(self.preview_btn)
        self.preview_note = QLabel(
            "Replaces whatever the viewer is currently showing. Only ever "
            "one image in the virtual channel.")
        self.preview_note.setWordWrap(True)
        self.preview_note.setStyleSheet("color: #666; font-style: italic;")
        pl.addWidget(self.preview_note)
        lay.addWidget(pv)

        act = QGroupBox("Then choose")
        al = QVBoxLayout(act)
        self.save_btn = QPushButton("Save Crop As...")
        self.save_btn.clicked.connect(self._save)
        al.addWidget(self.save_btn)

        self.apply_btn = QPushButton("Apply to Current Image")
        self.apply_btn.clicked.connect(self._apply)
        al.addWidget(self.apply_btn)
        al.addWidget(self._muted(
            "Makes the crop the working image: split, downsample and the "
            "viewer all use it from then on. Instant — the crop is a window "
            "onto the same file, not a copy."))

        self.cancel_btn = QPushButton("Cancel (no crop)")
        self.cancel_btn.clicked.connect(self._cancel)
        al.addWidget(self.cancel_btn)
        lay.addWidget(act)

    @staticmethod
    def _muted(text):
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: #666; font-style: italic;")
        return lab

    # ── state ─────────────────────────────────────────────────────────

    def _extent(self):
        """(depth, h, w) of the UNCROPPED current source."""
        return self.owner.uncropped_extent()

    def _populate(self, prefill=None):
        d, h, w = self._extent()
        # Guard the type, not just None: Qt hands `checked=False` to any slot
        # that accepts an argument, which is how this used to die silently.
        if not isinstance(prefill, (tuple, list)) or len(prefill) != 6:
            prefill = None
        c = prefill if prefill is not None else (self.owner.crop or
                                                 (0, d, 0, h, 0, w))
        for k, v in zip(('z0', 'z1', 'y0', 'y1', 'x0', 'x1'), c):
            self.fields[k].setText(str(int(v)))
        self._update_header()

    def _read(self):
        d, h, w = self._extent()
        vals = []
        for k, dflt in zip(('z0', 'z1', 'y0', 'y1', 'x0', 'x1'),
                           (0, d, 0, h, 0, w)):
            t = self.fields[k].text().strip()
            try:
                vals.append(int(t))
            except ValueError:
                vals.append(dflt)
        return clamp_crop(tuple(vals), d, h, w)

    def _update_header(self):
        d, h, w = self._extent()
        c = self._read()
        out = crop_shape(d, (h, w), c)
        vt = self.owner.vt
        nb = int(np.prod(out)) * (vt.dtype.itemsize if vt else 1)
        self.header.setText(
            f"Source (Z, Y, X): ({d}, {h}, {w})<br>"
            f"<b>Crop: {out} = {human_bytes(nb)}</b>")

    # ── actions ───────────────────────────────────────────────────────

    def _toggle_select(self):
        if self.select_btn.isChecked():
            if not self.owner.bridge.active:
                QMessageBox.information(
                    self, "Show the Image First",
                    "Use 'Show in Viewer' before selecting a region, so there "
                    "is something to draw the box on.")
                self.select_btn.setChecked(False)
                return
            self.owner.start_region_select(self._on_region)
            self.select_btn.setText("Stop Selecting")
        else:
            self.owner.stop_region_select()
            self.select_btn.setText("Select Region by Dragging")

    def _on_region(self, bounds):
        """
        Viewer coords -> absolute source coords.

        What is on screen is the CURRENT crop, so an active crop's origin has
        to be added back before these numbers mean anything against the file.
        """
        x0, x1, y0, y1 = bounds
        off = self.owner.bridge.crop
        dy, dx = (0, 0) if off is None else (off[2], off[4])
        d, h, w = self._extent()

        def _num(key, dflt):
            try:
                return int(self.fields[key].text())
            except (ValueError, TypeError):
                return dflt

        c = clamp_crop((_num('z0', 0), _num('z1', d),
                        int(round(y0)) + dy, int(round(y1)) + dy,
                        int(round(x0)) + dx, int(round(x1)) + dx), d, h, w)
        for k, v in zip(('y0', 'y1', 'x0', 'x1'), c[2:]):
            self.fields[k].blockSignals(True)
            self.fields[k].setText(str(int(v)))
            self.fields[k].blockSignals(False)
        self._update_header()

    def _preview(self):
        self.owner.preview_crop(self._read())

    def _save(self):
        if self.owner.save_crop(self._read(), parent=self):
            self.owner.stop_region_select()
            self.accept()

    def _apply(self):
        self.owner.stop_region_select()
        self.owner.apply_crop(self._read())
        self.accept()

    def _cancel(self):
        self.owner.stop_region_select()
        # Undo any preview taken while this dialog was open.
        if self.owner.crop != self._entry_crop or self.owner.bridge.crop is not None:
            self.owner.preview_crop(self._entry_crop, silent=True)
        self.reject()

    def closeEvent(self, event):
        self.owner.stop_region_select()
        self.owner._crop_dialog = None
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════
#  Viewer bridge — render a plane into the parent ImageViewerWindow
# ══════════════════════════════════════════════════════════════════════

PRESET_COLORS = [
    ("Yellow",  (1.0, 1.0, 0.0)),
    ("Magenta", (1.0, 0.0, 1.0)),
    ("Orange",  (1.0, 0.5, 0.0)),
    ("Cyan",    (0.0, 1.0, 1.0)),
    ("Green",   (0.0, 1.0, 0.0)),
    ("Red",     (1.0, 0.0, 0.0)),
    ("Blue",    (0.3, 0.5, 1.0)),
    ("White",   (1.0, 1.0, 1.0)),
]

# One plane larger than this is refused for display; the user is pointed at
# a pyramid level instead. Whole-slide level-0 planes can be gigabytes.
MAX_DISPLAY_PLANE_BYTES = 384 * 1024 ** 2

# Longest edge handed to pyqtgraph. Beyond this the GPU upload dominates and
# there is nothing to see anyway.
MAX_DISPLAY_EDGE = 3000


class ViewerBridge:
    """
    Draws one plane of a VirtualTiff into the parent viewer as a purely
    visual overlay, via a display hook.

    Nothing is written into the host's channel_data, so nothing can be
    saved, analysed, or otherwise mistaken for a real channel. Everything
    it owns is released by purge().

    Reads exactly one plane at a time and caches it, because the display
    hook fires on every pan, zoom and redraw -- without the cache a tiled
    QPTIFF would re-decode a plane per mouse move.
    """

    def __init__(self, api):
        import pyqtgraph as pg
        self.api = api
        self.pg = pg

        self.vt = None                # VirtualTiff being displayed
        self.scratch_path = None      # decompressed copy, if any
        self.scratch = None           # memmap over it
        self.axis = None              # channel axis, or None
        self.channel = None           # channel index, or None
        self.crop = None              # (z0,z1,y0,y1,x0,x1), absolute

        self.color = PRESET_COLORS[0][1]
        self.opacity = 0.85
        self.bmin, self.bmax = 0.0, 1.0

        self.image_item = None
        self._in_view = False
        self._hook_installed = False
        self._cache_key = None
        self._cache_plane = None
        self._range = None            # (lo, hi) for contrast
        self._owned_slider = None     # (max, value) to restore on purge
        self.active = False

    # ── plumbing ──────────────────────────────────────────────────────

    def _win(self):
        return self.api.get_unsafe_window()

    def _install_hook(self):
        if self._hook_installed:
            return
        self.api.register_display_hook(self.on_display)
        self._hook_installed = True

    def _remove_hook(self):
        if not self._hook_installed:
            return
        try:
            hooks = getattr(self._win(), '_plugin_display_hooks', None)
            if hooks and self.on_display in hooks:
                hooks.remove(self.on_display)
        except Exception:
            pass
        self._hook_installed = False

    # ── show / purge ──────────────────────────────────────────────────

    def show(self, path, series, level, axis, channel, parent=None,
             scratch_dir=None, crop=None):
        """
        Point the bridge at one plane stack. Returns True on success.

        `scratch_dir` non-None means decode this series to a disk-backed copy
        first, which makes slice scrubbing responsive on compressed files.
        """
        self.purge(refresh=False)

        vt = VirtualTiff(path, series, level)

        plane_bytes = int(np.prod(vt.plane_shape)) * vt.dtype.itemsize
        if plane_bytes > MAX_DISPLAY_PLANE_BYTES:
            vt.close()
            if parent:
                QMessageBox.warning(
                    parent, "Plane Too Large to Display",
                    f"A single plane of this image is "
                    f"{human_bytes(plane_bytes)}.\n\n"
                    "Displaying it would have to decode the whole plane at "
                    "once. Pick a lower-resolution pyramid level from the "
                    "list on the left and try again.")
            return False

        self.vt = vt
        self.axis = axis
        self.channel = channel
        self.crop = crop

        # Optional decompress-to-scratch for responsive scrubbing.
        if scratch_dir:
            try:
                import uuid
                os.makedirs(scratch_dir, exist_ok=True)
                self.scratch_path = os.path.join(
                    scratch_dir, f"vl_view_{uuid.uuid4().hex}.dat")
                self.scratch = vt.series.asarray(out=self.scratch_path)
            except Exception as e:
                self.scratch = None
                self.scratch_path = None
                if parent:
                    QMessageBox.warning(
                        parent, "Scratch Decode Failed",
                        f"{type(e).__name__}: {e}\n\n"
                        "Falling back to per-plane decoding, which will make "
                        "slice changes slower.")

        self.image_item = self.pg.ImageItem()
        self._range = None
        self._cache_key = None
        self._cache_plane = None
        self.active = True

        self._take_slider()
        self._install_hook()
        self.api.refresh_display()
        return True

    def purge(self, refresh=True):
        """Release everything: view item, hook, scratch file, memmap, cache."""
        self.active = False

        if self.image_item is not None:
            try:
                self._win().view.removeItem(self.image_item)
            except Exception:
                pass
        self.image_item = None
        self._in_view = False

        self._remove_hook()
        self._restore_slider()

        self._cache_key = None
        self._cache_plane = None
        self._range = None
        self.crop = None

        if self.scratch is not None:
            try:
                base = self.scratch
                while isinstance(getattr(base, 'base', None), np.memmap):
                    base = base.base
                if getattr(base, '_mmap', None) is not None:
                    base._mmap.close()
            except Exception:
                pass
        self.scratch = None
        if self.scratch_path:
            try:
                os.remove(self.scratch_path)
            except Exception:
                pass
        self.scratch_path = None

        if self.vt is not None:
            try:
                self.vt.close()
            except Exception:
                pass
        self.vt = None

        if refresh:
            try:
                self.api.refresh_display()
            except Exception:
                pass

    # ── slider ownership ──────────────────────────────────────────────

    def _depth(self):
        if self.vt is None:
            return 1
        if self.axis is not None and self.channel is not None:
            idxs = self.vt.plane_indices(axis=self.axis, value=self.channel)
        else:
            idxs = self.vt.plane_indices()
        return max(1, len(crop_z(idxs, self.crop)))

    def set_crop(self, crop):
        """
        Re-window without reopening anything.  Clears the plane cache and
        re-takes the slider, because both depend on the crop.
        """
        if self.vt is None:
            return
        self.crop = crop
        self._cache_key = None
        self._cache_plane = None
        self._range = None
        self._restore_slider()
        self._take_slider()
        try:
            self.api.refresh_display()
        except Exception:
            pass

    def _take_slider(self):
        """
        Extend the host slider to cover our depth when the host has no data
        of its own. If real channels are loaded we leave the slider alone
        and simply clamp -- their data is what the slider is for.
        """
        win = self._win()
        if self.api.get_shape() is not None:
            return
        try:
            sl = win.slice_slider
            self._owned_slider = (sl.maximum(), sl.value(), sl.isEnabled())
            sl.setEnabled(True)
            sl.setMinimum(0)
            sl.setMaximum(max(0, self._depth() - 1))
            sl.setValue(0)
            win.current_slice = 0
            h, w = self.vt.plane_shape[:2]
            if self.crop is not None:
                h = self.crop[3] - self.crop[2]
                w = self.crop[5] - self.crop[4]
            win.view.setRange(xRange=(0, w), yRange=(0, h), padding=0)
        except Exception:
            self._owned_slider = None

    def _restore_slider(self):
        if self._owned_slider is None:
            return
        try:
            mx, val, enabled = self._owned_slider
            sl = self._win().slice_slider
            sl.setMaximum(mx)
            sl.setValue(min(val, mx))
            sl.setEnabled(enabled)
        except Exception:
            pass
        self._owned_slider = None

    # ── plane access ──────────────────────────────────────────────────

    def _plane(self, z):
        key = (self.vt.series_index, self.vt.level, self.axis,
               self.channel, self.crop, z)
        if key == self._cache_key:
            return self._cache_plane

        if self.axis is not None and self.channel is not None:
            idxs = self.vt.plane_indices(axis=self.axis, value=self.channel)
        else:
            idxs = self.vt.plane_indices()
        idxs = crop_z(idxs, self.crop)
        if z >= len(idxs):
            return None
        flat = idxs[z]

        if self.scratch is not None:
            lead = self.vt.leading_shape
            idx = np.unravel_index(flat, lead) if lead else ()
            plane = np.asarray(self.scratch[idx])
        else:
            plane = self.vt.plane(flat)

        plane = crop_yx(plane, self.crop)
        if plane.ndim == 3:                       # RGB -> luminance
            plane = plane[..., :3].mean(axis=-1)

        self._cache_key = key
        self._cache_plane = plane
        return plane

    def _contrast(self, plane):
        if self._range is None:
            lo = float(np.min(plane))
            hi = float(np.max(plane))
            self._range = (lo, max(hi, lo + 1.0))
        return self._range

    def reset_contrast(self):
        self._range = None

    # ── display hook ──────────────────────────────────────────────────

    def on_display(self, view, current_slice, view_range):
        if not self.active or self.vt is None or self.image_item is None:
            return
        try:
            plane = self._plane(int(current_slice))
            if plane is None:
                if self._in_view:
                    view.removeItem(self.image_item)
                    self._in_view = False
                return

            h, w = plane.shape[:2]

            # Crop to what is actually on screen, so zooming reveals detail
            # instead of just scaling a fixed-size preview.
            try:
                (x0, x1), (y0, y1) = view_range[0], view_range[1]
                x0 = max(0, int(np.floor(x0))); x1 = min(w, int(np.ceil(x1)))
                y0 = max(0, int(np.floor(y0))); y1 = min(h, int(np.ceil(y1)))
                if x1 <= x0 or y1 <= y0:
                    x0, y0, x1, y1 = 0, 0, w, h
            except Exception:
                x0, y0, x1, y1 = 0, 0, w, h

            sub = plane[y0:y1, x0:x1]
            ds = max(1, int(np.ceil(max(sub.shape) / MAX_DISPLAY_EDGE)))
            if ds > 1:
                sub = sub[::ds, ::ds]

            lo, hi = self._contrast(plane)
            vmin = lo + (hi - lo) * self.bmin
            vmax = max(lo + (hi - lo) * self.bmax, vmin + 1e-6)
            norm = np.clip((np.asarray(sub, dtype=np.float32) - vmin)
                           / (vmax - vmin), 0, 1)

            r, g, b = self.color
            cmap = self.pg.ColorMap(pos=np.array([0, 1]),
                                    color=np.array([[0, 0, 0, 0],
                                                    [r, g, b, 1]]) * 255)
            self.image_item.setImage(norm.T, levels=(0, 1))
            self.image_item.setLookupTable(cmap.getLookupTable(0, 1, 256))
            self.image_item.setOpacity(self.opacity)
            self.image_item.setRect(x0, y0, x1 - x0, y1 - y0)

            if not self._in_view:
                view.addItem(self.image_item)
                self._in_view = True
        except Exception:
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
#  Show options popout
# ══════════════════════════════════════════════════════════════════════

class ShowOptionsDialog(QDialog):
    """
    Small popout for choosing what goes into the viewer, kept separate so
    the main window is not cluttered with display controls that only matter
    once something is being shown.
    """

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.setWindowTitle("Show in Viewer")
        self.setMinimumWidth(460)
        self._build()
        self._reload_sources()

    def _build(self):
        lay = QVBoxLayout(self)

        src = QGroupBox("What to show")
        g = QGridLayout(src)
        g.addWidget(QLabel("Image:"), 0, 0)
        self.src_combo = QComboBox()
        self.src_combo.currentIndexChanged.connect(self._on_source_changed)
        g.addWidget(self.src_combo, 0, 1, 1, 3)

        self.chan_label = QLabel("Channel:")
        g.addWidget(self.chan_label, 1, 0)
        self.chan_combo = QComboBox()
        g.addWidget(self.chan_combo, 1, 1, 1, 3)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #666; font-style: italic;")
        g.addWidget(self.note, 2, 0, 1, 4)
        lay.addWidget(src)

        look = QGroupBox("Appearance")
        g2 = QGridLayout(look)
        g2.addWidget(QLabel("Colour:"), 0, 0)
        self.color_combo = QComboBox()
        for name, _ in PRESET_COLORS:
            self.color_combo.addItem(name)
        self.color_combo.currentIndexChanged.connect(self._apply_look)
        g2.addWidget(self.color_combo, 0, 1)

        g2.addWidget(QLabel("Opacity:"), 0, 2)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.85)
        self.opacity.valueChanged.connect(self._apply_look)
        g2.addWidget(self.opacity, 0, 3)

        g2.addWidget(QLabel("Black point (Low End Contrast):"), 1, 0)
        self.bmin = QDoubleSpinBox()
        self.bmin.setRange(0.0, 0.99); self.bmin.setSingleStep(0.05)
        self.bmin.valueChanged.connect(self._apply_look)
        g2.addWidget(self.bmin, 1, 1)

        g2.addWidget(QLabel("White point (High End Contrast):"), 1, 2)
        self.bmax = QDoubleSpinBox()
        self.bmax.setRange(0.01, 1.0); self.bmax.setSingleStep(0.05)
        self.bmax.setValue(0.2)
        self.bmax.valueChanged.connect(self._apply_look)
        g2.addWidget(self.bmax, 1, 3)
        lay.addWidget(look)

        self.scratch_check = QCheckBox(
            "Decode to a scratch file first (faster slice scrubbing)")
        self.scratch_check.setChecked(False)
        lay.addWidget(self.scratch_check)
        self.scratch_note = QLabel("")
        self.scratch_note.setWordWrap(True)
        self.scratch_note.setStyleSheet("color: #666; font-style: italic;")
        lay.addWidget(self.scratch_note)

        row = QHBoxLayout()
        self.show_btn = QPushButton("Show")
        self.show_btn.clicked.connect(self._show)
        self.hide_btn = QPushButton("Hide")
        self.hide_btn.clicked.connect(self._hide)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(self.show_btn)
        row.addWidget(self.hide_btn)
        row.addStretch()
        row.addWidget(close_btn)
        lay.addLayout(row)

    # ── population ────────────────────────────────────────────────────

    def _reload_sources(self):
        self.src_combo.blockSignals(True)
        self.src_combo.clear()
        for i, e in enumerate(self.owner.survey):
            tag = f"S{e['series']}" + (f" L{e['level']}" if e['level'] else "")
            self.src_combo.addItem(
                f"{tag}  {e['shape']}  {e['axes']}  "
                f"{human_bytes(e['nbytes'])}", i)
        # Default to whatever is selected in the main window.
        row = self.owner.series_list.currentRow()
        if 0 <= row < self.src_combo.count():
            self.src_combo.setCurrentIndex(row)
        self.src_combo.blockSignals(False)
        self._on_source_changed()

    def _on_source_changed(self):
        i = self.src_combo.currentData()
        if i is None or i >= len(self.owner.survey):
            return
        e = self.owner.survey[i]
        try:
            vt = VirtualTiff(self.owner.path, e['series'], e['level'])
        except Exception as ex:
            self.note.setText(f"Could not open: {ex}")
            return
        try:
            axis, reason = detect_channel_axis(vt)
            names = vt.channel_names()

            self.chan_combo.blockSignals(True)
            self.chan_combo.clear()
            if axis is None:
                self.chan_label.setVisible(False)
                self.chan_combo.setVisible(False)
                self.chan_combo.addItem("(single image)", None)
                self.show_btn.setText("Show Image")
                self.note.setText("No channel axis detected — the whole "
                                  "stack will be shown.")
            else:
                self.chan_label.setVisible(True)
                self.chan_combo.setVisible(True)
                for c in range(vt.shape[axis]):
                    nm = names[c] if c < len(names) and names[c] else f"C{c}"
                    self.chan_combo.addItem(f"{c}: {nm}", c)
                self.show_btn.setText("Show Channel")
                self.note.setText(f"Channel axis {axis} — {reason}")
            self.chan_combo.blockSignals(False)
            self._axis = axis

            plane_bytes = int(np.prod(vt.plane_shape)) * vt.dtype.itemsize
            if plane_bytes > MAX_DISPLAY_PLANE_BYTES:
                self.note.setText(
                    self.note.text() + f"\n\nOne plane is "
                    f"{human_bytes(plane_bytes)} — too large to display. "
                    f"Choose a lower pyramid level.")
                self.show_btn.setEnabled(False)
            else:
                self.show_btn.setEnabled(True)

            compressed = 'NONE' not in vt.compression.upper()
            self.scratch_check.setEnabled(vt.mode != 'MEMMAP')
            if vt.mode == 'MEMMAP':
                self.scratch_check.setChecked(False)
                self.scratch_note.setText(
                    "Memory-mapped — slice changes are already instant.")
            elif compressed:
                self.scratch_check.setChecked(True)
                self.scratch_note.setText(
                    f"Compressed: each slice change re-decodes a plane. "
                    f"Decoding to scratch writes about "
                    f"{human_bytes(vt.nbytes)} to disk once, and the file is "
                    f"removed when this window closes.")
            else:
                self.scratch_check.setChecked(False)
                self.scratch_note.setText(
                    "Uncompressed — per-plane reads are fast enough.")
        finally:
            vt.close()

    # ── actions ───────────────────────────────────────────────────────

    def _apply_look(self):
        b = self.owner.bridge
        b.color = PRESET_COLORS[self.color_combo.currentIndex()][1]
        b.opacity = self.opacity.value()
        b.bmin = self.bmin.value()
        b.bmax = max(self.bmax.value(), self.bmin.value() + 0.01)
        if b.active:
            self.owner.api.refresh_display()

    def _show(self):
        i = self.src_combo.currentData()
        if i is None:
            return
        e = self.owner.survey[i]
        axis = getattr(self, '_axis', None)
        chan = self.chan_combo.currentData() if axis is not None else None

        scratch_dir = None
        if self.scratch_check.isChecked() and self.scratch_check.isEnabled():
            import tempfile
            scratch_dir = QFileDialog.getExistingDirectory(
                self, "Choose a scratch directory", tempfile.gettempdir())
            if not scratch_dir:
                return

        self._apply_look()
        crop = self.owner.crop if (e['series'] == (self.owner.vt.series_index
                                                   if self.owner.vt else -1)
                                   and e['level'] == (self.owner.vt.level
                                                      if self.owner.vt else -1)
                                   ) else None
        ok = self.owner.bridge.show(
            self.owner.path, e['series'], e['level'], axis, chan,
            parent=self, scratch_dir=scratch_dir, crop=crop)
        if ok:
            self.owner._update_show_state()

    def _hide(self):
        self.owner.stop_region_select()
        self.owner.bridge.purge()
        self.owner._update_show_state()


# ══════════════════════════════════════════════════════════════════════
#  Dialog
# ══════════════════════════════════════════════════════════════════════

class VirtualLoaderDialog(QDialog):

    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Virtual Loader")
        self.setMinimumSize(900, 700)

        self.path = None
        self.survey = []
        self.vt = None
        self._thread = None
        self._worker = None
        self.bridge = ViewerBridge(api)
        self._show_dialog = None
        self._crop_dialog = None
        self.crop = None              # working crop, absolute to the source
        self.selector = None          # RegionSelector, created on demand

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # File
        fbox = QGroupBox("File")
        fl = QHBoxLayout(fbox)
        self.open_btn = QPushButton("Open TIFF...")
        self.open_btn.clicked.connect(self._open_file)
        self.path_label = QLabel("No file loaded")
        self.path_label.setWordWrap(True)
        fl.addWidget(self.open_btn)
        fl.addWidget(self.path_label, 1)
        root.addWidget(fbox)

        split = QSplitter(Qt.Orientation.Horizontal)

        # Left: series/level picker
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel(
            "<b>Images in this file</b><br>"
            "<i>Whole-slide formats hold several images — full resolution, "
            "thumbnails, and pyramid levels. Pick the one to work on.</i>"))
        self.series_list = QListWidget()
        self.series_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.series_list.currentRowChanged.connect(self._on_series_changed)
        ll.addWidget(self.series_list, 1)

        # Channels
        cbox = QGroupBox("Channel axis")
        cl = QGridLayout(cbox)
        cl.addWidget(QLabel("Axis:"), 0, 0)
        self.axis_combo = QComboBox()
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        cl.addWidget(self.axis_combo, 0, 1)
        self.axis_reason = QLabel("")
        self.axis_reason.setWordWrap(True)
        self.axis_reason.setStyleSheet("color: #666; font-style: italic;")
        cl.addWidget(self.axis_reason, 1, 0, 1, 2)
        ll.addWidget(cbox)

        split.addWidget(left)

        # Right: metadata
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("<b>Metadata</b>"))
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.info.setStyleSheet("font-family: monospace;")
        rl.addWidget(self.info, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        # Actions
        abox = QGroupBox("Actions")
        al = QGridLayout(abox)

        self.split_btn = QPushButton("Split Channels to Directory...")
        self.split_btn.clicked.connect(self._do_split)
        self.split_btn.setEnabled(False)
        al.addWidget(self.split_btn, 0, 0, 1, 2)

        # Master show button. The options live in a popout so they do not
        # clutter this window until the user actually wants them.
        self.show_btn = QPushButton("Show in Viewer...")
        self.show_btn.clicked.connect(self._open_show_options)
        self.show_btn.setEnabled(False)
        al.addWidget(self.show_btn, 0, 2, 1, 2)

        self.crop_btn = QPushButton("Crop...")
        self.crop_btn.clicked.connect(lambda: self._open_crop())
        self.crop_btn.setEnabled(False)
        al.addWidget(self.crop_btn, 0, 4)

        self.showing_label = QLabel("")
        self.showing_label.setStyleSheet("color: #666; font-style: italic;")
        al.addWidget(self.showing_label, 0, 5)

        al.addWidget(QLabel("Downsample factors (1.0 = unchanged):"), 1, 0, 1, 6)
        self.fz = QDoubleSpinBox(); self.fy = QDoubleSpinBox(); self.fx = QDoubleSpinBox()
        for sp, lab, col in ((self.fz, "Z", 0), (self.fy, "Y", 2), (self.fx, "X", 4)):
            sp.setRange(0.001, 1.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.05)
            sp.setValue(0.5)
            al.addWidget(QLabel(lab), 2, col)
            al.addWidget(sp, 2, col + 1)

        al.addWidget(QLabel("Interpolation:"), 3, 0)
        self.order_combo = QComboBox()
        self.order_combo.addItems([
            "Linear (order=1) — for raw intensity data",
            "Nearest (order=0) — for label / segmentation data",
        ])
        al.addWidget(self.order_combo, 3, 1, 1, 3)

        al.addWidget(QLabel("Channel:"), 3, 4)
        self.chan_combo = QComboBox()
        al.addWidget(self.chan_combo, 3, 5)

        self.compress_check = QCheckBox("Compress output (zlib)")
        al.addWidget(self.compress_check, 4, 0, 1, 2)

        self.est_label = QLabel("")
        self.est_label.setStyleSheet("color: #666;")
        al.addWidget(self.est_label, 4, 2, 1, 4)
        for sp in (self.fz, self.fy, self.fx):
            sp.valueChanged.connect(self._update_estimate)
        self.chan_combo.currentIndexChanged.connect(self._update_estimate)

        self.down_btn = QPushButton("Downsample and Save As...")
        self.down_btn.clicked.connect(self._do_downsample)
        self.down_btn.setEnabled(False)
        al.addWidget(self.down_btn, 5, 0, 1, 6)

        root.addWidget(abox)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.status = QLabel("")
        root.addWidget(self.status)

        brow = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        brow.addStretch()
        brow.addWidget(self.cancel_btn)
        root.addLayout(brow)

    # ── loading ───────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF (not loaded into memory)", "", TIFF_FILTER)
        if not path:
            return
        try:
            self.survey = survey_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Could Not Read File",
                                 f"{type(e).__name__}: {e}")
            return
        if not self.survey:
            QMessageBox.warning(self, "Empty", "No image series found.")
            return

        self.path = path
        self.path_label.setText(path)

        self.series_list.blockSignals(True)
        self.series_list.clear()
        for e in self.survey:
            tag = f"S{e['series']}"
            if e['level']:
                tag += f" L{e['level']}"
            label = (f"{tag}  {e['shape']}  {e['axes']}  "
                     f"{e['dtype']}  {human_bytes(e['nbytes'])}")
            if e['name']:
                label += f"   [{e['name']}]"
            self.series_list.addItem(QListWidgetItem(label))
        self.series_list.blockSignals(False)
        self.series_list.setCurrentRow(0)

    def _on_series_changed(self, row):
        if row < 0 or not self.survey:
            return
        e = self.survey[row]
        if self.vt is not None:
            self.vt.close()
            self.vt = None
        try:
            self.vt = VirtualTiff(self.path, e['series'], e['level'])
        except Exception as ex:
            QMessageBox.critical(self, "Could Not Open Series",
                                 f"{type(ex).__name__}: {ex}")
            return

        # A crop is defined against one source; switching source drops it.
        self.crop = None

        self.info.setPlainText(self.vt.info_text())

        # Axis combo
        guess, reason = detect_channel_axis(self.vt)
        self.axis_combo.blockSignals(True)
        self.axis_combo.clear()
        self.axis_combo.addItem("None (single channel)", None)
        for i, s in enumerate(self.vt.leading_shape):
            name = self.vt.axes[i] if i < len(self.vt.axes) else '?'
            self.axis_combo.addItem(f"Axis {i} ('{name}', size {s})", i)
        idx = 0 if guess is None else guess + 1
        self.axis_combo.setCurrentIndex(min(idx, self.axis_combo.count() - 1))
        self.axis_combo.blockSignals(False)
        self.axis_reason.setText(reason)

        self._on_axis_changed()

    def _on_axis_changed(self):
        axis = self.axis_combo.currentData()
        names = self.vt.channel_names() if self.vt else []

        self.chan_combo.blockSignals(True)
        self.chan_combo.clear()
        if axis is None:
            self.chan_combo.addItem("All (single channel)", None)
            self.split_btn.setEnabled(False)
        else:
            n = self.vt.shape[axis]
            self.chan_combo.addItem("All channels", None)
            for c in range(n):
                nm = names[c] if c < len(names) and names[c] else f"C{c}"
                self.chan_combo.addItem(f"{c}: {nm}", c)
            self.split_btn.setEnabled(True)
        self.chan_combo.blockSignals(False)

        self.down_btn.setEnabled(self.vt is not None)
        self.show_btn.setEnabled(self.vt is not None)
        self.crop_btn.setEnabled(self.vt is not None)
        # Label follows what the file actually contains: a channel picker is
        # only meaningful when there are channels.
        self.show_btn.setText(
            "Show Channel in Viewer..." if axis is not None
            else "Show in Viewer...")
        self._update_estimate()

    def _update_estimate(self):
        if self.vt is None:
            self.est_label.setText("")
            return
        axis = self.axis_combo.currentData()
        chan = self.chan_combo.currentData()
        shape = list(self.vt.shape)
        if axis is not None and chan is not None:
            shape.pop(axis)
        elif axis is not None:
            shape.pop(axis)
        if self.crop is not None:
            d, h, w = self.uncropped_extent()
            shape = list(crop_shape(d, (h, w), self.crop))
        fz, fy, fx = self.fz.value(), self.fy.value(), self.fx.value()
        if len(shape) >= 3:
            out = (max(1, int(round(shape[0] * fz))),
                   max(1, int(round(shape[-2] * fy))),
                   max(1, int(round(shape[-1] * fx))))
        else:
            out = (1, max(1, int(round(shape[-2] * fy))),
                   max(1, int(round(shape[-1] * fx))))
        nbytes = int(np.prod(out)) * self.vt.dtype.itemsize
        self.est_label.setText(
            f"Output {out} = {human_bytes(nbytes)}"
            + ("" if fits_in_ram(nbytes) else "   (may not fit in RAM)"))

    # ── actions ───────────────────────────────────────────────────────

    def _do_split(self):
        if self.vt is None:
            return
        axis = self.axis_combo.currentData()
        if axis is None:
            QMessageBox.information(self, "No Channel Axis",
                                    "Set a channel axis first.")
            return
        outdir = QFileDialog.getExistingDirectory(
            self, "Choose output directory for channels")
        if not outdir:
            return

        n = self.vt.shape[axis]
        if n > 64:
            r = QMessageBox.question(
                self, "That is a lot of channels",
                f"Axis {axis} has {n} entries, which would write {n} files.\n\n"
                "If this is really a Z axis rather than a channel axis, "
                "cancel and pick a different axis.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return

        self._start(_SplitWorker({
            'path': self.path,
            'series': self.vt.series_index,
            'level': self.vt.level,
            'axis': axis,
            'outdir': outdir,
            'compression': 'zlib' if self.compress_check.isChecked() else None,
            'names': self.vt.channel_names(),
            'crop': self.crop,
        }), done_msg="Channels written")

    def _do_downsample(self):
        if self.vt is None:
            return
        fz, fy, fx = self.fz.value(), self.fy.value(), self.fx.value()
        if (fz, fy, fx) == (1.0, 1.0, 1.0):
            QMessageBox.information(self, "Nothing to Do",
                                    "All factors are 1.0.")
            return
        outpath, _ = QFileDialog.getSaveFileName(
            self, "Save downsampled TIFF as", "", "TIFF Files (*.tif *.tiff)")
        if not outpath:
            return
        if not outpath.lower().endswith(('.tif', '.tiff')):
            outpath += '.tif'
        if os.path.abspath(outpath) == os.path.abspath(self.path):
            QMessageBox.warning(self, "Cannot Overwrite Source",
                                "Choose a different destination.")
            return

        axis = self.axis_combo.currentData()
        chan = self.chan_combo.currentData()
        if axis is not None and chan is None:
            QMessageBox.information(
                self, "Pick One Channel",
                "Downsampling writes a single 3D stack. Choose one channel, "
                "or set the channel axis to None.\n\n"
                "To get every channel, use Split Channels first.")
            return

        self._start(_DownsampleWorker({
            'path': self.path,
            'series': self.vt.series_index,
            'level': self.vt.level,
            'factors': (fz, fy, fx),
            'order': 1 if self.order_combo.currentIndex() == 0 else 0,
            'outpath': outpath,
            'axis': axis,
            'channel': chan,
            'compression': 'zlib' if self.compress_check.isChecked() else None,
            'scratch': None,
            'crop': self.crop,
        }), done_msg="Downsampled file written")

    # ── worker plumbing ───────────────────────────────────────────────

    def _start(self, worker, done_msg=""):
        self._done_msg = done_msg
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.split_btn.setEnabled(False)
        self.down_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._thread = QThread()
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.status.connect(self.status.setText)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, cur, total):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(cur)

    def _reset(self):
        self.progress.setVisible(False)
        self.open_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._on_axis_changed()

    def _on_finished(self, result):
        self._reset()
        if result:
            self.status.setText(self._done_msg)

    def _on_error(self, msg):
        self._reset()
        self.status.setText("Error — see details.")
        QMessageBox.critical(self, "Virtual Loader Error", msg)

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
        self.status.setText("Cancelling...")

    # ── cropping ──────────────────────────────────────────────────────

    def uncropped_extent(self):
        """(depth, h, w) of the current source, ignoring any active crop."""
        if self.vt is None:
            return (1, 1, 1)
        axis = self.axis_combo.currentData()
        chan = self.chan_combo.currentData()
        if axis is not None and chan is not None:
            depth = len(self.vt.plane_indices(axis=axis, value=chan))
        elif axis is not None:
            depth = len(self.vt.plane_indices(axis=axis, value=0))
        else:
            depth = self.vt.n_pages
        h, w = self.vt.plane_shape[:2]
        return (depth, h, w)

    def _crop_source(self):
        """(axis, channel) the crop applies to. Cropping needs one channel."""
        axis = self.axis_combo.currentData()
        chan = self.chan_combo.currentData()
        if axis is not None and chan is None:
            chan = 0
        return axis, chan

    def _open_crop(self, prefill=None):
        if not isinstance(prefill, (tuple, list)) or len(prefill) != 6:
            prefill = None
        if self.vt is None:
            return
        if self._crop_dialog is None:
            self._crop_dialog = CropOptionsDialog(self, parent=self,
                                                  prefill=prefill)
        elif prefill is not None:
            self._crop_dialog._populate(prefill)
        self._crop_dialog.show()
        self._crop_dialog.raise_()

    def start_region_select(self, callback):
        if self.selector is None:
            self.selector = RegionSelector(self.api, callback)
        else:
            self.selector.on_region = callback
        self.selector.start()

    def stop_region_select(self):
        if self.selector is not None:
            try:
                self.selector.stop()
            except Exception:
                traceback.print_exc()

    def preview_crop(self, crop, silent=False):
        """
        Show `crop` in the viewer without committing to it.

        Routed through the bridge so the invariant holds: exactly one image
        occupies the virtual channel at any time. If nothing is being shown
        yet, start showing.
        """
        was_selecting = self.selector is not None and self.selector.active
        if was_selecting:
            self.selector.stop()
        if self.bridge.active:
            self.bridge.set_crop(crop)
        elif not silent:
            axis, chan = self._crop_source()
            self.bridge.show(self.path, self.vt.series_index, self.vt.level,
                             axis, chan, parent=self, crop=crop)
        if was_selecting and self._crop_dialog is not None:
            # Coordinates just moved under it; redraw against the new extent.
            self.selector.start()
        self._update_show_state()

    def apply_crop(self, crop):
        """Make the crop the working image for every operation."""
        self.crop = crop
        if self.bridge.active:
            self.bridge.set_crop(crop)
        self._update_estimate()
        self._update_show_state()
        d, h, w = self.uncropped_extent()
        self.status.setText(
            f"Crop applied: {crop_shape(d, (h, w), crop)} "
            f"(z {crop[0]}:{crop[1]}, y {crop[2]}:{crop[3]}, "
            f"x {crop[4]}:{crop[5]})")

    def save_crop(self, crop, parent=None):
        outpath, _ = QFileDialog.getSaveFileName(
            parent or self, "Save cropped TIFF as", "",
            "TIFF Files (*.tif *.tiff)")
        if not outpath:
            return False
        if not outpath.lower().endswith(('.tif', '.tiff')):
            outpath += '.tif'
        if os.path.abspath(outpath) == os.path.abspath(self.path):
            QMessageBox.warning(parent or self, "Cannot Overwrite Source",
                                "Choose a different destination.")
            return False
        axis, chan = self._crop_source()
        self._start(_CropSaveWorker({
            'path': self.path,
            'series': self.vt.series_index,
            'level': self.vt.level,
            'crop': crop,
            'axis': axis,
            'channel': chan,
            'outpath': outpath,
            'compression': 'zlib' if self.compress_check.isChecked() else None,
        }), done_msg="Crop written")
        return True

    # ── viewer bridge ─────────────────────────────────────────────────

    def _open_show_options(self):
        if not self.survey:
            return
        if self._show_dialog is None:
            self._show_dialog = ShowOptionsDialog(self, parent=self)
            self._show_dialog.finished.connect(self._on_show_dialog_closed)
        else:
            self._show_dialog._reload_sources()
        self._show_dialog.show()
        self._show_dialog.raise_()

    def _on_show_dialog_closed(self, _result):
        # Closing the options popout does not stop the display; the overlay
        # only goes away on Hide, or when this whole window closes.
        pass

    def _update_show_state(self):
        b = self.bridge
        if b.active and b.vt is not None:
            tag = f"S{b.vt.series_index}"
            if b.vt.level:
                tag += f" L{b.vt.level}"
            if b.channel is not None:
                tag += f" ch{b.channel}"
            extra = " (scratch)" if b.scratch is not None else ""
            if b.crop is not None:
                extra += f" cropped {crop_shape(0, b.vt.plane_shape, b.crop)}"
            self.showing_label.setText(f"Showing: {tag}{extra}")
        else:
            self.showing_label.setText("")

    # ── lifecycle ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)

        # Purge the viewer overlay before anything else. The host keeps no
        # reference to it, so this is the only thing that removes the image
        # item, unhooks the display callback, restores the slider, closes the
        # memmap and deletes the scratch file.
        try:
            self.stop_region_select()
        except Exception:
            traceback.print_exc()
        try:
            self.bridge.purge()
        except Exception:
            traceback.print_exc()

        if self._crop_dialog is not None:
            try:
                self._crop_dialog.close()
            except Exception:
                pass
            self._crop_dialog = None

        if self._show_dialog is not None:
            try:
                self._show_dialog.close()
            except Exception:
                pass
            self._show_dialog = None

        if self.vt is not None:
            self.vt.close()
            self.vt = None
        try:
            if self in _open_instances:
                _open_instances.remove(self)
        except Exception:
            pass
        super().closeEvent(event)
