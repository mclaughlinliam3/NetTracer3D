"""
napari_viewer_widget.py

Interactive 3D viewer widget using napari, linked to the NetTracer3D main window.
Supports clicking to select 3D objects (nodes/edges), highlighting selections,
and bidirectional synchronization with the 2D main window.

Usage:
    from . import napari_viewer_widget as nvw
    viewer_widget = nvw.NapariViewerWidget(parent=main_window)
    viewer_widget.launch(arrays_3d, colors, scale, ...)
"""

import numpy as np
from functools import partial

try:
    import os
    os.environ['QT_API'] = 'pyqt6'
    import napari
    from napari.utils.notifications import show_info
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False

from PyQt6.QtCore import (
    QObject, QEvent, QTimer, Qt, QRunnable, QThreadPool, pyqtSignal,
)

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ======================================================================
# Numba-accelerated highlight kernel (optional)
# ======================================================================

# ----------------------------------------------------------------------
# Label lookup table
# ----------------------------------------------------------------------
#
# The kernels below test membership with a single array lookup rather than
# scanning the selected-label list per voxel.  Cost becomes O(voxels)
# instead of O(voxels x labels), so highlighting 300 objects costs the
# same per voxel as highlighting one.
#
# Layout: ``lut[v] == 1`` iff label ``v`` is selected.  The table is
# allocated with two spare slots so that out-of-range label values can be
# clipped onto a guaranteed-zero sentinel at the end, and index 0
# (background) is always zero.

# Refuse to build a table larger than this (bytes); falls back to the
# older per-label scan.  Only reachable with absurdly sparse label values.
_MAX_LUT_ENTRIES = 100_000_000


def _build_lut(labels):
    """Return a uint8 lookup table for *labels*, or None if unusable."""
    ints = [int(l) for l in labels if int(l) > 0]
    if not ints:
        return None
    n = max(ints) + 2          # +1 for inclusive, +1 for the clip sentinel
    if n > _MAX_LUT_ENTRIES:
        return None
    lut = np.zeros(n, dtype=np.uint8)
    lut[ints] = 1
    return lut


if HAS_NUMBA:
    # NOTE: ``nogil=True`` matters — the full-volume rebuild runs on a
    # worker thread, and without it the kernel would hold the GIL for its
    # whole duration and stall the Qt GUI thread anyway.
    @njit(parallel=True, cache=True, nogil=True)
    def _numba_highlight_lut(highlight, label_data, lut,
                             z0, z1, y0, y1, x0, x1):
        """Write 255 into *highlight* wherever ``lut[label_data] == 1``,
        within the bounding box [z0:z1, y0:y1, x0:x1]."""
        n = lut.shape[0]
        for z in prange(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = int(label_data[z, y, x])
                    if v > 0 and v < n and lut[v] == 1:
                        highlight[z, y, x] = 255

    @njit(parallel=True, cache=True, nogil=True)
    def _numba_clear_lut(highlight, label_data, lut,
                         z0, z1, y0, y1, x0, x1):
        """Clear (set to 0) voxels whose label is flagged in *lut*."""
        n = lut.shape[0]
        for z in prange(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = label_data[z, y, x]
                    if v > 0 and v < n and lut[v] == 1:
                        highlight[z, y, x] = 0
else:
    _numba_highlight_lut = None
    _numba_clear_lut = None


def _compute_bbox_dict(label_data):
    """Return {label: (z0,z1,y0,y1,x0,x1)} for every non-zero label
    using scipy.ndimage.find_objects (single pass)."""
    from scipy.ndimage import find_objects
    slices = find_objects(label_data)
    bboxes = {}
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        label = i + 1  # find_objects is 1-indexed
        bboxes[label] = (
            sl[0].start, sl[0].stop,
            sl[1].start, sl[1].stop,
            sl[2].start, sl[2].stop,
        )
    return bboxes


def _merge_bboxes(bb_list):
    """Return a single (z0,z1,y0,y1,x0,x1) enclosing all bboxes."""
    z0 = min(b[0] for b in bb_list)
    z1 = max(b[1] for b in bb_list)
    y0 = min(b[2] for b in bb_list)
    y1 = max(b[3] for b in bb_list)
    x0 = min(b[4] for b in bb_list)
    x1 = max(b[5] for b in bb_list)
    return (z0, z1, y0, y1, x0, x1)


def _bbox_volume(bb):
    return (bb[1]-bb[0]) * (bb[3]-bb[2]) * (bb[5]-bb[4])


# ======================================================================
# Background worker plumbing
# ======================================================================

class _WorkerSignals(QObject):
    """Signal carrier for :class:`_FunctionWorker`.

    Created on the GUI thread, so Qt delivers these with a queued
    connection: ``emit`` is called from the pool thread but the slot runs
    on the GUI thread.
    """

    finished = pyqtSignal(object, int)   # (result, generation)
    failed = pyqtSignal(object, int)     # (exception, generation)


class _FunctionWorker(QRunnable):
    """Run ``fn(*args)`` on a QThreadPool thread and report back.

    *generation* is echoed back so the receiver can discard results from a
    viewer session that has since been closed or superseded.
    """

    def __init__(self, fn, signals, generation, *args):
        super().__init__()
        self._fn = fn
        self._signals = signals
        self._generation = int(generation)
        self._args = args
        self.setAutoDelete(True)

    def run(self):
        try:
            result = self._fn(*self._args)
        except BaseException as exc:  # noqa: BLE001 - must not kill the pool
            try:
                self._signals.failed.emit(exc, self._generation)
            except RuntimeError:
                pass  # receiver already destroyed
            return
        try:
            self._signals.finished.emit(result, self._generation)
        except RuntimeError:
            pass


# ======================================================================
# Qt event filter — intercepts right-clicks before napari/vispy see them
# ======================================================================

class _RightClickFilter(QObject):
    """Installed on the napari canvas widget in Select mode.

    """

    def __init__(self, napari_widget):
        super().__init__()
        self._nw = napari_widget

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton:
                # Show context menu on next event-loop tick
                QTimer.singleShot(0, self._nw._show_context_menu_at_cursor)
                return True  # consume — napari/vispy never see it
        # Also swallow the matching release so nothing gets confused
        if event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.RightButton:
                return True
        return False  # everything else passes through


# ======================================================================
# Docked control panel widget
# ======================================================================

class _SelectionControlWidget:
    """
    A small docked Qt widget that shows the current interaction mode
    (Navigate vs Select) with a large toggle button, plus an active-
    channel selector.  Lives inside the napari window so the user always
    knows which mode they are in.
    """

    def __init__(self, napari_widget):
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
            QGroupBox, QHBoxLayout,
        )
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont

        self._nw = napari_widget  # back-reference to NapariViewerWidget

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Mode toggle button ---
        self.mode_btn = QPushButton("NAVIGATE MODE")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setChecked(False)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.mode_btn.setFont(font)
        self.mode_btn.setMinimumHeight(40)
        self._style_navigate()
        self.mode_btn.clicked.connect(self._on_toggle)
        layout.addWidget(self.mode_btn)

        # --- Shortcut hint ---
        hint = QLabel("Press  S  to toggle mode")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(hint)

        # --- Active channel selector ---
        ch_group = QGroupBox("Click targets")
        ch_layout = QHBoxLayout(ch_group)
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Nodes (Ch 0)", "Edges (Ch 1)"])
        self.channel_combo.currentIndexChanged.connect(self._on_channel_change)
        ch_layout.addWidget(self.channel_combo)
        layout.addWidget(ch_group)

        # --- Bounding box acceleration ---
        bbox_group = QGroupBox("Highlight Acceleration")
        bbox_layout = QVBoxLayout(bbox_group)

        self.bbox_btn = QPushButton("Compute Bounding Boxes")
        self.bbox_btn.setToolTip(
            "Pre-compute per-label bounding boxes for the currently\n"
            "selected channel.  Dramatically speeds up highlighting\n"
            "on large volumes."
        )
        self.bbox_btn.clicked.connect(self._on_compute_bboxes)
        bbox_layout.addWidget(self.bbox_btn)

        self.bbox_node_label = QLabel("Nodes: not computed")
        self.bbox_node_label.setStyleSheet("font-size: 10px; color: #e88;")
        bbox_layout.addWidget(self.bbox_node_label)

        self.bbox_edge_label = QLabel("Edges: not computed")
        self.bbox_edge_label.setStyleSheet("font-size: 10px; color: #e88;")
        bbox_layout.addWidget(self.bbox_edge_label)

        layout.addWidget(bbox_group)

        # --- Selection info ---
        self.info_label = QLabel("No selection")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 10px; color: #aaa;")
        #layout.addWidget(self.info_label)

        # --- Busy indicator (this window only) ---
        self.busy_label = QLabel("")
        self.busy_label.setWordWrap(True)
        self.busy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.busy_label.setStyleSheet(
            "font-size: 10px; color: #ec9; font-weight: bold;")
        self.busy_label.setVisible(False)
        layout.addWidget(self.busy_label)

        layout.addStretch()

    # -- styling helpers --------------------------------------------------

    def _style_navigate(self):
        self.mode_btn.setStyleSheet(
            "QPushButton { background-color: #2a5a2a; color: white; "
            "border: 2px solid #3a7a3a; border-radius: 6px; }"
            "QPushButton:hover { background-color: #3a7a3a; }"
        )
        self.mode_btn.setText("\U0001F9ED  NAVIGATE MODE")

    def _style_select(self):
        self.mode_btn.setStyleSheet(
            "QPushButton { background-color: #6a2a2a; color: white; "
            "border: 2px solid #9a4a4a; border-radius: 6px; }"
            "QPushButton:hover { background-color: #8a3a3a; }"
        )
        self.mode_btn.setText("\U0001F3AF  SELECT MODE")

    # -- callbacks --------------------------------------------------------

    def _on_toggle(self):
        select = self.mode_btn.isChecked()
        self._nw._set_select_mode(select)
        if select:
            self._style_select()
        else:
            self._style_navigate()

    def set_mode_visual(self, select_mode):
        """Update button appearance without re-triggering the callback."""
        self.mode_btn.blockSignals(True)
        self.mode_btn.setChecked(select_mode)
        self.mode_btn.blockSignals(False)
        if select_mode:
            self._style_select()
        else:
            self._style_navigate()

    def _on_channel_change(self, idx):
        if self._nw.parent and hasattr(self._nw.parent, "set_active_channel"):
            self._nw.parent.set_active_channel(idx)

    def set_busy(self, busy, message="Updating highlight…"):
        """Put *this* panel into a busy state.  The parent application
        keeps running; the work is on a background thread."""
        self.busy_label.setText(message if busy else "")
        self.busy_label.setVisible(bool(busy))
        self.mode_btn.setEnabled(not busy)
        self.channel_combo.setEnabled(not busy)
        self.bbox_btn.setEnabled(not busy)

    def _on_compute_bboxes(self):
        """Compute bounding boxes for the channel currently selected in
        the combo box.  Runs on a background thread."""
        idx = self.channel_combo.currentIndex()
        self.bbox_btn.setEnabled(False)
        self.bbox_btn.setText("Computing…")
        self._nw._compute_bboxes_for_channel(idx)

    def update_bbox_status(self):
        """Refresh the bbox status labels from the parent widget state."""
        if self._nw._node_bboxes is not None:
            n = len(self._nw._node_bboxes)
            self.bbox_node_label.setText(f"Nodes: {n} labels indexed ✓")
            self.bbox_node_label.setStyleSheet("font-size: 10px; color: #8e8;")
        else:
            self.bbox_node_label.setText("Nodes: not computed")
            self.bbox_node_label.setStyleSheet("font-size: 10px; color: #e88;")

        if self._nw._edge_bboxes is not None:
            n = len(self._nw._edge_bboxes)
            self.bbox_edge_label.setText(f"Edges: {n} labels indexed ✓")
            self.bbox_edge_label.setStyleSheet("font-size: 10px; color: #8e8;")
        else:
            self.bbox_edge_label.setText("Edges: not computed")
            self.bbox_edge_label.setStyleSheet("font-size: 10px; color: #e88;")

        self.bbox_btn.setEnabled(True)
        self.bbox_btn.setText("Compute Bounding Boxes")

    def update_info(self, nodes, edges):
        parts = []
        if nodes:
            parts.append(f"Nodes: {nodes}")
        if edges:
            parts.append(f"Edges: {edges}")
        self.info_label.setText("\n".join(parts) if parts else "No selection")


# ======================================================================
# Main widget
# ======================================================================

class NapariViewerWidget:
    """
    Interactive 3D napari viewer linked to a NetTracer3D ImageViewerWindow.

    Follows the same parent-callback pattern as NetworkGraphWidget /
    UMAPGraphWidget:
    - parent.clicked_values is updated directly
    - parent.evaluate_mini() triggers 2D highlight
    - parent.highlight_in_subgraphs() syncs sibling widgets
    - parent.handle_info() updates the info panel
    - select_nodes(indices) receives highlight updates from siblings

    Attributes:
        rendered (bool): Whether the viewer is currently open.
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.viewer = None
        self.rendered = False

        # Raw label arrays for value lookup (ZYX)
        self._node_data = None
        self._edge_data = None
        self._scale = [1, 1, 1]

        # Napari layer references
        self._highlight_layer = None
        self._image_layers = []

        # Current selection (mirrors parent.clicked_values)
        self._selected_nodes = []
        self._selected_edges = []

        # Re-entrancy guard
        self._updating = False

        # Interaction mode
        self._select_mode = False

        # Docked control panel
        self._control = None

        # Right-click event filter (installed on canvas in select mode)
        self._right_click_filter = None
        self._canvas_native = None

        # Labels layers (add_labels path, optional)
        self._node_labels_layer = None
        self._edge_labels_layer = None

        # Bounding-box acceleration for highlight
        self._node_bboxes = None   # dict {label: (z0,z1,y0,y1,x0,x1)} or None
        self._edge_bboxes = None
        self._prev_selected_nodes = []
        self._prev_selected_edges = []

        # ---- Background work -----------------------------------------
        # Only the *full-volume rebuild* runs off-thread; that is the slow
        # default path that used to freeze the parent window.  The
        # bounding-box incremental path stays synchronous — it is fast by
        # construction, and keeping it on the GUI thread avoids the
        # double-buffering complexity that threading it would require.
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)

        # The array the layer currently displays, plus a spare the worker
        # paints into.  They swap on completion, so the worker never
        # writes to an array the renderer is reading.
        self._hl_data = None
        self._hl_scratch = None

        # True while the highlight layer is showing an externally supplied
        # image — adopted from the parent at launch, or pushed in by an
        # upstream caller — rather than a rendering of the selection.
        # Such an image bears no relation to any label, so no bounding box
        # can clear it: the next real selection change has to wipe and
        # repaint the whole volume instead of patching it incrementally.
        self._hl_foreign = False

        # The contrast limits napari chose for the empty highlight layer.
        # A foreign image may widen them; this is what gets put back when
        # the selection takes the layer over again.
        self._hl_limits_default = None

        # Sampling applied at launch.  Foreign images arrive at the
        # parent's full resolution, so this is what puts them on the same
        # grid as the volume already on screen.
        self._down_factor = None
        self._down_order = 0

        self._hl_signals = _WorkerSignals()
        self._hl_signals.finished.connect(self._on_rebuild_done)
        self._hl_signals.failed.connect(self._on_rebuild_failed)

        self._bbox_signals = _WorkerSignals()
        self._bbox_signals.finished.connect(self._on_bboxes_done)
        self._bbox_signals.failed.connect(self._on_bboxes_failed)

        self._hl_running = False
        self._hl_pending = False    # a request arrived while busy
        self._busy = False
        self._bbox_running = False

        # Bumped whenever the session is invalidated; in-flight results
        # carrying a stale generation are dropped.
        self._generation = 0

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(
        self,
        arrays_3d=None,
        arrays_4d=None,
        down_factor=None,
        order=0,
        xy_scale=1,
        z_scale=1,
        colors=None,
        box=False,
        node_data=None,
        edge_data=None,
        names_3d=None,
        names_4d=None,
        nodes_as_labels=False,
        edges_as_labels=False,
    ):
        """
        Open the napari viewer with the provided data.

        Args:
            arrays_3d: List of 3D arrays to display as image layers.
            arrays_4d: List of 4D (RGB/RGBA) arrays.  Each is split into
                       separate R / G / B image layers (napari workaround).
            down_factor: Optional downsample factor.
            order: Interpolation order for downsampling (0 = nearest).
            xy_scale, z_scale: Physical voxel scaling.
            colors: Colormaps for each *arrays_3d* entry.
            box: Whether to show a bounding box.
            node_data: Raw node label array (channel_data[0]).
            edge_data: Raw edge label array (channel_data[1]).
            names_3d: Display names for each *arrays_3d* entry
                      (e.g. ["Nodes", "Edges", "Highlight"]).
            names_4d: Display names for each *arrays_4d* entry
                      (e.g. ["Overlay 1", "Overlay 2"]).
            nodes_as_labels: If True, add *node_data* as a napari Labels
                layer (solid isosurface rendering, per-label colours,
                paint/erase tools) instead of only using it for picking.
            edges_as_labels: If True, add *edge_data* as a napari Labels
                layer with the same benefits as *nodes_as_labels*.
        """
        if not HAS_NAPARI:
            raise ImportError(
                "napari is not installed. Install with: pip install napari[all]"
            )

        # Remember the sampling so highlight images handed to us later
        # can be matched to the grid the viewer actually renders on.
        self._down_factor = down_factor
        self._down_order = order
        self._hl_foreign = False
        self._hl_limits_default = None

        if colors is None:
            colors = ["red", "green", "white", "cyan", "yellow"]

        # Compute scale & downsample
        if down_factor is not None:
            from nettracer3d.nettracer import downsample
            arrays_3d = (
                [downsample(a, down_factor, order=order) for a in arrays_3d]
                if arrays_3d else []
            )
            arrays_4d = (
                [downsample(a, down_factor, order=order) for a in arrays_4d]
                if arrays_4d else []
            )
            if node_data is not None:
                node_data = downsample(node_data, down_factor, order=0)
            if edge_data is not None:
                edge_data = downsample(edge_data, down_factor, order=0)
            self._scale = [
                z_scale * down_factor,
                xy_scale * down_factor,
                xy_scale * down_factor,
            ]
        else:
            self._scale = [z_scale, xy_scale, xy_scale]
            arrays_3d = arrays_3d if arrays_3d else []
            arrays_4d = arrays_4d if arrays_4d else []

        self._node_data = node_data
        self._edge_data = edge_data

        # ---- Create viewer ----
        self.viewer = napari.Viewer(
            ndisplay=3, title="NetTracer3D - Interactive 3D Viewer"
        )
        self._image_layers = []
        shape = None

        # ---- Determine which parent channels are present in arrays_3d ----
        # Parent channel indices: 0 = nodes, 1 = edges.  A channel is
        # absent from arrays_3d when its data is None or its visibility
        # is False.  We need to map from "nodes" / "edges" to the actual
        # index inside arrays_3d so we can skip the right image layer
        # when a labels flag is set.
        _node_present = True
        _edge_present = True

        if self.parent is not None:
            ch_data = getattr(self.parent, "channel_data", None)
            ch_vis  = getattr(self.parent, "channel_visible", None)
            if ch_data is not None:
                if len(ch_data) > 0 and ch_data[0] is None:
                    _node_present = False
                if len(ch_data) > 1 and ch_data[1] is None:
                    _edge_present = False
            if ch_vis is not None:
                if len(ch_vis) > 0 and not ch_vis[0]:
                    _node_present = False
                if len(ch_vis) > 1 and not ch_vis[1]:
                    _edge_present = False

        # Build a mapping from role → arrays_3d index.  Only present
        # channels occupy slots; the rest are packed in order after them.
        _node_arr_idx = None
        _edge_arr_idx = None
        _slot = 0
        if _node_present:
            _node_arr_idx = _slot
            _slot += 1
        if _edge_present:
            _edge_arr_idx = _slot
            _slot += 1

        # Decide which arrays_3d indices to skip (replaced by labels).
        _skip_image = set()
        if nodes_as_labels and _node_present and _node_arr_idx is not None:
            _skip_image.add(_node_arr_idx)
        if edges_as_labels and _edge_present and _edge_arr_idx is not None:
            _skip_image.add(_edge_arr_idx)

        # ---- Add 3D image layers ----
        for i, (arr, color) in enumerate(zip(arrays_3d, colors)):
            shape = arr.shape
            if i in _skip_image:
                continue
            if names_3d and i < len(names_3d):
                name = names_3d[i]
            else:
                name = f"Channel {i}"
            layer = self.viewer.add_image(
                arr,
                scale=self._scale,
                colormap=color,
                rendering="mip",
                blending="additive",
                opacity=0.5,
                name=name,
            )
            self._image_layers.append(layer)

        # ---- Add 4D (RGB/RGBA) arrays, split into R/G/B channels ----
        rgb_colormaps = ["red", "green", "blue"]
        rgb_labels = ["R", "G", "B"]
        if arrays_4d:
            if names_4d is None:
                names_4d = [f"Overlay {j+1}" for j in range(len(arrays_4d))]
            for j, arr in enumerate(arrays_4d):
                if arr.shape[-1] not in [3, 4]:
                    print(f"Warning: {names_4d[j]} is not RGB/RGBA, skipping.")
                    continue
                if arr.shape[-1] == 4:
                    arr = arr[:, :, :, :3]
                shape = arr.shape[:3]
                base_name = names_4d[j] if j < len(names_4d) else f"Overlay {j+1}"
                for c in range(3):
                    layer = self.viewer.add_image(
                        arr[:, :, :, c],
                        scale=self._scale,
                        colormap=rgb_colormaps[c],
                        rendering="mip",
                        blending="additive",
                        opacity=0.5,
                        name=f"{base_name} ({rgb_labels[c]})",
                    )
                    self._image_layers.append(layer)

        # Bounding box
        if box and shape is not None:
            bbox = self._generate_bounding_box(shape)
            self.viewer.add_image(
                bbox,
                scale=self._scale,
                colormap="white",
                rendering="mip",
                blending="additive",
                opacity=0.5,
                name="Bounding Box",
            )

        # ---- Optional Labels layers (solid isosurface rendering) ----
        self._node_labels_layer = None
        self._edge_labels_layer = None

        if nodes_as_labels and _node_present and node_data is not None:
            lbl = node_data if np.issubdtype(node_data.dtype, np.integer) \
                  else node_data.astype(np.int32)
            node_lbl_name = (names_3d[_node_arr_idx]
                             if names_3d and _node_arr_idx < len(names_3d)
                             else "Node Labels")
            self._node_labels_layer = self.viewer.add_labels(
                lbl,
                scale=self._scale,
                name=node_lbl_name,
                opacity=0.7,
            )

        if edges_as_labels and _edge_present and edge_data is not None:
            lbl = edge_data if np.issubdtype(edge_data.dtype, np.integer) \
                  else edge_data.astype(np.int32)
            edge_lbl_name = (names_3d[_edge_arr_idx]
                             if names_3d and _edge_arr_idx < len(names_3d)
                             else "Edge Labels")
            self._edge_labels_layer = self.viewer.add_labels(
                lbl,
                scale=self._scale,
                name=edge_lbl_name,
                opacity=0.7,
            )

        # Highlight layer
        if shape is None and self._node_data is not None:
            shape = self._node_data.shape
        if shape is None and self._edge_data is not None:
            shape = self._edge_data.shape
        if shape is not None:
            self._hl_data = np.zeros(shape, dtype=np.uint8)
            self._hl_scratch = None      # allocated lazily on first rebuild
            self._highlight_layer = self.viewer.add_image(
                self._hl_data,
                scale=self._scale,
                colormap="yellow",
                rendering="mip",
                blending="additive",
                opacity=0.7,
                name="Selection Highlight",
            )
            # Napari fixes the contrast limits from the empty volume it
            # was handed.  Keep them so they can be restored after a
            # foreign image widens them.
            try:
                self._hl_limits_default = tuple(
                    self._highlight_layer.contrast_limits
                )
            except Exception:
                self._hl_limits_default = None

        # ---- Mouse callback (only added when entering Select mode) ----
        # Do NOT append here — _set_select_mode will add it when needed.

        # ---- Grab canvas widget for event filter ----
        try:
            self._canvas_native = self.viewer.window._qt_viewer.canvas.native
            self._right_click_filter = _RightClickFilter(self)
        except Exception:
            self._canvas_native = None
            self._right_click_filter = None

        # ---- Keybind: S to toggle select/navigate ----
        @self.viewer.bind_key("s")
        def _toggle_mode(viewer):
            self._set_select_mode(not self._select_mode)

        # ---- Navigate mode by default (camera interactive) ----
        self._select_mode = False
        self._set_camera_interactive(True)
        self.viewer.status = (
            "Navigate mode \u2014 rotate / pan freely.  Press S for Select mode."
        )

        # ---- Dock the control panel ----
        self._control = _SelectionControlWidget(self)
        self.viewer.window.add_dock_widget(
            self._control.widget,
            name="Current Mode:",
            area="right",
        )

        # ---- Handle viewer close ----
        self.viewer.window._qt_window.destroyed.connect(self._on_close)
        self.rendered = True

        # ---- Initial highlight contents ----
        # If the parent is already holding a highlight overlay, adopt it
        # verbatim, whatever it depicts.  It may well have nothing to do
        # with the current selection — that is the point: what the user
        # was looking at before opening this viewer is what they should
        # still be looking at.  It stays until the selection actually
        # changes, at which point the normal render takes the layer back.
        adopted = self.adopt_parent_highlight()

        # No parent overlay — fall back to rendering the existing
        # selection, as before.
        if not adopted and self.parent is not None:
            existing = self.parent.clicked_values.get("nodes", [])
            if existing:
                self.select_nodes(existing)

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _set_select_mode(self, select):
        """Switch between Select and Navigate modes.

        Instead of toggling camera.interactive (which is unreliable across
        napari / vispy versions), we add or remove our mouse callback
        entirely.  When removed, napari's default camera controls take
        over with zero interference.
        """
        self._select_mode = select

        if select:
            # Register our callback so clicks go to selection logic
            if self._on_click not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self._on_click)
            # Install right-click event filter on canvas
            if self._canvas_native and self._right_click_filter:
                self._canvas_native.installEventFilter(self._right_click_filter)
            # Disable camera so clicks don't rotate/zoom
            self._set_camera_interactive(False)
        else:
            # Remove our callback entirely — napari gets full control
            while self._on_click in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self._on_click)
            # Remove right-click event filter
            if self._canvas_native and self._right_click_filter:
                self._canvas_native.removeEventFilter(self._right_click_filter)
            # Re-enable camera
            self._set_camera_interactive(True)

        if self._control:
            self._control.set_mode_visual(select)

        if select:
            self.viewer.status = (
                "Select mode \u2014 click to pick objects.  "
                "Ctrl+click to add/remove.  Press S for Navigate."
            )
            try:
                show_info("Select mode: click objects to select; right click for menu")
            except Exception:
                pass
        else:
            self.viewer.status = (
                "Navigate mode \u2014 rotate / pan freely.  Press S for Select."
            )
            try:
                show_info("Navigate mode: rotate and pan")
            except Exception:
                pass

    def _set_camera_interactive(self, interactive):
        """Reliably toggle camera mouse interaction at both napari and
        vispy levels."""
        try:
            self.viewer.camera.interactive = interactive
        except Exception:
            pass
        # Also poke the underlying vispy camera directly \u2014 napari's
        # wrapper sometimes fails to propagate the flag.
        try:
            canvas = self.viewer.window._qt_viewer.canvas
            canvas.view.camera.interactive = interactive
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Mouse callbacks
    # ------------------------------------------------------------------

    def _on_click(self, viewer, event):
        """Handle left-clicks in Select mode.

        This callback is only registered when Select mode is active
        (removed entirely in Navigate mode).  Right-clicks are handled
        separately by a Qt event filter to avoid corrupting napari's
        mouse state.
        """
        # ---- Only handle left click ----
        if event.button != 1:
            return

        # ---- This viewer is busy; ignore selection input ----
        # Camera navigation still works; only picking is locked out.
        if self._busy:
            self.viewer.status = (
                "Highlight update in progress — selection is locked.")
            return

        if self._node_data is None and self._edge_data is None:
            return

        # ---- Ray-cast to find label under cursor ----
        label_value, sort = self._pick_label(viewer, event)

        # ---- Modifier logic ----
        ctrl = self._has_ctrl(event)

        if label_value == 0:
            # Clicked background
            if not ctrl:
                self._clear_selection()
            return

        if ctrl:
            # Toggle: deselect if already selected, else add
            if sort == "node":
                if label_value in self._selected_nodes:
                    self._selected_nodes.remove(label_value)
                    self.viewer.status = f"Deselected node {label_value}"
                else:
                    self._selected_nodes.append(label_value)
                    self.viewer.status = f"Added node {label_value}"
            else:
                if label_value in self._selected_edges:
                    self._selected_edges.remove(label_value)
                    self.viewer.status = f"Deselected edge {label_value}"
                else:
                    self._selected_edges.append(label_value)
                    self.viewer.status = f"Added edge {label_value}"
        else:
            # Replace selection
            self._selected_nodes = []
            self._selected_edges = []
            if sort == "node":
                self._selected_nodes = [label_value]
                self.viewer.status = f"Selected node {label_value}"
            else:
                self._selected_edges = [label_value]
                self.viewer.status = f"Selected edge {label_value}"

        self._update_highlight()
        self._push_to_parent(sort)

    # ------------------------------------------------------------------
    # 3D picking via ray marching
    # ------------------------------------------------------------------

    def _pick_label(self, viewer, event):
        """Cast a ray through the volume and return (label_value, sort).

        Uses the camera view direction and cursor position to march
        through the label array.  This is far more reliable than a
        single-point lookup because MIP rendering means the visible
        feature can be at *any* depth along the viewing ray.
        """
        active = getattr(self.parent, "active_channel", 0) if self.parent else 0

        if active == 0 and self._node_data is not None:
            val = self._ray_march(viewer, event, self._node_data)
            if val != 0:
                return val, "node"
        elif active == 1 and self._edge_data is not None:
            val = self._ray_march(viewer, event, self._edge_data)
            if val != 0:
                return val, "edge"
        else:
            # Fallback: try both
            if self._node_data is not None:
                val = self._ray_march(viewer, event, self._node_data)
                if val != 0:
                    return val, "node"
            if self._edge_data is not None:
                val = self._ray_march(viewer, event, self._edge_data)
                if val != 0:
                    return val, "edge"

        return 0, "node"

    def _ray_march(self, viewer, event, label_data):
        """March a ray through *label_data* and return the first non-zero
        value encountered, or 0 if nothing is hit."""
        pos = np.array(viewer.cursor.position, dtype=np.float64)
        view_dir = self._get_view_direction(viewer, event)

        scale = np.array(self._scale, dtype=np.float64)

        # World \u2192 data coordinates
        data_pos = pos / scale
        data_dir = view_dir / scale
        norm = np.linalg.norm(data_dir)
        if norm < 1e-12:
            return 0
        data_dir /= norm

        shape = label_data.shape
        diag = np.sqrt(sum(s * s for s in shape))
        step = 0.5  # sub-voxel stepping

        # March in both directions from the cursor entry point.
        # Direction 1 (into the volume) is most likely to hit, but we
        # also check the reverse in case the cursor landed on the far
        # face of the bounding box.
        for sign in (1.0, -1.0):
            inside = False
            t = 0.0
            while t <= diag:
                pt = data_pos + sign * t * data_dir
                iz = int(round(pt[0]))
                iy = int(round(pt[1]))
                ix = int(round(pt[2]))
                if 0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]:
                    inside = True
                    v = label_data[iz, iy, ix]
                    if v != 0:
                        return int(v)
                elif inside:
                    break  # exited the volume
                t += step

        return 0

    @staticmethod
    def _get_view_direction(viewer, event):
        """Return the camera's view direction as a numpy array in world
        coordinates (z, y, x)."""
        # Napari \u22650.4.18 attaches view_direction to the mouse event.
        if hasattr(event, "view_direction") and event.view_direction is not None:
            vd = np.array(event.view_direction, dtype=np.float64)
            if np.linalg.norm(vd) > 1e-12:
                return vd

        # Fallback: compute from camera angles (turntable convention)
        angles = viewer.camera.angles  # (azimuth, elevation, roll)
        az = np.radians(angles[0])
        el = np.radians(angles[1])
        # napari world order is (z, y, x)
        return np.array([
            -np.sin(el),
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
        ], dtype=np.float64)

    @staticmethod
    def _has_ctrl(event):
        """Check whether Ctrl is held, handling both napari event styles."""
        mods = getattr(event, "modifiers", None)
        if mods is None:
            return False
        if isinstance(mods, (list, tuple, set, frozenset)):
            return "Control" in mods
        # String or flags
        return "Control" in str(mods)

    # ------------------------------------------------------------------
    # Selection management
    # ------------------------------------------------------------------

    def select_nodes(self, node_indices, edge_indices = None):
        """Called by parent (highlight_in_subgraphs) when selection
        changes elsewhere.  Updates the 3D highlight layer."""
        if self._updating or not self.rendered:
            return
        self._selected_nodes = list(node_indices) if node_indices else []
        self._selected_edges = list(edge_indices) if edge_indices else []
        self._update_highlight()
        if self._control:
            self._control.update_info(self._selected_nodes, self._selected_edges)

    def select_edges(self, edge_indices):
        """Update edge selection from external source."""
        if self._updating or not self.rendered:
            return
        self._selected_edges = list(edge_indices) if edge_indices else []
        self._update_highlight()
        if self._control:
            self._control.update_info(self._selected_nodes, self._selected_edges)

    def take_new_highlight(self, highlight):
        """Adopt an externally supplied highlight volume, as-is.

        Unconditional and unvalidated — the caller is asserting that the
        array is already on the viewer's grid.  Use
        :meth:`render_highlight_image` for anything that needs coercing,
        or that should defer to a highlight already on screen.
        """
        self._install_highlight_volume(np.ascontiguousarray(highlight))

    # ------------------------------------------------------------------
    # Foreign highlight images
    # ------------------------------------------------------------------
    #
    # Everything below deals with images that did not come from the
    # selection: the parent's overlay at launch, or a volume handed over
    # by an upstream caller.  The layer is shared with the selection
    # renderer, so the two have to agree on who owns it — that is what
    # ``_hl_foreign`` tracks.

    def render_highlight_image(self, image, force=False, binarize=None):
        """Render an arbitrary image into the highlight layer.

        The polite version of :meth:`take_new_highlight`: it declines if a
        highlight is already on screen, so an upstream caller can offer an
        image without stamping on the user's current selection.

        Args:
            image: 3D volume, or a 4D RGB/RGBA volume (collapsed to its
                brightest channel).  Downsampled to match the viewer if
                the viewer was launched downsampled.
            force: Replace whatever is on the layer instead of declining.
            binarize: True renders every non-zero voxel at full
                brightness, False preserves relative intensity, None
                (default) decides per dtype — masks and label volumes are
                binarized, greyscale images keep their intensities.

        Returns:
            bool: True if the image was rendered.
        """
        if not self.rendered or self._highlight_layer is None:
            return False
        if not force and self.has_highlight():
            return False

        volume = self._prepare_highlight_volume(image, binarize=binarize)
        if volume is None:
            return False
        return self._install_highlight_volume(volume)

    def adopt_parent_highlight(self):
        """Copy the parent's highlight overlay onto the highlight layer.

        Called at launch.  The overlay is taken regardless of what is
        selected; only its absence falls through to the selection render.

        Returns:
            bool: True if an overlay was adopted.
        """
        if self.parent is None or self._highlight_layer is None:
            return False

        overlay = getattr(self.parent, "highlight_overlay", None)
        if overlay is None:
            return False

        volume = self._prepare_highlight_volume(overlay)
        if volume is None:
            return False

        # Mirror the parent's selection rather than blanking it.  The
        # parent tends to echo its selection back at us just after launch,
        # and that echo must not read as a change — otherwise the image we
        # just adopted would be wiped before anyone saw it.
        try:
            clicked = getattr(self.parent, "clicked_values", {}) or {}
            nodes = list(clicked.get("nodes", []) or [])
            edges = list(clicked.get("edges", []) or [])
        except Exception:
            nodes, edges = [], []

        return self._install_highlight_volume(
            volume, sync_selection=(nodes, edges)
        )

    def has_highlight(self):
        """Whether the highlight layer is currently showing anything."""
        if self._highlight_layer is None:
            return False
        if self._hl_foreign:
            return True
        if self._selected_nodes or self._selected_edges:
            return True
        if self._hl_running or self._hl_pending:
            return True

        data = self._hl_data
        if data is None:
            try:
                data = self._highlight_layer.data
            except Exception:
                return False
        try:
            return bool(np.any(data))
        except Exception:
            return False

    def clear_highlight(self):
        """Empty the highlight layer without touching the selection."""
        if self._hl_foreign:
            self._drop_foreign_highlight()
        elif self._hl_data is not None:
            try:
                self._hl_data[...] = 0
                self._prev_selected_nodes = []
                self._prev_selected_edges = []
                if self._highlight_layer is not None:
                    self._highlight_layer.refresh()
            except Exception:
                pass

    def _install_highlight_volume(self, volume, sync_selection=None):
        """Put *volume* on the highlight layer and mark it foreign.

        *sync_selection* is the (nodes, edges) the image is being shown
        alongside.  Recording it as both the current *and* the previous
        selection makes an echo of that same selection read as "nothing
        changed", so the image survives it, while any genuine change still
        triggers the normal repaint.  None declares the image unrelated to
        any selection and clears the selection instead.
        """
        if self._highlight_layer is None:
            return False

        nodes, edges = sync_selection if sync_selection else ([], [])
        self._selected_nodes = list(nodes)
        self._selected_edges = list(edges)
        self._prev_selected_nodes = list(nodes)
        self._prev_selected_edges = list(edges)

        # A rebuild still in flight describes the buffers we are about to
        # replace, so invalidate it: its result gets dropped rather than
        # swapped in over this image.
        if self._hl_running:
            self._generation += 1
            self._hl_running = False
        self._hl_pending = False

        self._hl_data = np.ascontiguousarray(volume)
        self._hl_scratch = None          # shape may have changed

        try:
            self._highlight_layer.data = self._hl_data
            self._widen_highlight_contrast()
            self._highlight_layer.refresh()
        except Exception as exc:
            print(f"Could not render highlight image: {exc}")
            return False

        self._hl_foreign = True
        if self._busy:
            self._set_busy(False)
        if self._control:
            self._control.update_info(self._selected_nodes,
                                      self._selected_edges)
        return True

    def _drop_foreign_highlight(self):
        """Wipe an externally supplied image off the highlight layer."""
        self._hl_foreign = False
        self._restore_highlight_contrast()
        if self._hl_data is not None:
            try:
                self._hl_data[...] = 0
            except Exception:
                pass
        # The layer now depicts an empty selection, which is what the
        # incremental painter has to believe in order to work from here.
        self._prev_selected_nodes = []
        self._prev_selected_edges = []
        if self._highlight_layer is not None:
            try:
                self._highlight_layer.refresh()
            except Exception:
                pass

    def _highlight_shape(self):
        """Shape of the grid the highlight layer renders on, or None."""
        if self._hl_data is not None:
            return tuple(self._hl_data.shape)
        if self._highlight_layer is not None:
            try:
                return tuple(self._highlight_layer.data.shape)
            except Exception:
                return None
        return None

    def _prepare_highlight_volume(self, image, binarize=None):
        """Coerce *image* into something the highlight layer can display.

        Three things separate an upstream overlay from what this layer
        renders: an RGB(A) trailing axis, the downsampling applied at
        launch, and a value range that does not line up with the 0-255 the
        selection painter writes.

        Returns None (having printed why) if the image cannot be put on
        the layer's grid — a missing highlight beats one that is silently
        misaligned with the volume underneath it.
        """
        if image is None:
            return None

        try:
            arr = np.asarray(image)
        except Exception as exc:
            print(f"Highlight image rejected — not array-like: {exc}")
            return None

        if arr.size == 0:
            return None

        # RGB/RGBA collapses to its brightest channel; the highlight layer
        # is single-channel with a colormap of its own.
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            arr = arr[..., :3].max(axis=-1)

        target = self._highlight_shape()

        # A lone 2D plane only makes sense against a single-slice volume.
        if (arr.ndim == 2 and target is not None
                and len(target) == 3 and target[0] == 1):
            arr = arr[None, ...]

        if arr.ndim != 3:
            print(f"Highlight image rejected — expected a 3D volume, got "
                  f"shape {arr.shape}.")
            return None

        if target is not None and tuple(arr.shape) != target:
            arr = self._resample_to_target(arr, target)
            if arr is None:
                return None

        return self._coerce_highlight_dtype(arr, binarize=binarize)

    def _resample_to_target(self, arr, target):
        """Put *arr* on *target*'s grid via the launch downsample, or
        give up.  Nearest-neighbour, so mask edges stay where they are."""
        if self._down_factor:
            try:
                from nettracer3d.nettracer import downsample
                arr = downsample(arr, self._down_factor, order=0)
            except Exception as exc:
                print(f"Highlight image rejected — downsample failed: {exc}")
                return None

        if tuple(arr.shape) != tuple(target):
            print(f"Highlight image rejected — shape {tuple(arr.shape)} "
                  f"does not match the viewer volume {tuple(target)}.")
            return None
        return arr

    @staticmethod
    def _coerce_highlight_dtype(arr, binarize=None):
        """Map *arr* onto the uint8 0-255 range the highlight layer uses.

        *binarize* True lights every non-zero voxel fully (right for a
        mask), False rescales onto 0-255 keeping relative intensity (right
        for a greyscale image), and None picks between them by dtype.
        """
        def as_mask(a):
            return np.ascontiguousarray((a != 0).astype(np.uint8) * 255)

        if binarize:
            return as_mask(arr)

        if arr.dtype == bool:
            return as_mask(arr)

        if np.issubdtype(arr.dtype, np.floating):
            with np.errstate(invalid="ignore"):
                peak = float(np.nanmax(arr))
            if not np.isfinite(peak):
                return as_mask(np.isfinite(arr) & (arr != 0))
            if peak <= 0:
                return np.zeros(arr.shape, np.uint8)
            # Probabilities and normalised masks live in [0, 1]; anything
            # wider is rescaled by its own peak.
            scale = 255.0 if peak <= 1.0 else 255.0 / peak
            out = np.clip(np.nan_to_num(arr) * scale, 0, 255)
            return np.ascontiguousarray(out.astype(np.uint8))

        peak = int(arr.max())
        if peak <= 1:
            return as_mask(arr)          # 0/1 mask stored as an integer

        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr)

        if binarize is None:
            # Wider-than-byte integers are label maps far more often than
            # they are brightnesses, and a label map rescaled by its
            # highest label renders almost entirely black.
            return as_mask(arr)

        out = np.clip(arr * (255.0 / peak), 0, 255)
        return np.ascontiguousarray(out.astype(np.uint8))

    def _widen_highlight_contrast(self):
        """Open the layer's contrast limits up to the image's range.

        They were fixed when the layer was built from an empty uint8
        volume, so an image with a wider range would render flat against
        them."""
        if self._highlight_layer is None or self._hl_data is None:
            return
        try:
            peak = float(self._hl_data.max())
        except Exception:
            return
        if peak <= 0:
            return
        try:
            lo, hi = self._highlight_layer.contrast_limits
            if peak > float(hi):
                self._highlight_layer.contrast_limits = (float(lo), peak)
        except Exception:
            pass

    def _restore_highlight_contrast(self):
        """Put back the contrast limits the selection render expects."""
        if self._highlight_layer is None or self._hl_limits_default is None:
            return
        try:
            self._highlight_layer.contrast_limits = self._hl_limits_default
        except Exception:
            pass

    # ------------------------------------------------------------------

    def _clear_selection(self):
        """Clear all selections, update highlight + parent."""
        self._selected_nodes = []
        self._selected_edges = []
        # An explicit clear means the user wants an empty layer, even when
        # what is on it came from outside the selection system and would
        # otherwise be left alone by _update_highlight.
        if self._hl_foreign:
            self._drop_foreign_highlight()
        self._update_highlight()
        if self._control:
            self._control.update_info([], [])
        if self.parent:
            self.parent.clicked_values["nodes"] = []
            self.parent.clicked_values["edges"] = []
            self.parent.evaluate_mini()

    def _push_to_parent(self, sort="node"):
        """Push current selection to the parent window and trigger its
        highlight / info pipeline."""
        if self.parent is None:
            return

        self._updating = True
        try:
            self.parent.clicked_values["nodes"] = list(self._selected_nodes)
            self.parent.clicked_values["edges"] = list(self._selected_edges)

            self.parent.evaluate_mini()

            try:
                if self._selected_nodes:
                    self.parent.create_table_node_selection(self._selected_nodes)
            except:
                pass

            if sort == "node" and self._selected_nodes:
                try:
                    self.parent.highlight_value_in_tables(self._selected_nodes[-1])
                    self.parent.handle_info("node")
                except Exception:
                    pass
            elif sort == "edge" and self._selected_edges:
                try:
                    self.parent.highlight_value_in_tables(self._selected_edges[-1])
                    self.parent.handle_info("edge")
                except Exception:
                    pass

            self.parent.highlight_in_subgraphs(self._selected_nodes)
            self._navigate_parent_to_selection(sort)

            if self._control:
                self._control.update_info(
                    self._selected_nodes, self._selected_edges
                )
        finally:
            self._updating = False

    def _navigate_parent_to_selection(self, sort):
        """Move the 2D slice slider to the last selected object's centroid."""
        try:
            import sys
            gui_module = sys.modules.get(self.parent.__class__.__module__)
            if gui_module is None:
                return
            my_network = getattr(gui_module, "my_network", None)
            if my_network is None:
                return

            if sort == "node" and self._selected_nodes:
                label = self._selected_nodes[-1]
                if (my_network.node_centroids is not None
                        and label in my_network.node_centroids):
                    self.parent.slice_slider.setValue(
                        int(my_network.node_centroids[label][0])
                    )
            elif sort == "edge" and self._selected_edges:
                label = self._selected_edges[-1]
                if (my_network.edge_centroids is not None
                        and label in my_network.edge_centroids):
                    self.parent.slice_slider.setValue(
                        int(my_network.edge_centroids[label][0])
                    )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Bounding-box computation
    # ------------------------------------------------------------------

    def _compute_bboxes_for_channel(self, channel_idx):
        """Kick off per-label bounding boxes for channel 0 (nodes) or 1
        (edges) on a background thread.  Returns immediately."""
        if self._bbox_running:
            return
        data = self._node_data if channel_idx == 0 else self._edge_data
        if data is None:
            if self._control:
                self._control.update_bbox_status()
            return

        self._bbox_running = True
        self._set_busy(True, "Computing bounding boxes…")
        self._pool.start(_FunctionWorker(
            self._bbox_compute, self._bbox_signals, self._generation,
            channel_idx, data,
        ))

    @staticmethod
    def _bbox_compute(channel_idx, data):
        """Worker-thread body.  Pure numpy/scipy, no Qt."""
        return channel_idx, _compute_bbox_dict(data)

    def _on_bboxes_done(self, result, generation):
        # The flag tracks whether a job is outstanding, so clear it even
        # when the result is stale — otherwise a generation bump (from
        # adopting a highlight image, say) would leave it stuck True and
        # block bbox computation for the rest of the session.
        self._bbox_running = False
        if generation != self._generation:
            return
        channel_idx, bboxes = result
        if channel_idx == 0:
            self._node_bboxes = bboxes
        else:
            self._edge_bboxes = bboxes
        if self._control:
            self._control.update_bbox_status()
        if not self._hl_running:
            self._set_busy(False)

    def _on_bboxes_failed(self, exc, generation):
        self._bbox_running = False
        if generation != self._generation:
            return
        print(f"Bounding box computation failed: {exc}")
        if self._control:
            self._control.update_bbox_status()
        if not self._hl_running:
            self._set_busy(False)

    # ------------------------------------------------------------------
    # Highlight rendering
    # ------------------------------------------------------------------

    def _update_highlight(self):
        """Bring the 3D highlight layer in line with the current selection.

        Two paths:

        * **Incremental** (bounding boxes available for every changed
          channel) — clears removed labels and paints added ones over
          their boxes.  Cheap, so it runs synchronously right here.
        * **Full rebuild** (a channel with no bbox index changed) — wipes
          and repaints the whole volume.  This is the slow default path,
          so it is handed to a worker thread; the parent window keeps
          running and only this viewer goes busy.
        """
        if self._highlight_layer is None:
            return

        # A rebuild is already in flight — record that the selection moved
        # again and re-evaluate when it lands.  Doing incremental work on
        # the live array now would be undone by the pending swap.
        if self._hl_running:
            self._hl_pending = True
            return

        prev_n = set(self._prev_selected_nodes)
        prev_e = set(self._prev_selected_edges)
        cur_n = set(self._selected_nodes)
        cur_e = set(self._selected_edges)

        nodes_changed = (prev_n != cur_n)
        edges_changed = (prev_e != cur_e)
        if not nodes_changed and not edges_changed:
            # Nothing moved.  This is also where a foreign image survives
            # the parent echoing the current selection back at us.
            return

        # The selection genuinely changed, so a foreign image on the layer
        # has been superseded.  It matches no label, so no bounding box
        # can clear it — only a full wipe and repaint gets rid of it.
        if self._hl_foreign:
            self._hl_foreign = False
            self._restore_highlight_contrast()
            self._start_full_rebuild()
            return

        have_node_bb = self._node_bboxes is not None
        have_edge_bb = self._edge_bboxes is not None

        # A channel without a bbox index can't be selectively cleared from
        # the shared highlight array, so any change to it forces a full
        # wipe + repaint.
        needs_full = ((nodes_changed and not have_node_bb
                       and self._node_data is not None)
                      or (edges_changed and not have_edge_bb
                          and self._edge_data is not None))

        if needs_full:
            self._start_full_rebuild()
            return

        # ---- Incremental, on the GUI thread ----
        highlight = self._hl_data
        if highlight is None:
            highlight = self._hl_data = self._highlight_layer.data

        self._incremental_update(
            highlight,
            self._node_data, self._node_bboxes,
            self._edge_data, self._edge_bboxes,
            list(prev_n - cur_n) if nodes_changed else [],
            list(cur_n - prev_n) if nodes_changed else [],
            list(prev_e - cur_e) if edges_changed else [],
            list(cur_e - prev_e) if edges_changed else [],
            list(cur_n), list(cur_e),
        )

        self._prev_selected_nodes = list(self._selected_nodes)
        self._prev_selected_edges = list(self._selected_edges)
        self._highlight_layer.refresh()

    # ------------------------------------------------------------------
    # Threaded full rebuild
    # ------------------------------------------------------------------

    def _start_full_rebuild(self):
        """Dispatch a whole-volume repaint to the worker pool."""
        base = self._hl_data
        if base is None:
            base = self._hl_data = self._highlight_layer.data
        if base is None:
            return

        scratch = self._hl_scratch
        if (scratch is None or scratch.shape != base.shape
                or scratch.dtype != base.dtype):
            scratch = self._hl_scratch = np.zeros(base.shape, base.dtype)

        # Snapshot the selection so later mutations can't reach the worker.
        target = (tuple(self._selected_nodes), tuple(self._selected_edges))

        self._hl_running = True
        self._hl_pending = False
        self._set_busy(True)

        self._pool.start(_FunctionWorker(
            self._rebuild_compute, self._hl_signals, self._generation,
            scratch, target,
        ))

    def _rebuild_compute(self, scratch, target):
        """Worker-thread body: repaint *scratch* from scratch.

        Touches only numpy arrays — never Qt widgets, never the napari
        layer.
        """
        nodes, edges = target
        node_data = self._node_data
        edge_data = self._edge_data
        node_bb = self._node_bboxes
        edge_bb = self._edge_bboxes

        scratch[:] = 0

        if node_data is not None and nodes:
            lut = _build_lut(nodes)
            if node_bb:
                self._paint_all_selected(scratch, node_data, node_bb,
                                         list(nodes), lut)
            else:
                self._paint_fullvol(scratch, node_data, lut)

        if edge_data is not None and edges:
            lut = _build_lut(edges)
            if edge_bb:
                self._paint_all_selected(scratch, edge_data, edge_bb,
                                         list(edges), lut)
            else:
                self._paint_fullvol(scratch, edge_data, lut)

        return target

    def _on_rebuild_done(self, target, generation):
        """Swap the freshly painted volume in.  Runs on the GUI thread."""
        if generation != self._generation:
            return  # stale session

        self._hl_running = False

        # Swap: the scratch we just painted becomes the displayed array,
        # and the old displayed array becomes the next scratch.
        self._hl_scratch, self._hl_data = self._hl_data, self._hl_scratch
        self._prev_selected_nodes = list(target[0])
        self._prev_selected_edges = list(target[1])

        if self._highlight_layer is not None:
            try:
                self._highlight_layer.data = self._hl_data
                self._highlight_layer.refresh()
            except Exception:
                pass

        # The selection may have moved while we worked.  Re-evaluating can
        # legitimately start nothing (e.g. the parent echoed the same
        # selection back, so there is no longer any difference to paint),
        # so clear the busy state on the flag rather than in an else
        # branch — otherwise the viewer stays locked forever.
        if self._hl_pending:
            self._hl_pending = False
            self._update_highlight()

        if not self._hl_running and not self._bbox_running:
            self._set_busy(False)

    def _on_rebuild_failed(self, exc, generation):
        if generation != self._generation:
            return
        self._hl_running = False
        self._hl_pending = False
        # The scratch buffer is half-painted; force a clean rebuild next
        # time by making the recorded previous selection impossible to
        # match incrementally.
        self._prev_selected_nodes = []
        self._prev_selected_edges = []
        print(f"Highlight rebuild failed: {exc}")
        self._set_busy(False)

    def _set_busy(self, busy, message="Updating highlight…"):
        """Freeze *this* viewer's controls only; the parent stays live."""
        self._busy = bool(busy)
        try:
            if self._control is not None:
                self._control.set_busy(busy, message)
            if self.viewer is not None:
                self.viewer.status = (
                    "Updating highlight — selection is locked until this "
                    "finishes.  You can still rotate the view."
                    if busy else "Highlight updated."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Incremental bbox-based highlight helpers
    # ------------------------------------------------------------------

    def _incremental_update(self, highlight,
                            node_data, node_bb, edge_data, edge_bb,
                            removed_nodes, added_nodes,
                            removed_edges, added_edges,
                            cur_nodes, cur_edges):
        """Bring *highlight* from the previous selection to the current one
        by touching only the affected bounding boxes.

        Both channels are handled together because the highlight volume is
        shared between them: clearing a removed label also erases voxels
        that a *still-selected* label in the other channel covers.  Every
        cleared box is therefore repaired afterwards by repainting the
        surviving selection over it.  Without that repair, deselecting an
        edge silently punches holes in the highlighted nodes.
        """
        cur_node_lut = _build_lut(cur_nodes) if cur_nodes else None
        cur_edge_lut = _build_lut(cur_edges) if cur_edges else None

        # --- Clear removed labels, remembering the boxes we touched ---
        repair_boxes = []

        for lbls, data, bbs in ((removed_nodes, node_data, node_bb),
                                (removed_edges, edge_data, edge_bb)):
            if not lbls or data is None or not bbs:
                continue
            lut = _build_lut(lbls)
            for lbl in lbls:
                bb = bbs.get(lbl)
                if bb is None:
                    continue
                self._clear_bbox(highlight, data, lut, bb)
                repair_boxes.append(bb)

        # --- Repair collateral damage inside the cleared boxes ---
        for bb in repair_boxes:
            if node_data is not None and cur_node_lut is not None:
                self._paint_bbox(highlight, node_data, cur_node_lut, bb)
            if edge_data is not None and cur_edge_lut is not None:
                self._paint_bbox(highlight, edge_data, cur_edge_lut, bb)

        # --- Paint newly added labels ---
        # The LUT covers the whole current selection; painting a
        # neighbouring selected label early is harmless since it would be
        # painted anyway.
        if added_nodes and node_data is not None and node_bb:
            self._paint_all_selected(highlight, node_data, node_bb,
                                     added_nodes, cur_node_lut)
        if added_edges and edge_data is not None and edge_bb:
            self._paint_all_selected(highlight, edge_data, edge_bb,
                                     added_edges, cur_edge_lut)

    def _paint_all_selected(self, highlight, label_data, bboxes, labels,
                            lut=None):
        """Paint *labels* into highlight over their bounding boxes, with
        overlap-aware grouping."""
        if lut is None:
            lut = _build_lut(labels)
        if lut is None:
            return

        lbl_bb_pairs = []
        for lbl in labels:
            bb = bboxes.get(lbl)
            if bb is not None:
                lbl_bb_pairs.append((lbl, bb))

        if not lbl_bb_pairs:
            return

        # One kernel call per box.  Boxes are NOT merged into superboxes.
        # Merging existed to make the old per-voxel label scan cheaper by
        # visiting one contiguous region instead of many; with the LUT the
        # per-voxel cost is O(1) regardless, so merging buys nothing while
        # the merge search itself is quadratic in the number of selected
        # labels.  Measured on 4000 small labels: 43 s with merging,
        # 12.6 ms without.
        for _lbl, bb in lbl_bb_pairs:
            self._paint_bbox(highlight, label_data, lut, bb)

    # Voxels per slab for the non-numba path.  Bounds the size of the
    # temporary index array so a full-volume pass doesn't allocate a
    # second copy of the whole label volume.
    _SLAB_VOXELS = 8_000_000

    # Above this many selected labels, the table lookup beats np.isin.
    _ISIN_MAX_LABELS = 64

    def _apply_lut(self, highlight, label_data, lut, bb, value):
        """Write *value* into *highlight* wherever ``lut[label_data] == 1``
        inside *bb*.  ``value`` is 255 to paint, 0 to clear."""
        if lut is None:
            return
        z0, z1, y0, y1, x0, x1 = bb
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return

        if HAS_NUMBA and _numba_highlight_lut is not None:
            kernel = (_numba_highlight_lut if value else _numba_clear_lut)
            kernel(highlight, label_data, lut, z0, z1, y0, y1, x0, x1)
            return

        # --- numpy fallback, processed in z-slabs ---
        # Slabbing bounds the temporary arrays; a full-volume pass would
        # otherwise allocate a second copy of the whole label volume.
        hi = lut.shape[0] - 1
        sel = np.flatnonzero(lut)
        # np.isin beats table lookup for a handful of labels (it compiles
        # to a few vectorised comparisons) but degrades as the count
        # grows, where the table's O(1) per voxel wins.  Threshold
        # measured at roughly break-even.
        use_isin = sel.size <= self._ISIN_MAX_LABELS

        plane = max(1, (y1 - y0) * (x1 - x0))
        step = max(1, self._SLAB_VOXELS // plane)
        for zs in range(z0, z1, step):
            ze = min(zs + step, z1)
            sub = label_data[zs:ze, y0:y1, x0:x1]
            if not np.issubdtype(sub.dtype, np.integer):
                sub = sub.astype(np.int64)
            if use_isin:
                mask = np.isin(sub, sel)
            else:
                # Out-of-range and negative values clip onto slots that
                # are guaranteed zero (index 0 is background, the last
                # slot is the spare sentinel), so clipping is safe.
                mask = lut[np.clip(sub, 0, hi)].astype(bool, copy=False)
            highlight[zs:ze, y0:y1, x0:x1][mask] = value

    def _paint_fullvol(self, highlight, label_data, lut):
        """Paint the whole volume from *lut* (no bounding boxes known)."""
        s = label_data.shape
        self._apply_lut(highlight, label_data, lut,
                        (0, s[0], 0, s[1], 0, s[2]), 255)

    def _paint_bbox(self, highlight, label_data, lut, bb):
        """Set highlight=255 for flagged voxels within bb."""
        self._apply_lut(highlight, label_data, lut, bb, 255)

    def _clear_bbox(self, highlight, label_data, lut, bb):
        """Set highlight=0 for flagged voxels within bb."""
        self._apply_lut(highlight, label_data, lut, bb, 0)

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------

    def _show_context_menu_at_cursor(self):
        """Called by the Qt event filter.  Shows the context menu at the
        current mouse cursor position."""
        self._show_context_menu()

    def _show_context_menu(self, event=None):
        """Right-click context menu that delegates to the parent's
        existing handler methods."""
        if self.parent is None:
            return

        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QCursor
        import sys

        gui_module = sys.modules.get(self.parent.__class__.__module__)
        my_network = (
            getattr(gui_module, "my_network", None) if gui_module else None
        )

        menu = QMenu()

        # --- Find ---
        menu.addAction("Find Node/Edge/Community").triggered.connect(
            self.parent.handle_find
        )

        # --- Neighbors ---
        nb = menu.addMenu("Show Neighbors")
        nb.addAction("Neighboring Nodes").triggered.connect(
            self.parent.handle_show_neighbors
        )
        nb.addAction("Neighboring Nodes + Edges").triggered.connect(
            lambda: self.parent.handle_show_neighbors(edges=True)
        )
        nb.addAction("Neighboring Edges").triggered.connect(
            lambda: self.parent.handle_show_neighbors(edges=True, nodes=False)
        )

        # --- Connected component ---
        cc = menu.addMenu("Show Connected Component(s)")
        cc.addAction("Just nodes").triggered.connect(
            self.parent.handle_show_component
        )
        cc.addAction("Nodes + Edges").triggered.connect(
            lambda: self.parent.handle_show_component(edges=True)
        )
        cc.addAction("Just edges").triggered.connect(
            lambda: self.parent.handle_show_component(edges=True, nodes=False)
        )

        # --- Community ---
        cm = menu.addMenu("Show Nodes' Community(s)")
        cm.addAction("Just nodes").triggered.connect(
            self.parent.handle_show_communities
        )
        cm.addAction("Nodes + Edges").triggered.connect(
            lambda: self.parent.handle_show_communities(edges=True)
        )

        # --- Identity submenu ---
        if my_network and my_network.node_identities is not None:
            id_menu = menu.addMenu("Show Identity")
            seen = set()
            for v in my_network.node_identities.values():
                seen.update(v)
            for item in sorted(seen):
                act = id_menu.addAction(f"ID: {item}")
                act.triggered.connect(
                    partial(self.parent.handle_show_identities, sort=item)
                )

        # --- Community submenu ---
        if my_network and my_network.communities is not None:
            cm_sub = menu.addMenu("Show Community")
            for com_id in sorted(set(my_network.communities.values())):
                act = cm_sub.addAction(f"Com: {com_id}")
                act.triggered.connect(
                    partial(
                        self.parent.handle_show_communities_menu,
                        community=com_id,
                    )
                )

        # --- Select All ---
        sel = menu.addMenu("Select All")
        sel.addAction("Nodes").triggered.connect(
            lambda: self.parent.handle_select_all(edges=False, nodes=True)
        )
        sel.addAction("Nodes + Edges").triggered.connect(
            lambda: self.parent.handle_select_all(edges=True)
        )
        sel.addAction("Edges").triggered.connect(
            lambda: self.parent.handle_select_all(edges=True, nodes=False)
        )
        sel.addAction("Nodes in Network").triggered.connect(
            lambda: self.parent.handle_select_all(
                edges=False, nodes=True, network=True
            )
        )
        sel.addAction("Nodes + Edges in Network").triggered.connect(
            lambda: self.parent.handle_select_all(edges=True, network=True)
        )
        sel.addAction("Edges in Network").triggered.connect(
            lambda: self.parent.handle_select_all(
                edges=True, nodes=False, network=True
            )
        )

        # --- Selection operations ---
        n_n = len(self.parent.clicked_values.get("nodes", []))
        n_e = len(self.parent.clicked_values.get("edges", []))
        if n_n > 0 or n_e > 0:
            ops = menu.addMenu("Selection")
            if n_n > 1 or n_e > 1:
                ops.addAction("Combine Object Labels").triggered.connect(
                    self.parent.handle_combine
                )
            ops.addAction("Split Non-Touching Labels").triggered.connect(
                self.parent.handle_seperate
            )
            ops.addAction("Delete Selection").triggered.connect(
                self.parent.handle_delete
            )
            if n_n > 1:
                ops.addAction("Link Nodes").triggered.connect(
                    self.parent.handle_link
                )
                ops.addAction("Split Nodes").triggered.connect(
                    self.parent.handle_split
                )
            ops.addAction("Add to New Community").triggered.connect(
                self.parent.new_coms
            )
            ops.addAction("Add to New Identity").triggered.connect(
                self.parent.new_iden_method
            )
            ops.addAction("Override Channel with Selection").triggered.connect(
                self.parent.handle_override
            )

        # --- Highlight in network ---
        if (self.parent.highlight_overlay is not None
                or self.parent.mini_overlay_data is not None):
            menu.addAction(
                "Add highlight in network selection"
            ).triggered.connect(self.parent.handle_highlight_select)

        cursor_pos = QCursor.pos()
        menu.exec(cursor_pos)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        """Purge all state so the next launch starts completely fresh."""
        # Remove event filter if still installed
        if self._canvas_native and self._right_click_filter:
            try:
                self._canvas_native.removeEventFilter(self._right_click_filter)
            except Exception:
                pass

        # Invalidate in-flight work; stale results are dropped by the
        # signal handlers.  We deliberately do NOT block on a running job
        # here — waiting would freeze the GUI, which is the whole point.
        self._generation += 1
        try:
            self._pool.clear()      # drop jobs that have not started yet
        except Exception:
            pass
        self._hl_running = False
        self._hl_pending = False
        self._busy = False
        self._bbox_running = False
        self._hl_data = None
        self._hl_scratch = None
        self._hl_foreign = False
        self._hl_limits_default = None
        self._down_factor = None
        self._down_order = 0

        self.rendered = False
        self.viewer = None
        self._highlight_layer = None
        self._image_layers = []
        self._node_labels_layer = None
        self._edge_labels_layer = None
        self._node_data = None
        self._edge_data = None
        self._scale = [1, 1, 1]
        self._selected_nodes = []
        self._selected_edges = []
        self._updating = False
        self._select_mode = False
        self._control = None
        self._right_click_filter = None
        self._canvas_native = None
        self._node_bboxes = None
        self._edge_bboxes = None
        self._prev_selected_nodes = []
        self._prev_selected_edges = []

        # Clear the parent's reference so it knows to create a new one
        if self.parent is not None and hasattr(self.parent, "napari_viewer"):
            if self.parent.napari_viewer is self:
                self.parent.napari_viewer = None

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        self._on_close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _in_bounds(idx, shape):
        return all(0 <= idx[i] < shape[i] for i in range(3))

    @staticmethod
    def _generate_bounding_box(shape, foreground_value=1, background_value=0):
        """
        Generate a 3D bounding box array with edges connecting the corners.
        
        Parameters:
        -----------
        shape : tuple
            Shape of the array in format (Z, Y, X)
        foreground_value : int or float, default=1
            Value to use for the bounding box edges and corners
        background_value : int or float, default=0
            Value to use for the background
        
        Returns:
        --------
        numpy.ndarray
            3D array with bounding box edges
        """
        if len(shape) > 3:
            shape = (shape[0], shape[1], shape[2])

        z_size, y_size, x_size = shape
        
        # Create empty array filled with background value
        box_array = np.full(shape, background_value, dtype=np.float64)
        
        # Define the 8 corners of the 3D box
        corners = [
            (0, 0, 0),           # corner 0
            (0, 0, x_size-1),    # corner 1
            (0, y_size-1, 0),    # corner 2
            (0, y_size-1, x_size-1),  # corner 3
            (z_size-1, 0, 0),    # corner 4
            (z_size-1, 0, x_size-1),  # corner 5
            (z_size-1, y_size-1, 0),  # corner 6
            (z_size-1, y_size-1, x_size-1)  # corner 7
        ]
        
        # Set corner values
        for corner in corners:
            box_array[corner] = foreground_value
        
        # Define edges connecting adjacent corners
        # Each edge connects two corners that differ by only one coordinate
        edges = [
            # Bottom face edges (z=0)
            (0, 1), (1, 3), (3, 2), (2, 0),
            # Top face edges (z=max)
            (4, 5), (5, 7), (7, 6), (6, 4),
            # Vertical edges connecting bottom to top
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Draw edges using linspace
        for start_idx, end_idx in edges:
            start_corner = corners[start_idx]
            end_corner = corners[end_idx]
            
            # Calculate the maximum distance along any axis to determine number of points
            max_distance = max(
                abs(end_corner[0] - start_corner[0]),
                abs(end_corner[1] - start_corner[1]),
                abs(end_corner[2] - start_corner[2])
            )
            num_points = max_distance + 1
            
            # Generate points along the edge using linspace
            z_points = np.linspace(start_corner[0], end_corner[0], num_points, dtype=int)
            y_points = np.linspace(start_corner[1], end_corner[1], num_points, dtype=int)
            x_points = np.linspace(start_corner[2], end_corner[2], num_points, dtype=int)
            
            # Set foreground values along the edge
            for z, y, x in zip(z_points, y_points, x_points):
                box_array[int(z), int(y), int(x)] = foreground_value
        
        return box_array