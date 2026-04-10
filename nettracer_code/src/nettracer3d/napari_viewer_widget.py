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

from PyQt6.QtCore import QObject, QEvent, QTimer, Qt

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ======================================================================
# Numba-accelerated highlight kernel (optional)
# ======================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _numba_highlight_bbox(highlight, label_data, labels,
                              z0, z1, y0, y1, x0, x1):
        """Write 255 into *highlight* wherever *label_data* matches any
        value in *labels*, but only within the bounding box [z0:z1, y0:y1, x0:x1].
        """
        for z in prange(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = label_data[z, y, x]
                    for k in range(labels.shape[0]):
                        if v == labels[k]:
                            highlight[z, y, x] = 255
                            break

    @njit(parallel=True, cache=True)
    def _numba_clear_bbox(highlight, label_data, labels,
                          z0, z1, y0, y1, x0, x1):
        """Clear (set to 0) voxels in *highlight* that match any label in
        *labels*, within the bounding box."""
        for z in prange(z0, z1):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = label_data[z, y, x]
                    for k in range(labels.shape[0]):
                        if v == labels[k]:
                            highlight[z, y, x] = 0
                            break
else:
    _numba_highlight_bbox = None
    _numba_clear_bbox = None


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

    def _on_compute_bboxes(self):
        """Compute bounding boxes for the channel currently selected in
        the combo box."""
        idx = self.channel_combo.currentIndex()
        self.bbox_btn.setEnabled(False)
        self.bbox_btn.setText("Computing…")
        # Use a single-shot timer so the UI repaints before the heavy work
        QTimer.singleShot(50, partial(self._nw._compute_bboxes_for_channel, idx))

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

        # Bounding-box acceleration for highlight
        self._node_bboxes = None   # dict {label: (z0,z1,y0,y1,x0,x1)} or None
        self._edge_bboxes = None
        self._prev_selected_nodes = []
        self._prev_selected_edges = []

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
        """
        if not HAS_NAPARI:
            raise ImportError(
                "napari is not installed. Install with: pip install napari[all]"
            )

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

        # ---- Add 3D image layers (nodes, edges, highlight, etc.) ----
        for i, (arr, color) in enumerate(zip(arrays_3d, colors)):
            shape = arr.shape
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

        # Highlight layer
        if shape is None and self._node_data is not None:
            shape = self._node_data.shape
        if shape is None and self._edge_data is not None:
            shape = self._edge_data.shape
        if shape is not None:
            self._highlight_layer = self.viewer.add_image(
                np.zeros(shape, dtype=np.uint8),
                scale=self._scale,
                colormap="yellow",
                rendering="mip",
                blending="additive",
                opacity=0.7,
                name="Selection Highlight",
            )

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

        # Show existing selection if any
        if self.parent is not None:
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

    def _clear_selection(self):
        """Clear all selections, update highlight + parent."""
        self._selected_nodes = []
        self._selected_edges = []
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
        """Compute per-label bounding boxes for channel 0 (nodes) or 1 (edges)."""
        try:
            if channel_idx == 0 and self._node_data is not None:
                self._node_bboxes = _compute_bbox_dict(self._node_data)
            elif channel_idx == 1 and self._edge_data is not None:
                self._edge_bboxes = _compute_bbox_dict(self._edge_data)
        except Exception as e:
            print(f"Bounding box computation failed: {e}")
        if self._control:
            self._control.update_bbox_status()

    # ------------------------------------------------------------------
    # Highlight rendering
    # ------------------------------------------------------------------

    def _update_highlight(self):
        """Rebuild the 3D highlight layer from current selections.

        When bounding boxes have been precomputed for a channel, uses
        incremental updates: only clears removed labels and paints newly
        added labels instead of rebuilding from scratch.  Falls back to
        full-volume ``np.isin`` for channels without bboxes.
        """
        if self._highlight_layer is None:
            return

        highlight = self._highlight_layer.data

        # Determine what changed since last call
        prev_n = set(self._prev_selected_nodes)
        prev_e = set(self._prev_selected_edges)
        cur_n = set(self._selected_nodes)
        cur_e = set(self._selected_edges)

        nodes_changed = (prev_n != cur_n)
        edges_changed = (prev_e != cur_e)

        have_node_bb = self._node_bboxes is not None
        have_edge_bb = self._edge_bboxes is not None

        # Decide strategy: if a channel WITHOUT bboxes changed, we must
        # do a full wipe+rebuild (since we can't selectively clear its
        # old voxels from the shared highlight array).  If only channels
        # WITH bboxes changed, we can do a pure incremental update.
        nonbb_changed = ((nodes_changed and not have_node_bb
                          and self._node_data is not None)
                         or (edges_changed and not have_edge_bb
                             and self._edge_data is not None))

        if nonbb_changed:
            # Full rebuild — wipe everything, repaint all current selections
            highlight[:] = 0

            # Nodes: use bboxes if available, else full isin
            if self._node_data is not None and self._selected_nodes:
                if have_node_bb:
                    self._paint_all_selected(
                        highlight, self._node_data, self._node_bboxes,
                        self._selected_nodes)
                else:
                    self._paint_fullvol(highlight, self._node_data,
                                        self._selected_nodes)

            # Edges: use bboxes if available, else full volume search
            if self._edge_data is not None and self._selected_edges:
                if have_edge_bb:
                    self._paint_all_selected(
                        highlight, self._edge_data, self._edge_bboxes,
                        self._selected_edges)
                else:
                    self._paint_fullvol(highlight, self._edge_data,
                                        self._selected_edges)
        else:
            # Pure incremental — only bbox channels changed
            if nodes_changed and self._node_data is not None and have_node_bb:
                self._incremental_update(
                    highlight, self._node_data, self._node_bboxes,
                    list(prev_n - cur_n), list(cur_n - prev_n),
                )
            if edges_changed and self._edge_data is not None and have_edge_bb:
                self._incremental_update(
                    highlight, self._edge_data, self._edge_bboxes,
                    list(prev_e - cur_e), list(cur_e - prev_e),
                )

        # Snapshot for next diff
        self._prev_selected_nodes = list(self._selected_nodes)
        self._prev_selected_edges = list(self._selected_edges)

        self._highlight_layer.data = highlight
        self._highlight_layer.refresh()

    # ------------------------------------------------------------------
    # Incremental bbox-based highlight helpers
    # ------------------------------------------------------------------

    def _incremental_update(self, highlight, label_data, bboxes,
                            removed_labels, added_labels):
        """Clear *removed_labels* and paint *added_labels* using their
        precomputed bounding boxes.

        When multiple bboxes to paint overlap significantly, they are
        merged into a single superbox if that reduces total search volume.
        """
        # --- Clear removed labels ---
        for lbl in removed_labels:
            bb = bboxes.get(lbl)
            if bb is None:
                continue
            self._clear_bbox(highlight, label_data, [lbl], bb)

        if not added_labels:
            return

        # --- Paint added labels, with overlap-aware grouping ---
        self._paint_all_selected(highlight, label_data, bboxes, added_labels)

    def _paint_all_selected(self, highlight, label_data, bboxes, labels):
        """Paint all *labels* into highlight using their bounding boxes,
        with overlap-aware grouping."""
        lbl_bb_pairs = []
        for lbl in labels:
            bb = bboxes.get(lbl)
            if bb is not None:
                lbl_bb_pairs.append((lbl, bb))

        if not lbl_bb_pairs:
            return

        if len(lbl_bb_pairs) == 1:
            lbl, bb = lbl_bb_pairs[0]
            self._paint_bbox(highlight, label_data, [lbl], bb)
            return

        groups = self._group_bboxes(lbl_bb_pairs)
        for group_labels, group_bb in groups:
            self._paint_bbox(highlight, label_data, group_labels, group_bb)

    @staticmethod
    def _group_bboxes(lbl_bb_pairs):
        """Return list of (labels_list, combined_bb) groups.

        Greedily merge pairs whose superbox volume is less than or equal
        to the sum of the individual bbox volumes, since searching one
        large contiguous region is more cache-friendly and avoids repeat
        overlap work.
        """
        # Start with each label as its own group
        groups = [([lbl], bb) for lbl, bb in lbl_bb_pairs]

        merged = True
        while merged:
            merged = False
            new_groups = []
            used = [False] * len(groups)
            for i in range(len(groups)):
                if used[i]:
                    continue
                cur_labels, cur_bb = groups[i]
                for j in range(i + 1, len(groups)):
                    if used[j]:
                        continue
                    other_labels, other_bb = groups[j]
                    combined = _merge_bboxes([cur_bb, other_bb])
                    vol_combined = _bbox_volume(combined)
                    vol_separate = _bbox_volume(cur_bb) + _bbox_volume(other_bb)
                    if vol_combined <= vol_separate:
                        cur_labels = cur_labels + other_labels
                        cur_bb = combined
                        used[j] = True
                        merged = True
                new_groups.append((cur_labels, cur_bb))
                used[i] = True
            groups = new_groups

        return groups

    def _paint_fullvol(self, highlight, label_data, labels):
        """Paint the full volume — uses numba if available, else np.isin."""
        s = label_data.shape
        if HAS_NUMBA and _numba_highlight_bbox is not None:
            _numba_highlight_bbox(
                highlight, label_data,
                np.array(labels, dtype=label_data.dtype),
                0, s[0], 0, s[1], 0, s[2],
            )
        else:
            mask = np.isin(label_data, labels)
            np.maximum(highlight, mask.astype(np.uint8) * 255, out=highlight)

    def _paint_bbox(self, highlight, label_data, labels, bb):
        """Set highlight=255 for voxels matching any label within bb."""
        z0, z1, y0, y1, x0, x1 = bb
        if HAS_NUMBA and _numba_highlight_bbox is not None:
            _numba_highlight_bbox(
                highlight, label_data,
                np.array(labels, dtype=label_data.dtype),
                z0, z1, y0, y1, x0, x1,
            )
        else:
            sub_labels = label_data[z0:z1, y0:y1, x0:x1]
            mask = np.isin(sub_labels, labels)
            np.maximum(
                highlight[z0:z1, y0:y1, x0:x1],
                mask.astype(np.uint8) * 255,
                out=highlight[z0:z1, y0:y1, x0:x1],
            )

    def _clear_bbox(self, highlight, label_data, labels, bb):
        """Set highlight=0 for voxels matching any label within bb."""
        z0, z1, y0, y1, x0, x1 = bb
        if HAS_NUMBA and _numba_clear_bbox is not None:
            _numba_clear_bbox(
                highlight, label_data,
                np.array(labels, dtype=label_data.dtype),
                z0, z1, y0, y1, x0, x1,
            )
        else:
            sub_labels = label_data[z0:z1, y0:y1, x0:x1]
            mask = np.isin(sub_labels, labels)
            highlight[z0:z1, y0:y1, x0:x1][mask] = 0

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

        self.rendered = False
        self.viewer = None
        self._highlight_layer = None
        self._image_layers = []
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