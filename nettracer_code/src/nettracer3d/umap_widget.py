import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMenu,
                              QSizePolicy, QApplication, QScrollArea, QLabel, QFrame,
                              QFileDialog, QMessageBox, QMainWindow, QDialog, QFormLayout,
                              QGroupBox, QComboBox, QLineEdit, QCheckBox, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QPointF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QPainterPath, QPolygonF, QCloseEvent
import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem, PlotCurveItem, GraphicsLayoutWidget
import colorsys
import random
import copy
import json
import math
from . import nettracer_gui as netg

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


class UMAPGraphWidget(QWidget):
    """Interactive UMAP visualization widget with lasso selection and dynamic node sizing."""

    node_selected = pyqtSignal(object)  # Emits list of selected node IDs

    def __init__(self, parent=None,
                 community_dict=None,
                 identity_dict=None,
                 labels=False,
                 node_size=10,
                 color_mode='community'):
        """
        Parameters
        ----------
        parent : QWidget
            Parent window (expected to have clicked_values, evaluate_mini, handle_info).
        community_dict : dict, optional
            {node_id: community_label} for community colouring.
        identity_dict : dict, optional
            {node_id: identity_label} for identity colouring.
        labels : bool
            Whether to render text labels on nodes.
        node_size : float
            Base node radius in px.
        color_mode : str
            'community' or 'identity' – which dict drives the initial colouring.
        """
        super().__init__(parent)

        self.parent_window = parent
        self.community_dict = community_dict or {}
        self.identity_dict = identity_dict or {}
        self.labels = labels
        self.node_size = node_size
        self.color_mode = color_mode  # 'community' or 'identity'

        # --- data ---
        self.node_ids = []              # ordered list of node IDs matching embedding rows
        self.embedding = None           # np.ndarray (N, 2)
        self.node_positions = {}        # {node_id: np.array([x, y])}
        self.node_colors = []           # hex colours parallel to node_ids
        self.selected_nodes = set()
        self.rendered = False

        # --- caching for fast updates ---
        self.cached_spots = []
        self.cached_node_to_index = {}  # node_id -> spot index
        self.cached_brushes = {}        # node_id -> {'normal': brush, 'selected': brush}
        self.last_selected_set = set()
        self.cached_sizes_for_lod = []

        # --- interaction mode ---
        self.selection_mode = True
        self.zoom_mode = False

        # --- lasso selection ---
        self.lasso_points = []          # list of QPointF in *view* coords
        self.lasso_path_item = None     # PlotCurveItem for the dashed line
        self.is_lasso_selecting = False
        self.lasso_start_pos = None     # first point (view coords)
        self.lasso_close_threshold = 15 # pixels to consider "closed"

        # --- area selection (click‐and‐drag start detection) ---
        self.click_timer = None
        self.last_mouse_pos = None      # scene coords of press

        # --- middle mouse panning ---
        self.temp_pan_active = False

        # --- wheel zoom timer ---
        self.wheel_timer = None
        self.was_in_selection_before_wheel = False

        # --- labels ---
        self.label_items = {}
        self.label_data = []

        # --- zoom LOD ---
        self.current_zoom_factor = 1.0

        # --- saved embedding ---
        self._saved_embedding = None    # dict loaded from file or set externally

        self._setup_ui()

    # ------------------------------------------------------------------ UI --
    def _setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # -- left: graph container --
        graph_container = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(2)

        # pyqtgraph
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground('w')
        self.plot = self.graphics_widget.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.showGrid(x=False, y=False)

        # placeholder text
        self.loading_text = pg.TextItem(
            text="No UMAP computed",
            color=(100, 100, 100),
            anchor=(0.5, 0.5)
        )
        self.loading_text.setPos(0, 0)
        self.plot.addItem(self.loading_text)

        # mouse wiring
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.vb.setMenuEnabled(False)
        self.plot.vb.setMouseMode(pg.ViewBox.PanMode)

        # scatter
        self.scatter = ScatterPlotItem(size=10, pen=pg.mkPen(None),
                                       brush=pg.mkBrush(74, 144, 226, 200))
        self.plot.addItem(self.scatter)
        self.scatter.sigClicked.connect(self._on_node_clicked)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self.plot.sigRangeChanged.connect(self._on_view_changed)

        self.base_node_sizes = []

        graph_layout.addWidget(self.graphics_widget, stretch=1)

        # control panel
        control_panel = self._create_control_panel()
        graph_layout.addWidget(control_panel)

        graph_container.setLayout(graph_layout)
        layout.addWidget(graph_container, stretch=1)

        # -- right: legend --
        self.legend_container = QWidget()
        self.legend_layout = QVBoxLayout()
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_container.setLayout(self.legend_layout)
        self.legend_container.setMaximumWidth(0)
        layout.addWidget(self.legend_container)

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        # event filter for custom mouse handling
        self.graphics_widget.viewport().installEventFilter(self)
        self.plot.scene().installEventFilter(self)

    # --------------------------------------------------------- control panel --
    def _create_control_panel(self):
        panel = QWidget()
        panel_layout = QHBoxLayout()
        panel_layout.setContentsMargins(2, 2, 2, 2)
        panel_layout.setSpacing(2)

        self.select_btn = QPushButton("🖱️")
        self.select_btn.setToolTip("Lasso Selection Tool")
        self.select_btn.setCheckable(True)
        self.select_btn.setChecked(True)
        self.select_btn.setMaximumSize(32, 32)
        self.select_btn.clicked.connect(self._toggle_selection_mode)

        self.pan_btn = QPushButton("✋")
        self.pan_btn.setToolTip("Pan Tool")
        self.pan_btn.setCheckable(True)
        self.pan_btn.setChecked(False)
        self.pan_btn.setMaximumSize(32, 32)
        self.pan_btn.clicked.connect(self._toggle_pan_mode)

        self.zoom_btn = QPushButton("🔍")
        self.zoom_btn.setToolTip("Zoom Tool (Left: Out, Right: In)")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setMaximumSize(32, 32)
        self.zoom_btn.clicked.connect(self._toggle_zoom_mode)

        self.home_btn = QPushButton("🏠")
        self.home_btn.setToolTip("Reset View")
        self.home_btn.setMaximumSize(32, 32)
        self.home_btn.clicked.connect(self._reset_view)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setToolTip("Color Mode Settings")
        self.settings_btn.setMaximumSize(32, 32)
        self.settings_btn.clicked.connect(self._on_settings_clicked)

        self.save_btn = QPushButton("💾")
        self.save_btn.setToolTip("Save UMAP Embedding")
        self.save_btn.setMaximumSize(32, 32)
        self.save_btn.clicked.connect(self._save_embedding)

        panel_layout.addWidget(self.select_btn)
        panel_layout.addWidget(self.pan_btn)
        panel_layout.addWidget(self.zoom_btn)
        panel_layout.addWidget(self.home_btn)
        panel_layout.addWidget(self.settings_btn)
        panel_layout.addWidget(self.save_btn)
        panel_layout.addStretch()

        panel.setLayout(panel_layout)
        panel.setMaximumHeight(40)
        return panel

    # ------------------------------------------------------ settings / options --
    def _on_settings_clicked(self):
        """Open the UMAP display settings dialog."""
        dlg = netg.UMAPDisplayDialog(self, parent_window=self.parent_window)
        dlg.exec()

    def set_color_mode(self, mode, new_dict=None):
        """
        Switch colouring between 'community' and 'identity'.

        Parameters
        ----------
        mode : str
            'community' or 'identity'
        new_dict : dict, optional
            New {node: label} dict.  If None, uses whatever is already stored.
        """
        self.color_mode = mode
        if new_dict is not None:
            if mode == 'community':
                self.community_dict = new_dict
            else:
                self.identity_dict = new_dict

        # Quick re‐colour (embedding stays the same)
        if self.rendered and len(self.node_ids) > 0:
            self._recolor_nodes()

    # --------------------------------------------------- save / load embedding --
    def _save_embedding(self):
        """Save node IDs and their 2‑D embeddings to a JSON file."""
        if self.embedding is None or len(self.node_ids) == 0:
            QMessageBox.warning(self, "Nothing to save",
                                "Compute a UMAP embedding first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save UMAP Embedding", "", "JSON Files (*.json)")
        if not filename:
            return
        if not filename.endswith('.json'):
            filename += '.json'

        data = {
            'node_ids': [int(n) if isinstance(n, (int, np.integer)) else n
                         for n in self.node_ids],
            'embedding': self.embedding.tolist()
        }
        try:
            with open(filename, 'w') as f:
                json.dump(data, f)
            QMessageBox.information(self, "Saved",
                                    f"Embedding saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_from_save(self, filepath_or_dict):
        """
        Load a previously saved embedding and render it.

        Parameters
        ----------
        filepath_or_dict : str or dict
            Either a JSON file path or a dict with keys 'node_ids' and 'embedding'.
        """
        if isinstance(filepath_or_dict, str):
            with open(filepath_or_dict, 'r') as f:
                data = json.load(f)
        else:
            data = filepath_or_dict

        node_ids = data['node_ids']
        embedding = np.array(data['embedding'], dtype=np.float64)

        self._apply_embedding(node_ids, embedding)

    # ------------------------------------------------- main public API ----------
    def set_data(self, cluster_data,
                 community_dict=None, identity_dict=None,
                 color_mode=None,
                 umap_kwargs=None):
        """
        Compute UMAP and render.

        Parameters
        ----------
        cluster_data : dict
            {node_id: 1‑D np.ndarray} – the composition vectors to embed.
        community_dict : dict, optional
            {node_id: community_label}.
        identity_dict : dict, optional
            {node_id: identity_label}.
        color_mode : str, optional
            'community' or 'identity'.
        umap_kwargs : dict, optional
            Extra kwargs forwarded to umap.UMAP (e.g. n_neighbors, min_dist).
        """
        import umap as umap_lib  # heavy import deferred

        if community_dict is not None:
            self.community_dict = community_dict
        if identity_dict is not None:
            self.identity_dict = identity_dict
        if color_mode is not None:
            self.color_mode = color_mode

        # Remove loading text
        self._remove_loading_text()

        # Show computing indicator
        self.loading_text = pg.TextItem(
            text="Computing UMAP…",
            color=(100, 100, 100),
            anchor=(0.5, 0.5)
        )
        self.loading_text.setPos(0, 0)
        self.plot.addItem(self.loading_text)
        QApplication.processEvents()

        # --- compute embedding (in main thread – intentional) ---
        node_ids = list(cluster_data.keys())
        compositions = np.array([cluster_data[nid] for nid in node_ids])

        kw = dict(n_components=2, random_state=42)
        if umap_kwargs:
            kw.update(umap_kwargs)

        reducer = umap_lib.UMAP(**kw)
        embedding = reducer.fit_transform(compositions)

        self._apply_embedding(node_ids, embedding)

    # ----------------------------------------------------- internal rendering ---
    def _apply_embedding(self, node_ids, embedding):
        """Store embedding and render everything."""
        self._clear_plot()

        self.node_ids = list(node_ids)
        self.embedding = np.array(embedding, dtype=np.float64)
        self.node_positions = {nid: self.embedding[i]
                               for i, nid in enumerate(self.node_ids)}

        # Build colours + spots
        self._build_and_render()

    def _build_and_render(self):
        """Compute colours, spots, labels and push to scatter."""
        n = len(self.node_ids)
        if n == 0:
            return

        # --- dynamic base point size (smaller for large graphs) ---
        if n < 50:
            point_size = self.node_size
        elif n < 200:
            point_size = max(4, self.node_size * 0.7)
        elif n < 1000:
            point_size = max(3, self.node_size * 0.45)
        else:
            point_size = max(2, self.node_size * 0.25)

        # --- colours ---
        active_dict = self._active_color_dict()
        color_map = self._generate_community_colors(active_dict)

        colors_hex = []
        alphas = []
        for nid in self.node_ids:
            if self.color_mode == 'colorless':
                colors_hex.append('#4A90E2')
                alphas.append(200)
            else:
                label = active_dict.get(nid, None)
                if label is None:
                    colors_hex.append('#808080')
                    alphas.append(100)  # reduced alpha for unassigned
                else:
                    c = color_map.get(label, '#808080')
                    colors_hex.append(c)
                    if c == '#808080':
                        alphas.append(100)
                    else:
                        alphas.append(200)

        self.node_colors = colors_hex

        # --- spots & brush cache ---
        spots = []
        brush_cache = {}
        sizes = []
        for i, nid in enumerate(self.node_ids):
            hex_c = colors_hex[i].lstrip('#')
            r, g, b = (int(hex_c[j:j+2], 16) for j in (0, 2, 4))
            alpha = alphas[i]

            normal_brush = pg.mkBrush(r, g, b, alpha)
            selected_brush = pg.mkBrush(255, 255, 0, 255)

            brush_cache[nid] = {'normal': normal_brush, 'selected': selected_brush}

            spot = {
                'pos': self.embedding[i],
                'size': point_size,
                'brush': normal_brush,
                'data': nid
            }
            spots.append(spot)
            sizes.append(point_size)

        self.cached_spots = spots
        self.cached_brushes = brush_cache
        self.cached_sizes_for_lod = sizes[:]
        self.cached_node_to_index = {spot['data']: i for i, spot in enumerate(spots)}

        # --- render scatter ---
        self.scatter.setData(spots=self.cached_spots)
        self.scatter.setZValue(10)

        # --- labels ---
        self.label_data = []
        for i, nid in enumerate(self.node_ids):
            self.label_data.append({
                'node': nid,
                'text': str(nid),
                'pos': self.embedding[i]
            })

        if self.labels and n < 100:
            self._update_labels_in_viewport(n)

        # --- legend ---
        self._rebuild_legend(active_dict, color_map)

        self.rendered = True
        self.current_zoom_factor = 1.0

        # Reset view
        self.plot.blockSignals(True)
        self._reset_view()
        self.plot.blockSignals(False)

        # Re‐apply any selection the parent tracks
        if (self.parent_window is not None
                and hasattr(self.parent_window, 'clicked_values')
                and len(self.parent_window.clicked_values.get('nodes', [])) > 0):
            self.select_nodes(self.parent_window.clicked_values['nodes'])

    def _recolor_nodes(self):
        """Quick re‐colour without recomputing the embedding."""
        if not self.cached_spots or len(self.node_ids) == 0:
            return

        active_dict = self._active_color_dict()
        color_map = self._generate_community_colors(active_dict)

        colors_hex = []
        for i, nid in enumerate(self.node_ids):
            label = active_dict.get(nid, None)
            if label is None:
                hex_c = '#808080'
                alpha = 100
            else:
                hex_c = color_map.get(label, '#808080')
                alpha = 100 if hex_c == '#808080' else 200

            colors_hex.append(hex_c)
            hx = hex_c.lstrip('#')
            r, g, b = (int(hx[j:j+2], 16) for j in (0, 2, 4))

            normal_brush = pg.mkBrush(r, g, b, alpha)
            selected_brush = pg.mkBrush(255, 255, 0, 255)

            self.cached_brushes[nid] = {'normal': normal_brush, 'selected': selected_brush}

            if nid in self.selected_nodes:
                self.cached_spots[i]['brush'] = selected_brush
            else:
                self.cached_spots[i]['brush'] = normal_brush

        self.node_colors = colors_hex
        self.scatter.setData(spots=self.cached_spots)

        # rebuild legend
        self._rebuild_legend(active_dict, color_map)

    # ------------------------------------------------------ colour helpers ------
    def _active_color_dict(self):
        """Return the dict that should drive colouring right now."""
        if self.color_mode == 'colorless':
            return {}
        if self.color_mode == 'identity' and self.identity_dict:
            return self.identity_dict
        return self.community_dict

    def _generate_community_colors(self, my_dict):
        """
        Consistent with NetworkGraphWidget._generate_community_colors:
        deterministic shuffle with Random(42), HSV hues, community 0 → brown.
        """
        from collections import Counter

        if not my_dict:
            return {}

        try:
            unique_labels = sorted(set(my_dict.values()))
        except TypeError:
            str_dict = {n: str(v) for n, v in my_dict.items()}
            unique_labels = sorted(set(str_dict.values()))

        shuffled = random.Random(42).sample(unique_labels, len(unique_labels))
        rgb_list = self._generate_distinct_colors_rgb(len(unique_labels))
        color_map = {label: rgb_list[i] for i, label in enumerate(shuffled)}

        # Community / label 0 ➔ brown
        if 0 in unique_labels:
            color_map[0] = '#8B4513'

        return color_map

    @staticmethod
    def _generate_distinct_colors_rgb(n_colors):
        colors = []
        for i in range(n_colors):
            hue = i / max(n_colors, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append('#{:02x}{:02x}{:02x}'.format(
                int(r * 255), int(g * 255), int(b * 255)))
        return colors

    # ------------------------------------------------------- legend -------------
    def _rebuild_legend(self, active_dict, color_map):
        """Rebuild the side legend widget."""
        # Clear old
        for i in reversed(range(self.legend_layout.count())):
            w = self.legend_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        if not active_dict or not color_map:
            self.legend_container.setMaximumWidth(0)
            return

        unique_labels = sorted(set(active_dict.values()))

        legend_widget = QWidget()
        ll = QVBoxLayout()
        ll.setContentsMargins(5, 5, 5, 5)
        ll.setSpacing(2)

        title_text = "Identity" if self.color_mode == 'identity' else "Community"
        title = QLabel(title_text)
        title.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 3px;")
        ll.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(200)
        scroll.setMinimumWidth(150)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)

        items_widget = QWidget()
        items_layout = QVBoxLayout()
        items_layout.setContentsMargins(2, 2, 2, 2)
        items_layout.setSpacing(3)

        for label in unique_labels:
            row = QWidget()
            rl = QHBoxLayout()
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(5)

            box = QLabel()
            box.setFixedSize(16, 16)
            c = color_map.get(label, '#808080')
            box.setStyleSheet(f"background-color: {c}; border: 1px solid #888;")

            lbl = QLabel(str(label))
            lbl.setStyleSheet("font-size: 9pt;")

            rl.addWidget(box)
            rl.addWidget(lbl)
            rl.addStretch()
            row.setLayout(rl)
            items_layout.addWidget(row)

        # Check if there are unassigned nodes
        has_unassigned = any(nid not in active_dict for nid in self.node_ids)
        if has_unassigned:
            row = QWidget()
            rl = QHBoxLayout()
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(5)
            box = QLabel()
            box.setFixedSize(16, 16)
            box.setStyleSheet("background-color: #808080; border: 1px solid #888;")
            lbl = QLabel("Unassigned")
            lbl.setStyleSheet("font-size: 9pt;")
            rl.addWidget(box)
            rl.addWidget(lbl)
            rl.addStretch()
            row.setLayout(rl)
            items_layout.addWidget(row)

        items_layout.addStretch()
        items_widget.setLayout(items_layout)
        scroll.setWidget(items_widget)

        ll.addWidget(scroll)
        legend_widget.setLayout(ll)

        self.legend_layout.addWidget(legend_widget)
        self.legend_container.setMaximumWidth(200)

    # ------------------------------------------------------- node selection -----
    def select_nodes(self, nodes, add_to_selection=False):
        if not add_to_selection:
            self.selected_nodes.clear()

        if not nodes:
            self.selected_nodes = set()
        else:
            valid = set(self.cached_node_to_index.keys())
            for node in nodes:
                if node in valid:
                    self.selected_nodes.add(node)

        self._render_nodes()
        self.node_selected.emit(list(self.selected_nodes))

    def clear_selection(self):
        self.selected_nodes.clear()
        self._render_nodes()
        self.node_selected.emit([])

    def get_selected_nodes(self):
        return list(self.selected_nodes)

    def get_selected_node(self):
        if self.selected_nodes:
            return next(iter(self.selected_nodes))
        return None

    def push_selection(self):
        if self.parent_window is not None and hasattr(self.parent_window, 'clicked_values'):
            self.parent_window.clicked_values['nodes'] = list(self.selected_nodes)
            if hasattr(self.parent_window, 'evaluate_mini'):
                self.parent_window.evaluate_mini(subgraph_push=True)
            if hasattr(self.parent_window, 'handle_info'):
                self.parent_window.handle_info('node')

    # ------------------------------------------------------- render helpers -----
    def _render_nodes(self):
        """Only update brushes for nodes that changed selection state."""
        if not self.cached_spots or not self.cached_brushes:
            return

        newly_selected = self.selected_nodes - self.last_selected_set
        newly_deselected = self.last_selected_set - self.selected_nodes

        if not newly_selected and not newly_deselected:
            return

        for node in newly_selected:
            if node in self.cached_node_to_index:
                idx = self.cached_node_to_index[node]
                self.cached_spots[idx]['brush'] = self.cached_brushes[node]['selected']

        for node in newly_deselected:
            if node in self.cached_node_to_index:
                idx = self.cached_node_to_index[node]
                self.cached_spots[idx]['brush'] = self.cached_brushes[node]['normal']

        self.scatter.setData(spots=self.cached_spots)
        self.last_selected_set = self.selected_nodes.copy()

    # ------------------------------------------------------- view / zoom --------
    def _on_view_changed(self):
        if not self.node_positions or len(self.node_positions) == 0:
            return

        view_range = self.plot.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]

        pos_array = self.embedding
        if pos_array is None or len(pos_array) == 0:
            return

        full_x = pos_array[:, 0].max() - pos_array[:, 0].min()
        full_y = pos_array[:, 1].max() - pos_array[:, 1].min()

        if full_x > 0 and full_y > 0:
            zoom_x = full_x / x_range if x_range > 0 else 1
            zoom_y = full_y / y_range if y_range > 0 else 1
            zoom_factor = max(zoom_x, zoom_y)

            zoom_changed = abs(zoom_factor - self.current_zoom_factor) / max(self.current_zoom_factor, 0.01) > 0.05
            if zoom_changed:
                self.current_zoom_factor = zoom_factor
                self._update_lod_rendering()
            else:
                if self.labels:
                    self._update_labels_for_zoom()

    def _update_lod_rendering(self):
        """Adjust node sizes based on zoom level — small when zoomed out, large when zoomed in."""
        if not self.cached_spots or not self.cached_sizes_for_lod:
            return

        # zoom_factor ~1.0 at full extent, grows as you zoom in
        # We want nodes to be noticeably smaller at full zoom-out and grow
        # significantly as the user zooms in.
        zf = self.current_zoom_factor

        if zf <= 0.5:
            # Very zoomed out — shrink nodes
            scale_factor = 0.5
        elif zf <= 1.0:
            # Around default view — interpolate from 0.5× to 1.0×
            scale_factor = 0.5 + 0.5 * (zf / 1.0)
        else:
            # Zoomed in — grow nodes with a sqrt curve for smooth scaling
            # At 2× zoom → ~1.4×, at 4× → ~2×, at 10× → ~3.2×, at 25× → ~5×
            scale_factor = 1.0 + (math.sqrt(zf) - 1.0) * 1.0

        # Clamp to avoid absurdly large nodes
        scale_factor = min(scale_factor, 8.0)

        for i, base_size in enumerate(self.cached_sizes_for_lod):
            self.cached_spots[i]['size'] = base_size * scale_factor

        if self.labels:
            self._update_labels_for_zoom()

        self.scatter.setData(spots=self.cached_spots)

    # ------------------------------------------------------- labels -------------
    def _update_labels_for_zoom(self):
        if not self.labels or not self.label_data:
            return

        num_nodes = len(self.label_data)
        if num_nodes < 100:
            zoom_threshold = 0
        elif num_nodes < 500:
            zoom_threshold = 1.5
        else:
            zoom_threshold = 3.0

        if self.current_zoom_factor > zoom_threshold:
            self._update_labels_in_viewport(num_nodes)
        else:
            if self.label_items:
                for li in self.label_items.values():
                    self.plot.removeItem(li)
                self.label_items.clear()

    def _update_labels_in_viewport(self, num_nodes):
        view_range = self.plot.viewRange()
        x_min, x_max = view_range[0]
        y_min, y_max = view_range[1]

        visible_set = set()
        labels_to_render = []

        for info in self.label_data:
            px, py = info['pos'][0], info['pos'][1]
            if x_min <= px <= x_max and y_min <= py <= y_max:
                visible_set.add(info['node'])
                labels_to_render.append(info)

        max_visible = 200 if num_nodes >= 500 else 1000
        if len(labels_to_render) > max_visible:
            for li in self.label_items.values():
                self.plot.removeItem(li)
            self.label_items.clear()
            return

        current_set = set(self.label_items.keys())
        for node in (current_set - visible_set):
            if node in self.label_items:
                self.plot.removeItem(self.label_items[node])
                del self.label_items[node]

        for info in labels_to_render:
            node = info['node']
            if node not in current_set:
                ti = pg.TextItem(text=info['text'], color=(0, 0, 0), anchor=(0.5, 0.5))
                ti.setPos(info['pos'][0], info['pos'][1])
                ti.setZValue(20)
                self.plot.addItem(ti)
                self.label_items[node] = ti

    # ------------------------------------------------------- mode toggles -------
    def _toggle_selection_mode(self):
        self.selection_mode = self.select_btn.isChecked()
        if self.selection_mode:
            self.pan_btn.setChecked(False)
            self.zoom_btn.setChecked(False)
            self.zoom_mode = False
            self.plot.setCursor(Qt.CursorShape.ArrowCursor)
            self.plot.vb.setMenuEnabled(False)
            self.plot.setMouseEnabled(x=False, y=False)
        else:
            if not self.pan_btn.isChecked() and not self.zoom_btn.isChecked():
                self.pan_btn.click()

    def _toggle_pan_mode(self):
        if self.pan_btn.isChecked():
            self.select_btn.setChecked(False)
            self.zoom_btn.setChecked(False)
            self.selection_mode = False
            self.zoom_mode = False
            self.plot.vb.setMenuEnabled(True)
            self.plot.setCursor(Qt.CursorShape.OpenHandCursor)
            self.plot.setMouseEnabled(x=True, y=True)
        else:
            if not self.select_btn.isChecked() and not self.zoom_btn.isChecked():
                self.select_btn.click()

    def _toggle_zoom_mode(self):
        self.zoom_mode = self.zoom_btn.isChecked()
        if self.zoom_mode:
            self.select_btn.setChecked(False)
            self.pan_btn.setChecked(False)
            self.selection_mode = False
            self.plot.setCursor(Qt.CursorShape.CrossCursor)
            self.plot.vb.setMenuEnabled(False)
            self.plot.setMouseEnabled(x=False, y=False)
        else:
            if not self.pan_btn.isChecked() and not self.select_btn.isChecked():
                self.select_btn.click()

    # ------------------------------------------------------- reset / clear ------
    def _reset_view(self):
        if not self.node_positions:
            return
        if self.embedding is None or len(self.embedding) == 0:
            return

        x_min, y_min = self.embedding.min(axis=0)
        x_max, y_max = self.embedding.max(axis=0)

        padding = 0.1
        xr = x_max - x_min
        yr = y_max - y_min

        self.plot.setXRange(x_min - padding * xr, x_max + padding * xr, padding=0)
        self.plot.setYRange(y_min - padding * yr, y_max + padding * yr, padding=0)

    def _clear_plot(self):
        """Clear all rendered items (keeps widget alive)."""
        self._remove_loading_text()

        self.scatter.clear()

        for li in list(self.label_items.values()):
            try:
                self.plot.removeItem(li)
            except Exception:
                pass
        self.label_items.clear()

        # Remove any stray TextItems
        items_to_remove = [item for item in self.plot.items
                           if isinstance(item, pg.TextItem)]
        for item in items_to_remove:
            self.plot.removeItem(item)

        # Remove lasso path if present
        if self.lasso_path_item is not None:
            self.plot.removeItem(self.lasso_path_item)
            self.lasso_path_item = None
        self.lasso_points.clear()
        self.is_lasso_selecting = False

        # Clear legend
        try:
            for i in reversed(range(self.legend_layout.count())):
                w = self.legend_layout.itemAt(i).widget()
                if w:
                    w.setParent(None)
        except Exception:
            pass
        self.legend_container.setMaximumWidth(0)

        # Clear caches
        self.node_positions.clear()
        self.selected_nodes.clear()
        self.cached_spots.clear()
        self.cached_node_to_index.clear()
        self.cached_brushes.clear()
        self.last_selected_set.clear()
        self.cached_sizes_for_lod.clear()
        self.label_data.clear()
        self.rendered = False

    def _remove_loading_text(self):
        if hasattr(self, 'loading_text') and self.loading_text is not None:
            self.plot.removeItem(self.loading_text)
            self.loading_text = None

    # ------------------------------------------------------- mouse events -------
    def _on_node_clicked(self, scatter, points, ev):
        if not self.selection_mode or len(points) == 0:
            return

        clicked_node = points[0].data()
        modifiers = ev.modifiers()
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier

        if ctrl:
            if clicked_node in self.selected_nodes:
                self.selected_nodes.remove(clicked_node)
            else:
                self.selected_nodes.add(clicked_node)
        else:
            self.selected_nodes.clear()
            self.selected_nodes.add(clicked_node)

        self.push_selection()
        self._render_nodes()
        self.node_click_flag = True
        self.node_selected.emit(list(self.selected_nodes))

    def _on_plot_clicked(self, ev):
        if not self.selection_mode or getattr(self, 'node_click_flag', False):
            self.node_click_flag = False
            return

        if ev.button() != Qt.MouseButton.LeftButton:
            return

        modifiers = ev.modifiers()
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
        if not ctrl:
            self.selected_nodes.clear()
            self._render_nodes()
            self.push_selection()
            self.node_selected.emit([])

    def _on_mouse_moved(self, pos):
        """Update lasso path as mouse moves, auto-closing on self-intersection."""
        if not self.is_lasso_selecting:
            return

        view_pos = self.plot.vb.mapSceneToView(pos)
        self.lasso_points.append(view_pos)
        self._draw_lasso()

        # Check if the newest segment crosses any older segment
        cross_idx = self._find_segment_crossing()
        if cross_idx >= 0:
            self._auto_close_lasso(cross_idx)

    # ------------------------------------------------------- lasso selection ----
    def _start_lasso(self, scene_pos):
        """Begin drawing a lasso from the given scene position."""
        self.is_lasso_selecting = True
        self.node_click_flag = False
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        self.lasso_start_pos = view_pos
        self.lasso_points = [view_pos]

        # Create a dashed‐line PlotCurveItem
        if self.lasso_path_item is not None:
            self.plot.removeItem(self.lasso_path_item)

        pen = pg.mkPen(color=(0, 100, 255), width=2, style=Qt.PenStyle.DashLine)
        self.lasso_path_item = PlotCurveItem(pen=pen)
        self.lasso_path_item.setZValue(30)
        self.plot.addItem(self.lasso_path_item)

    def _draw_lasso(self):
        """Redraw the dashed lasso line from accumulated points."""
        if self.lasso_path_item is None or len(self.lasso_points) < 2:
            return

        xs = [p.x() for p in self.lasso_points]
        ys = [p.y() for p in self.lasso_points]
        self.lasso_path_item.setData(x=np.array(xs), y=np.array(ys))

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        """
        Test if line segment (p1→p2) intersects (p3→p4).
        Returns (True, t) where t is the parameter along p3→p4 at the
        intersection, or (False, 0) if no intersection.
        Each point is (x, y).
        """
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p4[0] - p3[0]
        dy2 = p4[1] - p3[1]

        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-12:
            return False, 0  # parallel

        dx3 = p3[0] - p1[0]
        dy3 = p3[1] - p1[1]

        t = (dx3 * dy2 - dy3 * dx2) / denom
        u = (dx3 * dy1 - dy3 * dx1) / denom

        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return True, u
        return False, 0

    def _find_segment_crossing(self):
        """
        Check if the newest lasso segment (points[-2] → points[-1]) crosses
        any earlier segment.  Skip the most recent segments to avoid false
        positives from tight drawing.

        Returns the index of the earlier segment's START point where the
        crossing was found, or -1 if none.
        """
        n = len(self.lasso_points)
        if n < 6:
            return -1

        # The new segment: second-to-last → last point
        a = self.lasso_points[-2]
        b = self.lasso_points[-1]
        seg_new = (a.x(), a.y()), (b.x(), b.y())

        # Skip the last few segments to prevent self-crossing on tight curves
        # Check segments 0→1, 1→2, …, up to (n-6)→(n-5)
        check_up_to = n - 5

        for i in range(0, check_up_to):
            c = self.lasso_points[i]
            d = self.lasso_points[i + 1]
            seg_old = (c.x(), c.y()), (d.x(), d.y())

            hit, _ = self._segments_intersect(
                seg_new[0], seg_new[1], seg_old[0], seg_old[1]
            )
            if hit:
                return i

        return -1

    def _auto_close_lasso(self, crossing_segment_idx):
        """
        The lasso path crossed itself.  Form a closed polygon from the
        crossing point to the end and select nodes inside it.
        """
        if not self.is_lasso_selecting:
            return

        # The loop runs from crossing_segment_idx to the end of lasso_points
        loop_points = self.lasso_points[crossing_segment_idx:]

        if len(loop_points) >= 3:
            poly = QPolygonF([QPointF(p.x(), p.y()) for p in loop_points])
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()

            selected_in_lasso = []
            for nid, pos in self.node_positions.items():
                if path.contains(QPointF(float(pos[0]), float(pos[1]))):
                    selected_in_lasso.append(nid)

            modifiers = QApplication.keyboardModifiers()
            ctrl = modifiers & Qt.KeyboardModifier.ControlModifier

            if not ctrl:
                self.selected_nodes = set()
            self.selected_nodes.update(selected_in_lasso)
            self.push_selection()
            self._render_nodes()
            self.node_selected.emit(list(self.selected_nodes))

        # Clean up
        self._cleanup_lasso()

    def _cleanup_lasso(self):
        """Remove lasso visuals and reset state."""
        if self.lasso_path_item is not None:
            self.plot.removeItem(self.lasso_path_item)
            self.lasso_path_item = None
        self.lasso_points.clear()
        self.is_lasso_selecting = False
        self.lasso_start_pos = None

    def _finish_lasso(self, event):
        """
        Called on mouse release.  If the lasso hasn't already auto-closed
        via self-intersection, check if the end is near the start as a
        fallback, otherwise just cancel.
        """
        if not self.is_lasso_selecting:
            return

        # Fallback: close to start point?
        if len(self.lasso_points) >= 3 and self.lasso_start_pos is not None:
            last_view = self.lasso_points[-1]
            first_view = self.lasso_start_pos
            scene_last = self.plot.vb.mapViewToScene(last_view)
            scene_first = self.plot.vb.mapViewToScene(first_view)
            dx = scene_last.x() - scene_first.x()
            dy = scene_last.y() - scene_first.y()
            pixel_dist = math.sqrt(dx * dx + dy * dy)

            if pixel_dist < self.lasso_close_threshold:
                poly = QPolygonF([QPointF(p.x(), p.y()) for p in self.lasso_points])
                path = QPainterPath()
                path.addPolygon(poly)
                path.closeSubpath()

                selected_in_lasso = []
                for nid, pos in self.node_positions.items():
                    if path.contains(QPointF(float(pos[0]), float(pos[1]))):
                        selected_in_lasso.append(nid)

                modifiers = event.modifiers()
                ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
                if not ctrl:
                    self.selected_nodes = set()
                self.selected_nodes.update(selected_in_lasso)
                self.push_selection()
                self._render_nodes()
                self.node_selected.emit(list(self.selected_nodes))

        self._cleanup_lasso()

    # ------------------------------------------------------- zoom helpers -------
    def _zoom_at_point(self, scene_pos, scale_factor):
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        vr = self.plot.viewRange()
        xc = (vr[0][0] + vr[0][1]) / 2
        yc = (vr[1][0] + vr[1][1]) / 2
        xs = vr[0][1] - vr[0][0]
        ys = vr[1][1] - vr[1][0]

        nxs = xs / scale_factor
        nys = ys / scale_factor

        xo = (view_pos.x() - xc) * (1 - 1 / scale_factor)
        yo = (view_pos.y() - yc) * (1 - 1 / scale_factor)

        nxc = xc + xo
        nyc = yc + yo

        self.plot.setXRange(nxc - nxs / 2, nxc + nxs / 2, padding=0)
        self.plot.setYRange(nyc - nys / 2, nyc + nys / 2, padding=0)

    # ------------------------------------------------------- temp pan -----------
    def _start_temp_pan(self):
        self.temp_pan_active = True
        self.plot.setMouseEnabled(x=True, y=True)

    def _end_temp_pan(self):
        self.temp_pan_active = False
        if self.selection_mode or self.zoom_mode:
            self.plot.setMouseEnabled(x=False, y=False)

    def _handle_wheel_in_selection(self, event):
        if not self.was_in_selection_before_wheel:
            self.was_in_selection_before_wheel = True
            self.plot.setMouseEnabled(x=True, y=True)

        if not self.wheel_timer:
            self.wheel_timer = QTimer()
            self.wheel_timer.setSingleShot(True)
            self.wheel_timer.timeout.connect(self._end_wheel_zoom)
        self.wheel_timer.start(1)

    def _end_wheel_zoom(self):
        if self.was_in_selection_before_wheel and (self.selection_mode or self.zoom_mode):
            self.plot.setMouseEnabled(x=False, y=False)
            self.was_in_selection_before_wheel = False

    # --------------------------------------------------- event filter -----------
    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent

        if obj != self.plot.scene():
            return super().eventFilter(obj, event)

        if event.type() == QEvent.Type.GraphicsSceneMousePress:
            # Middle mouse → temp pan
            if event.button() == Qt.MouseButton.MiddleButton:
                if self.selection_mode or self.zoom_mode:
                    self._start_temp_pan()
                    return False

            # Left button → start lasso / zoom rect detection
            elif event.button() == Qt.MouseButton.LeftButton and (self.selection_mode or self.zoom_mode):
                self.last_mouse_pos = event.scenePos()
                if not self.click_timer:
                    self.click_timer = QTimer()
                    self.click_timer.setSingleShot(True)
                    self.click_timer.timeout.connect(self._on_long_press)
                self.click_timer.start(200)

        elif event.type() == QEvent.Type.GraphicsSceneMouseMove:
            # Detect significant movement → start lasso
            if (self.selection_mode or self.zoom_mode) and self.click_timer and self.click_timer.isActive():
                if self.last_mouse_pos:
                    cur = event.scenePos()
                    dx = abs(cur.x() - self.last_mouse_pos.x())
                    dy = abs(cur.y() - self.last_mouse_pos.y())
                    if dx > 10 or dy > 10:
                        self.click_timer.stop()
                        if not self.is_lasso_selecting:
                            if self.selection_mode:
                                self._start_lasso(self.last_mouse_pos)
                            elif self.zoom_mode:
                                self._start_area_selection_rect(self.last_mouse_pos)

        elif event.type() == QEvent.Type.GraphicsSceneMouseRelease:
            if event.button() == Qt.MouseButton.MiddleButton:
                if self.temp_pan_active:
                    self._end_temp_pan()
                    return False

            elif event.button() == Qt.MouseButton.LeftButton and (self.selection_mode or self.zoom_mode):
                if self.click_timer and self.click_timer.isActive():
                    self.click_timer.stop()

                if self.is_lasso_selecting:
                    self._finish_lasso(event)
                    return True
                elif self._is_zoom_rect_selecting():
                    self._finish_zoom_rect(event)
                    return True

            # Zoom mode click zoom
            if self.zoom_mode and not self._is_zoom_rect_selecting():
                if event.button() == Qt.MouseButton.LeftButton:
                    self._zoom_at_point(event.scenePos(), 2)
                    return True
                elif event.button() == Qt.MouseButton.RightButton:
                    self._zoom_at_point(event.scenePos(), 0.5)
                    return True

        elif event.type() == QEvent.Type.GraphicsSceneWheel:
            if self.selection_mode or self.zoom_mode:
                self._handle_wheel_in_selection(event)
                return False

        return super().eventFilter(obj, event)

    def _on_long_press(self):
        if self.selection_mode and self.last_mouse_pos:
            self._start_lasso(self.last_mouse_pos)
        elif self.zoom_mode and self.last_mouse_pos:
            self._start_area_selection_rect(self.last_mouse_pos)

    # ------------------------------------------------ zoom rect (zoom mode only) -
    def _start_area_selection_rect(self, scene_pos):
        """Rectangle area select used only in zoom mode."""
        self._zoom_rect_active = True
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        self._zoom_rect_start = view_pos

        if hasattr(self, '_zoom_rect_roi') and self._zoom_rect_roi is not None:
            self.plot.removeItem(self._zoom_rect_roi)

        self._zoom_rect_roi = pg.ROI(
            [view_pos.x(), view_pos.y()], [0, 0],
            pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine),
            movable=False, resizable=False
        )
        self._zoom_rect_roi.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.plot.addItem(self._zoom_rect_roi)

    def _is_zoom_rect_selecting(self):
        return getattr(self, '_zoom_rect_active', False)

    def _finish_zoom_rect(self, event):
        if not self._is_zoom_rect_selecting():
            return
        roi = getattr(self, '_zoom_rect_roi', None)
        if roi is None:
            self._zoom_rect_active = False
            return

        rp = roi.pos()
        rs = roi.size()
        x_min, y_min = rp[0], rp[1]
        x_max, y_max = x_min + rs[0], y_min + rs[1]
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        self.plot.removeItem(roi)
        self._zoom_rect_roi = None
        self._zoom_rect_active = False

        xr = x_max - x_min
        yr = y_max - y_min
        if xr > 0 and yr > 0:
            pad = 0.05
            self.plot.setXRange(x_min - pad * xr, x_max + pad * xr, padding=0)
            self.plot.setYRange(y_min - pad * yr, y_max + pad * yr, padding=0)

    # ----------------------------------------------------------- utility --------
    def update_params(self, community_dict=None, identity_dict=None,
                      labels=None, node_size=None, color_mode=None):
        """Update visualization parameters without recomputing the embedding."""
        if community_dict is not None:
            self.community_dict = community_dict
        if identity_dict is not None:
            self.identity_dict = identity_dict
        if labels is not None:
            self.labels = labels
        if node_size is not None:
            self.node_size = node_size
        if color_mode is not None:
            self.color_mode = color_mode

    def show_in_window(self, title="UMAP Visualization", width=1000, height=800):
        self.popup_window = _UMAPPopupWindow(self, parent=None)
        self.popup_window.setWindowTitle(title)
        self.popup_window.setGeometry(100, 100, width, height)
        self.popup_window.setCentralWidget(self)
        self.show()
        self.popup_window.show()
        return self.popup_window




class _UMAPPopupWindow(QMainWindow):
    """Popup window that hides + clears the UMAP widget instead of destroying it."""
    def __init__(self, umap_widget, parent=None):
        super().__init__(parent)
        self._umap_widget = umap_widget

    def closeEvent(self, event: QCloseEvent):
        # Take the widget back out before the window destroys it
        self.setCentralWidget(None)
        self._umap_widget.setParent(self._umap_widget.parent_window)
        self._umap_widget.hide()
        self._umap_widget._clear_plot()
        self._umap_widget.node_ids.clear()
        self._umap_widget.embedding = None
        event.accept()
        
# ----------------------------------------------------------------- Example ----
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    class DummyParent(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("UMAP Widget Demo")
            self.setGeometry(100, 100, 1000, 800)
            self.clicked_values = {'nodes': []}

            self.umap_widget = UMAPGraphWidget(
                parent=self,
                labels=True,
                color_mode='community'
            )
            self.setCentralWidget(self.umap_widget)

            # Generate some fake data
            np.random.seed(0)
            n_nodes = 200
            n_features = 20
            cluster_data = {}
            community_dict = {}
            identity_dict = {}
            for i in range(n_nodes):
                cluster_data[i] = np.random.rand(n_features)
                community_dict[i] = i % 5
                identity_dict[i] = f"Type_{i % 3}"

            # A few nodes with no community
            del community_dict[0]
            del community_dict[1]

            self.umap_widget.set_data(
                cluster_data,
                community_dict=community_dict,
                identity_dict=identity_dict,
                color_mode='community'
            )

            self.umap_widget.node_selected.connect(self._on_sel)

        def _on_sel(self, nodes):
            if nodes:
                print(f"Selected: {nodes}")

        def evaluate_mini(self, **kw):
            pass

        def handle_info(self, *a, **kw):
            pass

    app = QApplication(sys.argv)
    win = DummyParent()
    win.show()
    sys.exit(app.exec())