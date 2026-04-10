import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QSizePolicy, QApplication, QFileDialog,
                              QMessageBox, QMainWindow)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF
from PyQt6.QtGui import QPainterPath, QPolygonF, QCloseEvent
import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem, PlotCurveItem
import json
import math


class FlowCytometryWidget(QWidget):
    """Interactive flow-cytometry-style scatter plot.

    Each axis represents the expression of a marker, displayed on an
    asinh scale (inverse hyperbolic sine — the standard computational
    approximation to the biexponential/logicle transform).  Raw
    compensated intensities are passed in and transformed internally.

    The asinh transform is logarithmic for large values, linear near
    zero, and handles negatives naturally — giving better separation
    across the full dynamic range than pure log10.

    Parameters
    ----------
    data : dict
        ``{node_id: [{x_marker_name: raw_intensity}, {y_marker_name: raw_intensity}]}``
        The two single-key dicts specify the X and Y values respectively.
        Marker names are read from the first node and used as axis labels.
    parent : QWidget, optional
        Parent window.  If it exposes ``clicked_values``, ``evaluate_mini``,
        and ``handle_info`` they will be used for selection push-back.
    node_size : float
        Base node radius in pixels.
    cofactor : float
        Divisor for the asinh transform: ``asinh(x / cofactor)``.
        ~150 for fluorescence cytometry, ~5 for CyTOF / mass cytometry.
    background : str
        ``'white'``, ``'black'``, or ``'green'``.
    """

    node_selected = pyqtSignal(object)  # emits list of selected node IDs

    def __init__(self, parent=None, data=None, node_size=10,
                 cofactor=150, background='white'):
        super().__init__(parent)

        self.parent_window = parent
        self.node_size = node_size
        self.cofactor = cofactor

        _bg_map = {'white': 'w', 'black': '#1a1a2e', 'green': '#c8d5a3'}
        self.background = _bg_map.get(background, 'w')

        # --- data ---
        self.node_ids = []
        self.embedding = None           # Nx2 array (transformed intensities)
        self.node_positions = {}        # {node_id: np.array([x, y])}
        self.selected_nodes = set()
        self.rendered = False
        self.x_marker = ''
        self.y_marker = ''

        # --- raw data kept for save ---
        self._source_data = {}

        # --- caching ---
        self.cached_node_to_index = {}
        self.last_selected_set = set()

        # --- fast array rendering ---
        self._pos_array = None
        self._size_array = None
        self._brush_list = None
        self._normal_brush_list = None
        self._data_list = None
        self._base_point_size = 10.0

        self._selected_brush = pg.mkBrush(255, 255, 0, 255)
        self._highlight_size_boost = 3.0

        # --- LOD debounce ---
        self._lod_timer = None

        # --- interaction mode ---
        self.selection_mode = True
        self.zoom_mode = False

        # --- lasso ---
        self.lasso_points = []
        self.lasso_path_item = None
        self.is_lasso_selecting = False
        self.lasso_start_pos = None
        self.lasso_close_threshold = 15

        # --- click detection ---
        self.click_timer = None
        self.last_mouse_pos = None

        # --- middle mouse pan ---
        self.temp_pan_active = False

        # --- wheel zoom ---
        self.wheel_timer = None
        self.was_in_selection_before_wheel = False

        # --- zoom LOD ---
        self.current_zoom_factor = 1.0

        self._setup_ui()

        if data is not None:
            self.set_data(data)

    # ------------------------------------------------------------------ UI --
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # pyqtgraph widget
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(self.background)
        self.plot = self.graphics_widget.addPlot()
        self.plot.setAspectLocked(False)

        # axis labels (will be set once data arrives)
        self.plot.setLabel('bottom', '')
        self.plot.setLabel('left', '')
        self.plot.showGrid(x=True, y=True, alpha=0.15)

        # placeholder
        self.loading_text = pg.TextItem(
            text="No data loaded",
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

        # base scatter
        self.scatter = ScatterPlotItem(size=10, pen=pg.mkPen(None),
                                        brush=pg.mkBrush(74, 144, 226, 200))
        self.plot.addItem(self.scatter)
        self.scatter.sigClicked.connect(self._on_node_clicked)

        # highlight overlay
        self.highlight_scatter = ScatterPlotItem(size=12, pen=pg.mkPen(None),
                                                  brush=pg.mkBrush(255, 255, 0, 255))
        self.highlight_scatter.setZValue(15)
        self.plot.addItem(self.highlight_scatter)
        self.highlight_scatter.sigClicked.connect(self._on_highlight_node_clicked)

        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self.plot.sigRangeChanged.connect(self._on_view_changed)

        layout.addWidget(self.graphics_widget, stretch=1)

        # control panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)

        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)

        # event filter
        self.graphics_widget.viewport().installEventFilter(self)
        self.plot.scene().installEventFilter(self)

    # ------------------------------------------------------- control panel --
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
        self.zoom_btn.setToolTip("Zoom Tool")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.setMaximumSize(32, 32)
        self.zoom_btn.clicked.connect(self._toggle_zoom_mode)

        self.home_btn = QPushButton("🏠")
        self.home_btn.setToolTip("Reset View")
        self.home_btn.setMaximumSize(32, 32)
        self.home_btn.clicked.connect(self._reset_view)

        self.save_btn = QPushButton("💾")
        self.save_btn.setToolTip("Save Plot Data")
        self.save_btn.setMaximumSize(32, 32)
        self.save_btn.clicked.connect(self._save_data)

        panel_layout.addWidget(self.select_btn)
        panel_layout.addWidget(self.pan_btn)
        panel_layout.addWidget(self.zoom_btn)
        panel_layout.addWidget(self.home_btn)
        panel_layout.addWidget(self.save_btn)
        panel_layout.addStretch()

        panel.setLayout(panel_layout)
        panel.setMaximumHeight(40)
        return panel

    # ---------------------------------------------------- save ----------------
    def _save_data(self):
        """Save the source data dict to a JSON file."""
        if not self._source_data:
            QMessageBox.warning(self, "Nothing to save",
                                "Load data first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Flow Cytometry Data", "",
            "JSON Files (*.json)")
        if not filename:
            return
        if not filename.endswith('.json'):
            filename += '.json'

        # Make keys JSON-serialisable (str)
        out = {}
        for node_id, pair in self._source_data.items():
            key = str(node_id) if not isinstance(node_id, str) else node_id
            serialisable_pair = []
            for d in pair:
                serialisable_pair.append(
                    {k: float(v) for k, v in d.items()})
            out[key] = serialisable_pair

        try:
            with open(filename, 'w') as f:
                json.dump(out, f, indent=2)
            QMessageBox.information(self, "Saved",
                                    f"Data saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    # ------------------------------------------------- main public API ------
    def _asinh_transform(self, values):
        """Inverse hyperbolic sine transform: asinh(x / cofactor).

        Behaves like log for large values, linear near zero, and handles
        negatives naturally (no clipping).  This is the standard
        computational approximation to the biexponential/logicle
        transform used in FlowJo.

        Typical cofactor values:
            ~150  for fluorescence cytometry
            ~5    for mass cytometry (CyTOF)
        """
        return np.arcsinh(values / self.cofactor)

    def set_data(self, data):
        """
        Load and render flow cytometry data.

        Parameters
        ----------
        data : dict
            ``{node_id: [{x_marker: raw_intensity}, {y_marker: raw_intensity}]}``
            Values are raw (compensated) fluorescence intensities.
            They will be asinh-transformed for display.
        """
        self._source_data = data
        self._clear_plot()
        self._remove_loading_text()

        node_ids = list(data.keys())
        if not node_ids:
            return

        # Extract marker names from the first entry
        first = data[node_ids[0]]
        self.x_marker = list(first[0].keys())[0]
        self.y_marker = list(first[1].keys())[0]

        # Build Nx2 embedding — asinh-transformed raw intensities
        xs_raw = np.array([data[n][0][self.x_marker] for n in node_ids], dtype=np.float64)
        ys_raw = np.array([data[n][1][self.y_marker] for n in node_ids], dtype=np.float64)
        xs = self._asinh_transform(xs_raw)
        ys = self._asinh_transform(ys_raw)
        embedding = np.column_stack([xs, ys])

        self.node_ids = node_ids
        self.embedding = embedding
        self.node_positions = {nid: embedding[i]
                               for i, nid in enumerate(node_ids)}

        # Axis labels
        self.plot.setLabel('bottom', f'{self.x_marker}  (asinh, cofactor={self.cofactor})')
        self.plot.setLabel('left', f'{self.y_marker}  (asinh, cofactor={self.cofactor})')

        self._build_and_render()

    def load_from_file(self, filepath):
        """Load previously saved JSON data and render it."""
        try:
            with open(filepath, 'r') as f:
                raw = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
            return

        # Attempt to restore original key types (int if possible)
        data = {}
        for k, v in raw.items():
            try:
                key = int(k)
            except (ValueError, TypeError):
                key = k
            data[key] = v

        self.set_data(data)

    # --------------------------------------------------- internal render ----
    def _build_and_render(self):
        n = len(self.node_ids)
        if n == 0:
            return

        # Dynamic point size
        if n < 50:
            point_size = self.node_size
        elif n < 200:
            point_size = max(4, self.node_size * 0.7)
        elif n < 1000:
            point_size = max(3, self.node_size * 0.45)
        else:
            point_size = max(2, self.node_size * 0.25)
        self._base_point_size = float(point_size)

        # Uniform colour (flow cytometry blue)
        default_brush = pg.mkBrush(74, 144, 226, 180)
        normal_brushes = [default_brush] * n

        self._pos_array = self.embedding.copy()
        self._size_array = np.full(n, point_size, dtype=np.float64)
        self._normal_brush_list = list(normal_brushes)
        self._brush_list = list(normal_brushes)
        self._data_list = list(self.node_ids)

        self.cached_node_to_index = {nid: i for i, nid in enumerate(self.node_ids)}

        self.highlight_scatter.clear()

        self.scatter.setData(
            pos=self._pos_array,
            size=self._size_array,
            brush=self._brush_list,
            data=self._data_list,
            pen=None,
        )
        self.scatter.setZValue(10)

        self.rendered = True
        self.current_zoom_factor = 1.0
        self.graphics_widget.setBackground(self.background)

        self.plot.blockSignals(True)
        self._reset_view()
        self.plot.blockSignals(False)

        # Re-apply parent selection if present
        if (self.parent_window is not None
                and hasattr(self.parent_window, 'clicked_values')
                and len(self.parent_window.clicked_values.get('nodes', [])) > 0):
            self.select_nodes(self.parent_window.clicked_values['nodes'])

    # ------------------------------------------------------- node selection -
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

    def push_selection(self):
        try:
            if self.parent_window is not None and hasattr(self.parent_window, 'clicked_values'):
                self.parent_window.clicked_values['nodes'] = list(self.selected_nodes)
                if hasattr(self.parent_window, 'evaluate_mini'):
                    self.parent_window.evaluate_mini(subgraph_push=True)
                if hasattr(self.parent_window, 'handle_info'):
                    self.parent_window.handle_info('node')
        except Exception:
            pass

    # ------------------------------------------------------- render helpers -
    def _flush_brushes(self):
        if self._pos_array is not None:
            self.scatter.setData(
                pos=self._pos_array,
                size=self._size_array,
                brush=self._brush_list,
                data=self._data_list,
                pen=None,
            )

    def _render_nodes(self):
        if self._pos_array is None:
            return
        newly_selected = self.selected_nodes - self.last_selected_set
        newly_deselected = self.last_selected_set - self.selected_nodes
        if not newly_selected and not newly_deselected:
            return
        self._update_highlight_scatter()
        self.last_selected_set = self.selected_nodes.copy()

    def _update_highlight_scatter(self):
        if not self.selected_nodes:
            self.highlight_scatter.clear()
            return
        cached_idx = self.cached_node_to_index
        indices = [cached_idx[n] for n in self.selected_nodes if n in cached_idx]
        if not indices:
            self.highlight_scatter.clear()
            return
        idx_arr = np.array(indices, dtype=np.int64)
        sel_pos = self._pos_array[idx_arr]
        sel_data = [self.node_ids[i] for i in indices]
        current_size = float(self._size_array[0]) if len(self._size_array) > 0 else self._base_point_size
        sel_size = current_size + self._highlight_size_boost
        self.highlight_scatter.setData(
            pos=sel_pos,
            size=sel_size,
            brush=self._selected_brush,
            data=sel_data,
            pen=None,
        )

    # ------------------------------------------------------- view / zoom ----
    def _on_view_changed(self):
        if not self.node_positions or self.embedding is None:
            return
        view_range = self.plot.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]
        pos_array = self.embedding
        full_x = pos_array[:, 0].max() - pos_array[:, 0].min()
        full_y = pos_array[:, 1].max() - pos_array[:, 1].min()
        if full_x > 0 and full_y > 0:
            zoom_x = full_x / x_range if x_range > 0 else 1
            zoom_y = full_y / y_range if y_range > 0 else 1
            zoom_factor = max(zoom_x, zoom_y)
            zoom_changed = abs(zoom_factor - self.current_zoom_factor) / max(self.current_zoom_factor, 0.01) > 0.05
            if zoom_changed:
                self.current_zoom_factor = zoom_factor
                self._schedule_lod_update()

    def _schedule_lod_update(self):
        if self._lod_timer is None:
            self._lod_timer = QTimer()
            self._lod_timer.setSingleShot(True)
            self._lod_timer.timeout.connect(self._apply_lod_update)
        self._lod_timer.start(30)

    def _apply_lod_update(self):
        self._update_lod_rendering()

    def _update_lod_rendering(self):
        if self._size_array is None or len(self._size_array) == 0:
            return
        zf = self.current_zoom_factor
        if zf <= 0.5:
            scale_factor = 0.5
        elif zf <= 1.0:
            scale_factor = 0.5 + 0.5 * (zf / 1.0)
        else:
            scale_factor = 1.0 + (math.sqrt(zf) - 1.0) * 1.0
        scale_factor = min(scale_factor, 8.0)
        new_size = self._base_point_size * scale_factor
        try:
            if (self.scatter.data is not None
                    and len(self.scatter.data) == len(self._size_array)):
                self._size_array[:] = new_size
                self.scatter.data['size'] = new_size
                self.scatter.updateSpots()
                self.scatter.prepareGeometryChange()
                self.scatter.bounds = [None, None]
                self.scatter.update()
            else:
                self._size_array[:] = new_size
                self._flush_brushes()
        except (AttributeError, TypeError):
            self._size_array[:] = new_size
            self._flush_brushes()
        if self.selected_nodes:
            self._update_highlight_scatter()

    # ------------------------------------------------------- mode toggles ---
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

    # ------------------------------------------------------- reset / clear --
    def _reset_view(self):
        if self.embedding is None or len(self.embedding) == 0:
            return
        x_min, y_min = self.embedding.min(axis=0)
        x_max, y_max = self.embedding.max(axis=0)
        padding = 0.1
        xr = x_max - x_min or 1.0
        yr = y_max - y_min or 1.0
        self.plot.setXRange(x_min - padding * xr, x_max + padding * xr, padding=0)
        self.plot.setYRange(y_min - padding * yr, y_max + padding * yr, padding=0)

    def _clear_plot(self):
        self._remove_loading_text()
        self.scatter.clear()
        self.highlight_scatter.clear()

        # Remove stray TextItems
        items_to_remove = [item for item in self.plot.items
                           if isinstance(item, pg.TextItem)]
        for item in items_to_remove:
            self.plot.removeItem(item)

        if self.lasso_path_item is not None:
            self.plot.removeItem(self.lasso_path_item)
            self.lasso_path_item = None
        self.lasso_points.clear()
        self.is_lasso_selecting = False

        self.node_positions.clear()
        self.selected_nodes.clear()
        self.cached_node_to_index.clear()
        self.last_selected_set.clear()
        self.rendered = False

        self._pos_array = None
        self._size_array = None
        self._brush_list = None
        self._normal_brush_list = None
        self._data_list = None

        if self._lod_timer is not None and self._lod_timer.isActive():
            self._lod_timer.stop()

    def _remove_loading_text(self):
        if hasattr(self, 'loading_text') and self.loading_text is not None:
            self.plot.removeItem(self.loading_text)
            self.loading_text = None

    # ------------------------------------------------------- mouse events ---
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

    def _on_highlight_node_clicked(self, scatter, points, ev):
        if not self.selection_mode or len(points) == 0:
            return
        clicked_node = points[0].data()
        modifiers = ev.modifiers()
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
        if ctrl:
            self.selected_nodes.discard(clicked_node)
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
        if not self.is_lasso_selecting:
            return
        view_pos = self.plot.vb.mapSceneToView(pos)
        self.lasso_points.append(view_pos)
        self._draw_lasso()
        cross_idx = self._find_segment_crossing()
        if cross_idx >= 0:
            self._auto_close_lasso(cross_idx)

    # ------------------------------------------------------- lasso ----------
    def _start_lasso(self, scene_pos):
        self.is_lasso_selecting = True
        self.node_click_flag = False
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        self.lasso_start_pos = view_pos
        self.lasso_points = [view_pos]
        if self.lasso_path_item is not None:
            self.plot.removeItem(self.lasso_path_item)
        pen = pg.mkPen(color=(0, 100, 255), width=2, style=Qt.PenStyle.DashLine)
        self.lasso_path_item = PlotCurveItem(pen=pen)
        self.lasso_path_item.setZValue(30)
        self.plot.addItem(self.lasso_path_item)

    def _draw_lasso(self):
        if self.lasso_path_item is None or len(self.lasso_points) < 2:
            return
        xs = [p.x() for p in self.lasso_points]
        ys = [p.y() for p in self.lasso_points]
        self.lasso_path_item.setData(x=np.array(xs), y=np.array(ys))

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p4[0] - p3[0]
        dy2 = p4[1] - p3[1]
        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-12:
            return False, 0
        dx3 = p3[0] - p1[0]
        dy3 = p3[1] - p1[1]
        t = (dx3 * dy2 - dy3 * dx2) / denom
        u = (dx3 * dy1 - dy3 * dx1) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return True, u
        return False, 0

    def _find_segment_crossing(self):
        n = len(self.lasso_points)
        if n < 6:
            return -1
        a = self.lasso_points[-2]
        b = self.lasso_points[-1]
        seg_new = (a.x(), a.y()), (b.x(), b.y())
        check_up_to = n - 5
        for i in range(0, check_up_to):
            c = self.lasso_points[i]
            d = self.lasso_points[i + 1]
            seg_old = (c.x(), c.y()), (d.x(), d.y())
            hit, _ = self._segments_intersect(
                seg_new[0], seg_new[1], seg_old[0], seg_old[1])
            if hit:
                return i
        return -1

    def _auto_close_lasso(self, crossing_segment_idx):
        if not self.is_lasso_selecting:
            return
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
        self._cleanup_lasso()

    def _cleanup_lasso(self):
        if self.lasso_path_item is not None:
            self.plot.removeItem(self.lasso_path_item)
            self.lasso_path_item = None
        self.lasso_points.clear()
        self.is_lasso_selecting = False
        self.lasso_start_pos = None

    def _finish_lasso(self, event):
        if not self.is_lasso_selecting:
            return
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

    # ------------------------------------------------------- zoom helpers ---
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

    # ------------------------------------------------------- temp pan -------
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

    # --------------------------------------------------- zoom rect ----------
    def _start_area_selection_rect(self, scene_pos):
        self._zoom_rect_active = True
        view_pos = self.plot.vb.mapSceneToView(scene_pos)
        self._zoom_rect_start = view_pos
        if hasattr(self, '_zoom_rect_roi') and self._zoom_rect_roi is not None:
            self.plot.removeItem(self._zoom_rect_roi)
        self._zoom_rect_roi = pg.ROI(
            [view_pos.x(), view_pos.y()], [0, 0],
            pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine),
            movable=False, resizable=False)
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

    # --------------------------------------------------- event filter -------
    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent

        if obj != self.plot.scene():
            return super().eventFilter(obj, event)

        if event.type() == QEvent.Type.GraphicsSceneMousePress:
            if event.button() == Qt.MouseButton.MiddleButton:
                if self.selection_mode or self.zoom_mode:
                    self._start_temp_pan()
                    return False
            elif event.button() == Qt.MouseButton.LeftButton and (self.selection_mode or self.zoom_mode):
                self.last_mouse_pos = event.scenePos()
                if not self.click_timer:
                    self.click_timer = QTimer()
                    self.click_timer.setSingleShot(True)
                    self.click_timer.timeout.connect(self._on_long_press)
                self.click_timer.start(200)

        elif event.type() == QEvent.Type.GraphicsSceneMouseMove:
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
            # Update zoom rect if active
            if self._is_zoom_rect_selecting():
                view_pos = self.plot.vb.mapSceneToView(event.scenePos())
                start = self._zoom_rect_start
                w = view_pos.x() - start.x()
                h = view_pos.y() - start.y()
                self._zoom_rect_roi.setSize([w, h])

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

    # ------------------------------------------------------- popup window ---
    def show_in_window(self, title="Flow Cytometry Plot", width=900, height=700):
        self.popup_window = _FlowPopupWindow(self, parent=None)
        self.popup_window.setWindowTitle(title)
        self.popup_window.setGeometry(100, 100, width, height)
        self.popup_window.setCentralWidget(self)
        self.show()
        self.popup_window.show()
        return self.popup_window


class _FlowPopupWindow(QMainWindow):
    """Popup window that hides the widget instead of destroying it."""
    def __init__(self, flow_widget, parent=None):
        super().__init__(parent)
        self._flow_widget = flow_widget

    def closeEvent(self, event: QCloseEvent):
        self.setCentralWidget(None)
        self._flow_widget.setParent(self._flow_widget.parent_window)
        self._flow_widget.hide()
        self._flow_widget._clear_plot()
        self._flow_widget.node_ids.clear()
        self._flow_widget.embedding = None
        event.accept()


# ----------------------------------------------------------------- Example --
if __name__ == "__main__":
    import sys

    class DummyParent(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Flow Cytometry Widget Demo")
            self.setGeometry(100, 100, 900, 700)
            self.clicked_values = {'nodes': []}

            # Generate fake flow cytometry data (raw fluorescence intensities)
            # Includes some near-zero / negative values (compensation artifacts)
            # to demonstrate the asinh transform's advantage over pure log
            np.random.seed(42)
            n_nodes = 300
            data = {}
            for i in range(n_nodes):
                if i < n_nodes // 3:
                    # CD4-bright / CD8-dim population
                    x_val = 10 ** np.random.normal(3.5, 0.4)   # ~1000–10000
                    y_val = np.random.normal(50, 80)            # dim, some negative
                elif i < 2 * n_nodes // 3:
                    # CD4-dim / CD8-bright population
                    x_val = np.random.normal(50, 80)
                    y_val = 10 ** np.random.normal(3.5, 0.4)
                else:
                    # Double-negative population (near zero on both)
                    x_val = np.random.normal(30, 60)
                    y_val = np.random.normal(30, 60)
                data[i] = [{'CD4': x_val}, {'CD8': y_val}]

            self.flow_widget = FlowCytometryWidget(
                parent=self,
                data=data,
                node_size=10,
                background='white'
            )
            self.setCentralWidget(self.flow_widget)

            self.flow_widget.node_selected.connect(self._on_sel)

        def _on_sel(self, nodes):
            if nodes:
                print(f"Selected {len(nodes)} nodes: {nodes[:10]}{'...' if len(nodes) > 10 else ''}")

        def evaluate_mini(self, **kw):
            pass

        def handle_info(self, *a, **kw):
            pass

    app = QApplication(sys.argv)
    win = DummyParent()
    win.show()
    sys.exit(app.exec())