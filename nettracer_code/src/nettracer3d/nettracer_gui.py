import sys
import networkx as nx
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, 
                            QHBoxLayout, QSlider, QMenuBar, QMenu, QDialog, 
                            QFormLayout, QLineEdit, QPushButton, QFileDialog,
                            QLabel, QComboBox, QMessageBox, QTableView, QInputDialog,
                            QMenu, QTabWidget, QGroupBox, QCheckBox, QScrollArea,
                            QListWidget, QListWidgetItem, QRadioButton, QButtonGroup,
                            QFrame, QSplitter, QWidget, QToolButton)
from PyQt6.QtCore import (QPoint, Qt, QAbstractTableModel, QTimer,  QThread, pyqtSignal, QObject, QCoreApplication, QEvent, QEventLoop, QMimeData)
from PyQt6 import QtCore
import numpy as np
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from qtrangeslider import QRangeSlider
from nettracer3d import nettracer as n3d
from nettracer3d import proximity as pxt
from nettracer3d import smart_dilate as sdl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from PyQt6.QtGui import (QFont, QCursor, QColor, QPixmap, QFontMetrics, QPainter, QPen, QShortcut, QKeySequence, QDrag, QAction)
import tifffile
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from nettracer3d import segmenter
try:
    from nettracer3d import segmenter_GPU as seg_GPU
except:
    pass
from nettracer3d import excelotron
import threading
import queue
from threading import Lock
from scipy import ndimage
import pyqtgraph as pg
import os
from . import painting
from . import stats as net_stats
from . import network_graph_widget as ngw
from . import umap_widget as umw
from . import threshold_predictor
import ast
import math
import random


class ImageViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NetTracer3D")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize channel data and states
        self.channel_data = [None] * 5
        self.channel_visible = [False] * 5
        self.current_slice = 0
        self.active_channel = 0  # Initialize active channel
        self.node_name = "Root_Nodes"
        self.last_saved = None
        self.last_saved_name = None
        self.last_load = None
        self.temp_chan = 0
        self.scale_bar = False

        self.color_dictionary = {
        # Reds
        "RED": (1, 0, 0),
        "LIGHT_RED": (1, 0.3, 0.3),
        "DARK_RED": (0.6, 0, 0),
        "CORAL": (1, 0.5, 0.3),
        
        # Oranges
        "ORANGE": (1, 0.5, 0),
        "LIGHT_ORANGE": (1, 0.7, 0.3),
        "DARK_ORANGE": (0.8, 0.3, 0),
        
        # Yellows
        "YELLOW": (1, 1, 0),
        "LIGHT_YELLOW": (1, 1, 0.5),
        "GOLD": (1, 0.84, 0),
        
        # Greens
        "GREEN": (0, 1, 0),
        "LIGHT_GREEN": (0.3, 1, 0.3),
        "DARK_GREEN": (0, 0.6, 0),
        "LIME": (0.6, 1, 0),
        "FOREST_GREEN": (0.13, 0.55, 0.13),
        
        # Blues
        "BLUE": (0, 0, 1),
        "LIGHT_BLUE": (0.3, 0.3, 1),
        "DARK_BLUE": (0, 0, 0.6),
        "ROYAL_BLUE": (0.25, 0.41, 0.88),
        "NAVY": (0, 0, 0.5),
        
        # Cyans
        "CYAN": (0, 1, 1),
        "LIGHT_CYAN": (0.5, 1, 1),
        "DARK_CYAN": (0, 0.6, 0.6),
        
        # Purples
        "PURPLE": (0.5, 0, 0.5),
        "LIGHT_PURPLE": (0.8, 0.6, 0.8),
        "VIOLET": (0.93, 0.51, 0.93),
        "MAGENTA": (1, 0, 1),
        
        # Neutrals
        "WHITE": (1, 1, 1),
        "GRAY": (0.5, 0.5, 0.5),
        "LIGHT_GRAY": (0.8, 0.8, 0.8),
        "DARK_GRAY": (0.2, 0.2, 0.2),
        }

        self.base_colors = [ #Channel colors
            self.color_dictionary['LIGHT_RED'],    # Lighter red
            self.color_dictionary['LIGHT_GREEN'],    # Lighter green
            self.color_dictionary['WHITE'],        # White
            self.color_dictionary['CYAN']         # Now cyan
        ]
        
        
        # Initialize selection state
        self.selecting = False
        self.selection_start = None
        self.selection_rect = None
        self.click_start_time = None  # Add this to track when click started
        self.selection_threshold = 1.0  # Time in seconds before starting rectangle selection
        self.background = None
        self.last_update_time = 0
        self.update_interval = 0.008  # 60 FPS
        
        # Initialize zoom mode state
        self.zoom_mode = False
        self.original_xlim = None
        self.original_ylim = None
        self.zoom_changed = False

        # Pan mode state
        self.pan_mode = False
        #self.pan_modes = True
        self.panning = False
        self.pan_start = None
        self.img_width = None
        self.img_height = None
        self.pre_pan_channel_state = None  # Store which channels were visible before pan
        self.is_pan_preview = False        # Track if we're in pan preview mode
        self.pan_background_image = None     # Store the rendered composite image
        self.pan_zoom_state = None           # Store zoom state when pan began
        self.is_pan_preview = False          # Track if we're in pan preview mode

        #For ML segmenting mode
        self.brush_mode = False
        self.can = False
        self.threed = False
        self.threedthresh = 5
        self.painting = False
        self.foreground = True
        self.machine_window = None
        self.brush_size = 1  # Start with 1 pixel
        self.min_brush_size = 1
        self.max_brush_size = 10
        
        # Store brightness/contrast values for each channel
        self.channel_brightness = [{
            'min': 0,
            'max': 1
        } for _ in range(5)]
        
        # Create the brightness dialog but don't show it yet
        self.brightness_dialog = BrightnessContrastDialog(self)
        
        self.min_max = {
            0: [0,1],
            1: [0,1],
            2: [0,1],
            3: [0,1]
        }

        self.volume_dict = {
            0: None,
            1: None,
            2: None,
            3: None
        } #For storing thresholding information

        self.radii_dict = {
            0: None,
            1: None,
            2: None,
            3: None
        }

        self.surface_area_dict = {
            0: None,
            1: None,
            2: None,
            3: None
        }

        self.sphericity_dict = {
            0: None,
            1: None,
            2: None,
            3: None
        }

        self.branch_dict = {
            0: None,
            1: None

        }
        self.stats_dict = {}

        self.original_shape = None #For undoing resamples
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Create active channel selector
        active_channel_widget = QWidget()
        active_channel_layout = QHBoxLayout(active_channel_widget)
        
        active_label = QLabel("Active Image:")
        active_channel_layout.addWidget(active_label)
        
        self.active_channel_combo = QComboBox()
        self.active_channel_combo.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2"])
        self.active_channel_combo.setCurrentIndex(0)
        self.active_channel_combo.currentIndexChanged.connect(self.set_active_channel)
        # Initially disable the combo box
        self.active_channel_combo.setEnabled(True)
        active_channel_layout.addWidget(self.active_channel_combo)
        
        control_layout.addWidget(active_channel_widget)

        # Create zoom button and pan button
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)


        self.toggle_scale = QPushButton("📏")
        self.toggle_scale.setToolTip("Toggle Scalebar/Grid")
        self.toggle_scale.setFixedSize(32, 32)
        self.toggle_scale.clicked.connect(self.toggle_scalebar)
        self.toggle_scale.setCheckable(True)
        self.toggle_scale.setChecked(False)
        control_layout.addWidget(self.toggle_scale)

        self.reset_view = QPushButton("🏠")
        self.reset_view.setToolTip("Reset View")
        self.reset_view.setFixedSize(32, 32)
        self.reset_view.clicked.connect(self.home)
        control_layout.addWidget(self.reset_view)

        # "Create" zoom button
        self.zoom_button = QPushButton("🔍")
        self.zoom_button.setToolTip("Zoom Mode")
        self.zoom_button.setCheckable(True)
        self.zoom_button.setFixedSize(40, 40)
        self.zoom_button.clicked.connect(self.toggle_zoom_mode)
        buttons_layout.addWidget(self.zoom_button)
        self.resizing = False

        self.pan_button = QPushButton("✋")
        self.pan_button.setToolTip("Pan Mode")
        self.pan_button.setCheckable(True)
        self.pan_button.setFixedSize(40, 40)
        self.pan_button.clicked.connect(self.toggle_pan_mode)
        buttons_layout.addWidget(self.pan_button)

        self.high_button = QPushButton("👁️")
        self.high_button.setToolTip("Toggle Highlight")
        self.high_button.setCheckable(True)
        self.high_button.setFixedSize(40, 40)
        self.high_button.clicked.connect(self.toggle_highlight)
        self.high_button.setChecked(True)
        buttons_layout.addWidget(self.high_button)
        self.highlight = True
        self.needs_mini = False

        self.pen_button = QPushButton("🖊️")
        self.pen_button.setToolTip("Draw Mode")
        self.pen_button.setCheckable(True)
        self.pen_button.setFixedSize(40, 40)
        self.pen_button.clicked.connect(self.toggle_brush_mode)
        buttons_layout.addWidget(self.pen_button)

        self.thresh_button = QPushButton("✏️")
        self.thresh_button.setToolTip("Threshold/Segment")
        self.thresh_button.setFixedSize(40, 40)
        self.thresh_button.clicked.connect(self.show_thresh_dialog)
        buttons_layout.addWidget(self.thresh_button)

        control_layout.addWidget(buttons_widget)

        self.preview = False #Whether in preview mode or not
        self.targs = None #Targets for preview mode
                
        # Create channel buttons
        self.channel_buttons = []
        self.delete_buttons = []  # New list to store delete buttons
        self.channel_names = ["Nodes", "Edges", "Overlay 1", "Overlay 2"]

        # Create channel toggles with delete buttons
        for i in range(4):
            # Create container for each channel's controls
            channel_container = QWidget()
            channel_layout = QHBoxLayout(channel_container)
            channel_layout.setSpacing(2)  # Reduce spacing between buttons
            
            # Create toggle button
            btn = QPushButton(f"{self.channel_names[i]}")
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, ch=i: self.toggle_channel(ch))
            self.channel_buttons.append(btn)
            channel_layout.addWidget(btn)
            
            # Create delete button
            delete_btn = QPushButton("×")  # Using × character for delete
            delete_btn.setFixedSize(20, 20)  # Make it small and square
            delete_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: gray;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: red;
                }
                QPushButton:disabled {
                    color: lightgray;
                }
            """)
            delete_btn.setEnabled(False)
            delete_btn.clicked.connect(lambda checked, ch=i: self.delete_channel(ch))
            self.delete_buttons.append(delete_btn)
            channel_layout.addWidget(delete_btn)
            
            control_layout.addWidget(channel_container)

        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        


        # Create left panel for image and controls
        left_panel = QWidget()
        self.left_layout = QVBoxLayout(left_panel)

        class WheelViewBox(pg.ViewBox):
            wheel_signal = QtCore.Signal()
            pan_finished_signal = QtCore.Signal()
            
            def __init__(self, parent_window, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent_window = parent_window
                
            def wheelEvent(self, ev, axis=None):
                from PyQt6.QtCore import Qt
                
                # Get modifiers
                modifiers = ev.modifiers()
                ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                alt_pressed = bool(modifiers & Qt.KeyboardModifier.AltModifier)
                
                # Determine scroll direction
                scroll_up = ev.delta() > 0
                
                # Handle different modifier combinations
                if self.parent_window.brush_mode:
                    # Alt: adjust 3D threshold
                    if alt_pressed and self.parent_window.threed and not ctrl_pressed and not shift_pressed:
                        self.parent_window.on_threed_scroll(scroll_up)
                        ev.accept()
                        return
                    
                    # Ctrl (no shift): adjust brush size
                    elif ctrl_pressed and not shift_pressed:
                        self.parent_window.on_brush_size_scroll(scroll_up)
                        ev.accept()
                        return
                
                # Shift: scroll through slices
                if shift_pressed:
                    step_multiplier = 3 if ctrl_pressed else 1
                    self.parent_window.on_slice_scroll(scroll_up, step_multiplier)
                    ev.accept()
                    return
                
                # No modifiers (or unhandled combination): default zoom behavior
                super().wheelEvent(ev, axis)
                self.wheel_signal.emit()
                ev.accept()
            
            def mousePressEvent(self, ev):
                if self.parent_window.pan_mode:
                    super().mousePressEvent(ev)
                else:
                    result = self.parent_window.on_mouse_press(ev)
                    if result:
                        super().mousePressEvent(ev)
                    else:
                        ev.accept()
            
            def mouseMoveEvent(self, ev):
                if self.parent_window.pan_mode:
                    super().mouseMoveEvent(ev)
                    ev.accept()
                else:
                    self.parent_window.on_mouse_move(ev.scenePos())
                    ev.accept()
            
            def mouseReleaseEvent(self, ev):
                if self.parent_window.pan_mode:
                    super().mouseReleaseEvent(ev)
                else:
                    self.parent_window.on_mouse_release(ev)
                    ev.accept()


        # Create custom ViewBox instance first
        self.view = WheelViewBox(self)
        self.view.setAspectLocked(True)

        # Create the graphics widget
        self.graphics_widget = pg.GraphicsLayoutWidget()

        # Create custom ViewBox instance first
        self.view = WheelViewBox(self)
        self.view.setAspectLocked(True)

        # Add a plot item with our custom ViewBox
        self.plot_item = self.graphics_widget.addPlot(viewBox=self.view)

        self.plot_item.hideAxis('left')
        self.plot_item.hideAxis('bottom')

        # Add image item to the view
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)

        # Add widget to layout
        self.left_layout.addWidget(self.graphics_widget)

        # Enable drag and drop for the graphics widget
        self.graphics_widget.setAcceptDrops(True)

        # Connect wheel event
        self.view.wheel_signal.connect(self.on_wheel_event)
        self.view.sigRangeChangedManually.connect(self.on_range_changed_manually)


        # Create debounce timer for wheel events
        self.wheel_timer = QTimer()
        self.previous_zoom_level = None  # Track zoom level to detect direction
        self.wheel_timer.setSingleShot(True)
        self.wheel_timer.timeout.connect(self.on_wheel_finished)
        self.is_wheeling = False
        # Connect scroll event for pyqtgraph
        self.graphics_widget.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.view.pan_finished_signal.connect(self.on_pan_finished)

        self.pan_timer = QTimer()
        self.pan_timer.setSingleShot(True)
        self.pan_timer.timeout.connect(self.on_pan_finished)
        self.is_panning = False


        self.left_layout.addWidget(control_panel)

        # Add timer for debouncing slice updates
        self._slice_update_timer = QTimer()
        self._slice_update_timer.setSingleShot(True)  # Only fire once after last trigger
        self._slice_update_timer.timeout.connect(self._do_slice_update)
        self.pending_slice = None  # Store the latest requested slice
        
        # Create container for slider and arrow buttons
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add left arrow button
        self.left_arrow = QPushButton("←")
        self.left_arrow.setFixedSize(30, 30)
        self.left_arrow.pressed.connect(self.start_left_scroll)
        self.left_arrow.released.connect(self.stop_continuous_scroll)
        slider_layout.addWidget(self.left_arrow)
        
        # Add slider for depth navigation
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_slice)
        slider_layout.addWidget(self.slice_slider)
        
        # Add right arrow button
        self.right_arrow = QPushButton("→")
        self.right_arrow.setFixedSize(30, 30)
        self.right_arrow.pressed.connect(self.start_right_scroll)
        self.right_arrow.released.connect(self.stop_continuous_scroll)
        slider_layout.addWidget(self.right_arrow)
        
        # Initialize continuous scroll timer
        self.continuous_scroll_timer = QTimer()
        self.continuous_scroll_timer.timeout.connect(self.continuous_scroll)
        self.scroll_direction = 0  # 0: none, -1: left, 1: right
        
        self.left_layout.addWidget(slider_container)

        
        main_layout.addWidget(left_panel)
        
        # Create right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create tabbed data widget for top right
        self.tabbed_data = TabbedDataWidget(self)
        right_layout.addWidget(self.tabbed_data)
        # Initialize data_table property to None - it will be set when tabs are added
        self.data_table = []

        # Create table control panel
        table_control = QWidget()
        table_control_layout = QHBoxLayout(table_control)

        # Create toggle buttons for tables
        self.network_button = QPushButton("Network Table")
        self.network_button.setCheckable(True)
        self.network_button.setChecked(True)
        self.network_button.clicked.connect(self.show_network_table)

        self.network_graph_button = QPushButton("Network Graph")
        self.network_graph_button.setCheckable(True)
        self.network_graph_button.setChecked(False)
        self.network_graph_button.clicked.connect(self.show_network_graph)

        self.selection_button = QPushButton("Selection Table")
        self.selection_button.setCheckable(True)
        self.selection_button.clicked.connect(self.show_selection_table)

        self.selection_graph_button = QPushButton("Selection Graph")
        self.selection_graph_button.setCheckable(True)
        self.selection_graph_button.clicked.connect(self.show_selection_graph)

        # Add buttons to control layout
        table_control_layout.addWidget(self.network_button)
        table_control_layout.addWidget(self.network_graph_button)
        table_control_layout.addWidget(self.selection_button)
        table_control_layout.addWidget(self.selection_graph_button)

        # Add control panel to right layout
        right_layout.addWidget(table_control)

        # CREATE A CONTAINER FOR SWITCHABLE VIEWS
        self.view_container = QWidget()
        self.view_layout = QVBoxLayout(self.view_container)
        self.view_layout.setContentsMargins(0, 0, 0, 0)



        # Create graph widgets
        self.network_graph_widget = ngw.NetworkGraphWidget(
            parent=self,
            weight=True,
            geometric=False,
            centroids=None,
            communities=False,
            community_dict=None,
            labels=True,
            z_size = True,
            popout = True
        )

        self.selection_graph_widget = ngw.NetworkGraphWidget(
            parent=self,
            weight=True,
            geometric=False,
            centroids=None,
            communities=False,
            community_dict=None,
            labels=True,
            z_size = True,
            popout = True
        )

        self.umap_widgets = []

        # Create both table views
        self.network_table = CustomTableView(self, subgraph = self.network_graph_widget)
        self.selection_table = CustomTableView(self, subgraph = self.selection_graph_widget)
        empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
        self.selection_table.setModel(PandasModel(empty_df))
        self.network_table.setAlternatingRowColors(True)
        self.selection_table.setAlternatingRowColors(True)
        self.network_table.setSortingEnabled(True)
        self.selection_table.setSortingEnabled(True)

        # Add all views to the container
        self.view_layout.addWidget(self.network_table)
        self.view_layout.addWidget(self.network_graph_widget)
        self.view_layout.addWidget(self.selection_table)
        self.view_layout.addWidget(self.selection_graph_widget)

        # Initially show only network table
        self.network_table.show()
        self.network_graph_widget.hide()
        self.selection_table.hide()
        self.selection_graph_widget.hide()

        # Add the container to the right layout
        right_layout.addWidget(self.view_container)

        # Store reference to currently active view
        self.active_view = self.network_table
        
        main_layout.addWidget(right_panel)
        
        # Create menu bar
        self.create_menu_bar()

        # Initialize clicked values dictionary
        self.clicked_values = {
            'nodes': [],
            'edges': []
        }
        
        # Initialize measurement tracking
        self.measurement_points = []  # List to store point pairs
        self.angle_measurements = []  # NEW: List to store angle trios
        self.current_point = None  # Store first point of current pair/trio
        self.current_second_point = None  # Store second point when building trio
        self.current_pair_index = 0  # Track pair numbering
        self.current_trio_index = 0  # Track trio numbering
        self.measurement_mode = "distance"  # "distance" or "angle" mode

        # Add these new methods for handling neighbors and components (FOR RIGHT CLICKIGN)
        self.show_neighbors_clicked = None
        self.show_component_clicked = None

        # Initialize highlight overlay
        self.highlight_overlay = None
        self.highlight_bounds = None  # Store bounds for positioning
        self.mini_overlay = False # If the program is currently drawing the overlay by frame this will be true
        self.mini_overlay_xy = False
        self.mini_overlay_data = None #Actual data for mini overlay
        self.mini_thresh = (500*500*500) # Array volume to start using mini overlays for
        self.shape = None

        self.excel_manager = ExcelotronManager(self)
        self.excel_manager.data_received.connect(self.handle_excel_data)
        self.prev_coms = None
        
        # Background caching for blitting
        self.paint_session_active = False
        
        # Batch paint operations
        self.paint_batch = []
        self.last_paint_pos = None

        self.resume = False
        self._first_pan_done = False
        self.thresh_window_ref = None
        self.disable_pan = False
        self.grid_ready = False
        self.remove_scale = False
        self.remove_grid = False
        self.original_dims = None
        self.thresh_min = None
        self.thresh_max = None
        self.thresh_vals = None
        self.temp_graph_widgets = []

        #Deprecated:
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Load graph
        #self.graph_widget.set_graph(my_network.network)
        #self.graph_widget.load_graph()

    def toggle_grid(self, show=None, alpha=0.3):
        """Toggle measurement grid with x/y axis labels."""
        if show is None:
            show = not getattr(self, 'grid_visible', False)
        
        self.grid_visible = show
        
        # Show/hide grid
        self.plot_item.showGrid(x=show, y=show, alpha=alpha)
        
        # Show/hide axes
        self.plot_item.showAxis('left', show=show)
        self.plot_item.showAxis('bottom', show=show)

    def dragEnterEvent(self, event):
        """Handle drag enter event to accept file drops"""
        from PyQt6.QtCore import Qt
        
        if event.mimeData().hasUrls():
            # Check if at least one URL is a local file
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event):
        """Handle drop event to load dragged image files or directories"""
        from PyQt6.QtCore import Qt
        import os
        
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    
                    # Check if it's a directory
                    if os.path.isdir(file_path):
                        # TODO: Handle directory - specify what to do with directories here
                        print(f"Loading session: {file_path}")
                        self.load_from_network_obj(file_path)
                        break
                    
                    # Check if it's a valid image file
                    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.nii')
                    if file_path.lower().endswith(valid_extensions):
                        # Load the image file
                        self.load_channel(self.active_channel, filename=file_path)
                        print(f"Loading image: {file_path}")
                        break  # Load only the first valid image
                    else:
                        print(f"Invalid file type: {file_path}")
            
            event.acceptProposedAction()
        else:
            event.ignore()

    def on_range_changed_manually(self):
        """Called when view range changes due to user interaction (wheel or pan)."""
        # If we're wheeling, ignore
        if self.is_wheeling:
            return
        
        # If we just finished wheeling recently, ignore (prevents false pan detection)
        if hasattr(self, '_wheel_finished_time'):
            if time.time() - self._wheel_finished_time < 1:  # grace period
                return

        if self.pan_mode:
            self.graphics_widget.setCursor(Qt.CursorShape.ClosedHandCursor)

        # Otherwise it's a pan event
        if not self.is_panning:
            self.is_panning = True
        
        self.pan_timer.stop()
        self.pan_timer.start(150)

    def on_wheel_event(self):
        """Handle wheel scroll - do quick low-res update only when zooming out"""
        view_range = self.view.viewRange()
        current_zoom_level = (view_range[0][1] - view_range[0][0]) * (view_range[1][1] - view_range[1][0])
        
        zooming_out = self.previous_zoom_level is not None and current_zoom_level > self.previous_zoom_level
        if zooming_out and not self.is_wheeling:
            self.update_display(quick_wheel_update=True)

        self.is_wheeling = True
        if self.machine_window is not None:
            if self.machine_window.segmentation_worker is not None:
                if not self.machine_window.segmentation_worker._paused:
                    self.resume = True
                self.machine_window.segmentation_worker.pause()
        
        self.previous_zoom_level = current_zoom_level
        
        self.wheel_timer.stop()
        self.wheel_timer.start(150)

    def on_pan_finished(self):
        """Called once when pan finishes (after 150ms of no panning)."""
        self.is_panning = False

        view_range = self.view.viewRange()
        current_xlim = view_range[0]
        current_ylim = view_range[1]

        if self.resume:
            self.machine_window.segmentation_worker.resume()
            self.resume = False

        if self.disable_pan:
            self.pan_button.setChecked(False)
            self.toggle_pan_mode()
            self.disable_pan = False
            if self.penning:
                self.pen_button.click()
            return
    
        if self.pan_mode:
            self.graphics_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        
        self.update_display()

    def on_wheel_finished(self):
        """Called after wheel events stop - do high-res update"""
        import time

        self._wheel_finished_time = time.time()
        self.is_wheeling = False
        if self.resume:
            self.machine_window.segmentation_worker.resume()
            self.resume = False
        if self.disable_pan:
            self.pan_button.setChecked(False)
            self.toggle_pan_mode()
            self.disable_pan = False
            if self.penning:
                self.pen_button.click()
            return
        self.update_display()

    def on_slice_scroll(self, scroll_up, step_multiplier=1):
        """Handle slice scrolling with mouse wheel."""
        step = step_multiplier if scroll_up else -step_multiplier
        
        new_slice = self.current_slice + step
        
        # Check bounds
        if new_slice < 0 or new_slice > self.slice_slider.maximum():
            return
        
        self.current_slice = new_slice
        self.slice_slider.setValue(new_slice)
        
        view_range = self.view.viewRange()
        self.update_display(preserve_zoom=(view_range[0], view_range[1]))

    def on_brush_size_scroll(self, scroll_up):
        """Handle brush size adjustment with mouse wheel."""
        step = 1 if scroll_up else -1
        new_size = self.brush_size + step
        
        # Clamp to valid range
        new_size = max(self.min_brush_size, min(self.max_brush_size, new_size))
        
        self.brush_size = new_size
        self.update_brush_cursor()

    def on_threed_scroll(self, scroll_up):
        """Handle 3D threshold adjustment with mouse wheel."""
        import math
        
        step = 1 if scroll_up else -1
        self.threedthresh += step
        
        # Round to appropriate odd integer
        if scroll_up:
            # Round up to nearest odd
            self.threedthresh = math.ceil(self.threedthresh)
            if self.threedthresh % 2 == 0:
                self.threedthresh += 1
        else:
            # Round down to nearest odd, but not below 1
            self.threedthresh = math.floor(self.threedthresh)
            if self.threedthresh % 2 == 0:
                self.threedthresh -= 1
            self.threedthresh = max(1, self.threedthresh)
        
        self.update_brush_cursor()


    def load_file(self):
        """Load CSV or Excel file and convert to dictionary format."""
        try:
            # Open file dialog
            file_filter = "Spreadsheet Files (*.csv *.xlsx *.json);;CSV Files (*.csv);;Excel Files (*.xlsx);;UMAP Embedding (*.json)"
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load File",
                "",
                file_filter
            )
            
            if not filename:
                return
            
            if filename.endswith('.csv') or filename.endswith('.xlsx'):
                # Read the file
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filename)
                else:
                    QMessageBox.warning(self, "Error", "Please select a CSV or Excel file.")
                    return
                
                if df.empty:
                    QMessageBox.warning(self, "Error", "The file appears to be empty.")
                    return
                
                # Extract headers
                headers = df.columns.tolist()
                if len(headers) < 1:
                    QMessageBox.warning(self, "Error", "File must have at least 1 column.")
                    return
                
                # Extract filename without extension for title
                import os
                title = os.path.splitext(os.path.basename(filename))[0]
                
                if len(headers) == 1:
                    # Single column: pass header to metric, column data as list to data, nothing to value
                    metric = headers[0]
                    data = df.iloc[:, 0].tolist()  # First column as list
                    value = None
                    
                    df = self.format_for_upperright_table(data=data, metric=metric, value=value, title=title)
                    return df
                else:
                    # Multiple columns: create dictionary as before
                    # First column header (for metric parameter)
                    metric = headers[0]
                    
                    # Remaining headers (for value parameter)
                    value = headers[1:]
                    
                    # Create dictionary
                    data_dict = {}
                    
                    for index, row in df.iterrows():
                        key = row.iloc[0]  # First column value as key
                        
                        if len(headers) == 2:
                            # If only 2 columns, store single value
                            data_dict[key] = row.iloc[1]
                        else:
                            # If more than 2 columns, store as list
                            data_dict[key] = row.iloc[1:].tolist()
                    
                    if len(value) == 1:
                        value = value[0]
                    
                    # Call the parent method
                    df = self.format_for_upperright_table(data=data_dict, metric=metric, value=value, title=title)
                    return df
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"File '{title}' loaded successfully with {len(df)} entries."
                )
            
            if filename.endswith('.json'):

                if my_network.node_identities:
                    neighbor_classes = {}

                    for item in my_network.node_identities.keys():
                        if item in my_network.node_identities:
                            idens = my_network.node_identities[item]
                            neighbor_classes[item] = random.choice(idens)
                else:
                    neighbor_classes = None

                umap = umw.UMAPGraphWidget(
                    parent=self.parent(),
                    labels=False
                )

                self.umap_widgets.append(umap)

                umap.update_params(
                    community_dict=my_network.communities,
                    identity_dict=neighbor_classes
                )

                umap.load_from_save(filename)

                umap.show_in_window("UMAP Visualization")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load file: {str(e)}"
            )
        
    def popup_canvas(self):
        """Pop the canvas out into its own window"""
        if hasattr(self, 'popup_window') and self.popup_window.isVisible():
            # If popup already exists, just bring it to front
            self.popup_window.raise_()
            self.popup_window.activateWindow()
            if hasattr(self, 'control_popup_window'):
                self.control_popup_window.raise_()
                self.control_popup_window.activateWindow()
            # Also bring machine window to front if it exists
            if self.machine_window is not None:
                self.machine_window.raise_()
                self.machine_window.activateWindow()
            return
        
        self.is_popped = True

        # Store original widget size policy before popping out
        self.original_widget_size_policy = self.graphics_widget.sizePolicy()
        
        # Create popup window for canvas
        self.popup_window = QMainWindow()
        self.popup_window.setWindowTitle("NetTracer3D - Canvas View")
        self.popup_window.setGeometry(200, 200, 1000, 800)  # Bigger size
        
        # Install event filters for both window management and keyboard shortcuts
        self.popup_window.installEventFilter(self)
        
        # Create popup window for control panel and slider
        self.control_popup_window = QMainWindow()
        self.control_popup_window.setWindowTitle("NetTracer3D - Controls")
        self.control_popup_window.setGeometry(1220, 200, 400, 200)  # Bigger height for slider
        
        # Install event filter on control window too
        self.control_popup_window.installEventFilter(self)
        
        # Make control window non-closeable while popped out
        self.control_popup_window.setWindowFlags(
            self.control_popup_window.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )
        
        # Set control panel as child of canvas popup for natural stacking
        self.control_popup_window.setParent(self.popup_window, Qt.WindowType.Window)
        
        # Remove graphics widget from left panel
        self.graphics_widget.setParent(None)
        
        # Remove control panel from left panel
        # First find the control_panel widget
        control_panel = None
        for i in range(self.left_layout.count()):
            widget = self.left_layout.itemAt(i).widget()
            if widget and hasattr(widget, 'findChild') and widget.findChild(QComboBox):
                control_panel = widget
                break
        
        # Remove slider container from left panel
        slider_container = None
        for i in range(self.left_layout.count()):
            widget = self.left_layout.itemAt(i).widget()
            if widget and hasattr(widget, 'findChild') and widget.findChild(QSlider):
                slider_container = widget
                break
        
        if control_panel:
            control_panel.setParent(None)
            self.popped_control_panel = control_panel
            
        if slider_container:
            slider_container.setParent(None)
            self.popped_slider_container = slider_container
        
        # Move the graphics widget to popup window
        self.popup_window.setCentralWidget(self.graphics_widget)
        
        # Create a container widget for the control popup to hold both control panel and slider
        control_container = QWidget()
        control_container_layout = QVBoxLayout(control_container)
        
        # Add control panel to container
        if control_panel:
            control_container_layout.addWidget(control_panel)
            
        # Add slider container to container
        if slider_container:
            control_container_layout.addWidget(slider_container)
        
        # Set the container as the central widget
        self.control_popup_window.setCentralWidget(control_container)
        
        # Create placeholder for left panel
        placeholder = QLabel("Canvas and controls are popped out\nClick to return both")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                font-size: 14px;
                color: #666;
            }
        """)
        placeholder.mousePressEvent = lambda event: self.return_canvas()
        
        # Add placeholder to left layout
        self.left_layout.insertWidget(0, placeholder)  # Insert at canvas position
        self.canvas_placeholder = placeholder
        
        # Create keyboard shortcuts for popup windows
        self.create_popup_shortcuts()
        
        # Show both popup windows
        self.popup_window.show()
        self.control_popup_window.show()
        
        # Ensure proper initial window order
        self.ensure_window_order()
        
        # Connect close event to return canvas (only canvas window can be closed)
        self.popup_window.closeEvent = self.on_popup_close

    def eventFilter(self, obj, event):
        """Filter events to manage window stacking and keyboard shortcuts"""
        # Handle keyboard events for popup windows
        if (obj == self.popup_window or obj == self.control_popup_window) and event.type() == QEvent.Type.KeyPress:
            # Forward key events to main window's keyPressEvent method
            self.keyPressEvent(event)
            return True  # Event handled
        
        # Handle scroll events for popup canvas - PyQtGraph handles this automatically
        # No need to forward scroll events as the WheelViewBox will handle them
        
        # Existing window stacking code
        if obj == self.popup_window:
            if event.type() == QEvent.Type.WindowActivate:
                # Canvas popup was activated, raise our preferred windows
                QTimer.singleShot(0, self.ensure_window_order)
            elif event.type() == QEvent.Type.FocusIn:
                # Canvas got focus, raise controls
                QTimer.singleShot(0, self.ensure_window_order)
        
        return super().eventFilter(obj, event)

    def create_popup_shortcuts(self):
        """Create keyboard shortcuts for popup windows"""
        if not hasattr(self, 'popup_shortcuts'):
            self.popup_shortcuts = []
        
        # Define shortcuts - using your existing keyPressEvent logic
        shortcuts_config = [
            ('Z', lambda: self.zoom_button.click()),
            ('Ctrl+Z', self.handle_undo),
            ('X', lambda: self.high_button.click()),
            ('Shift+F', self.handle_find),
            ('Ctrl+S', self.handle_resave),
            ('Ctrl+L', lambda: self.load_from_network_obj(directory=self.last_load)),
            ('F', lambda: self.toggle_can() if self.brush_mode and self.machine_window is None else None),
            ('D', lambda: self.toggle_threed() if self.brush_mode and self.machine_window is None else None),
            ('A', lambda: self.machine_window.switch_foreground() if self.machine_window is not None else None)
        ]
        
        # Create shortcuts for both popup windows
        for key_seq, func in shortcuts_config:
            # Canvas popup window shortcuts
            shortcut1 = QShortcut(QKeySequence(key_seq), self.popup_window)
            shortcut1.activated.connect(func)
            self.popup_shortcuts.append(shortcut1)
            
            # Control popup window shortcuts
            shortcut2 = QShortcut(QKeySequence(key_seq), self.control_popup_window)
            shortcut2.activated.connect(func)
            self.popup_shortcuts.append(shortcut2)

    def handle_undo(self):
        """Handle undo shortcut"""

        if self.brush_mode or self.machine_window is not None:
            self.pm.undo_last_virtual_stroke()
            return

        try:
            self.load_channel(self.last_change[1], self.last_change[0], True)
        except:
            pass

    def ensure_window_order(self):
        """Ensure control panel and machine window stay above canvas"""
        if hasattr(self, 'control_popup_window') and self.control_popup_window.isVisible():
            self.control_popup_window.raise_()
        
        if self.machine_window is not None and self.machine_window.isVisible():
            self.machine_window.raise_()

    def return_canvas(self):
        """Return canvas and control panel to main window"""
        if hasattr(self, 'popup_window'):
            # Clean up popup shortcuts
            if hasattr(self, 'popup_shortcuts'):
                for shortcut in self.popup_shortcuts:
                    shortcut.deleteLater()
                del self.popup_shortcuts
            
            # Remove event filters when returning
            self.popup_window.removeEventFilter(self)
            if hasattr(self, 'control_popup_window'):
                self.control_popup_window.removeEventFilter(self)
            
            # Remove graphics widget from popup
            self.graphics_widget.setParent(None)
            self.is_popped = False
            
            # Remove control panel from popup
            if hasattr(self, 'popped_control_panel') and hasattr(self, 'control_popup_window'):
                self.popped_control_panel.setParent(None)
                
            # Remove slider container from popup
            if hasattr(self, 'popped_slider_container') and hasattr(self, 'control_popup_window'):
                self.popped_slider_container.setParent(None)
            
            # Remove placeholder
            if hasattr(self, 'canvas_placeholder'):
                self.canvas_placeholder.setParent(None)
                del self.canvas_placeholder
            
            # Reset widget size policy
            if hasattr(self, 'original_widget_size_policy'):
                self.graphics_widget.setSizePolicy(self.original_widget_size_policy)
            
            # Reset graphics widget minimum and maximum sizes to allow proper resizing
            self.graphics_widget.setMinimumSize(0, 0)
            self.graphics_widget.setMaximumSize(16777215, 16777215)
            
            # Return graphics widget to left panel
            self.left_layout.insertWidget(0, self.graphics_widget)  # Insert at top
            
            # Return control panel to left panel (after canvas)
            if hasattr(self, 'popped_control_panel'):
                self.left_layout.insertWidget(1, self.popped_control_panel)  # Insert after canvas
                del self.popped_control_panel
                
            # Return slider container to left panel (after control panel)
            if hasattr(self, 'popped_slider_container'):
                self.left_layout.insertWidget(2, self.popped_slider_container)  # Insert after control panel
                del self.popped_slider_container
            
            # Force layout update
            self.graphics_widget.updateGeometry()
            self.graphics_widget.update()
            
            # Reset the main window layout to ensure proper proportions
            main_widget = self.centralWidget()
            if main_widget:
                main_widget.updateGeometry()
                main_widget.update()
            
            # Close both popup windows
            self.popup_window.close()
            if hasattr(self, 'control_popup_window'):
                self.control_popup_window.close()
                del self.control_popup_window
            
            # Clean up stored size references
            if hasattr(self, 'original_widget_size_policy'):
                del self.original_widget_size_policy

    def on_popup_close(self, event):
        """Return canvas when popup is closed"""
        self.return_canvas()
        event.accept()

    def start_left_scroll(self):
        """Start scrolling left when left arrow is pressed."""
        # Single increment first
        current_value = self.slice_slider.value()
        if current_value > self.slice_slider.minimum():
            self.slice_slider.setValue(current_value - 1)
        # Then start continuous scroll
        self.scroll_direction = -1
        self.continuous_scroll_timer.start(200)  # 200ms interval for steady pace
        
    def start_right_scroll(self):
        """Start scrolling right when right arrow is pressed."""
        # Single increment first
        current_value = self.slice_slider.value()
        if current_value < self.slice_slider.maximum():
            self.slice_slider.setValue(current_value + 1)
        # Then start continuous scroll
        self.scroll_direction = 1
        self.continuous_scroll_timer.start(200)  # 200ms interval for steady pace
        
    def stop_continuous_scroll(self):
        """Stop continuous scrolling when arrow is released."""
        self.continuous_scroll_timer.stop()
        self.scroll_direction = 0
        
    def continuous_scroll(self):
        """Handle continuous scrolling while arrow is held."""
        current_value = self.slice_slider.value()
        new_value = current_value + self.scroll_direction
        
        if self.scroll_direction < 0 and new_value >= self.slice_slider.minimum():
            self.slice_slider.setValue(new_value)
        elif self.scroll_direction > 0 and new_value <= self.slice_slider.maximum():
            self.slice_slider.setValue(new_value)


    def confirm_mini_thresh(self):

        try:

            if self.shape[0] * self.shape[1] * self.shape[2] > self.mini_thresh:
                self.mini_overlay = True
                return True
            else:
                return False
        except:
            return False

    def evaluate_mini(self, mode = 'nodes', subgraph_push = False):
        if self.confirm_mini_thresh():
            self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
        else:
            self.create_highlight_overlay(node_indices=self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
        if subgraph_push:
            self.create_table_node_selection(self.clicked_values['nodes'])


    def highlight_in_subgraphs(self, node_indices):
        if self.network_graph_widget.rendered:
            try:
                self.network_graph_widget.select_nodes(node_indices)
            except:
                pass
        if self.selection_graph_widget.rendered:
            try:
                self.selection_graph_widget.select_nodes(node_indices)
            except:
                pass
        for umap in self.umap_widgets:
            if umap.rendered:
                try:
                    umap.select_nodes(node_indices)
                except:
                    pass

        for graph in self.temp_graph_widgets:
            try:
                if graph.rendered:
                    graph.select_nodes(node_indices)
            except:
                pass

    def table_subgraph(self, table_widget, table):

        new_model = PandasModel(table)
        table_widget.setModel(new_model)

        try:
            list1, list2, list3 = table.T.values.tolist()
            temp_network = n3d.Network_3D()
            temp_network.network_lists = [list1, list2, list3]
            table_widget.subgraph.set_graph(temp_network.network)
        except:
            import traceback
            traceback.print_exc()
            pass

    def create_table_node_selection(self, nodes):

        # Get the existing DataFrame from the model
        original_df = self.network_table.model()._data

        # Create mask for rows for nodes in question
        mask = (
            (original_df.iloc[:, 0].isin(nodes) & original_df.iloc[:, 1].isin(nodes))
            )
        
        # Filter the DataFrame to only include direct connections
        filtered_df = original_df[mask].copy()
        
        # Create new model with filtered DataFrame and update selection table
        self.table_subgraph(self.selection_table, filtered_df)

    def create_highlight_overlay(self, node_indices=None, edge_indices=None, overlay1_indices=None, overlay2_indices=None, bounds=False):
        """
        Create a binary overlay highlighting specific nodes and/or edges using parallel processing.
        
        Args:
            node_indices (list): List of node indices to highlight
            edge_indices (list): List of edge indices to highlight
        """

        self.mini_overlay = False
        self.mini_overlay_data = None

        if not self.high_button.isChecked():
            if len(self.clicked_values['edges']) > 0:
                self.format_for_upperright_table(self.clicked_values['edges'], title='Selected Edges')
            if len(self.clicked_values['nodes']) > 0:
                self.format_for_upperright_table(self.clicked_values['nodes'], title='Selected Nodes')
            return

        def process_chunk(chunk_data, indices_to_check):
            """Process a single chunk of the array to create highlight mask"""
            # Make a copy of indices for this thread to avoid cache contention
            indices_copy = np.array(indices_to_check, copy=True)
            mask = np.isin(chunk_data, indices_copy)
            return mask * 255

        def process_chunk_bounds(chunk_data, indices_to_check):
            """Process a single chunk of the array to create highlight mask"""
            # Make a copy of indices for this thread to avoid cache contention
            indices_copy = np.array(indices_to_check, copy=True)
            mask = (chunk_data >= indices_copy[0]) & (chunk_data <= indices_copy[1])
            return mask * 255

        self.highlight_in_subgraphs(node_indices)
        if node_indices is not None:
            if 0 in node_indices:
                node_indices.remove(0)
        if edge_indices is not None:
            if 0 in edge_indices:
                edge_indices.remove(0)
        if overlay1_indices is not None:
            if 0 in overlay1_indices:
                overlay1_indices.remove(0)
        if overlay2_indices is not None:
            if 0 in overlay2_indices:
                overlay2_indices.remove(0)

        if node_indices is None:
            node_indices = []
        if edge_indices is None:
            edge_indices = []
        if overlay1_indices is None:
            overlay1_indices = []
        if overlay2_indices is None:
            overlay2_indices = []
            
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
        
        if not node_indices and not edge_indices and not overlay1_indices and not overlay2_indices and self.machine_window is None:
            self.highlight_overlay = None
            self.highlight_bounds = None
            self.update_display(preserve_zoom=(current_xlim, current_ylim))
            return
            
        # Get the shape of the full array from any existing channel
        for channel in self.channel_data:
            if channel is not None:
                full_shape = channel.shape
                break
        else:
            return  # No valid channels to get shape from
            
        # Initialize full-size overlay
        self.highlight_overlay = np.zeros(full_shape, dtype=np.uint8)
        
        # Get number of CPU cores
        num_cores = mp.cpu_count()
        
        # Calculate chunk size along y-axis
        chunk_size = full_shape[1] // num_cores
        if chunk_size < 1:
            chunk_size = 1
        
        def process_channel(channel_data, indices, array_shape):
            if channel_data is None or not indices:
                return None
                
            # Create chunks
            chunks = []
            for i in range(0, array_shape[1], chunk_size):
                end = min(i + chunk_size, array_shape[1])
                chunks.append(channel_data[:, i:end, :])
                
            # Process chunks in parallel using ThreadPoolExecutor
            if not bounds:
                # Each thread gets its own copy via the process_chunk function
                process_func = partial(process_chunk, indices_to_check=indices)
            else:
                # Prepare bounds indices
                bounds_indices = list(indices)
                if len(bounds_indices) == 1:
                    bounds_indices.insert(0, 0)
                # Each thread gets its own copy via the process_chunk_bounds function
                process_func = partial(process_chunk_bounds, indices_to_check=bounds_indices)
            
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                chunk_results = list(executor.map(process_func, chunks))
                
            # Reassemble the chunks
            return np.concatenate(chunk_results, axis=1)
        
        # Process nodes and edges in parallel using multiprocessing
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            future_nodes = executor.submit(process_channel, self.channel_data[0], node_indices, full_shape)
            future_edges = executor.submit(process_channel, self.channel_data[1], edge_indices, full_shape)
            future_overlay1 = executor.submit(process_channel, self.channel_data[2], overlay1_indices, full_shape)
            future_overlay2 = executor.submit(process_channel, self.channel_data[3], overlay2_indices, full_shape)

            # Get results
            node_overlay = future_nodes.result()
            edge_overlay = future_edges.result()
            overlay1_overlay = future_overlay1.result()
            overlay2_overlay = future_overlay2.result()
            
        # Combine results
        if node_overlay is not None:
            self.highlight_overlay = np.maximum(self.highlight_overlay, node_overlay).astype(np.uint8)
        if edge_overlay is not None:
            self.highlight_overlay = np.maximum(self.highlight_overlay, edge_overlay).astype(np.uint8)
        if overlay1_overlay is not None:
            self.highlight_overlay = np.maximum(self.highlight_overlay, overlay1_overlay).astype(np.uint8)
        if overlay2_overlay is not None:
            self.highlight_overlay = np.maximum(self.highlight_overlay, overlay2_overlay).astype(np.uint8)

        # Update display
        self.update_display(preserve_zoom=(current_xlim, current_ylim))

    def create_highlight_overlay_slice(self, indices, bounds = False, update = True):

        def crop_and_downsample(image, y_start, y_end, x_start, x_end, factor):
            # Crop to visible region
            if len(image.shape) == 2:
                cropped = image[y_start:y_end, x_start:x_end]
            elif len(image.shape) == 3:
                cropped = image[y_start:y_end, x_start:x_end, :]
            else:
                cropped = image
            
            # Downsample if needed
            if factor == 1:
                return cropped
            
            if len(cropped.shape) == 2:
                return cropped[::factor, ::factor]
            elif len(cropped.shape) == 3:
                return cropped[::factor, ::factor, :]
            else:
                return cropped

        """Highlight overlay generation method specific for the segmenter interactive mode"""

        self.mini_overlay_data = None
        self.highlight_overlay = None
        if self.shape[0] * self.shape[1] * self.shape[2] > self.mini_thresh:
            self.mini_overlay = True

        array = self.channel_data[self.active_channel]
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

        current_slice = array[self.current_slice, :, :]
        full_shape = array.shape

        if self.mini_overlay:
            current_slice = crop_and_downsample(current_slice, self.y_min_padded, self.y_max_padded,
            self.x_min_padded, self.x_max_padded, self.downsample_factor)
        elif self.highlight_overlay is None:
            self.highlight_overlay = np.zeros(full_shape, dtype=np.uint8)
        slice_shape = current_slice.shape

        # Process the slice based on bounds parameter
        if not bounds:
            mask = np.isin(current_slice, indices)
            overlay = mask * 255
        else:
            if len(indices) == 1:
                indices.insert(0, 0)
            mask = (current_slice >= indices[0]) & (current_slice <= indices[1])
            overlay = mask * 255

        if self.active_channel == 0:
            self.highlight_in_subgraphs(indices)

        try:
            if not self.mini_overlay:
                self.highlight_overlay[self.current_slice, :, :] = overlay
                self.update_display()
            else:
                self.mini_overlay_data = overlay
                # Update display
                if update:
                    self.update_display(needs_mini = False)
        except:
            pass

        if my_network.network is not None:
            try:
                if self.active_channel == 0:

                    # Get the existing DataFrame from the model
                    original_df = self.network_table.model()._data
                    
                    # Create mask for rows where one column is any original node AND the other column is any neighbor
                    mask = (
                        (original_df.iloc[:, 0].isin(indices)) &
                        (original_df.iloc[:, 1].isin(indices)))
                    
                    # Filter the DataFrame to only include direct connections
                    filtered_df = original_df[mask].copy()
                    
                    # Create new model with filtered DataFrame and update selection table
                    self.table_subgraph(self.selection_table, filtered_df)
                    
                    # Switch to selection table
                    #self.selection_button.click()
            except:
                pass


    def create_mini_overlay(self, node_indices = None, edge_indices = None, update = True):

        """
        Create a highlight overlay one slice at a time.
        
        Args:
            node_indices (list): List of node indices to highlight
            edge_indices (list): List of edge indices to highlight
        """


        def crop_and_downsample(image, y_start, y_end, x_start, x_end, factor):
            # Crop to visible region
            if len(image.shape) == 2:
                cropped = image[y_start:y_end, x_start:x_end]
            elif len(image.shape) == 3:
                cropped = image[y_start:y_end, x_start:x_end, :]
            else:
                cropped = image
            
            # Downsample if needed
            if factor == 1:
                return cropped
            
            if len(cropped.shape) == 2:
                return cropped[::factor, ::factor]
            elif len(cropped.shape) == 3:
                return cropped[::factor, ::factor, :]
            else:
                return cropped


        if not self.high_button.isChecked():

            if len(self.clicked_values['edges']) > 0:
                self.format_for_upperright_table(self.clicked_values['edges'], title = 'Selected Edges')
                self.needs_mini = True
            if len(self.clicked_values['nodes']) > 0:
                self.format_for_upperright_table(self.clicked_values['nodes'], title = 'Selected Nodes')
                self.needs_mini = True

            return

        self.highlight_in_subgraphs(node_indices)
        if node_indices is not None:
            if 0 in node_indices:
                node_indices.remove(0)
        if edge_indices is not None:
            if 0 in edge_indices:
                edge_indices.remove(0)

        if node_indices is None:
            node_indices = []
        if edge_indices is None:
            edge_indices = []

                    
        if not node_indices and not edge_indices: #Theoretically this can't be called because it uses full highlight overlay method for empty clicks
            self.mini_overlay_data = None
            self.mini_overlay = False
            self.update_display()
            return
            

        if self.channel_data[0] is None and self.channel_data[1] is None:
            return
        
        if self.channel_data[0] is not None:
            slice_node = crop_and_downsample(self.channel_data[0][self.current_slice, :, :], self.y_min_padded, self.y_max_padded,
            self.x_min_padded, self.x_max_padded, self.downsample_factor)
            full_shape = slice_node.shape
        if self.channel_data[1] is not None:
            slice_edge = crop_and_downsample(self.channel_data[1][self.current_slice, :, :], self.y_min_padded, self.y_max_padded,
            self.x_min_padded, self.x_max_padded, self.downsample_factor)
            full_shape = slice_edge.shape

        # Initialize full-size overlay
        self.mini_overlay_data = np.zeros(full_shape, dtype=np.uint8)

        # Process nodes
        if self.channel_data[0] is not None and node_indices:
            node_mask = np.isin(slice_node, node_indices)
            node_overlay = node_mask * 255
            self.mini_overlay_data = np.maximum(self.mini_overlay_data, node_overlay)
        
        # Process edges
        if self.channel_data[1] is not None and edge_indices:
            edge_mask = np.isin(slice_edge, edge_indices)
            edge_overlay = edge_mask * 255
            self.mini_overlay_data = np.maximum(self.mini_overlay_data, edge_overlay)

        if update:
            self.update_display(needs_mini = False)




#METHODS RELATED TO RIGHT CLICK:
    
    def create_context_menu(self, event):
        """Create and show context menu at mouse position."""
        if self.channel_data[self.active_channel] is not None:
            x_idx = int(round(event.xdata))
            y_idx = int(round(event.ydata))
            
            try:
                # Create context menu
                context_menu = QMenu(self)

                find = context_menu.addAction("Find Node/Edge/community")
                find.triggered.connect(self.handle_find)
                
                # Create "Show Neighbors" submenu
                neighbors_menu = QMenu("Show Neighbors", self)
                
                # Add submenu options
                show_neighbor_nodes = neighbors_menu.addAction("Show Neighboring Nodes")
                show_neighbor_all = neighbors_menu.addAction("Show Neighboring Nodes and Edges")
                show_neighbor_edge = neighbors_menu.addAction("Show Neighboring Edges")
                
                context_menu.addMenu(neighbors_menu)

                component_menu = QMenu("Show Connected Component(s)", self)
                show_component_nodes = component_menu.addAction("Just nodes")
                show_component_edges = component_menu.addAction("Nodes + Edges")
                show_component_only_edges = component_menu.addAction("Just edges")
                context_menu.addMenu(component_menu)

                community_menu = QMenu("Show Nodes' Community(s)", self)
                show_community_nodes = community_menu.addAction("Just nodes")
                show_community_edges = community_menu.addAction("Nodes + Edges")
                context_menu.addMenu(community_menu)

                if my_network.node_identities is not None:
                    identity_menu = QMenu("Show Identity", self)
                    
                    # Collect all unique identities
                    all_idens = my_network.node_identities.values()
                    seen = set()
                    for iden in all_idens:
                        seen.update(iden)
                    idens = sorted(seen)
                    
                    # Add individual identity options
                    for item in idens:
                        show_identity = identity_menu.addAction(f"ID: {item}")
                        show_identity.triggered.connect(lambda checked, item=item: self.handle_show_identities(sort=item))
                    
                    # Add separator before custom option
                    if idens:  # Only add separator if there are identities
                        identity_menu.addSeparator()
                    
                    # Add custom option
                    custom_action = identity_menu.addAction("Custom...")
                    custom_action.triggered.connect(self.show_identity_selection_dialog)
                    
                    context_menu.addMenu(identity_menu)

                select_all_menu = QMenu("Select All", self)
                select_nodes = select_all_menu.addAction("Nodes")
                select_both = select_all_menu.addAction("Nodes + Edges")
                select_edges = select_all_menu.addAction("Edges")
                select_net_nodes = select_all_menu.addAction("Nodes in Network")
                select_net_both = select_all_menu.addAction("Nodes + Edges in Network")
                select_net_edges = select_all_menu.addAction("Edges in Network")
                context_menu.addMenu(select_all_menu)

                if len(self.clicked_values['nodes']) > 0 or len(self.clicked_values['edges']) > 0:
                    highlight_menu = QMenu("Selection", self)
                    if len(self.clicked_values['nodes']) > 1 or len(self.clicked_values['edges']) > 1:
                        combine_obj = highlight_menu.addAction("Combine Object Labels")
                        combine_obj.triggered.connect(self.handle_combine)
                    split_obj = highlight_menu.addAction("Split Non-Touching Labels")
                    split_obj.triggered.connect(self.handle_seperate)
                    delete_obj = highlight_menu.addAction("Delete Selection")
                    delete_obj.triggered.connect(self.handle_delete)
                    if len(self.clicked_values['nodes']) > 1:
                        link_nodes = highlight_menu.addAction("Link Nodes")
                        link_nodes.triggered.connect(self.handle_link)
                        delink_nodes = highlight_menu.addAction("Split Nodes")
                        delink_nodes.triggered.connect(self.handle_split)
                    new_coms = highlight_menu.addAction("Add to New Community")
                    new_coms.triggered.connect(self.new_coms)
                    override_obj = highlight_menu.addAction("Override Channel with Selection")
                    override_obj.triggered.connect(self.handle_override)
                    context_menu.addMenu(highlight_menu)

                # Create measurement submenu
                measure_menu = context_menu.addMenu("Measurements")
                
                distance_menu = measure_menu.addMenu("Distance")
                if self.current_point is None:
                    show_point_menu = distance_menu.addAction("Place First Point")
                    show_point_menu.triggered.connect(
                        lambda: self.place_distance_point(x_idx, y_idx, self.current_slice))
                elif (self.current_point is not None and 
                      hasattr(self, 'measurement_mode') and 
                      self.measurement_mode == "distance"):
                    show_point_menu = distance_menu.addAction("Place Second Point")
                    show_point_menu.triggered.connect(
                        lambda: self.place_distance_point(x_idx, y_idx, self.current_slice))

                # Angle measurement options
                angle_menu = measure_menu.addMenu("Angle")
                if self.current_point is None:
                    angle_first = angle_menu.addAction("Place First Point (A)")
                    angle_first.triggered.connect(
                        lambda: self.place_angle_point(x_idx, y_idx, self.current_slice))
                elif (self.current_point is not None and 
                      self.current_second_point is None and 
                      hasattr(self, 'measurement_mode') and 
                      self.measurement_mode == "angle"):
                    angle_second = angle_menu.addAction("Place Second Point (B - Vertex)")
                    angle_second.triggered.connect(
                        lambda: self.place_angle_point(x_idx, y_idx, self.current_slice))
                elif (self.current_point is not None and 
                      self.current_second_point is not None and 
                      hasattr(self, 'measurement_mode') and 
                      self.measurement_mode == "angle"):
                    angle_third = angle_menu.addAction("Place Third Point (C)")
                    angle_third.triggered.connect(
                        lambda: self.place_angle_point(x_idx, y_idx, self.current_slice))
                
                
                show_remove_menu = measure_menu.addAction("Remove All Measurements")
                show_remove_menu.triggered.connect(self.handle_remove_all_measurements)
                
                context_menu.addMenu(measure_menu)
                
                # Connect actions to callbacks
                show_neighbor_nodes.triggered.connect(self.handle_show_neighbors)
                show_neighbor_all.triggered.connect(lambda: self.handle_show_neighbors(edges=True))
                show_neighbor_edge.triggered.connect(lambda: self.handle_show_neighbors(edges = True, nodes = False))
                show_component_nodes.triggered.connect(self.handle_show_component)
                show_component_edges.triggered.connect(lambda: self.handle_show_component(edges = True))
                show_component_only_edges.triggered.connect(lambda: self.handle_show_component(edges = True, nodes = False))
                show_community_nodes.triggered.connect(self.handle_show_communities)
                show_community_edges.triggered.connect(lambda: self.handle_show_communities(edges = True))
                select_nodes.triggered.connect(lambda: self.handle_select_all(edges = False, nodes = True))
                select_both.triggered.connect(lambda: self.handle_select_all(edges = True))
                select_edges.triggered.connect(lambda: self.handle_select_all(edges = True, nodes = False))
                select_net_nodes.triggered.connect(lambda: self.handle_select_all(edges = False, nodes = True, network = True))
                select_net_both.triggered.connect(lambda: self.handle_select_all(edges = True, network = True))
                select_net_edges.triggered.connect(lambda: self.handle_select_all(edges = True, nodes = False, network = True))
                if self.highlight_overlay is not None or self.mini_overlay_data is not None:
                    highlight_select = context_menu.addAction("Add highlight in network selection")
                    highlight_select.triggered.connect(self.handle_highlight_select)
                
                cursor_pos = QCursor.pos()
                context_menu.exec(cursor_pos)
                
            except IndexError:
                pass

    def place_distance_point(self, x, y, z):
        """Place a measurement point for distance measurement."""
        if self.current_point is None:
            # This is the first point
            self.current_point = (x, y, z)
            
            # Create and store the artists
            pt = self.ax.plot(x, y, 'yo', markersize=8)[0]
            txt = self.ax.text(x, y+5, f"D{self.current_pair_index}", 
                        color='yellow', ha='center', va='bottom')
            
            # Add to measurement_artists so they can be managed by update_display
            if not hasattr(self, 'measurement_artists'):
                self.measurement_artists = []
            self.measurement_artists.extend([pt, txt])
            
            self.measurement_mode = "distance"
        else:
            # This is the second point
            x1, y1, z1 = self.current_point
            x2, y2, z2 = x, y, z
            
            # Calculate distance
            distance = np.sqrt(((x2-x1)*my_network.xy_scale)**2 + 
                              ((y2-y1)*my_network.xy_scale)**2 + 
                              ((z2-z1)*my_network.z_scale)**2)
            distance2 = np.sqrt(((x2-x1))**2 + ((y2-y1))**2 + ((z2-z1))**2)
            
            # Store the point pair with type indicator
            self.measurement_points.append({
                'pair_index': self.current_pair_index,
                'point1': self.current_point,
                'point2': (x2, y2, z2),
                'distance': distance,
                'distance2': distance2,
                'type': 'distance'  # Added type tracking
            })
            
            # Draw second point and line, storing the artists
            pt2 = self.ax.plot(x2, y2, 'yo', markersize=8)[0]
            txt2 = self.ax.text(x2, y2+5, f"D{self.current_pair_index}", 
                        color='yellow', ha='center', va='bottom')
            
            # Add to measurement_artists
            self.measurement_artists.extend([pt2, txt2])
            
            if z1 == z2:  # Only draw line if points are on same slice
                line = self.ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.5)[0]
                self.measurement_artists.append(line)
                            
            # Update measurement display
            self.update_measurement_display()
            
            # Reset for next pair
            self.current_point = None
            self.current_pair_index += 1
            self.measurement_mode = "distance"

        self.update_display()

    def place_angle_point(self, x, y, z):
        """Place a measurement point for angle measurement."""
        if not hasattr(self, 'measurement_artists'):
            self.measurement_artists = []
            
        if self.current_point is None:
            # First point (A)
            self.current_point = (x, y, z)
            
            # Create and store artists
            pt = self.ax.plot(x, y, 'go', markersize=8)[0]
            txt = self.ax.text(x, y+5, f"A{self.current_trio_index}", 
                        color='green', ha='center', va='bottom')
            self.measurement_artists.extend([pt, txt])
            
            self.measurement_mode = "angle"
            
        elif self.current_second_point is None:
            # Second point (B - vertex)
            self.current_second_point = (x, y, z)
            x1, y1, z1 = self.current_point
            
            # Create and store artists
            pt = self.ax.plot(x, y, 'go', markersize=8)[0]
            txt = self.ax.text(x, y+5, f"B{self.current_trio_index}", 
                        color='green', ha='center', va='bottom')
            self.measurement_artists.extend([pt, txt])
            
            # Draw line from A to B
            if z1 == z:
                line = self.ax.plot([x1, x], [y1, y], 'g--', alpha=0.7)[0]
                self.measurement_artists.append(line)
            
        else:
            # Third point (C)
            x1, y1, z1 = self.current_point  # Point A
            x2, y2, z2 = self.current_second_point  # Point B (vertex)
            x3, y3, z3 = x, y, z  # Point C
            
            # Calculate angles and distances
            angle_data = self.calculate_3d_angle(
                (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)
            )
            
            # Store the trio
            self.angle_measurements.append({
                'trio_index': self.current_trio_index,
                'point_a': (x1, y1, z1),
                'point_b': (x2, y2, z2),  # vertex
                'point_c': (x3, y3, z3),
                **angle_data
            })
            
            # Also add the two distances as separate pairs with type indicator
            dist_ab = np.sqrt(((x2-x1)*my_network.xy_scale)**2 + 
                             ((y2-y1)*my_network.xy_scale)**2 + 
                             ((z2-z1)*my_network.z_scale)**2)
            dist_bc = np.sqrt(((x3-x2)*my_network.xy_scale)**2 + 
                             ((y3-y2)*my_network.xy_scale)**2 + 
                             ((z3-z2)*my_network.z_scale)**2)
            
            dist_ab_voxel = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            dist_bc_voxel = np.sqrt((x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2)
            
            self.measurement_points.extend([
                {
                    'pair_index': f"A{self.current_trio_index}-B{self.current_trio_index}",
                    'point1': (x1, y1, z1),
                    'point2': (x2, y2, z2),
                    'distance': dist_ab,
                    'distance2': dist_ab_voxel,
                    'type': 'angle'  # Added type tracking
                },
                {
                    'pair_index': f"B{self.current_trio_index}-C{self.current_trio_index}",
                    'point1': (x2, y2, z2),
                    'point2': (x3, y3, z3),
                    'distance': dist_bc,
                    'distance2': dist_bc_voxel,
                    'type': 'angle'  # Added type tracking
                }
            ])
            
            # Draw third point and line, storing artists
            pt3 = self.ax.plot(x3, y3, 'go', markersize=8)[0]
            txt3 = self.ax.text(x3, y3+5, f"C{self.current_trio_index}", 
                        color='green', ha='center', va='bottom')
            self.measurement_artists.extend([pt3, txt3])
            
            if z2 == z3:  # Draw line from B to C if on same slice
                line = self.ax.plot([x2, x3], [y2, y3], 'g--', alpha=0.7)[0]
                self.measurement_artists.append(line)
            
            # Update measurement display
            self.update_measurement_display()
            
            # Reset for next trio
            self.current_point = None
            self.current_second_point = None
            self.current_trio_index += 1
            self.measurement_mode = "angle"

        self.update_display()



    def calculate_3d_angle(self, point_a, point_b, point_c):
        """Calculate 3D angle at vertex B between points A-B-C."""
        x1, y1, z1 = point_a
        x2, y2, z2 = point_b  # vertex
        x3, y3, z3 = point_c
        
        # Apply scaling
        scaled_a = np.array([x1 * my_network.xy_scale, y1 * my_network.xy_scale, z1 * my_network.z_scale])
        scaled_b = np.array([x2 * my_network.xy_scale, y2 * my_network.xy_scale, z2 * my_network.z_scale])
        scaled_c = np.array([x3 * my_network.xy_scale, y3 * my_network.xy_scale, z3 * my_network.z_scale])
        
        # Create vectors from vertex B
        vec_ba = scaled_a - scaled_b
        vec_bc = scaled_c - scaled_b
        
        # Calculate angle using dot product
        dot_product = np.dot(vec_ba, vec_bc)
        magnitude_ba = np.linalg.norm(vec_ba)
        magnitude_bc = np.linalg.norm(vec_bc)
        
        # Avoid division by zero
        if magnitude_ba == 0 or magnitude_bc == 0:
            return {'angle_degrees': 0}
        
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        
        angle_radians = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle_radians)
        
        return {'angle_degrees': angle_degrees}

    def handle_remove_all_measurements(self):
        """Remove all measurement points and angles."""
        self.measurement_points = []
        self.angle_measurements = []
        self.current_point = None
        self.current_second_point = None
        self.current_pair_index = 0
        self.current_trio_index = 0
        self.measurement_mode = "distance"
        self.update_display()
        self.update_measurement_display()

    def update_measurement_display(self):
        """Update the measurement information display in the top right widget."""
        # Distance measurements
        if not self.measurement_points:
            distance_df = pd.DataFrame()
        else:
            distance_data = []
            for point in self.measurement_points:
                x1, y1, z1 = point['point1']
                x2, y2, z2 = point['point2']
                distance_data.append({
                    'Pair ID': point['pair_index'],
                    'Point 1 (X,Y,Z)': f"({x1:.1f}, {y1:.1f}, {z1})",
                    'Point 2 (X,Y,Z)': f"({x2:.1f}, {y2:.1f}, {z2})",
                    'Scaled Distance': f"{point['distance']:.2f}",
                    'Voxel Distance': f"{point['distance2']:.2f}"
                })
            distance_df = pd.DataFrame(distance_data)
        
        # Angle measurements
        if not self.angle_measurements:
            angle_df = pd.DataFrame()
        else:
            angle_data = []
            for angle in self.angle_measurements:
                xa, ya, za = angle['point_a']
                xb, yb, zb = angle['point_b']
                xc, yc, zc = angle['point_c']
                angle_data.append({
                    'Trio ID': f"A{angle['trio_index']}-B{angle['trio_index']}-C{angle['trio_index']}",
                    'Point A (X,Y,Z)': f"({xa:.1f}, {ya:.1f}, {za})",
                    'Point B (X,Y,Z)': f"({xb:.1f}, {yb:.1f}, {zb})",
                    'Point C (X,Y,Z)': f"({xc:.1f}, {yc:.1f}, {zc})",
                    'Angle (°)': f"{angle['angle_degrees']:.1f}"
                })
            angle_df = pd.DataFrame(angle_data)
        
        # Create tables
        if not distance_df.empty:
            distance_table = CustomTableView(self)
            distance_table.setModel(PandasModel(distance_df))
            self.tabbed_data.add_table("Distance Measurements", distance_table)
            for column in range(distance_table.model().columnCount(None)):
                distance_table.resizeColumnToContents(column)
        
        if not angle_df.empty:
            angle_table = CustomTableView(self)
            angle_table.setModel(PandasModel(angle_df))
            self.tabbed_data.add_table("Angle Measurements", angle_table)
            for column in range(angle_table.model().columnCount(None)):
                angle_table.resizeColumnToContents(column)

    def show_network_graph(self):
        """Switch to display the network graph."""
        if not self.network_graph_button.isChecked():
            self.network_graph_button.setChecked(True)
            return
        
        # Uncheck other buttons
        self.network_button.setChecked(False)
        self.selection_button.setChecked(False)
        self.selection_graph_button.setChecked(False)
        
        # Hide all other views
        self.network_table.hide()
        self.selection_table.hide()
        self.selection_graph_widget.hide()
        
        # Show network graph
        self.network_graph_widget.show()
        
        # Load graph (check if network exists)
        self.network_graph_widget.set_graph(my_network.network)
        
        self.active_view = self.network_graph_widget

    def show_selection_graph(self):
        """Switch to display the selection graph."""
        if not self.selection_graph_button.isChecked():
            self.selection_graph_button.setChecked(True)
            return
        
        # Uncheck other buttons
        self.network_button.setChecked(False)
        self.network_graph_button.setChecked(False)
        self.selection_button.setChecked(False)
        
        # Hide all other views
        self.network_table.hide()
        self.network_graph_widget.hide()
        self.selection_table.hide()
        
        # Show selection graph
        self.selection_graph_widget.show()
        
        self.active_view = self.selection_graph_widget

    def show_network_table(self):
        """Switch to display the main network table."""
        if not self.network_button.isChecked():
            self.network_button.setChecked(True)
            return
        
        # Uncheck other buttons
        self.network_graph_button.setChecked(False)
        self.selection_button.setChecked(False)
        self.selection_graph_button.setChecked(False)
        
        # Hide all other views
        self.network_graph_widget.hide()
        self.selection_table.hide()
        self.selection_graph_widget.hide()
        
        # Show network table
        self.network_table.show()
        
        self.active_view = self.network_table
        self.active_table = self.network_table

    def show_selection_table(self):
        """Switch to display the selection table."""
        if not self.selection_button.isChecked():
            self.selection_button.setChecked(True)
            return
        
        # Uncheck other buttons
        self.network_button.setChecked(False)
        self.network_graph_button.setChecked(False)
        self.selection_graph_button.setChecked(False)
        
        # Hide all other views
        self.network_table.hide()
        self.network_graph_widget.hide()
        self.selection_graph_widget.hide()
        
        # Show selection table
        self.selection_table.show()
        
        self.active_view = self.selection_table
        self.active_table = self.selection_table

    def handle_show_neighbors(self, edges=False, nodes = True):
        """Handle the Show Neighbors action."""

        try:
            if len(self.clicked_values['nodes']) > 0 or len(self.clicked_values['edges']) > 0:  # Check if we have any nodes selected

                old_nodes = copy.deepcopy(self.clicked_values['nodes']) 

                # Get the existing DataFrame from the model
                original_df = self.network_table.model()._data
                
                # Create mask for rows where one column is any original node AND the other column is any neighbor
                mask = (
                    (original_df.iloc[:, 0].isin(self.clicked_values['nodes'])) |
                    (original_df.iloc[:, 1].isin(self.clicked_values['nodes'])) |
                    (original_df.iloc[:, 2].isin(self.clicked_values['edges']))
                    )
                
                # Filter the DataFrame to only include direct connections
                filtered_df = original_df[mask].copy()
                
                # Create new model with filtered DataFrame and update selection table
                self.table_subgraph(self.selection_table, filtered_df)
                
                # Switch to selection table
                #self.selection_button.click()

                print(f"Found {len(filtered_df)} direct connections between nodes {old_nodes} and their neighbors")
                self.clicked_values['nodes'] = list(set(filtered_df.iloc[:, 0].to_list() + filtered_df.iloc[:, 1].to_list()))

                if not nodes:
                    self.clicked_values['nodes'] = old_nodes

                do_highlight = True

            else:

                do_highlight = False

            if do_highlight:
              
                # Create highlight overlay for visualization
                if edges:
                    edge_indices = filtered_df.iloc[:, 2].unique().tolist()
                    self.clicked_values['edges'] = edge_indices
                    self.evaluate_mini(mode = 'edges')
                else:
                    self.evaluate_mini()
                
        except Exception as e:
            print(f"Error showing neighbors: {e}")

    
    def handle_show_component(self, edges = False, nodes = True, isolate = False):
        """Handle the Show Component action."""

        try:

            old_nodes = copy.deepcopy(self.clicked_values['nodes'])

            if len(self.clicked_values['nodes']) == 0: #If we haven't clicked anything, this will return the largest connected component

                G = my_network.isolate_connected_component(gen_images = False)

                # Get the existing DataFrame from the model
                original_df = self.network_table.model()._data

                # Create mask for rows where one column is any original node AND the other column is any neighbor
                mask = (
                    (original_df.iloc[:, 0].isin(G.nodes()) & original_df.iloc[:, 1].isin(G.nodes()))
                    )
                
                # Filter the DataFrame to only include direct connections
                filtered_df = original_df[mask].copy()

                # Create new model with filtered DataFrame and update selection table
                self.table_subgraph(self.selection_table, filtered_df)

            else: #If we have clicked any nodes, we get the components of the clicked objects instead

                G = nx.Graph()

                for node in self.clicked_values['nodes']:

                    if node in G: #Meaning we've already done this component
                        continue
                    else: #Otherwise, get the graph and add it to the subgraph(s)
                        G1 = my_network.isolate_connected_component(gen_images = False, key = node)
                        G = nx.compose(G1, G)

                # Get the existing DataFrame from the model
                original_df = self.network_table.model()._data

                # Create mask for rows of this component
                mask = (
                    (original_df.iloc[:, 0].isin(G.nodes()) & original_df.iloc[:, 1].isin(G.nodes()))
                    )
                
                # Filter the DataFrame to only include direct connections
                filtered_df = original_df[mask].copy()
                
                # Create new model with filtered DataFrame and update selection table
                self.table_subgraph(self.selection_table, filtered_df)
                
                # Switch to selection table
                #self.selection_button.click()

            if not nodes:
                self.clicked_values['nodes'] = old_nodes
            else:
                self.clicked_values['nodes'] = G.nodes()

            if edges:
                edge_indices = filtered_df.iloc[:, 2].unique().tolist()
                self.clicked_values['edges'] = edge_indices
                self.evaluate_mini(mode = 'edges')
            else:
                self.evaluate_mini()

            if isolate:
                self.selection_table.set_selection_to_active()
                self.network_graph_widget._clear_graph()


        except Exception as e:

            print(f"Error finding component: {e}")

    def handle_show_communities(self, edges = False):

        def invert_dict(d):
            """For inverting the community dictionary"""
            inverted = {}
            for key, value in d.items():
                inverted.setdefault(value, []).append(key)
            return inverted

        try:

            if len(self.clicked_values['nodes']) > 0:

                if my_network.communities is None:
                    self.show_partition_dialog()

                communities = invert_dict(my_network.communities)

                targets = []

                for node in self.clicked_values['nodes']: #Get the communities we need

                    if node in targets:
                        continue
                    else:
                        targets.append(my_network.communities[node])

                nodes = []

                for com in targets: #Get the nodes for each community in question

                    for node in communities[com]:

                        nodes.append(node)

                nodes = list(set(nodes))

                try:
                    self.create_table_node_selection(nodes)
                except:
                    pass

                if edges:
                    edge_indices = filtered_df.iloc[:, 2].unique().tolist()
                    self.clicked_values['edges'] = edge_indices
                    if self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2] > self.mini_thresh:
                        self.mini_overlay = True
                        self.create_mini_overlay(node_indices = nodes, edge_indices = edge_indices)
                    else:
                        self.create_highlight_overlay(
                            node_indices=nodes,
                            edge_indices=edge_indices
                        )
                    self.clicked_values['nodes'] = nodes
                else:
                    if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                        self.mini_overlay = True
                        self.create_mini_overlay(node_indices = nodes, edge_indices = self.clicked_values['edges'])
                    else:
                        self.create_highlight_overlay(
                            node_indices = nodes,
                            edge_indices = self.clicked_values['edges']
                    )
                    self.clicked_values['nodes'] = nodes

        except Exception as e:
            print(f"Error showing communities: {e}")

    def handle_show_identities(self, sort):

        try:

            nodes = []


            if isinstance(sort, str):
                for node, idens in my_network.node_identities.items():
                    if sort in idens:
                        nodes.append(node)
            else:

                for node, idens in my_network.node_identities.items():
                    pos_gate = all(item in idens for item in sort['positive'])
                    neg_gate = all(item not in idens for item in sort['negative'])
                    if pos_gate and neg_gate:
                        nodes.append(node)

            neighbors = set()  # Use a set from the start to avoid duplicates
            nodes += self.clicked_values['nodes']

            try:
                if my_network.network.number_of_edges() < 100000: # Only make a selection for smaller graphs for performance
                    try:
                    
                        # Get the existing DataFrame from the model
                        original_df = self.network_table.model()._data
                        
                        # Create mask for pairs that have nodes of the ID in question
                        mask = (
                            (original_df.iloc[:, 0].isin(nodes)) | (original_df.iloc[:, 1].isin(nodes))
                        )

                        
                        # Filter the DataFrame to only include direct connections
                        filtered_df = original_df[mask].copy()
                        
                        # Create new model with filtered DataFrame and update selection table
                        self.table_subgraph(self.selection_table, filtered_df)
                        
                        # Switch to selection table
                        #self.selection_button.click()
                    except:
                        pass
            except:
                pass

            #print(f"Found {len(filtered_df)} direct connections between nodes of ID {sort} and their neighbors (of any ID)")

            if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                self.mini_overlay = True
                self.create_mini_overlay(node_indices = nodes, edge_indices = self.clicked_values['edges'])
            else:
                self.create_highlight_overlay(
                    node_indices = nodes,
                    edge_indices = self.clicked_values['edges']
                    )
            self.clicked_values['nodes'] = nodes

        except Exception as e:
            print(f"Error showing identities: {e}")

    def handle_find(self):

        class FindDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Find Node (or edge/com?)")
                self.setModal(True)
                
                layout = QFormLayout(self)

                self.targ = QLineEdit("")
                layout.addRow("Node/Edge ID:", self.targ)

                self.mode_selector = QComboBox()
                self.mode_selector.addItems(["nodes", "edges", "communities"])
                if self.parent().active_channel == 1:
                    self.mode_selector.setCurrentIndex(1)
                else:
                    self.mode_selector.setCurrentIndex(0)

                layout.addRow("Type to select:", self.mode_selector)

                run_button = QPushButton("Enter")
                run_button.clicked.connect(self.run)
                layout.addWidget(run_button)

            def run(self):

                try:

                    mode = self.mode_selector.currentIndex()

                    value = int(self.targ.text()) if self.targ.text().strip() else None

                    if value is None:
                        return

                    if mode == 1:

                        if my_network.edge_centroids is None:
                            self.parent().show_centroid_dialog()

                        num = (self.parent().channel_data[1].shape[0] * self.parent().channel_data[1].shape[1] * self.parent().channel_data[1].shape[2])

                        self.parent().clicked_values['edges'] = [value]
                        self.parent().handle_info(sort = 'edge')

                        if value in my_network.edge_centroids:

                            # Get centroid coordinates (Z, Y, X)
                            centroid = my_network.edge_centroids[value]
                            # Set the active channel to edges (1)
                            self.parent().set_active_channel(1)
                            # Toggle on the edges channel if it's not already visible
                            if not self.parent().channel_visible[1]:
                                self.parent().channel_buttons[1].setChecked(True)
                                self.parent().toggle_channel(1)
                            # Navigate to the Z-slice
                            self.parent().slice_slider.setValue(int(centroid[0]))
                            print(f"Found edge {value} at [Z,Y,X] -> {centroid}")

                        else:
                            print(f"Edge {value} not found in centroids dictionary")


                    else:


                        if my_network.node_centroids is None:
                            self.parent().show_centroid_dialog()

                        num = (self.parent().channel_data[0].shape[0] * self.parent().channel_data[0].shape[1] * self.parent().channel_data[0].shape[2])

                        if mode == 0:
                            self.parent().clicked_values['nodes'] = [value]
                            self.parent().handle_info(sort = 'node')
                        elif mode == 2:

                            coms = n3d.invert_dict(my_network.communities)
                            self.parent().clicked_values['nodes'] = coms[value]
                            com = value
                            value = coms[value][0]
                            self.parent().create_table_node_selection(self.parent().clicked_values['nodes'])


                        if value in my_network.node_centroids:
                            # Get centroid coordinates (Z, Y, X)
                            centroid = my_network.node_centroids[value]
                            # Set the active channel to nodes (0)
                            self.parent().set_active_channel(0) 
                            # Toggle on the nodes channel if it's not already visible
                            if not self.parent().channel_visible[0]:
                                self.parent().channel_buttons[0].setChecked(True)
                                self.parent().toggle_channel(0)
                            # Navigate to the Z-slice
                            self.parent().slice_slider.setValue(int(centroid[0]))
                            if mode == 0:
                                print(f"Found node {value} at [Z,Y,X] -> {centroid}")
                            elif mode == 2:
                                print(f"Found node {value} from community {com} at [Z,Y,X] -> {centroid}")

                            
                        else:
                            print(f"Node {value} not found in centroids dictionary")


                    if num > self.parent().mini_thresh:
                        self.parent().mini_overlay = True
                        self.parent().create_mini_overlay(node_indices = self.parent().clicked_values['nodes'], edge_indices = self.parent().clicked_values['edges'])
                    else:
                        self.parent().create_highlight_overlay(
                            node_indices=self.parent().clicked_values['nodes'], 
                            edge_indices=self.parent().clicked_values['edges']
                        )


                    
                    # Close the dialog after processing
                    self.accept()

                except Exception as e:

                    print(f"Error: {e}")
                
        dialog = FindDialog(self)
        dialog.exec()




    def handle_select_all(self, nodes = True, edges = False, network = False):

        try:

            if nodes:
                if not network:
                    nodes = list(np.unique(my_network.nodes))
                else:
                    nodes = list(set(my_network.network_lists[0] + my_network.network_lists[1]))
                if nodes[0] == 0:
                    del nodes[0]
                num = (self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2])
                print(f"Found {len(nodes)} node objects")
            else:
                nodes = []
            if edges:
                if not network:
                    edges = list(np.unique(my_network.edges))
                else:
                    edges = my_network.network_lists[2]
                num = (self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2])
                if edges[0] == 0:
                    del edges[0]
                print(f"Found {len(edges)} edge objects")
            else:
                edges = []

            self.clicked_values['nodes'] = nodes
            self.clicked_values['edges'] = edges


            if num > self.mini_thresh:
                self.mini_overlay = True
                self.create_mini_overlay(node_indices = nodes, edge_indices = edges)
            else:
                self.create_highlight_overlay(edge_indices = self.clicked_values['edges'], node_indices = self.clicked_values['nodes'])

        except Exception as e:
            print(f"Error: {e}")

    def handle_info(self, sort = 'node'):

        try:

            info_dict = {}

            if sort == 'node':

                label = self.clicked_values['nodes'][-1]

                info_dict['Label'] = label

                info_dict['Object Class'] = 'Node'

                if my_network.node_identities is not None:
                    try:
                        info_dict['ID'] = my_network.node_identities[label]
                    except:
                        pass

                if my_network.network is not None:
                    try:
                        info_dict['Degree'] = my_network.network.degree(label)
                    except:
                        pass


                if my_network.network is not None:
                    try:
                        info_dict['Neighbors'] = list(my_network.network.neighbors(label))
                    except:
                        pass

                if my_network.communities is not None:
                    try:
                        info_dict['Community'] = my_network.communities[label]
                    except:
                        pass

                if my_network.node_centroids is not None:
                    try:
                        info_dict['Centroid(Z,Y,X)'] = my_network.node_centroids[label]
                    except:
                        pass

                if self.volume_dict[0] is not None:
                    try:
                        info_dict['Volume (Scaled)'] = self.volume_dict[0][label]
                    except:
                        pass

                if self.radii_dict[0] is not None:
                    try:
                        info_dict['Max Radius (Scaled)'] = self.radii_dict[0][label]
                    except:
                        pass

                if self.surface_area_dict[0] is not None:
                    try:
                        info_dict['~Surface Area (Scaled; Jagged Faces)'] = self.surface_area_dict[0][label]
                    except:
                        pass

                if self.sphericity_dict[0] is not None:
                    try:
                        info_dict['Sphericity'] = self.sphericity_dict[0][label]
                    except:
                        pass

                if self.branch_dict[0] is not None:
                    try:
                        info_dict['Branch Length'] = self.branch_dict[0][0][label]
                    except:
                        pass
                    try:
                        info_dict['Branch Tortuosity'] = self.branch_dict[0][1][label]
                    except:
                        pass

                try:
                    for stat, stat_dict in self.stats_dict.items():
                        try:
                            info_dict[stat] = stat_dict[label]
                        except:
                            import traceback
                            print(traceback.format_exc())
                            pass
                except:
                    import traceback
                    print(traceback.format_exc())
                    pass


            elif sort == 'edge':

                label = self.clicked_values['edges'][-1]

                info_dict['Label'] = label

                info_dict['Object Class'] = 'Edge'

                try:
                    # Get the existing DataFrame from the model
                    original_df = self.network_table.model()._data
                    
                    # Create mask for rows where one column is any original node AND the other column is any neighbor
                    mask = (
                        (original_df.iloc[:, 0].isin(self.clicked_values['nodes'])) |
                        (original_df.iloc[:, 1].isin(self.clicked_values['nodes'])) |
                        (original_df.iloc[:, 2].isin(self.clicked_values['edges']))
                        )

                    filtered_df = original_df[mask].copy()
                    node_list = list(set(filtered_df.iloc[:, 0].to_list() + filtered_df.iloc[:, 1].to_list()))
                    info_dict["Num Nodes"] = len(node_list)
                    info_dict['Nodes'] = node_list
                except:
                    pass

                if my_network.edge_centroids is not None:
                    try:
                        info_dict['Centroid(Z,Y,X)'] = my_network.edge_centroids[label]
                    except:
                        pass

                if self.volume_dict[1] is not None:
                    try:
                        info_dict['Volume (Scaled)'] = self.volume_dict[1][label]
                    except:
                        pass

                if self.radii_dict[1] is not None:
                    try:
                        info_dict['~Radius (Scaled)'] = self.radii_dict[1][label]
                    except:
                        pass

                if self.surface_area_dict[1] is not None:
                    try:
                        info_dict['~Surface Area (Scaled; Jagged Faces)'] = self.surface_area_dict[1][label]
                    except:
                        pass

                if self.sphericity_dict[1] is not None:
                    try:
                        info_dict['Sphericity'] = self.sphericity_dict[1][label]
                    except:
                        pass

                if self.branch_dict[1] is not None:
                    try:
                        info_dict['Branch Length'] = self.branch_dict[1][0][label]
                    except:
                        pass
                    try:
                        info_dict['Branch Tortuosity'] = self.branch_dict[1][1][label]
                    except:
                        pass

            self.format_for_upperright_table(info_dict, title = f'Info on Object', sort = False)

        except:
            pass



    def handle_combine(self):

        try:

            self.clicked_values['nodes'].sort()
            nodes = copy.deepcopy(self.clicked_values['nodes'])
            self.clicked_values['edges'].sort()
            edges = copy.deepcopy(self.clicked_values['edges'])

            if len(nodes) > 1:
                new_nodes = nodes[0]

                mask = np.isin(self.channel_data[0], nodes)
                my_network.nodes[mask] = new_nodes
                self.load_channel(0, my_network.nodes, True)
                self.clicked_values['nodes'] = new_nodes

            if len(edges) > 1:
                new_edges = edges[0]

                mask = np.isin(self.channel_data[1], edges)
                my_network.edges[mask] = new_edges
                self.load_channel(1, my_network.edges, True)
                self.clicked_values['edges'] = new_edges

            try:

                for i in range(len(my_network.network_lists[0])):
                    if my_network.network_lists[0][i] in nodes and len(nodes) > 1:
                        my_network.network_lists[0][i] = new_nodes
                    if my_network.network_lists[1][i] in nodes and len(nodes) > 1:
                        my_network.network_lists[1][i] = new_nodes    
                    if my_network.network_lists[2][i] in edges and len(edges) > 1:
                        my_network.network_lists[2][i] = new_edges


                my_network.network_lists = my_network.network_lists

                if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                    empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
                    model = PandasModel(empty_df)
                    self.network_table.setModel(model)
                else:
                    model = PandasModel(my_network.network_lists)
                    self.network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.network_table.resizeColumnToContents(column)

                self.highlight_overlay = None
                self.update_display()

                self.show_centroid_dialog()

            except Exception as e:
                print(f"Error, could not update network: {e}")


        except Exception as e:
            print(f"An error has occured: {e}")


    def expand_bbox(self, bbox, array_shape, padding=1):
        """Expand bounding box by padding in each dimension, clamped to array bounds"""
        expanded = []
        for i, slice_obj in enumerate(bbox):
            start = max(0, slice_obj.start - padding)
            stop = min(array_shape[i], slice_obj.stop + padding)
            expanded.append(slice(start, stop, None))
        return tuple(expanded)

    def process_label_split_only(self, item, input_array):
        """Pass 1: Split disconnected components, identify largest"""
        orig_label, bbox = item
        
        try:
            # Extract subarray
            label_subarray = input_array[bbox]
            
            # Create binary mask for this label only
            binary_mask = label_subarray == orig_label
            
            if not np.any(binary_mask):
                return orig_label, bbox, None, 0, None
            
            # Find connected components
            labeled_cc, num_cc = n3d.label_objects(binary_mask)
            
            if num_cc == 0:
                return orig_label, bbox, None, 0, None
            
            # Find largest component
            volumes = np.bincount(labeled_cc.ravel())[1:]
            largest_cc_id = np.argmax(volumes) + 1
            
            return orig_label, bbox, labeled_cc, num_cc, largest_cc_id
            
        except Exception as e:
            print(f"Error processing label {orig_label}: {e}")
            return orig_label, bbox, None, 0, None

    def process_illegal_label_reassign(self, item, pass1_array, original_max_val):
        """Pass 2: Reassign illegal labels based on legal neighbors"""
        illegal_label, bbox = item
        
        try:
            # Expand bbox by 1
            expanded_bbox = self.expand_bbox(bbox, pass1_array.shape, padding=1)
            
            # Extract subarray
            subarray = pass1_array[expanded_bbox]
            
            # Create mask for this illegal label
            illegal_mask = subarray == illegal_label
            
            if not np.any(illegal_mask):
                return illegal_label, None
            
            # Dilate to find neighbors
            dilated_mask = n3d.dilate_3D_old(illegal_mask, 3, 3, 3)
            
            # Border region
            border_mask = dilated_mask & ~illegal_mask & (subarray > 0)
            
            # Get border labels
            border_labels = subarray * border_mask
            
            # Filter out illegal labels (> original_max_val) and background (0)
            legal_border_labels = border_labels.copy()
            legal_border_labels[border_labels > original_max_val] = 0
            
            # Count occurrences of legal neighbors
            unique_borders = np.bincount(legal_border_labels.ravel())[1:]  # Skip 0
            
            if len(unique_borders) > 0 and np.max(unique_borders) > 0:
                # Found legal neighbors, pick largest shared border
                chosen_label = np.argmax(unique_borders) + 1
                return illegal_label, chosen_label
            else:
                # No legal neighbors, keep current label
                return illegal_label, None
            
        except Exception as e:
            print(f"Error processing illegal label {illegal_label}: {e}")
            return illegal_label, None

    def separate_nontouching_objects(self, input_array, max_val=None, branches=False):
        """
        Two-pass algorithm:
        Pass 1: Split disconnected components (largest keeps label, others get new labels)
        Pass 2 (branches=True only): Reassign new labels based on legal neighbors
        """
        if max_val == None:
            max_val = np.max(input_array)
        print("Splitting nontouching objects - Pass 1")
        
        binary_mask = input_array > 0
        if not np.any(binary_mask):
            return np.zeros_like(input_array)
        
        unique_labels = np.unique(input_array[binary_mask])
        print(f"Processing {len(unique_labels)} unique labels")
        
        # Store original max_val for later
        original_max_val = int(max_val)
        
        # Get all bounding boxes at once
        bounding_boxes = ndimage.find_objects(input_array)
        
        # Prepare work items
        work_items = []
        for orig_label in unique_labels:
            bbox_index = orig_label - 1
            
            if (bbox_index >= 0 and 
                bbox_index < len(bounding_boxes) and 
                bounding_boxes[bbox_index] is not None):
                
                bbox = bounding_boxes[bbox_index]
                work_items.append((orig_label, bbox))
        
        if len(work_items) == 0:
            print("No valid work items found!")
            return np.zeros_like(input_array)
        
        # PASS 1: Split components, largest keeps label, others get new labels
        max_workers = min(mp.cpu_count(), len(work_items))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = lambda item: self.process_label_split_only(item, input_array)
            results = list(executor.map(process_func, work_items))
        
        # Reconstruct output array from pass 1
        current_label = original_max_val + 1
        pass1_array = np.zeros_like(input_array)
        
        for orig_label, bbox, labeled_sub, num_cc, largest_cc_id in results:
            if num_cc > 0 and labeled_sub is not None:
                for cc_id in range(1, num_cc + 1):
                    mask = labeled_sub == cc_id
                    
                    if cc_id == largest_cc_id:
                        # Largest component keeps original label
                        assigned_label = orig_label
                    else:
                        # Others get new incremental labels
                        assigned_label = current_label
                        current_label += 1
                    
                    try:
                        pass1_array[bbox][mask] = assigned_label
                    except:
                        # Handle dtype overflow
                        if assigned_label < 256:
                            dtype = np.uint8
                        elif assigned_label < 65535:
                            dtype = np.uint16
                        else:
                            dtype = np.uint32
                        pass1_array = pass1_array.astype(dtype)
                        pass1_array[bbox][mask] = assigned_label
        
        print(f"Pass 1 complete. Created {current_label - original_max_val - 1} new labels")
        
        # If branches=False, we're done
        if not branches:
            return pass1_array
        
        # PASS 2 (branches=True): Reassign illegal labels based on legal neighbors
        print("Pass 2: Reassigning illegal labels based on legal neighbors")
        
        # Find all labels that are > original_max_val (these are "illegal")
        illegal_mask = pass1_array > original_max_val
        if not np.any(illegal_mask):
            return pass1_array
        
        illegal_labels = np.unique(pass1_array[illegal_mask])
        print(f"Processing {len(illegal_labels)} illegal labels")
        
        # Get bounding boxes for illegal labels
        illegal_bboxes = ndimage.find_objects(pass1_array)
        
        # Prepare work items for pass 2
        work_items_pass2 = []
        for illegal_label in illegal_labels:
            bbox_index = illegal_label - 1
            
            if (bbox_index >= 0 and 
                bbox_index < len(illegal_bboxes) and 
                illegal_bboxes[bbox_index] is not None):
                
                bbox = illegal_bboxes[bbox_index]
                work_items_pass2.append((illegal_label, bbox))
        
        # Process illegal labels
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = lambda item: self.process_illegal_label_reassign(item, pass1_array, original_max_val)
            results_pass2 = list(executor.map(process_func, work_items_pass2))
        
        # Apply pass 2 results
        pass2_array = pass1_array.copy()
        
        def replace_labels_in_chunk(args):
            """Process a single chunk of the array"""
            pass1_chunk, results_pass2 = args
            
            # Create output chunk (copy of input)
            pass2_chunk = pass1_chunk.copy()
            
            # Find which labels actually exist in this chunk
            unique_labels = set(np.unique(pass1_chunk))
            
            for illegal_label, new_label in results_pass2:
                if new_label is not None and new_label != illegal_label:
                    # Only process if this label exists in this chunk
                    if illegal_label in unique_labels:
                        # Read from pass1_chunk, write to pass2_chunk
                        pass2_chunk[pass1_chunk == illegal_label] = new_label
            
            return pass2_chunk

        # Get number of CPU cores
        num_cores = mp.cpu_count()

        # Split array along y-axis (axis=1)
        chunks = np.array_split(pass1_array, num_cores, axis=1)

        # Prepare arguments for each worker
        chunk_args = [(chunk, results_pass2) for chunk in chunks]

        # Process chunks in parallel with threads
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            processed_chunks = list(executor.map(replace_labels_in_chunk, chunk_args))

        # Stack results back together along y-axis
        pass2_array = np.concatenate(processed_chunks, axis=1)
        
        print(f"Pass 2 complete")
        return pass2_array

    def handle_seperate(self):
        """
        Seperate objects in an array that share a label but do not touch
        """
        try:
            # Handle nodes
            if len(self.clicked_values['nodes']) > 0:
                
                # Create highlight overlay (this should preserve original label values)
                self.create_highlight_overlay(node_indices=self.clicked_values['nodes'])
                                
                # Create a boolean mask for where we have highlighted values
                highlight_mask = self.highlight_overlay != 0
                
                # Create array with just the highlighted values (preserving original labels)
                highlighted_nodes = np.where(highlight_mask, my_network.nodes, 0)
                                
                # Get non-highlighted part of the array
                non_highlighted = np.where(highlight_mask, 0, my_network.nodes)
                
                # Calculate max_val
                max_val = np.max(self.channel_data[0])
                                
                # Process highlighted part
                processed_highlights = self.separate_nontouching_objects(highlighted_nodes, max_val)
                                
                # Combine back with non-highlighted parts
                my_network.nodes = non_highlighted + processed_highlights
                                
                self.load_channel(0, my_network.nodes, True)
            
            # Handle edges
            if len(self.clicked_values['edges']) > 0:
 
                self.create_highlight_overlay(edge_indices=self.clicked_values['edges'])
                
                # Create a boolean mask for highlighted values
                highlight_mask = self.highlight_overlay != 0
                
                # Create array with just the highlighted values
                highlighted_edges = np.where(highlight_mask, my_network.edges, 0)
                
                # Get non-highlighted part of the array
                non_highlighted = np.where(highlight_mask, 0, my_network.edges)

                max_val = np.max(self.channel_data[1])
                
                # Process highlighted part
                processed_highlights = self.separate_nontouching_objects(highlighted_edges, max_val)
                
                # Combine back with non-highlighted parts
                my_network.edges = non_highlighted + processed_highlights

                self.load_channel(1, my_network.edges, True)
            
            self.highlight_overlay = None
            self.update_display()
            print("Network is not updated automatically, please recompute if necessary - this method has a high chance of disrupting the network. Identities are not automatically updated.")
            self.show_centroid_dialog()
        except Exception as e:
            print(f"Error separating: {e}")



    def handle_delete(self):
        try:
            if len(self.clicked_values['nodes']) > 0:
                self.create_highlight_overlay(node_indices=self.clicked_values['nodes'])
                mask = self.highlight_overlay == 0
                my_network.nodes = my_network.nodes * mask
                self.load_channel(0, my_network.nodes, True)

                clicked_nodes = set(self.clicked_values['nodes'])  # O(1) lookups

                if my_network.network_lists is not None:
                    src = np.array(my_network.network_lists[0])
                    dst = np.array(my_network.network_lists[1])
                    weights = np.array(my_network.network_lists[2])

                    keep = ~(np.isin(src, list(clicked_nodes)) | np.isin(dst, list(clicked_nodes)))

                    my_network.network_lists[0] = src[keep].tolist()
                    my_network.network_lists[1] = dst[keep].tolist()
                    my_network.network_lists[2] = weights[keep].tolist()

                # Batch dict deletions
                for node in clicked_nodes:
                    try:
                        my_network.node_centroids.pop(node, None)
                    except:
                        pass
                    try:
                        my_network.node_identities.pop(node, None)
                    except:
                        pass
                    try:
                        my_network.communities.pop(node, None)
                    except:
                        pass

            if len(self.clicked_values['edges']) > 0:
                self.create_highlight_overlay(edge_indices=self.clicked_values['edges'])
                mask = self.highlight_overlay == 0
                my_network.edges = my_network.edges * mask
                self.load_channel(1, my_network.edges, True)

                clicked_edges = set(self.clicked_values['edges'])  # O(1) lookups

                if my_network.network_lists is not None:
                    src = np.array(my_network.network_lists[0])
                    dst = np.array(my_network.network_lists[1])
                    weights = np.array(my_network.network_lists[2])

                    keep = ~np.isin(weights, list(clicked_edges))

                    my_network.network_lists[0] = src[keep].tolist()
                    my_network.network_lists[1] = dst[keep].tolist()
                    my_network.network_lists[2] = weights[keep].tolist()

                for edge in clicked_edges:
                    my_network.edge_centroids.pop(edge, None)

            my_network.network_lists = my_network.network_lists
            self.network_graph_widget.set_graph(my_network.network)
            self.selection_graph_widget.set_graph(None)
            empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
            model = PandasModel(empty_df)
            self.selection_table.setModel(model)
            if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
                model = PandasModel(empty_df)
                self.network_table.setModel(model)
            else:
                model = PandasModel(my_network.network_lists)
                self.network_table.setModel(model)
                for column in range(model.columnCount(None)):
                    self.network_table.resizeColumnToContents(column)
        except Exception as e:
            print(f"Error: {e}")

    def handle_link(self):

        try:
            nodes = self.clicked_values['nodes']
            from itertools import combinations
            pairs = list(combinations(nodes, 2))
            
            # Convert existing connections to a set of tuples for efficient lookup
            existing_connections = set()
            for n1, n2 in zip(my_network.network_lists[0], my_network.network_lists[1]):
                existing_connections.add((n1, n2))
                existing_connections.add((n2, n1))  # Add reverse pair too
            
            # Filter out existing connections
            new_pairs = []
            for pair in pairs:
                if pair not in existing_connections:
                    new_pairs.append(pair)
            
            # Add new connections
            for pair in new_pairs:
                my_network.network_lists[0].append(pair[0])
                my_network.network_lists[1].append(pair[1])
                my_network.network_lists[2].append(0)
            
            # Update the table
            if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
                model = PandasModel(empty_df)
                self.network_table.setModel(model)
            else:
                model = PandasModel(my_network.network_lists)
                self.network_table.setModel(model)
                # Adjust column widths to content
                for column in range(model.columnCount(None)):
                    self.network_table.resizeColumnToContents(column)
        except Exception as e:
            print(f"An error has occurred: {e}")


    def handle_split(self):
        try:
            nodes = self.clicked_values['nodes']

            from itertools import combinations

            pairs = list(combinations(nodes, 2))


            for i in range(len(my_network.network_lists[0]) - 1, -1, -1):
                print((my_network.network_lists[0][i], my_network.network_lists[1][i]))
                if (my_network.network_lists[0][i], my_network.network_lists[1][i]) in pairs or (my_network.network_lists[1][i], my_network.network_lists[0][i]) in pairs:
                    del my_network.network_lists[0][i]
                    del my_network.network_lists[1][i]
                    del my_network.network_lists[2][i]

            my_network.network_lists = my_network.network_lists

            if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
                model = PandasModel(empty_df)
                self.network_table.setModel(model)
            else:
                model = PandasModel(my_network.network_lists)
                self.network_table.setModel(model)
                # Adjust column widths to content
                for column in range(model.columnCount(None)):
                    self.network_table.resizeColumnToContents(column)
        except Exception as e:
            print(f"An error has occurred: {e}")

    def new_coms(self):

        if my_network.node_identities is None:
            my_network.node_identities = {}
            new_com = 1
        else:
            new_com = 1 + max(list(my_network.node_identities.values()))

        for node in self.clicked_values['nodes']:
            my_network.node_identities[node] = new_com

    def handle_override(self):
        dialog = OverrideDialog(self)
        dialog.exec()




    def handle_highlight_select(self):

        try:

            # Get the existing DataFrame from the model
            original_df = self.network_table.model()._data
            
            # Create mask for rows where one column is any original node AND the other column is any neighbor
            mask = (
                (original_df.iloc[:, 0].isin(self.clicked_values['nodes'])) |
                (original_df.iloc[:, 1].isin(self.clicked_values['nodes'])) |
                (original_df.iloc[:, 2].isin(self.clicked_values['edges']))

            )
            
            # Filter the DataFrame to only include direct connections
            filtered_df = original_df[mask].copy()
            
            # Create new model with filtered DataFrame and update selection table
            self.table_subgraph(self.selection_table, filtered_df)
            
            # Switch to selection table
            #self.selection_button.click()

            print("Selected nodes + edges have been isolated in the selection table, alongside their neighbors")

        except Exception as e:
            print(f"Error: {e}")


    def home(self):

        if self.original_xlim is None and self.original_dims is not None:
            self.original_xlim = (0, self.original_dims[1])
            self.original_ylim = (0, self.original_dims[0])
        self.view.setRange(xRange=self.original_xlim, yRange=self.original_ylim, padding=0)
        self.update_display()

    def toggle_scalebar(self):

        try:

            self.scale_bar = self.toggle_scale.isChecked()

            if self.grid_ready:
                self.toggle_grid()
                self.toggle_scale.setChecked(True)
                self.grid_ready = False
                self.remove_scale = True
                return
            if self.remove_scale:
                self._remove_scalebar()
                self.update_display(preserve_zoom=(self.ax.get_xlim(), self.ax.get_ylim()))
                self.toggle_scale.setChecked(True)
                self.remove_scale = False
                self.remove_grid = True
                return

            if self.scale_bar:
                self.grid_ready = True
                self._draw_scalebar()
                self.update_display(preserve_zoom=(self.ax.get_xlim(), self.ax.get_ylim()))
            else:
                self.coord_label.setText("                                                                                               ")
                self.grid_ready = False
                self.remove_grid = False
                self.toggle_grid()
                self._remove_scalebar()
                self.update_display(preserve_zoom=(self.ax.get_xlim(), self.ax.get_ylim()))
        except:
            self.scale_bar = False
            self.toggle_scale.setChecked(False)
            self._remove_scalebar()


    def toggle_highlight(self):
        self.highlight = self.high_button.isChecked()
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

        if self.high_button.isChecked() and self.machine_window is None and not self.preview:
            if self.highlight_overlay is None and ((len(self.clicked_values['nodes']) + len(self.clicked_values['edges'])) > 0):
                if self.needs_mini:
                    self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                    self.needs_mini = False
                else:
                    self.evaluate_mini()
            else:
                self.evaluate_mini()

        
        self.update_display(preserve_zoom=(current_xlim, current_ylim))

        
    def toggle_zoom_mode(self):
        """Toggle zoom mode on/off."""
        self.zoom_mode = self.zoom_button.isChecked()

        if self.zoom_mode:
            if self.pan_mode:
                self.pan_mode = False
                self.pan_button.setChecked(False)

            self.pen_button.setChecked(False)
            self.brush_mode = False
            self.can = False
            self.threed = False
            self.last_change = None
            if self.machine_window is not None:
                self.machine_window.silence_button()
            self.graphics_widget.setCursor(Qt.CursorShape.CrossCursor)
            if (hasattr(self, 'virtual_draw_operations') and self.virtual_draw_operations) or \
               (hasattr(self, 'virtual_erase_operations') and self.virtual_erase_operations) or \
               (hasattr(self, 'current_operation') and self.current_operation):
                # Finish current operation first
                if hasattr(self, 'current_operation') and self.current_operation:
                    self.pm.finish_current_virtual_operation()
                # Now convert to real data
                self.pm.convert_virtual_strokes_to_data()
                current_xlim = self.ax.get_xlim()
                current_ylim = self.ax.get_ylim()
                self.update_display(preserve_zoom=(current_xlim, current_ylim))

        else:
            if self.machine_window is None:
                self.graphics_widget.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.machine_window.toggle_brush_button()


    def toggle_pan_mode(self):
        """Toggle pan mode on/off."""
        self.pan_mode = self.pan_button.isChecked()
        if self.pan_mode:
            self.zoom_button.setChecked(False)
            self.pen_button.setChecked(False)
            self.zoom_mode = False
            self.can = False
            self.threed = False
            self.last_change = None
            self.brush_mode = False
            if (hasattr(self, 'completed_paint_strokes') and self.completed_paint_strokes) or \
               (hasattr(self, 'current_stroke_points') and self.current_stroke_points) or \
               (hasattr(self, 'virtual_paint_items') and self.virtual_paint_items) or \
               (hasattr(self, 'current_paint_items') and self.current_paint_items):
                if hasattr(self, 'current_stroke_points') and self.current_stroke_points:
                    self.pm.finish_current_virtual_operation()
                self.pm.convert_virtual_strokes_to_data()
                self.update_display()
            if self.machine_window is not None:
                self.machine_window.silence_button()
            self.graphics_widget.setCursor(Qt.CursorShape.OpenHandCursor)

            if self.machine_window is not None:
                if self.machine_window.segmentation_worker is not None:
                    if not self.machine_window.segmentation_worker._paused:
                        self.resume = True
                    self.machine_window.segmentation_worker.pause()

            self.update_display()

        else:
            self.setEnabled(True)
            self.update_display()
            if self.machine_window is None:
                self.graphics_widget.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.machine_window.toggle_brush_button()

    def toggle_brush_mode(self):
        """Toggle brush mode on/off"""
        self.brush_mode = self.pen_button.isChecked()
        if self.brush_mode:

            if self.pan_mode:
                self.pan_mode = False
                self.pan_button.setChecked(False)
            else:
                print("Pen mode enabled. Left click = Draw, Right click = Erase. 'Ctrl + Mousewheel' = Resize Pen. 'F' = Toggle Fill Can. 'D' = Toggle 3D Drawing. 'Ctrl + Z' = Undo Last")

            self.pm = painting.PaintManager(parent = self)

            # Start virtual paint session
            # Get current zoom to preserve it
            current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
            current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

            if self.pen_button.isChecked():
                channel = self.active_channel
            else:
                channel = 2

            self.pan_button.setChecked(False)
            self.zoom_button.setChecked(False)
            self.pan_mode = False
            self.zoom_mode = False
            self.update_brush_cursor()
        else:
            if (hasattr(self, 'completed_paint_strokes') and self.completed_paint_strokes) or \
               (hasattr(self, 'current_stroke_points') and self.current_stroke_points) or \
               (hasattr(self, 'virtual_paint_items') and self.virtual_paint_items) or \
               (hasattr(self, 'current_paint_items') and self.current_paint_items):
                if hasattr(self, 'current_stroke_points') and self.current_stroke_points:
                    self.pm.finish_current_virtual_operation()
                self.pm.convert_virtual_strokes_to_data()
            self.update_display()

            self.last_change = None
            self.can = False
            self.threed = False
            self.graphics_widget.setCursor(Qt.CursorShape.ArrowCursor)

    def toggle_can(self):

        if not self.can:
            self.can = True
            self.update_brush_cursor()
        else:
            self.can = False
            self.last_change = None
            self.update_brush_cursor()

    def toggle_threed(self):

        if not self.threed:
            print("3D Drawing Enabled. 'Alt + Mousewheel' = Alters Number of Slices to Draw On at Once. 'F' = Toggle Void Fill Can (Fills All Holes in the Z Stack Starting Above and Below Click)")
            self.threed = True
            self.threedthresh = 5
            self.update_brush_cursor()
        else:
            self.threed = False
            self.update_brush_cursor()


    def keyPressEvent(self, event):

        """Key press shortcuts for main class"""

        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            if (self.brush_mode or self.machine_window is not None) and not self.can:
                self.pm.undo_last_virtual_stroke()
                return
            else:
                try:
                    self.load_channel(self.last_change[1], self.last_change[0], True)
                except:
                    pass

            return  # Return to prevent triggering the regular Z key action below
        elif event.key() == Qt.Key_Z:
            self.zoom_button.click()
        elif self.machine_window is not None:
            if event.key() == Qt.Key_A:
                self.machine_window.switch_foreground()
            elif event.key() == Qt.Key_T:
                self.machine_window.train_model()
        elif event.key() == Qt.Key_X:
            self.high_button.click()
        elif event.key() == Qt.Key_F and event.modifiers() == Qt.ShiftModifier:
            self.handle_find()
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.handle_resave()
        elif event.key() == Qt.Key_L and event.modifiers() == Qt.ControlModifier:
            self.load_from_network_obj(directory = self.last_load)
        elif self.brush_mode and self.machine_window is None:
            if event.key() == Qt.Key_F:
                self.toggle_can()
            elif event.key() == Qt.Key_D:
                self.toggle_threed()
        elif event.key() == Qt.Key_Delete:
            self.handle_delete()
        elif event.key() == Qt.Key.Key_1:
            self.set_active_channel(0)
        elif event.key() == Qt.Key.Key_2:
            self.set_active_channel(1)
        elif event.key() == Qt.Key.Key_3:
            self.set_active_channel(2)
        elif event.key() == Qt.Key.Key_4:
            self.set_active_channel(3)

    def handle_resave(self, asbool = True):

        try:

            if self.last_saved is None:

                self.save_network_3d()

            else:
                my_network.dump(parent_dir=self.last_saved, name=self.last_save_name)

        except Exception as e:
            print(f"Error saving: {e}")

    def update_brush_cursor(self):
        """Update the cursor to show brush size"""
        if not self.brush_mode:
            return
        
        # Get font metrics first to determine text size
        font = QFont()
        font.setPointSize(14)
        font_metrics = QFontMetrics(font)
        thresh_text = str(self.threedthresh)
        text_rect = font_metrics.boundingRect(thresh_text)
        
        # Create a pixmap for the cursor - ensure it's large enough for text
        brush_size = self.brush_size * 2 + 2  # Add padding for border
        extra_width = max(0, text_rect.width() + 4 - brush_size)  # Extra width for text if needed
        extra_height = max(0, text_rect.height() + 4 - brush_size)  # Extra height for text if needed
        
        # Make sure pixmap is large enough for both brush and text
        total_width = brush_size + extra_width
        total_height = brush_size + extra_height
        pixmap = QPixmap(total_width, total_height)
        pixmap.fill(Qt.transparent)
        
        # Create painter for the pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing) 
        
        # Calculate center offset for brush ellipse to accommodate text
        x_offset = extra_width // 2
        y_offset = extra_height // 2
        
        # Draw circle
        if not self.threed:
            pen = QPen(Qt.white)
        else:
            pen = QPen(Qt.red)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.transparent)
        if not self.can:
            painter.drawEllipse(1 + x_offset, 1 + y_offset, brush_size-2, brush_size-2)
            
            # Draw threshold number when threed is True and can is False
            if self.threed:
                # Set text properties
                painter.setFont(font)
                painter.setPen(QPen(Qt.white))  # White text for visibility
                
                # Draw the text
                painter.drawText(2, font_metrics.ascent() + 2, thresh_text)
        else:
            painter.drawRect(1 + x_offset, 1 + y_offset, 8, 8) 
        
        # Create cursor from pixmap
        cursor = QCursor(pixmap)
        self.graphics_widget.setCursor(cursor)
        
        painter.end()

    def get_current_mouse_position(self):
        """Get current mouse position in data coordinates for PyQtGraph."""
        # Get the main application's current mouse position
        cursor_pos = QCursor.pos()
        
        # Convert global screen coordinates to graphics widget coordinates
        widget_pos = self.graphics_widget.mapFromGlobal(cursor_pos)
        
        # Check if the position is within the widget bounds
        if not (0 <= widget_pos.x() < self.graphics_widget.width() and 
                0 <= widget_pos.y() < self.graphics_widget.height()):
            return 0, 0  # Mouse is outside of the graphics widget
        
        try:
            # Convert widget coordinates to scene coordinates
            scene_pos = self.graphics_widget.mapToScene(widget_pos)
            
            # Check if within view bounds
            if not self.view.sceneBoundingRect().contains(scene_pos):
                return 0, 0
            
            # Convert scene coordinates to view (data) coordinates
            view_pos = self.view.mapSceneToView(scene_pos)
            
            # Get data coordinates
            x = view_pos.x()
            y = view_pos.y()
            
            # Check if within image bounds
            if self.original_dims is not None:
                if not (0 <= x < self.original_dims[1] and 0 <= y < self.original_dims[0]):
                    return 0, 0
            
            return x, y
        except:
            return 0, 0

    def handle_can(self, x, y):

        def fill_void_3d(x, y):
            """
            Intelligent void filling that propagates through Z-slices.
            Stops when the fill touches new edges (indicating it's spilling out of the void).
            """
            if not self.threed:
                return
            
            # Backup for undo
            ref = copy.deepcopy(self.channel_data[self.active_channel])
            the_slice = self.channel_data[self.active_channel]
            
            def fill_2d_slice(slice_data, x, y):
                """
                Fill a 2D slice at coordinate (x, y) and return the filled mask.
                Returns None if the point is already filled or out of bounds.
                """
                h, w = slice_data.shape
                
                # Check bounds
                if x < 0 or x >= w or y < 0 or y >= h:
                    return None
                
                # Check if already filled
                if slice_data[y, x] != 0:
                    return None
                
                # Invert to find voids
                inv = ~slice_data.astype(bool)
                
                # Label connected components
                labeled_array, num_features = ndimage.label(inv)
                
                # Get target label at clicked point
                target_label = labeled_array[y, x]
                
                if target_label > 0:
                    # Create mask of the clicked component
                    fill_mask = (labeled_array == target_label) * max(1, np.max(ref))
                    return fill_mask
                
                return None
            
            def get_touched_edges(mask):
                """
                Returns set of edges that the mask touches.
                Edges: 'top', 'bottom', 'left', 'right'
                """
                edges = set()
                h, w = mask.shape
                
                if np.any(mask[0, :]):      # Top edge
                    edges.add('top')
                if np.any(mask[h-1, :]):    # Bottom edge
                    edges.add('bottom')
                if np.any(mask[:, 0]):      # Left edge
                    edges.add('left')
                if np.any(mask[:, w-1]):    # Right edge
                    edges.add('right')
                
                return edges
            
            # Fill the initial slice
            initial_mask = fill_2d_slice(the_slice[self.current_slice], x, y)
            
            if initial_mask is None:
                return  # Nothing to fill (already filled or out of bounds)
            
            # Get the edges touched in the initial fill
            # These edges are "valid" and won't trigger stopping
            initial_edges = get_touched_edges(initial_mask)
            
            # Apply the initial fill
            the_slice[self.current_slice] = the_slice[self.current_slice] | initial_mask
            
            # Propagate FORWARD in Z (increasing slice numbers)
            for z in range(self.current_slice + 1, the_slice.shape[0]):
                # Try to fill at the same (x, y) position on this slice
                mask = fill_2d_slice(the_slice[z], x, y)
                
                if mask is None:
                    # No void at this position, stop propagating forward
                    break
                
                # Check which edges this mask touches
                touched_edges = get_touched_edges(mask)
                
                # Check if it touches NEW edges (not in initial_edges)
                new_edges = touched_edges - initial_edges
                
                if new_edges:
                    # Touches new edges - the void is spilling out!
                    # Don't fill this slice and stop propagating forward
                    break
                
                # Safe to fill this slice
                the_slice[z] = the_slice[z] | mask
            
            # Propagate BACKWARD in Z (decreasing slice numbers)
            for z in range(self.current_slice - 1, -1, -1):
                # Try to fill at the same (x, y) position on this slice
                mask = fill_2d_slice(the_slice[z], x, y)
                
                if mask is None:
                    # No void at this position, stop propagating backward
                    break
                
                # Check which edges this mask touches
                touched_edges = get_touched_edges(mask)
                
                # Check if it touches NEW edges (not in initial_edges)
                new_edges = touched_edges - initial_edges
                
                if new_edges:
                    # Touches new edges - the void is spilling out!
                    # Don't fill this slice and stop propagating backward
                    break
                
                # Safe to fill this slice
                the_slice[z] = the_slice[z] | mask
            
            # Save the backup and apply the changes
            self.last_change = [ref, self.active_channel]
            self.load_channel(self.active_channel, the_slice, True)


        if self.threed:
            fill_void_3d(x, y)

        else:

            ref = copy.deepcopy(self.channel_data[self.active_channel])

            the_slice = self.channel_data[self.active_channel][self.current_slice]

            # First invert the boolean array
            inv = n3d.invert_boolean(the_slice)
            
            # Label the connected components in the inverted array
            labeled_array, num_features = ndimage.label(inv)
            
            # Get the target label at the clicked point
            target_label = labeled_array[y][x]
            
            # Only fill if we clicked on a valid region (target_label > 0)
            if target_label > 0:
                # Create a mask of the connected component we clicked on
                fill_mask = (labeled_array == target_label) * max(1, np.max(ref))

                self.last_change = [ref, self.active_channel]
                
                # Add this mask to the original slice
                the_slice = the_slice | fill_mask  # Use logical OR to add the filled region
            
            # Update the channel data
            self.channel_data[self.active_channel][self.current_slice] = the_slice
            self.load_channel(self.active_channel, self.channel_data[self.active_channel], True)


    def on_mouse_move(self, pos):
        """Handle mouse movement in pyqtgraph"""
        # Map mouse position to view coordinates
        if not self.view.sceneBoundingRect().contains(pos):
            return
        
        mouse_point = self.view.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        try:
            if self.remove_grid or self.remove_scale:
                self.coord_label.setText(f"Z: {self.current_slice}, Y: {int(y)}, X: {int(x)}")
        except:
            pass

        try:
            # Check if within image bounds
            if self.original_dims is None:
                return
            if not (0 <= x < self.original_dims[1] and 0 <= y < self.original_dims[0]):
                return
        except:
            return

        current_time = time.time()
        self.rect_time = current_time
        
        # Selection rectangle handling
        if self.selection_start and not self.selecting and not self.pan_mode and not self.brush_mode:
            if (abs(x - self.selection_start[0]) > 1 or 
                abs(y - self.selection_start[1]) > 1):
                self.selecting = True

                self.pan_button.setCheckable(False)
                
                # Create selection rectangle using ROI
                width = abs(x - self.selection_start[0])
                height = abs(y - self.selection_start[1])
                x_min = min(self.selection_start[0], x)
                y_min = min(self.selection_start[1], y)
                
                self.selection_rect = pg.ROI([x_min, y_min], [width, height], 
                                             pen=pg.mkPen('w', style=pg.QtCore.Qt.PenStyle.DashLine),
                                             movable=False, removable=False)
                self.selection_rect.handleSize = 0  # Hide handles
                self.selection_rect.handlePen = pg.mkPen(None)
                self.view.addItem(self.selection_rect)
                
        if self.selecting and self.selection_rect is not None:
            # Throttle updates
            self.pan_button.setCheckable(False)
            if current_time - self.last_update_time < self.update_interval:
                return
            self.last_update_time = current_time
            
            x_min = min(self.selection_start[0], x)
            y_min = min(self.selection_start[1], y)
            width = abs(x - self.selection_start[0])
            height = abs(y - self.selection_start[1])
            
            self.selection_rect.setPos([x_min, y_min])
            self.selection_rect.setSize([width, height])


        elif self.painting and self.brush_mode:
            # Throttle updates
            current_time = time.time()
            if current_time - getattr(self, 'last_paint_update_time', 0) < 0.016:  # ~60fps
                return
            self.last_paint_update_time = current_time
            
            x_int, y_int = int(x), int(y)
            
            # Determine foreground/background for machine window mode
            foreground = getattr(self, 'foreground', True)
            
            # Add virtual paint stroke with interpolation
            brush_size = getattr(self, 'brush_size', 5)
            self.pm.add_virtual_paint_stroke(x_int, y_int, brush_size, self.erase, foreground)

    def on_mouse_release(self, event):
        """Handle mouse release events in pyqtgraph"""
        # Map to view coordinates
        self.pan_button.setCheckable(True)
        if not self.view.sceneBoundingRect().contains(event.scenePos()):
            return
        
        mouse_point = self.view.mapSceneToView(event.scenePos())
        x, y = mouse_point.x(), mouse_point.y()
        
        if self.zoom_mode:
            rect_condition = (time.time() - self.rect_time) > 0.01
        else:
            rect_condition = True

        
        if event.button() == Qt.MouseButton.LeftButton:
            if rect_condition and self.selecting and self.selection_rect is not None:
                # Get the rectangle bounds
                rect_pos = self.selection_rect.pos()
                rect_size = self.selection_rect.size()
                x0 = rect_pos[0]
                y0 = rect_pos[1]
                width = rect_size[0]
                height = rect_size[1]
                
                shift_pressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                
                if shift_pressed:
                    args = int(x0), int(x0 + width), int(y0), int(y0 + height)
                    self.show_crop_dialog(args)

                elif self.zoom_mode:  # Rectangle zoom
                    # Calculate aspect ratio to avoid zooming into very thin rectangles
                    aspect_ratio = width / height if height > 0 else float('inf')
                    
                    # Skip zoom if the rectangle is too narrow/thin
                    if width > 10 and height > 10 and 0.1 < aspect_ratio < 10:
                        new_xlim = [x0, x0 + width]
                        new_ylim = [y0, y0 + height]
                        self.view.setRange(xRange=new_xlim, yRange=new_ylim, padding=0)
                    
                    self.zoom_changed = True
                    
                    # Update display
                    view_range = self.view.viewRange()
                    self.previous_zoom_level = (view_range[0][1] - view_range[0][0]) * (view_range[1][1] - view_range[1][0])
                    self.update_display(preserve_zoom=(view_range[0], view_range[1]))
                
                # Get current slice data for active channel
                elif self.channel_data[self.active_channel] is not None:
                    data = self.channel_data[self.active_channel][self.current_slice]
                    
                    # Convert coordinates to array indices
                    x_min = max(0, int(x0))
                    y_min = max(0, int(y0))
                    x_max = min(data.shape[1], int(x0 + width))
                    y_max = min(data.shape[0], int(y0 + height))
                    
                    # Extract unique non-zero values in selection rectangle
                    selected_region = data[y_min:y_max, x_min:x_max]
                    selected_values = np.unique(selected_region)
                    selected_values = selected_values[selected_values != 0]
                    
                    # Check if ctrl is pressed
                    ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                    
                    # Update clicked_values based on active channel
                    if self.active_channel == 0:  # Nodes
                        if not ctrl_pressed:
                            self.clicked_values['nodes'] = []
                            self.clicked_values['edges'] = []
                        self.clicked_values['nodes'].extend(selected_values)
                        self.clicked_values['nodes'] = list(dict.fromkeys(self.clicked_values['nodes']))
                        
                        if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                            self.mini_overlay = True
                            self.create_mini_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                        else:
                            self.create_highlight_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                        
                        if len(self.clicked_values['nodes']) == 1:
                            self.highlight_value_in_tables(self.clicked_values['nodes'][-1])
                            self.handle_info('node')
                    
                    elif self.active_channel == 1:  # Edges
                        if not ctrl_pressed:
                            self.clicked_values['edges'] = []
                            self.clicked_values['nodes'] = []
                        self.clicked_values['edges'].extend(selected_values)
                        self.clicked_values['edges'] = list(dict.fromkeys(self.clicked_values['edges']))
                        
                        if self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2] > self.mini_thresh:
                            self.mini_overlay = True
                            self.create_mini_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                        else:
                            self.create_highlight_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                        
                        if len(self.clicked_values['edges']):
                            self.highlight_value_in_tables(self.clicked_values['edges'][-1])
                            self.handle_info('edge')
                    
                    try:
                        if len(self.clicked_values['nodes']) > 0 or len(self.clicked_values['edges']) > 0:
                            old_nodes = copy.deepcopy(self.clicked_values['nodes'])
                            original_df = self.network_table.model()._data
                            
                            mask = (
                                ((original_df.iloc[:, 0].isin(self.clicked_values['nodes'])) &
                                 (original_df.iloc[:, 1].isin(self.clicked_values['nodes']))) |
                                (original_df.iloc[:, 2].isin(self.clicked_values['edges']))
                            )
                            
                            filtered_df = original_df[mask].copy()
                            self.table_subgraph(self.selection_table, filtered_df)
                    except:
                        pass
            
            elif not self.selecting and self.selection_start:
                # Handle as a normal click
                self.on_mouse_click_pg(event, x, y)
            
            # Clean up selection
            self.selection_start = None
            self.selecting = False

            for item in list(self.view.addedItems):
                if isinstance(item, pg.ROI):
                    self.view.removeItem(item)
        
        # Handle brush mode cleanup
        if self.brush_mode and hasattr(self, 'painting') and self.painting:
            self.pm.connect_virtual_paint_points()
            self.pm.finish_current_stroke()
            self.pm.finish_current_virtual_operation()
            self.painting = False
            
            if self.erase:
                if (hasattr(self, 'completed_paint_strokes') and self.completed_paint_strokes) or \
                   (hasattr(self, 'current_stroke_points') and self.current_stroke_points) or \
                   (hasattr(self, 'virtual_paint_items') and self.virtual_paint_items) or \
                   (hasattr(self, 'current_paint_items') and self.current_paint_items):
                    if hasattr(self, 'current_stroke_points') and self.current_stroke_points:
                        self.pm.finish_current_virtual_operation()
                    self.pm.convert_virtual_strokes_to_data()
                    view_range = self.view.viewRange()
                    self.update_display()
            
    
    def on_mouse_click_pg(self, event, x, y):
        """Handle mouse clicks for data inspection (called from release if no drag)."""
        # Not in any special mode - handle value inspection

        if self.zoom_mode:
            # Initialize original bounds if needed
            if self.original_xlim is None and self.original_dims is not None:
                self.original_xlim = (0, self.original_dims[1])
                self.original_ylim = (0, self.original_dims[0])
            
            # Get current view range
            view_range = self.view.viewRange()
            current_xlim = view_range[0]
            current_ylim = view_range[1]
            
            if event.button() == Qt.MouseButton.LeftButton:  # Left click - zoom in
                x_range = (current_xlim[1] - current_xlim[0]) / 4
                y_range = (current_ylim[1] - current_ylim[0]) / 4
                
                new_xlim = [x - x_range, x + x_range]
                new_ylim = [y - y_range, y + y_range]
                
                self.view.setRange(xRange=new_xlim, yRange=new_ylim, padding=0)
                self.zoom_changed = True
                            
            # Update display with new zoom
            view_range = self.view.viewRange()
            self.update_display()

        elif self.channel_data[self.active_channel] is not None:
            try:
                # Get clicked value
                x_idx = int(round(x))
                y_idx = int(round(y))
                
                # Check if Ctrl key is pressed
                ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                
                if len(self.channel_data[self.active_channel].shape) != 4:
                    if self.channel_data[self.active_channel][self.current_slice, y_idx, x_idx] != 0:
                        clicked_value = self.channel_data[self.active_channel][self.current_slice, y_idx, x_idx]
                    else:
                        if not ctrl_pressed:
                            self.clicked_values = {
                                'nodes': [],
                                'edges': []
                            }
                            self.create_highlight_overlay()
                        return
                
                starting_vals = copy.deepcopy(self.clicked_values)
                
                # Store or remove the clicked value in the appropriate list
                if self.active_channel == 0:
                    if ctrl_pressed:
                        if clicked_value in self.clicked_values['nodes']:
                            self.clicked_values['nodes'].remove(clicked_value)
                        else:
                            self.clicked_values['nodes'].append(clicked_value)
                    else:
                        self.clicked_values = {'nodes': [clicked_value], 'edges': []}
                    latest_value = self.clicked_values['nodes'][-1] if self.clicked_values['nodes'] else None
                    self.handle_info('node')
                elif self.active_channel == 1:
                    if ctrl_pressed:
                        if clicked_value in self.clicked_values['edges']:
                            self.clicked_values['edges'].remove(clicked_value)
                        else:
                            self.clicked_values['edges'].append(clicked_value)
                    else:
                        self.clicked_values = {'nodes': [], 'edges': [clicked_value]}
                    latest_value = self.clicked_values['edges'][-1] if self.clicked_values['edges'] else None
                    self.handle_info('edge')
                
                # Try to find and highlight the latest value in the current table
                try:
                    found = self.highlight_value_in_tables(latest_value)
                except:
                    return
                
                # If not found in current table but exists in other table, switch
                try:
                    if not found:
                        other_table = self.selection_table if self.active_table == self.network_table else self.network_table
                        if other_table.model() is not None:
                            df = other_table.model()._data
                            if self.active_channel == 0:
                                exists_in_other = (df[df.columns[0]] == latest_value).any() or (df[df.columns[1]] == latest_value).any()
                            else:
                                exists_in_other = (df[df.columns[2]] == latest_value).any()
                            
                            if exists_in_other and not (self.network_graph_widget.isVisbile() or self.selection_graph_widget.is_Visible()):
                                if other_table == self.network_table:
                                    self.network_button.click()
                                else:
                                    self.selection_button.click()
                                self.highlight_value_in_tables(latest_value)
                except:
                    pass
                
                # Highlight the clicked element in the image
                if self.active_channel == 0 and starting_vals['nodes'] != self.clicked_values['nodes']:
                    if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                        self.mini_overlay = True
                        self.create_mini_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                    else:
                        self.create_highlight_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                elif self.active_channel == 1 and starting_vals['edges'] != self.clicked_values['edges']:
                    if self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2] > self.mini_thresh:
                        self.mini_overlay = True
                        self.create_mini_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                    else:
                        self.create_highlight_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
            
            except IndexError:
                pass         

    def on_mouse_press(self, event):
        """Handle mouse press events in pyqtgraph."""
        # Map to view coordinates
        if not self.view.sceneBoundingRect().contains(event.scenePos()):
            return False
        
        mouse_point = self.view.mapSceneToView(event.scenePos())
        x, y = mouse_point.x(), mouse_point.y()
        
        # Check bounds
        if self.original_dims is None:
            return False
        if not (0 <= x < self.original_dims[1] and 0 <= y < self.original_dims[0]):
            return False
        

        # Middle click -> toggle pan mode
        if event.button() == Qt.MouseButton.MiddleButton and not self.is_wheeling:
            if self.machine_window is None and self.brush_mode:
                self.penning = True
            else:
                self.penning = False
            self.pan_button.click()
            self.disable_pan = True
            return True

        #self.pan_button.setCheckable(False)

        if self.brush_mode and not (event.button() == Qt.MouseButton.MiddleButton):
            """Handle brush mode with virtual painting."""
            view_range = self.view.viewRange()
            current_xlim = view_range[0]
            current_ylim = view_range[1]
            
            if self.pen_button.isChecked():
                channel = self.active_channel
            else:
                channel = 2
            
            self.pm.initiate_paint_session(channel, current_xlim, current_ylim)
            
            if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
                if self.machine_window is not None:
                    if self.machine_window.segmentation_worker is not None:
                        if not self.machine_window.segmentation_worker._paused:
                            self.resume = True
                        self.machine_window.segmentation_worker.pause()
                
                x_int, y_int = int(x), int(y)
                
                # Get current zoom to preserve it
                view_range = self.view.viewRange()
                current_xlim = view_range[0]
                current_ylim = view_range[1]
                
                if event.button() == Qt.MouseButton.LeftButton and getattr(self, 'can', False):
                    self.update_display(preserve_zoom=(current_xlim, current_ylim))
                    self.handle_can(x_int, y_int)
                    return
                
                # Determine erase mode
                if event.button() == Qt.MouseButton.RightButton:
                    self.erase = True
                else:
                    self.erase = False
                
                # Determine foreground/background for machine window mode
                foreground = getattr(self, 'foreground', True)
                
                self.last_virtual_pos = (x_int, y_int)
                
                if self.pen_button.isChecked():
                    channel = self.active_channel
                else:
                    channel = 2
                
                self.pm.start_virtual_paint_session(channel, current_xlim, current_ylim)
                
                # Add first virtual paint stroke
                brush_size = getattr(self, 'brush_size', 5)
                self.pm.add_virtual_paint_stroke(x_int, y_int, brush_size, self.erase, foreground)
                
                # Update display with virtual paint
                self.painting = True
        
        elif not self.zoom_mode and event.button() == Qt.MouseButton.RightButton:  # Right click for context menu
            self.create_context_menu_pg(event, x, y)

        elif self.zoom_mode and event.button() == Qt.MouseButton.RightButton:  # Right click - zoom out
            
            if self.original_xlim is None and self.original_dims is not None:
                self.original_xlim = (0, self.original_dims[1])
                self.original_ylim = (0, self.original_dims[0])

            view_range = self.view.viewRange()
            current_xlim = view_range[0]
            current_ylim = view_range[1]

            x_range = (current_xlim[1] - current_xlim[0])
            y_range = (current_ylim[1] - current_ylim[0])
            
            new_xlim = [x - x_range, x + x_range]
            new_ylim = [y - y_range, y + y_range]
            
            # Check if zooming out would go beyond original bounds
            if (new_xlim[0] <= self.original_xlim[0] or 
                new_xlim[1] >= self.original_xlim[1] or
                new_ylim[0] <= self.original_ylim[0] or
                new_ylim[1] >= self.original_ylim[1]):
                # Reset to original view
                self.view.setRange(xRange=self.original_xlim, yRange=self.original_ylim, padding=0)
            else:
                self.view.setRange(xRange=new_xlim, yRange=new_ylim, padding=0)

            view_range = self.view.viewRange()
            self.update_display()
        
        elif event.button() == Qt.MouseButton.LeftButton:  # Left click
            # Store initial click position but don't start selection yet
            self.selection_start = (x, y)
            self.selecting = False

        return False

    def create_context_menu_pg(self, event, x, y):
        """Wrapper for context menu with pyqtgraph coordinates"""
        # Create a mock matplotlib-like event for compatibility
        class MockEvent:
            def __init__(self, x, y):
                self.xdata = x
                self.ydata = y
        
        mock_event = MockEvent(x, y)
        self.create_context_menu(mock_event)

    def highlight_value_in_tables(self, clicked_value):
        """Helper method to find and highlight a value in both tables."""

        try:
        
            if not self.network_table.model() and not self.selection_table.model():
                return False

            found = False
            tables_to_check = [self.network_table, self.selection_table]
            if not hasattr(self, 'active_table'):
                self.active_table = self.network_table
            active_table_index = tables_to_check.index(self.active_table)
            
            # Reorder tables to check active table first
            tables_to_check = tables_to_check[active_table_index:] + tables_to_check[:active_table_index]
            
            for table in tables_to_check:
                if table.model() is None:
                    continue
                    
                df = table.model()._data

                # Create appropriate masks based on active channel
                if self.active_channel == 0:  # Nodes channel
                    col1_matches = df[df.columns[0]] == clicked_value
                    col2_matches = df[df.columns[1]] == clicked_value
                    all_matches = col1_matches | col2_matches

                elif self.active_channel == 1:  # Edges channel
                    all_matches = df[df.columns[2]] == clicked_value

                else:
                    continue

                if all_matches.any():
                    # Get indices from the current dataframe's index
                    match_indices = df[all_matches].index.tolist()
                    
                    # If this is the active table, handle selection and scrolling
                    if table == self.active_table:
                        current_row = table.currentIndex().row()
                        
                        # Convert match_indices to row numbers (position in the visible table)
                        row_positions = [df.index.get_loc(idx) for idx in match_indices]
                        
                        # Find next match after current position
                        if current_row >= 0:
                            next_positions = [pos for pos in row_positions if pos > current_row]
                            row_pos = next_positions[0] if next_positions else row_positions[0]
                        else:
                            row_pos = row_positions[0]
                        
                        # Update selection and scroll
                        model_index = table.model().index(row_pos, 0)
                        table.scrollTo(model_index)
                        table.clearSelection()
                        table.selectRow(row_pos)
                        table.setCurrentIndex(model_index)

                        # Add highlighting for specific cells based on active channel
                        if self.active_channel == 0:  # Nodes channel
                            # Only highlight cells in columns 0 and 1 where the value matches
                            if df.iloc[row_pos, 0] == clicked_value:
                                table.model().highlight_cell(row_pos, 0)
                            if df.iloc[row_pos, 1] == clicked_value:
                                table.model().highlight_cell(row_pos, 1)
                        else:  # Edges channel
                            # Highlight the edge column
                            table.model().highlight_cell(row_pos, 2)
                    
                    found = True

            return found
        except:
            pass
                
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")

        # Create Save submenu
        save_menu = file_menu.addMenu("Save")
        network_save = save_menu.addAction("Save Current Session")
        network_save.triggered.connect(lambda: self.save_network_3d(False))
        for i in range(4):
            save_action = save_menu.addAction(f"Save {self.channel_names[i]}")
            save_action.triggered.connect(lambda checked, ch=i: self.save(ch, False))
        highlight_save = save_menu.addAction("Save Highlight Overlay")
        highlight_save.triggered.connect(lambda checked, ch=4: self.save(ch, False))

        # Create Save As submenu
        save_as_menu = file_menu.addMenu("Save As")
        network_saveas = save_as_menu.addAction("Save Current Session As")
        network_saveas.triggered.connect(lambda: self.save_network_3d(True))
        for i in range(4):
            save_action = save_as_menu.addAction(f"Save {self.channel_names[i]} As")
            save_action.triggered.connect(lambda checked, ch=i: self.save(ch))
        highlight_save = save_as_menu.addAction("Save Highlight Overlay As")
        highlight_save.triggered.connect(lambda checked, ch=4: self.save(ch))
        
        # Create Load submenu
        load_menu = file_menu.addMenu("Load")
        network_load = load_menu.addAction("Load Previous Session")
        network_load.triggered.connect(lambda: self.load_from_network_obj(None))
        for i in range(4):
            load_action = load_menu.addAction(f"Load {self.channel_names[i]}")
            load_action.triggered.connect(lambda checked, ch=i: self.load_channel(ch))
        load_action = load_menu.addAction("Load Full-Sized Highlight Overlay")
        load_action.triggered.connect(lambda: self.load_channel(channel_index = 4, load_highlight = True))
        load_action = load_menu.addAction("Load Network")
        load_action.triggered.connect(self.load_network)
        load_action = load_menu.addAction("Load From Excel Helper")
        load_action.triggered.connect(self.launch_excelotron)
        misc_menu = load_menu.addMenu("Load Misc Properties")
        load_action = misc_menu.addAction("Load Node IDs")
        load_action.triggered.connect(lambda: self.load_misc('Node Identities'))
        load_action = misc_menu.addAction("Load Node Centroids")
        load_action.triggered.connect(lambda: self.load_misc('Node Centroids'))
        load_action = misc_menu.addAction("Load Edge Centroids")
        load_action.triggered.connect(lambda: self.load_misc('Edge Centroids'))
        load_action = misc_menu.addAction("Load Node Communities")
        load_action.triggered.connect(lambda: self.load_misc('Communities'))
        node_identities = file_menu.addMenu('Images -> Node Identities')
        load_action = node_identities.addAction("Merge Labeled Images Into Nodes")
        load_action.triggered.connect(lambda: self.load_misc('Merge Nodes'))
        load_action = node_identities.addAction("Assign Node Identities From Overlap With Other Images")
        load_action.triggered.connect(self.show_merge_node_id_dialog)

        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analyze")
        network_menu = analysis_menu.addMenu("Network")
        netshow_action = network_menu.addAction("Show Network")
        netshow_action.triggered.connect(self.show_netshow_dialog)
        report_action = network_menu.addAction("Generic Network Report")
        report_action.triggered.connect(self.handle_report)
        partition_action = network_menu.addAction("Create Communities Based on Network? (Finds spatial or functional clusters)")
        partition_action.triggered.connect(self.show_partition_dialog)
        neigh_com_action = network_menu.addAction("Create Communities Based on Node's Immediate Neighbors? (Finds recurring motifs; would want many node identities first)")
        neigh_com_action.triggered.connect(self.handle_neigh_com)
        com_cell_action = network_menu.addAction("Create Communities Based on Cuboidal Proximity Cells? (For chaotic node arrangements)")
        com_cell_action.triggered.connect(self.handle_com_cell)
        com_identity_action = network_menu.addAction("Calculate Composition of Communities (and Show UMAP/Heatmaps)")
        com_identity_action.triggered.connect(self.handle_com_id)
        com_neighbor_action = network_menu.addAction("Convert Network Communities into Supercommunities? (ie Further cluster similar communities—Also Returns Compositional Heatmaps)")
        com_neighbor_action.triggered.connect(self.handle_com_neighbor)

        stats_menu = analysis_menu.addMenu("Stats")
        stats_net_menu = stats_menu.addMenu("Network Related")
        allstats_action = stats_net_menu.addAction("Calculate Generic Network Stats")
        allstats_action.triggered.connect(self.stats)
        histos_action = stats_net_menu.addAction("Network Statistic Histograms")
        histos_action.triggered.connect(self.histograms)
        radial_action = stats_net_menu.addAction("Radial Distribution Analysis")
        radial_action.triggered.connect(self.show_radial_dialog)
        netnear_action = stats_net_menu.addAction("Network Nearest Neighbors")
        netnear_action.triggered.connect(self.show_netnear_dialog)
        heatmap_action = stats_net_menu.addAction("Community Cluster Heatmap")
        heatmap_action.triggered.connect(self.show_heatmap_dialog)

        stats_space_menu = stats_menu.addMenu("Spatial")
        neighbor_id_action = stats_space_menu.addAction("Identity Distribution of Neighbors")
        neighbor_id_action.triggered.connect(self.show_neighbor_id_dialog)
        ripley_action = stats_space_menu.addAction("Ripley Clustering Analysis")
        ripley_action.triggered.connect(self.show_ripley_dialog)
        nearneigh_action = stats_space_menu.addAction("Average Nearest Neighbors (With Clustering Heatmaps)")
        nearneigh_action.triggered.connect(self.show_nearneigh_dialog)
        inter_action = stats_space_menu.addAction("Calculate Node < > Edge Interaction")
        inter_action.triggered.connect(self.show_interaction_dialog)

        stats_morph_menu = stats_menu.addMenu("Morphological")
        vol_action = stats_morph_menu.addAction("Calculate Volumes")
        vol_action.triggered.connect(self.volumes)
        rad_action = stats_morph_menu.addAction("Calculate Radii")
        rad_action.triggered.connect(self.show_rad_dialog)
        sa_action = stats_morph_menu.addAction("Calculate Surface Area")
        sa_action.triggered.connect(self.handle_sa)
        sphere_action = stats_morph_menu.addAction("Calculate Sphericities")
        sphere_action.triggered.connect(self.handle_sphericity)
        branch_stats = stats_morph_menu.addAction("Calculate Branch Stats (Lengths, Tortuosities)")
        branch_stats.triggered.connect(self.show_branchstat_dialog)

        sig_action = stats_menu.addAction("Significance Testing")
        sig_action.triggered.connect(self.sig_test)
        violin_action = stats_menu.addAction("Show Identity Violins/UMAP/Assign Intensity Communities")
        violin_action.triggered.connect(self.show_violin_dialog)


        overlay_menu = analysis_menu.addMenu("Data/Overlays")
        degree_action = overlay_menu.addAction("Get Degree Information")
        degree_action.triggered.connect(self.show_degree_dialog)
        hub_action = overlay_menu.addAction("Get Hub Information")
        hub_action.triggered.connect(self.show_hub_dialog)
        mother_action = overlay_menu.addAction("Get Mother Nodes")
        mother_action.triggered.connect(self.show_mother_dialog)
        community_code_action = overlay_menu.addAction("Code Communities")
        community_code_action.triggered.connect(lambda: self.show_code_dialog(sort = 'Community'))
        id_code_action = overlay_menu.addAction("Code Identities")
        id_code_action.triggered.connect(lambda: self.show_code_dialog(sort = 'Identity'))
        umap_action = overlay_menu.addAction("Centroid UMAP")
        umap_action.triggered.connect(self.handle_centroid_umap)

        rand_menu = analysis_menu.addMenu("Randomize")
        random_action = rand_menu.addAction("Generate Equivalent Random Network")
        random_action.triggered.connect(self.show_random_dialog)
        random_nodes = rand_menu.addAction("Scramble Nodes (Centroids)")
        random_nodes.triggered.connect(self.show_randnode_dialog)



        # Process menu
        process_menu = menubar.addMenu("Process")
        calculate_menu = process_menu.addMenu("Calculate Network")
        calc_all_action = calculate_menu.addAction("Calculate Connectivity Network (Find Node-Edge-Node Network)")
        calc_all_action.triggered.connect(self.show_calc_all_dialog)
        calc_prox_action = calculate_menu.addAction("Calculate Proximity Network (connect nodes by distance)")
        calc_prox_action.triggered.connect(self.show_calc_prox_dialog)
        calc_branch_action = calculate_menu.addAction("Calculate Branchpoint Network (Connect Branchpoints of Edge Image - Good for Nerves/Vessels)")
        calc_branch_action.triggered.connect(self.handle_calc_branch)
        calc_branchprox_action = calculate_menu.addAction("Calculate Branch Adjacency Network (Of Edges)")
        calc_branchprox_action.triggered.connect(self.handle_branchprox_calc)
        #calc_id_net_action = calculate_menu.addAction("Calculate Identity Network (beta)")
        #calc_id_net_action.triggered.connect(self.handle_identity_net_calc)
        centroid_action = calculate_menu.addAction("Calculate Centroids (Active Image)")
        centroid_action.triggered.connect(self.show_centroid_dialog)

        image_menu = process_menu.addMenu("Image")
        resize_action = image_menu.addAction("Resize (Up/Downsample)")
        resize_action.triggered.connect(self.show_resize_dialog)
        clean_action = image_menu.addAction("Clean Segmentation")
        clean_action.triggered.connect(self.show_clean_dialog)
        dilate_action = image_menu.addAction("Dilate")
        dilate_action.triggered.connect(self.show_dilate_dialog)
        erode_action = image_menu.addAction("Erode")
        erode_action.triggered.connect(self.show_erode_dialog)
        hole_action = image_menu.addAction("Fill Holes")
        hole_action.triggered.connect(self.show_hole_dialog)
        binarize_action = image_menu.addAction("Binarize")
        binarize_action.triggered.connect(self.show_binarize_dialog)
        label_action = image_menu.addAction("Label Objects")
        label_action.triggered.connect(self.show_label_dialog)
        slabel_action = image_menu.addAction("Neighbor Labels")
        slabel_action.triggered.connect(self.show_slabel_dialog)
        thresh_action = image_menu.addAction("Threshold/Segment")
        thresh_action.triggered.connect(self.show_thresh_dialog)
        mask_action = image_menu.addAction("Mask Channel")
        mask_action.triggered.connect(self.show_mask_dialog)
        crop_action = image_menu.addAction("Crop Channels")
        crop_action.triggered.connect(lambda: self.show_crop_dialog(args = None))
        type_action = image_menu.addAction("Channel dtype")
        type_action.triggered.connect(self.show_type_dialog)
        skeletonize_action = image_menu.addAction("Skeletonize")
        skeletonize_action.triggered.connect(self.show_skeletonize_dialog)
        dt_action = image_menu.addAction("Distance Transform (For binary images)")
        dt_action.triggered.connect(self.show_dt_dialog)
        watershed_action = image_menu.addAction("Binary Watershed")
        watershed_action.triggered.connect(self.show_watershed_dialog)
        gray_water_action = image_menu.addAction("Gray Watershed")
        gray_water_action.triggered.connect(self.show_gray_water_dialog)
        invert_action = image_menu.addAction("Invert")
        invert_action.triggered.connect(self.show_invert_dialog)
        z_proj_action = image_menu.addAction("Z Project")
        z_proj_action.triggered.connect(self.show_z_dialog)

        generate_menu = process_menu.addMenu("Generate")
        centroid_node_action = generate_menu.addAction("Generate Nodes (From Node Centroids)")
        centroid_node_action.triggered.connect(self.show_centroid_node_dialog)
        gennodes_action = generate_menu.addAction("Generate Nodes (From 'Edge' Vertices)")
        gennodes_action.triggered.connect(self.show_gennodes_dialog)
        branch_action = generate_menu.addAction("Label Branches")
        branch_action.triggered.connect(lambda: self.show_branch_dialog())
        filament_action = generate_menu.addAction("Trace Filaments (For Segmented Data)")
        filament_action.triggered.connect(self.show_filament_dialog)
        genvor_action = generate_menu.addAction("Generate Voronoi Diagram - goes in Overlay2")
        genvor_action.triggered.connect(self.voronoi)
        genhull_action = generate_menu.addAction("Generate Convex Hull - goes in Overlay2")
        genhull_action.triggered.connect(self.convex_hull)
        genhull_action = generate_menu.addAction("Generate Artificial Hexagonal Nodes - goes in Overlay2")
        genhull_action.triggered.connect(self.show_hexnode_dialog)

        modify_action = process_menu.addAction("Modify Network/Properties")
        modify_action.triggered.connect(self.show_modify_dialog)

        
        # Image menu
        image_menu = menubar.addMenu("Image")
        properties_action = image_menu.addAction("Properties")
        properties_action.triggered.connect(self.show_properties_dialog)
        brightness_action = image_menu.addAction("Adjust Brightness/Contrast")
        brightness_action.triggered.connect(self.show_brightness_dialog)
        color_action = image_menu.addAction("Channel Colors")
        color_action.triggered.connect(self.show_color_dialog)
        overlay_menu = image_menu.addMenu("Overlays")
        netoverlay_action = overlay_menu.addAction("Create Network Overlay")
        netoverlay_action.triggered.connect(self.show_netoverlay_dialog)
        idoverlay_action = overlay_menu.addAction("Create Node Number Overlay")
        idoverlay_action.triggered.connect(self.show_idoverlay_dialog)
        coloroverlay_action = overlay_menu.addAction("Color Nodes (or Edges)")
        coloroverlay_action.triggered.connect(self.show_coloroverlay_dialog)
        shuffle_action = overlay_menu.addAction("Shuffle")
        shuffle_action.triggered.connect(self.show_shuffle_dialog)
        arbitrary_action = image_menu.addAction("Select Objects")
        arbitrary_action.triggered.connect(self.show_arbitrary_dialog)
        show3d_action = image_menu.addAction("Show 3D (Requires Napari)")
        show3d_action.triggered.connect(self.show3d_dialog)
        cellpose_action = image_menu.addAction("Cellpose (Requires Cellpose GUI installed)")
        cellpose_action.triggered.connect(self.open_cellpose)

        # Help

        help_menu = menubar.addMenu("Help")
        documentation_action = help_menu.addAction("Documentation")
        documentation_action.triggered.connect(self.help_me)
        tutorial_action = help_menu.addAction("Tutorial")
        tutorial_action.triggered.connect(self.start_tutorial)
        documentation_action = help_menu.addAction("Youtube")
        documentation_action.triggered.connect(self.help_me_vid)

        # Initialize downsample factor
        self.downsample_factor = 1
        
        # Create container widget for corner controls
        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(5, 0, 5, 0)

        self.coord_label = QLabel(f"                                                                                               ")
        self.coord_label.setText(f"                                                                                               ")
        corner_layout.addWidget(self.coord_label)

        self.xy_scale_label = QLabel(f"xy_scale: {my_network.xy_scale}                   ")
        self.xy_scale_label.setText(f"xy_scale: {my_network.xy_scale}                   ")
        corner_layout.addWidget(self.xy_scale_label)

        self.z_scale_label = QLabel(f"z_scale: {my_network.z_scale}                   ")
        self.z_scale_label.setText(f"z_scale: {my_network.z_scale}                   ")
        corner_layout.addWidget(self.z_scale_label)

        self.z_label = QLabel(f"Slice {self.current_slice}")
        self.z_label.setText(f"Slice {self.current_slice}")
        corner_layout.addWidget(self.z_label)
        
        self.threed_button = QPushButton("3D") 
        self.threed_button.setFixedSize(40, 40)
        self.threed_button.clicked.connect(self.quick_3d)
        corner_layout.addWidget(self.threed_button)

        # Add after your other buttons
        self.popup_button = QPushButton("⤴") 
        self.popup_button.setFixedSize(40, 40)
        self.popup_button.setToolTip("Pop out canvas")
        self.popup_button.clicked.connect(self.popup_canvas)
        corner_layout.addWidget(self.popup_button)

        # Add some spacing
        corner_layout.addSpacing(10)

        # Add camera button
        self.cam_button = QPushButton("📷")
        self.cam_button.setToolTip("Screenshot")
        self.cam_button.setFixedSize(40, 40)
        self.cam_button.setStyleSheet("font-size: 24px;")
        self.cam_button.clicked.connect(self.snap)
        corner_layout.addWidget(self.cam_button)

        self.load_button = QPushButton("📁")
        self.load_button.setToolTip("Load Spreadsheet")
        self.load_button.setFixedSize(40, 40)
        self.load_button.setStyleSheet("font-size: 24px;")
        self.load_button.clicked.connect(self.load_file)
        corner_layout.addWidget(self.load_button)

        # Set as corner widget
        menubar.setCornerWidget(corner_widget, Qt.Corner.TopRightCorner)

    def on_downsample_changed(self, text):
        """Called whenever the text in the downsample input changes"""
        try:
            if text.strip() == "":
                self.downsample_factor = 1
            else:
                value = float(text)
                if value <= 0:
                    self.downsample_factor = 1
                else:
                    self.downsample_factor = int(value) if value == int(value) else value
        except (ValueError, TypeError):
            self.downsample_factor = 1

    def validate_downsample_input(self, text = None, update = True):
        """Called when user finishes editing (loses focus or presses Enter)"""
        if text:
            if text < 1:
                return
            self.downsample_factor = text
        else:
            try: # If enabled for manual display downsampling
                text = self.downsample_input.text().strip()
                if text == "":
                    # Empty input - set to default
                    self.downsample_factor = 1
                    self.downsample_input.setText("1")
                else:
                    value = int(text)
                    if value < 1:
                        # Invalid value - reset to default
                        self.downsample_factor = 1
                        self.downsample_input.setText("1")
                    else:
                        # Valid value - use it (prefer int if possible)
                        if value == int(value):
                            self.downsample_factor = int(value)
                            self.downsample_input.setText(str(int(value)))
                        else:
                            self.downsample_factor = value
                            self.downsample_input.setText(f"{value:.1f}")
            except:
                # Invalid input - reset to default
                self.downsample_factor = 1
        
        # Optional: Trigger display update if you want immediate effect
        if update:
            current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
            current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
            self.update_display(preserve_zoom=(current_xlim, current_ylim))

    def quick_3d(self):

        arrays_3d = []
        arrays_4d = []

        color_template = ['red', 'green', 'white', 'cyan', 'yellow']  # color list
        colors = []


        for i, channel in enumerate(self.channel_data):
            if channel is not None:

                if len(channel.shape) == 3:
                    visible = self.channel_buttons[i].isChecked()
                    if visible:
                        arrays_3d.append(channel)
                        colors.append(color_template[i])
                elif len(channel.shape) == 4:
                    visible = self.channel_buttons[i].isChecked()
                    if visible:
                        arrays_4d.append(channel)

        if self.thresh_window_ref is not None:
            self.thresh_window_ref.make_full_highlight()

        if self.highlight_overlay is not None or self.mini_overlay_data is not None:
            if self.mini_overlay == True:
                self.create_highlight_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
            arrays_3d.append(self.highlight_overlay)
            colors.append(color_template[4])
        
        n3d.show_3d(arrays_3d, arrays_4d, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale, colors = colors)


    def snap(self):
        try:
            # Check if we have any data to save
            data = False
            for thing in self.channel_data:
                if thing is not None:
                    data = True
                    break
            if not data:
                return
            
            # Get filename from user
            filename, _ = QFileDialog.getSaveFileName(
                self,
                f"Save Image As",
                "",
                "PNG Files (*.png);;TIFF Files (*.tif *.tiff);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            
            if filename:
                # Determine file extension
                if filename.lower().endswith(('.tif', '.tiff')):
                    format_type = 'tiff'
                elif filename.lower().endswith(('.jpg', '.jpeg')):
                    format_type = 'jpeg'
                elif filename.lower().endswith('.png'):
                    format_type = 'png'
                else:
                    filename += '.png'
                    format_type = 'png'
                
                # Temporarily render at full resolution if downsampled
                if self.downsample_factor > 1:
                    self.update_display(downsample = False)
                
                # Export the view as an image
                from pyqtgraph.exporters import ImageExporter
                
                exporter = ImageExporter(self.view)
                
                # Set high resolution
                exporter.parameters()['width'] = int(self.view.width() * 3)  # 3x resolution for quality
                
                # Export to file
                exporter.export(filename)
                
                print(f"View snapshot saved: {filename}")
        
        except Exception as e:
            print(f"Error saving snapshot: {e}")

    def _remove_scalebar(self):
        """Remove existing scalebar artists if present."""
        if hasattr(self, 'scalebar_artists') and self.scalebar_artists:
            for artist in self.scalebar_artists:
                try:
                    self.view.removeItem(artist)
                except:
                    pass
            self.scalebar_artists = None

    def _draw_scalebar(self):
        """Draw a scale bar and store artists in self.scalebar_artists."""
        # Remove any existing scalebar first
        self._remove_scalebar()
        
        # Initialize the list
        self.scalebar_artists = []
        
        # Get current view range (in pixel coordinates)
        view_range = self.view.viewRange()
        xlim = view_range[0]
        ylim = view_range[1]
        
        # Calculate view dimensions
        width_pixels = abs(xlim[1] - xlim[0])
        height_pixels = abs(ylim[1] - ylim[0])
        
        # Convert to actual units using xy_scale
        width_units = width_pixels * my_network.xy_scale
        
        # Determine a nice scale bar size (target ~15% of width)
        target_size = width_units * 0.15
        
        # Round to a nice number (1, 2, 5, 10, 20, 50, 100, etc.)
        magnitude = 10 ** np.floor(np.log10(target_size))
        normalized = target_size / magnitude
        if normalized < 1.5:
            nice_size = 1 * magnitude
        elif normalized < 3.5:
            nice_size = 2 * magnitude
        elif normalized < 7.5:
            nice_size = 5 * magnitude
        else:
            nice_size = 10 * magnitude
        
        # Convert back to pixels for drawing
        bar_length_pixels = nice_size / my_network.xy_scale
        
        # Position in bottom right corner with padding (5% margins)
        padding_x = width_pixels * 0.05
        padding_y = height_pixels * 0.05
        
        # Calculate position
        x_max = max(xlim)
        y_max = max(ylim)  # Bottom in inverted Y coordinates
        
        bar_x_start = x_max - padding_x - bar_length_pixels
        bar_x_end = x_max - padding_x
        bar_y = y_max - padding_y
        
        # Draw scale bar with outline for visibility
        # Black outline (thicker)
        outline = pg.PlotDataItem(
            [bar_x_start, bar_x_end], [bar_y, bar_y],
            pen=pg.mkPen(color='k', width=6)
        )
        self.view.addItem(outline)
        self.scalebar_artists.append(outline)
        
        # White main line
        line = pg.PlotDataItem(
            [bar_x_start, bar_x_end], [bar_y, bar_y],
            pen=pg.mkPen(color='w', width=4)
        )
        self.view.addItem(line)
        self.scalebar_artists.append(line)
        
        # Format the label text
        if nice_size >= 1:
            label_text = f'{nice_size:.0f}'
        else:
            label_text = f'{nice_size:.2f}'
        
        # Add text label above the bar
        text_offset = height_pixels * 0.03
        text_y = bar_y + text_offset  # Add to go visually up (Y is inverted)
        text_x = (bar_x_start + bar_x_end) / 2
        
        # Create text with background
        text = pg.TextItem(
            text=label_text,
            color=(255, 255, 255),
            anchor=(0.5, 1)  # Center, bottom
        )
        text.setPos(text_x, text_y)
        
        # Add semi-transparent black background to text
        from PyQt6.QtGui import QColor, QBrush
        text.fill = QBrush(QColor(0, 0, 0, 180))  # Black brush with alpha
        text.border = pg.mkPen(None)
        
        self.view.addItem(text)
        self.scalebar_artists.append(text)

    def open_cellpose(self):

        try:
            if self.shape[0] == 1:
                use_3d = False
                print("Launching 2D cellpose GUI")
            else:
                use_3d = True
                print("Launching 3D cellpose GUI")
        except:
            use_3d = True
            print("Launching 3D cellpose GUI")

        try:

            from . import cellpose_manager
            self.cellpose_launcher = cellpose_manager.CellposeGUILauncher(parent_widget=self)

            self.cellpose_launcher.launch_cellpose_gui(use_3d = use_3d)

        except:
            QMessageBox.critical(
                self,
                "Error",
                f"Error starting cellpose: {str(e)}\nNote: You may need to install cellpose with corresponding torch first - in your environment, please call 'pip install cellpose'. Please see: 'https://pytorch.org/get-started/locally/' to see what torch install command corresponds to your NVIDIA GPU"
            )
            pass


    def help_me(self):

        import webbrowser
        try:
            webbrowser.open('https://nettracer3d.readthedocs.io/en/latest/')
            return True
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False

    def help_me_vid(self):

        import webbrowser
        try:
            webbrowser.open('https://www.youtube.com/watch?v=_4uDy0mzG94&list=PLsrhxiimzKJMZ3_gTWkfrcAdJQQobUhj7')
            return True
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False

    def start_tutorial(self):
        """Open the tutorial selection dialog"""
        if not hasattr(self, 'tutorial_dialog'):
            self.tutorial_dialog = TutorialSelectionDialog(self)
        self.tutorial_dialog.show()
        self.tutorial_dialog.raise_()
        self.tutorial_dialog.activateWindow()


    def stats(self):
        """Method to get and display the network stats"""
        # Get the stats dictionary
        try:
            stats = my_network.get_network_stats()

            self.format_for_upperright_table(stats, title = 'Network Stats')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error finding stats: {e}")

    def histograms(self):
        """
        Show a PyQt6 window with buttons to select which histogram to generate.
        Only calculates the histogram that the user selects.
        """
        try:
            # Create QApplication if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            # Create and show the histogram selector window
            from . import histos
            self.histogram_selector = histos.HistogramSelector(self, self.stats_dict, my_network.network)
            self.histogram_selector.show()
            
            # Keep the window open (you might want to handle this differently based on your application structure)
            if not app.exec():
                pass  # Window was closed
                
        except Exception as e:
            print(f"Error creating histogram selector: {e}")

    def sig_test(self):
        # Get the existing QApplication instance
        app = QApplication.instance()
        
        # Create the statistical GUI window without starting a new event loop
        stats_window = net_stats.main(app)
        
        # Keep a reference so it doesn't get garbage collected
        self.stats_window = stats_window

    def volumes(self):


        if self.active_channel == 1:
            output = my_network.volumes('edges')
            self.format_for_upperright_table(output, metric='Edge ID', value = 'Voxel Volume (Scaled)', title = 'Edge Volumes')
            self.volume_dict[1] = output

        elif self.active_channel == 0:
            output = my_network.volumes('nodes')
            self.format_for_upperright_table(output, metric='Node ID', value = 'Voxel Volume (Scaled)', title = 'Node Volumes')
            self.volume_dict[0] = output

        elif self.active_channel == 2:
            output = my_network.volumes('network_overlay')
            self.format_for_upperright_table(output, metric='Object ID', value = 'Voxel Volume (Scaled)', title = 'Overlay 1 Volumes')
            self.volume_dict[2] = output

        elif self.active_channel == 3:
            output = my_network.volumes('id_overlay')
            self.format_for_upperright_table(output, metric='Object ID', value = 'Voxel Volume (Scaled)', title = 'Overlay 2 Volumes')
            self.volume_dict[3] = output

        

    def format_for_upperright_table(self, data, metric='Metric', value='Value', title=None, sort = True, save = False):
       """
       Format dictionary or list data for display in upper right table.
       
       Args:
           data: Dictionary with keys and single/multiple values, or a list of values
           metric: String for the key/index column header
           value: String or list of strings for value column headers (used for dictionaries only)
           title: Optional custom title for the tab
       """

       if 'dentit' in value:
            try:
                if type(list(my_network.node_identities.values())[0]) != str:
                    self.format_identities_for_table(data, node_col='Node', identity_col='Identities', title=title, sort=sort, save=save)
                    return
            except:
                print(type(list(my_network.node_identities.values())[0]))
                pass

       def convert_to_numeric(val):
           """Helper function to convert strings to numeric types when possible"""
           if isinstance(val, str):
               try:
                   # First try converting to int
                   if '.' not in val:
                       return int(val)
                   # If that fails or if there's a decimal point, try float
                   return float(val)
               except ValueError:
                   return val
           return val
       
       def format_number(x):
           """Smart formatting that removes trailing zeros"""
           if not isinstance(x, (float, np.float64)):
               return str(x)
           
           # Use more decimal places, then strip trailing zeros
           formatted = f"{x:.8f}".rstrip('0').rstrip('.')
           return formatted if formatted else "0"
       
       try:

           if isinstance(data, (list, tuple, np.ndarray)):
               # Handle list input - create single column DataFrame
               df = pd.DataFrame({
                   metric: [convert_to_numeric(val) for val in data]
               })
               
               # Format floating point numbers
               df[metric] = df[metric].apply(format_number)
               
           else:  # Dictionary input
               # Get sample value to determine structure
               sample_value = next(iter(data.values()))
               is_multi_value = isinstance(sample_value, (list, tuple, np.ndarray))
               
               if is_multi_value:
                   # Handle multi-value case
                   if isinstance(value, str):
                       # If single string provided for multi-values, generate numbered headers
                       n_cols = len(sample_value)
                       value_headers = [f"{value}_{i+1}" for i in range(n_cols)]
                   else:
                       # Use provided list of headers
                       value_headers = value
                       if len(value_headers) != len(sample_value):
                           raise ValueError("Number of headers must match number of values per key")
                   
                   # Create lists for each column
                   dict_data = {metric: list(data.keys())}
                   for i, header in enumerate(value_headers):
                       # Convert values to numeric when possible before adding to DataFrame
                       dict_data[header] = [convert_to_numeric(data[key][i]) for key in data.keys()]
                   
                   df = pd.DataFrame(dict_data)
                   
                   # Format floating point numbers in all value columns
                   for header in value_headers:
                       df[header] = df[header].apply(format_number)
                       
               else:
                   # Single-value case
                   df = pd.DataFrame({
                       metric: data.keys(),
                       value: [convert_to_numeric(val) for val in data.values()]
                   })
                   
                   # Format floating point numbers
                   df[value] = df[value].apply(format_number)
           
           # Create new table
           table = CustomTableView(self)
           table.setModel(PandasModel(df))

           if sort:
               try:
                   first_column_name = table.model()._data.columns[0]
                   table.sort_table(first_column_name, ascending=True)
               except:
                    pass
           
           # Add to tabbed widget
           if title is None:
               self.tabbed_data.add_table(f"{metric} Analysis", table)
               #print(list(self.tabbed_data.tables.values())[-1].model()._data) 
               #for reference, the above is how you access the data in the tabbed data viz
           else:
               self.tabbed_data.add_table(f"{title}", table)
           # Adjust column widths to content
           for column in range(table.model().columnCount(None)):
               table.resizeColumnToContents(column)

           if save:
                table.save_table_as('csv')
           return df

       except:
            pass


    def format_identities_for_table(self, data, node_col='Node', identity_col='Identities', title=None, sort=True, save=False):
        """
        Format node identity dictionary for display where values are lists shown in a single column.
        
        Args:
            data: Dictionary with keys (nodes) and list values (identities)
            node_col: String for the node column header
            identity_col: String for the identity column header
            title: Optional custom title for the tab
            sort: Whether to sort by first column
            save: Whether to save as CSV
        """
        # Create DataFrame directly - keep lists as lists
        df = pd.DataFrame({
            node_col: list(data.keys()),
            identity_col: list(data.values())
        })
        
        # Create new table
        table = CustomTableView(self)
        table.setModel(PandasModel(df))
        
        if sort:
            try:
                first_column_name = table.model()._data.columns[0]
                table.sort_table(first_column_name, ascending=True)
            except:
                pass
        
        # Add to tabbed widget
        if title is None:
            self.tabbed_data.add_table(f"{node_col} {identity_col}", table)
        else:
            self.tabbed_data.add_table(title, table)
        
        # Adjust column widths to content
        for column in range(table.model().columnCount(None)):
            table.resizeColumnToContents(column)
        
        if save:
            table.save_table_as('csv')
        
        return df
    
    def show_identity_selection_dialog(self):
        """Show dialog for selecting identities with gates"""
        # Collect all unique identities
        all_idens = my_network.node_identities.values()
        seen = set()
        for iden in all_idens:
            seen.update(iden)
        
        # Create and show dialog
        dialog = IdentitySelectionDialog(seen, parent=self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            classifier_dict = dialog.get_classifier_data()
            
            # Only proceed if at least one gate was added
            if classifier_dict['positive'] or classifier_dict['negative']:
                self.handle_show_identities(sort=classifier_dict)
            else:
                # Optional: show message if no gates selected
                QMessageBox.information(
                    self, 
                    "No Selection", 
                    "Please add at least one identity gate to filter nodes."
                )

    def show_merge_node_id_dialog(self):

        if my_network.nodes is None:
            QMessageBox.critical(
                self,
                "Error",
                "Please load your segmented cells into 'Nodes' channel first"
            )
            return
        else:
            dialog = MergeNodeIdDialog(self)
            dialog.exec()

    def show_multichan_dialog(self, data):
        dialog = MultiChanDialog(self, data)
        dialog.show()

    def show_gray_water_dialog(self):
        """Show the gray watershed parameter dialog."""
        dialog = GrayWaterDialog(self)
        dialog.exec()

    def show_watershed_dialog(self):
        """Show the watershed parameter dialog."""
        dialog = WatershedDialog(self)
        dialog.exec()

    def show_arbitrary_dialog(self):
        """Show the arbitrary selection dialog."""
        dialog = ArbitraryDialog(self)
        dialog.exec()

    def show_invert_dialog(self):
        """Show the watershed parameter dialog."""
        dialog = InvertDialog(self)
        dialog.exec()

    def show_z_dialog(self):
        """Show the z-proj dialog."""
        dialog = ZDialog(self)
        dialog.exec()

    def show_calc_all_dialog(self):
        """Show the calculate all parameter dialog."""
        dialog = CalcAllDialog(self)
        dialog.show()

    def show_calc_prox_dialog(self, tutorial_example = False):
        """Show the proximity calc dialog"""
        dialog = ProxDialog(self, tutorial_example = True)
        if tutorial_example:
            dialog.show()
        else:
            dialog.exec()

    def table_load_attrs(self):

        # Display network_lists in the network table
        try:
            if hasattr(my_network, 'network_lists'):
                model = PandasModel(my_network.network_lists)
                self.network_table.setModel(model)
                # Adjust column widths to content
                for column in range(model.columnCount(None)):
                    self.network_table.resizeColumnToContents(column)
        except Exception as e:
            print(f"Error loading network_lists: {e}")

        #Display the other things if they exist
        try:

            if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                try:
                    self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                except Exception as e:
                    print(f"Error loading node identity table: {e}")

            if hasattr(my_network, 'node_centroids') and my_network.node_centroids is not None:
                try:
                    self.format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                except Exception as e:
                    print(f"Error loading node centroid table: {e}")


            if hasattr(my_network, 'edge_centroids') and my_network.edge_centroids is not None:
                try:
                    self.format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
                except Exception as e:
                    print(f"Error loading edge centroid table: {e}")


        except Exception as e:
            print(f"An error has occured: {e}")

    def confirm_calcbranch_dialog(self, message):
        """Shows a dialog asking user to confirm if they want to proceed below"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Alert")
        msg.setInformativeText(message)
        msg.setWindowTitle("Proceed?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def handle_calc_branch(self):

        try:

            if self.channel_data[0] is not None or self.channel_data[3] is not None:
                if not self.confirm_calcbranch_dialog("Use of this feature will require additional use of the Nodes and Overlay 2 channels. Please save any data and return, or proceed if you do not need those channels' data"):
                    return

            if my_network.edges is None and my_network.nodes is not None:
                self.load_channel(1, my_network.nodes, data = True)
                self.delete_channel(0, False)

            self.show_gennodes_dialog()

            my_network.edges = (my_network.nodes == 0) * my_network.edges

            my_network.calculate_all(my_network.nodes, my_network.edges, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale, search = None, diledge = None, inners = False, remove_trunk = 0, ignore_search_region = True, other_nodes = None, label_nodes = True, directory = None, GPU = False, fast_dil = False, skeletonize = False, GPU_downsample = None)

            self.load_channel(1, my_network.edges, data = True)
            self.load_channel(0, my_network.nodes, data = True)

            self.clear_subgraphs()
            self.network_graph_widget.set_graph(my_network.network)

            self.table_load_attrs()

        except Exception as e:

            try:
                my_network.edges = my_network.id_overlay
                my_network.id_overlay = None
            except:
                pass

            print(f"Error calculating branchpoint network: {e}")

    def handle_branchprox_calc(self):

        try:

            if self.channel_data[0] is not None:
                if not self.confirm_calcbranch_dialog("Use of this feature will require additional use of the Nodes and Overlay 2 channels. Please save any data and return, or proceed if you do not need those channels' data"):
                    return

            if my_network.edges is None and my_network.nodes is not None:
                self.load_channel(1, my_network.nodes, data = True)
                self.delete_channel(0, False)

            self.show_branch_dialog(called = True)

            self.load_channel(0, my_network.edges, data = True)

            try:
                self.branch_dict[0] = self.branch_dict[1]
                self.branch_dict[1] = None
            except:
                pass

            self.delete_channel(1, False)

            my_network.morph_proximity(search = [3,3], fastdil = True)

            self.clear_subgraphs()
            self.network_graph_widget.set_graph(my_network.network)

            self.table_load_attrs()

        except Exception as e:

            print(f"Error calculating network: {e}")


    def show_centroid_dialog(self):
        """show the centroid dialog"""
        dialog = CentroidDialog(self)
        dialog.exec()

    def handle_identity_net_calc(self):

        """Not implemented - Wasn't useful"""

        try:

            def confirm_dialog():
                """Shows a dialog asking user to confirm and input connection limit"""
                from PyQt6.QtWidgets import QInputDialog
                
                value, ok = QInputDialog.getInt(
                    None,  # parent widget
                    "Confirm",  # window title
                    "Calculate Identity Network\n\n"
                    "Connect nodes that share an identity - useful for nodes that\n"
                    "overlap in identity to some degree.\n\n"
                    "Enter maximum connections per node within same identity:",
                    5,  # default value
                    1,  # minimum value
                    1000,  # maximum value
                    1   # step
                )
                
                if ok:
                    return True, value
                else:
                    return False, None

            confirm, val = confirm_dialog()

            if confirm:
                my_network.create_id_network(val)
                self.table_load_attrs()
            else:
                return

        except:
            pass

    def show_dilate_dialog(self, args = None, execute = False):
        """show the dilate dialog"""
        dialog = DilateDialog(self, args)
        if not execute:
            dialog.show()
        else:
            dialog.exec()

    def show_erode_dialog(self, args = None):
        """show the erode dialog"""
        dialog = ErodeDialog(self, args)
        dialog.exec()

    def show_hole_dialog(self):
        """show the hole dialog"""
        dialog = HoleDialog(self)
        dialog.exec()

    def show_filament_dialog(self):
        """show the filament dialog"""
        dialog = FilamentDialog(self)
        dialog.show()

    def show_hexnode_dialog(self):
        """show the hexnode dialog"""
        dialog = HexNodeDialog(self)
        dialog.show()


    def show_label_dialog(self):
        """Show the label dialog"""
        dialog = LabelDialog(self)
        dialog.exec()

    def show_slabel_dialog(self):
        """Show the slabel dialog"""
        dialog = SLabelDialog(self)
        dialog.exec()

    def show_thresh_dialog(self, tutorial_example = False):
        """Show threshold dialog"""
        if self.machine_window is not None:
            return

        dialog = ThresholdDialog(self)
        if not tutorial_example:
            dialog.exec()
        else:
            dialog.show()

    def show_machine_window_tutorial(self):
        dialog = MachineWindow(self, tutorial_example = True)
        dialog.show()


    def show_mask_dialog(self):
        """Show the mask dialog"""
        dialog = MaskDialog(self)
        dialog.exec()

    def show_crop_dialog(self, args = None):
        """Show the crop dialog"""
        dialog = CropDialog(self, args = args)
        dialog.exec()

    def show_type_dialog(self):
        """Show the type dialog"""
        try:
            dialog = TypeDialog(self)
            dialog.exec()
        except:
            pass

    def show_skeletonize_dialog(self):
        """show the skeletonize dialog"""
        dialog = SkeletonizeDialog(self)
        dialog.exec()

    def show_dt_dialog(self):
        """show the dt dialog"""
        dialog = DistanceDialog(self)
        dialog.exec()

    def show_centroid_node_dialog(self):
        """show the centroid node dialog"""
        dialog = CentroidNodeDialog(self)
        dialog.exec()


    def show_gennodes_dialog(self, down_factor = None, called = False, tutorial_example = False):
        """show the gennodes dialog"""
        gennodes = GenNodesDialog(self, down_factor = down_factor, called = called)
        if not tutorial_example:
            gennodes.exec()
        else:
            gennodes.show()

    def show_branch_dialog(self, called = False, tutorial_example = False):
        """Show the branch label dialog"""
        dialog = BranchDialog(self, called = called, tutorial_example = tutorial_example)
        if tutorial_example:
            dialog.show()
        else:
            dialog.exec()

    def voronoi(self):

        try:

            array = sdl.smart_dilate(self.channel_data[self.active_channel], use_dt_dil_amount = np.max(self.shape), fast_dil = False)
            self.load_channel(3, array, True)

        except Exception as e:
            print(f"Error generating voronoi: {e}")

    def convex_hull(self):

        try:
            self.load_channel(3, n3d.create_convex_hull_mask(self.channel_data[self.active_channel]), True)
        except Exception as e:
            print(f"Error: {e}")



    def show_modify_dialog(self):
        """Show the network modify dialog"""
        dialog = ModifyDialog(self)
        dialog.show()


    def show_binarize_dialog(self):
        """show the binarize dialog"""
        dialog = BinarizeDialog(self)
        dialog.exec()


    def show_resize_dialog(self):
        """show the resize dialog"""
        dialog = ResizeDialog(self)
        dialog.exec()

    def show_clean_dialog(self):
        dialog = CleanDialog(self)
        dialog.show()

    def show_properties_dialog(self):
        """Show the properties dialog"""
        dialog = PropertiesDialog(self)
        dialog.show()
    
    def show_brightness_dialog(self):
        """Show the brightness/contrast control dialog."""
        self.brightness_dialog.show()

    def show_color_dialog(self):
        """Show the color control dialog."""
        dialog = ColorDialog(self)
        dialog.exec()



    def show_netoverlay_dialog(self):
        """show the net overlay dialog"""
        dialog = NetOverlayDialog(self)
        dialog.exec()

    def show_idoverlay_dialog(self):
        """show the id overlay dialog"""
        dialog = IdOverlayDialog(self)
        dialog.exec()

    def show_coloroverlay_dialog(self):
        """show the color overlay dialog"""
        dialog = ColorOverlayDialog(self)
        dialog.exec()


    def show_shuffle_dialog(self):
        """Show the shuffle dialog"""
        dialog = ShuffleDialog(self)
        dialog.exec()

    def show3d_dialog(self):
        """Show the 3D control dialog"""
        dialog = Show3dDialog(self)
        dialog.exec()

    
    def load_misc(self, sort):
        """Loads various things"""

        def uncork(my_dict, trumper = None):

            if trumper is None:
                for thing in my_dict:
                    val = my_dict[thing]
                    new_val = val[0]
                    for i in range(1, len(val)):
                        try:
                            new_val += f" AND {val[i]}"
                        except:
                            break
                    my_dict[thing] = new_val
            elif trumper == '-':
                for key, value in my_dict.items():
                    my_dict[key] = value[0]
            elif trumper == '/':
                new_dict = {}
                max_val = max(my_dict.keys()) + 1
                for key, value in my_dict.items():
                    new_dict[key] = f'{value[0]}'
                    if len(value) > 1:
                        for i in range(1, len(value)):
                            new_dict[max_val] = f'{value[i]}'
                            try:
                                my_network.node_centroids[max_val] = my_network.node_centroids[key]
                            except:
                                pass
                            max_val += 1
                return new_dict
            else:
                for thing in my_dict:
                    val = my_dict[thing]
                    if trumper in val:
                        my_dict[thing] = trumper
                    else:
                        new_val = val[0]
                        for i in range(1, len(val)):
                            try:
                                new_val += f" AND {val[i]}"
                            except:
                                break
                        my_dict[thing] = new_val

            return my_dict

        if sort != 'Merge Nodes':

            try:

                filename, _ = QFileDialog.getOpenFileName(
                    self,
                    f"Load {sort}",
                    "",
                    "Spreadsheets (*.xlsx *.csv *.json)"
                )

                try:
                    if sort == 'Node Identities':
                        my_network.load_node_identities(file_path = filename)
                        self.network_graph_widget.identity_dict = my_network.node_identities
                        self.selection_graph_widget.identity_dict = my_network.node_identities

                        if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                            try:
                                self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                            except Exception as e:
                                print(f"Error loading node identity table: {e}")

                    elif sort == 'Node Centroids':
                        my_network.load_node_centroids(file_path = filename)
                        self.network_graph_widget.centroids = my_network.node_centroids
                        self.selection_graph_widget.centroids = my_network.node_centroids

                        if hasattr(my_network, 'node_centroids') and my_network.node_centroids is not None:
                            try:
                                self.format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                            except Exception as e:
                                print(f"Error loading node centroid table: {e}")

                    elif sort == 'Edge Centroids':
                        my_network.load_edge_centroids(file_path = filename)

                        if hasattr(my_network, 'edge_centroids') and my_network.edge_centroids is not None:
                            try:
                                self.format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
                            except Exception as e:
                                print(f"Error loading edge centroid table: {e}")
                    elif sort == 'Communities':
                        my_network.load_communities(file_path = filename)

                        self.network_graph_widget.community_dict = my_network.communities
                        self.selection_graph_widget.community_dict = my_network.communities

                        if hasattr(my_network, 'communities') and my_network.communities is not None:
                            self.prev_coms = None
                            try:
                                self.format_for_upperright_table(my_network.communities, 'NodeID', 'Community', 'Node Communities')
                            except Exception as e:
                                print(f"Error loading edge centroid table: {e}")


                except Exception as e:
                    print(f"An error has occured: {e}")

            except Exception as e:

                QMessageBox.critical(
                    self,
                    "Error Loading",
                    f"Failed to load {sort}: {str(e)}"
                )

        elif sort == 'Merge Nodes':
            try:
                if my_network.nodes is None:
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Please load your first set of nodes into the 'Nodes' channel first"
                    )
                    return
                if len(np.unique(my_network.nodes)) < 3:
                    self.show_label_dialog()
                
                # Create custom dialog
                dialog = QDialog(self)
                dialog.setWindowTitle("Merge Nodes Configuration")
                dialog.setModal(True)
                dialog.resize(400, 200)
                
                layout = QVBoxLayout(dialog)
                
                # Selection type
                type_layout = QHBoxLayout()
                type_label = QLabel("Selection Type:")
                type_combo = QComboBox()
                type_combo.addItems(["TIFF File", "Directory"])
                type_layout.addWidget(type_label)
                type_layout.addWidget(type_combo)
                layout.addLayout(type_layout)
                
                # Centroids checkbox
                centroids_layout = QHBoxLayout()
                centroids_check = QCheckBox("Compute node centroids for each image prior to merging")
                centroids_layout.addWidget(centroids_check)
                layout.addLayout(centroids_layout)
                
                # Down factor for centroid calculation
                down_factor_layout = QHBoxLayout()
                down_factor_label = QLabel("Down Factor (for centroid calculation downsampling):")
                down_factor_edit = QLineEdit()
                down_factor_edit.setText("1")  # Default value
                down_factor_edit.setPlaceholderText("Enter down factor (e.g., 1, 2, 4)")
                down_factor_layout.addWidget(down_factor_label)
                down_factor_layout.addWidget(down_factor_edit)
                layout.addLayout(down_factor_layout)
                
                # Buttons
                button_layout = QHBoxLayout()
                accept_button = QPushButton("Accept")
                cancel_button = QPushButton("Cancel")
                button_layout.addWidget(accept_button)
                button_layout.addWidget(cancel_button)
                layout.addLayout(button_layout)
                
                # Connect buttons
                accept_button.clicked.connect(dialog.accept)
                cancel_button.clicked.connect(dialog.reject)
                
                # Execute dialog
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    # Get values from dialog
                    selection_type = type_combo.currentText()
                    centroids = centroids_check.isChecked()
                    
                    # Validate and get down_factor
                    try:
                        down_factor = int(down_factor_edit.text())
                        if down_factor <= 0:
                            raise ValueError("Down factor must be positive")
                    except ValueError as e:
                        QMessageBox.critical(
                            self,
                            "Invalid Input",
                            f"Invalid down factor: {str(e)}"
                        )
                        return
                    
                    # Handle file/directory selection based on combo box choice
                    if selection_type == "TIFF File":
                        filename, _ = QFileDialog.getOpenFileName(
                            self,
                            "Select TIFF file",
                            "",
                            "TIFF files (*.tiff *.tif)"
                        )
                        if filename:
                            selected_path = filename
                        else:
                            return  # User cancelled file selection
                    else:  # Directory
                        file_dialog = QFileDialog(self)
                        file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
                        file_dialog.setOption(QFileDialog.Option.ReadOnly)
                        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
                        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
                        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                            selected_path = file_dialog.directory().absolutePath()
                        else:
                            return  # User cancelled directory selection
                    
                    if down_factor == 1:
                        down_factor = None
                    # Call merge_nodes with all parameters
                    my_network.merge_nodes(
                        selected_path, 
                        root_id=self.node_name, 
                        centroids=centroids,
                        down_factor=down_factor,
                        label_nodes = False
                    )
                    
                    self.load_channel(0, my_network.nodes, True)
                    
                    if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                        try:
                            self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                        except Exception as e:
                            print(f"Error loading node identity table: {e}")
                    
                    if centroids:
                        self.format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Merging",
                    f"Failed to load {sort}: {str(e)}"
                )


    def clear_subgraphs(self):

        self.network_graph_widget._clear_graph()
        self.selection_graph_widget._clear_graph()
        self.network_graph_widget.set_graph(None)
        self.network_graph_widget.community_dict = None
        self.network_graph_widget.identity_dict = None
        self.network_graph_widget.centroids = None
        self.selection_graph_widget.set_graph(None)
        self.selection_graph_widget.community_dict = None
        self.selection_graph_widget.identity_dict = None
        self.selection_graph_widget.centroids = None

    def update_graph_fields(self):

        #self.network_graph_widget.set_graph(my_network.network)
        self.network_graph_widget.community_dict = my_network.communities
        self.network_graph_widget.identity_dict = ngw.remove_dupe_ids(my_network.node_identities)
        self.network_graph_widget.centroids = my_network.node_centroids
        self.selection_graph_widget.community_dict = my_network.communities
        self.selection_graph_widget.identity_dict = ngw.remove_dupe_ids(my_network.node_identities)
        self.selection_graph_widget.centroids = my_network.node_centroids

    # Modify load_from_network_obj method
    def load_from_network_obj(self, directory = None):
        try: 

            if directory is None:

                directory = QFileDialog.getExistingDirectory(
                    self,
                    f"Select Directory for Network3D Object",
                    "",
                    QFileDialog.Option.ShowDirsOnly
                    )

            self.last_load = directory
            self.last_saved = os.path.dirname(directory)
            self.last_save_name = directory
            self.setWindowTitle(f"NetTracer3D - Session: {self.last_save_name}")            

            self.channel_data = [None] * 5
            if directory != "":

                self.reset(network = True, xy_scale = 1, z_scale = 1, nodes = True, edges = True, network_overlay = True, id_overlay = True, update = False)

                self.clear_subgraphs()
                my_network.assemble(directory)
                self.xy_scale_label.setText(f"xy_scale: {my_network.xy_scale:.2e}                   ")
                self.z_scale_label.setText(f"z_scale: {my_network.z_scale:.2e}                   ")
                self.network_graph_widget.set_graph(my_network.network)
                self.network_graph_widget.centroids = my_network.node_centroids
                self.selection_graph_widget.centroids = my_network.node_centroids
                if my_network.communities is not None:
                    self.prev_coms = None
                #self.network_graph_widget.load_graph()

                # Load image channelsTrue
                try:
                    self.load_channel(0, my_network.nodes, True)
                except Exception as e:
                    print(e)
                try:
                    self.load_channel(1, my_network.edges, True)
                except Exception as e:
                    print(e)
                try:
                    self.load_channel(2, my_network.network_overlay, True)
                except Exception as e:
                    print(e)
                try:
                    self.load_channel(3, my_network.id_overlay, True)
                except Exception as e:
                    print(e)
                self.update_display(home = True)

                # Update slider range based on new data
                for channel in self.channel_data:
                    if channel is not None:
                        self.slice_slider.setEnabled(True)
                        self.slice_slider.setMinimum(0)
                        self.slice_slider.setMaximum(self.shape[0] - 1)
                        self.slice_slider.setValue(0)
                        self.current_slice = 0
                        break

                # Display network_lists in the network table
                # Create empty DataFrame for network table if network_lists is None
                if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                    empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
                    model = PandasModel(empty_df)
                    self.network_table.setModel(model)
                else:
                    model = PandasModel(my_network.network_lists)
                    self.network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.network_table.resizeColumnToContents(column)

                if hasattr(my_network, 'node_centroids') and my_network.node_centroids is not None:
                    try:
                        self.format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                    except Exception as e:
                        print(f"Error loading node centroid table: {e}")

                if hasattr(my_network, 'edge_centroids') and my_network.edge_centroids is not None:
                    try:
                        self.format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
                    except Exception as e:
                        print(f"Error loading edge centroid table: {e}")

                if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                    try:
                        self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                    except Exception as e:
                        print(f"Error loading node identity table: {e}")


                if hasattr(my_network, 'communities') and my_network.communities is not None:
                    try:
                        self.format_for_upperright_table(my_network.communities, 'NodeID', 'Community', 'Node Communities')
                    except Exception as e:
                        print(f"Error loading node community table: {e}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Error Loading Previous Session",
                f"Failed to load Network 3D Object: {str(e)}"
            )



    def load_network(self):
        """Load in the network from a .xlsx (need to add .csv support)"""

        try:

            filename, _ = QFileDialog.getOpenFileName(
                self,
                f"Load Network",
                "",
                "Spreadsheets (*.xlsx *.csv *.json)"
            )

            my_network.load_network(file_path = filename)
            self.clear_subgraphs()
            self.network_graph_widget.set_graph(my_network.network)
            self.network_graph_widget.load_graph()

            # Display network_lists in the network table
            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.network_table.resizeColumnToContents(column)
            except Exception as e:
                print(f"Error loading network table: {e}")

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error Loading File",
                f"Failed to load network: {str(e)}"
            )

    def launch_excelotron(self):
        """Method to launch Excelotron - call this from a button or menu"""
        self.excel_manager.launch()
    
    def close_excelotron(self):
        """Method to close Excelotron"""
        self.excel_manager.close()
    
    def handle_excel_data(self, data_dict, property_name, add):
        """Handle data received from Excelotron"""
        print(f"Received data for property: {property_name}")
        print(f"Data keys: {list(data_dict.keys())}")

        if property_name == 'Node Centroids':

            try:

                if not add or my_network.node_centroids is None:
                    centroids = {}
                    max_val = 0
                else:
                    centroids = my_network.node_centroids
                    max_val = max(list(my_network.node_centroids.keys()))

                ys = data_dict['Y']
                xs = data_dict['X']
                if 'Numerical IDs' in data_dict:
                    nodes = data_dict['Numerical IDs']
                else:
                    nodes = np.arange(max_val + 1, max_val + len(ys) + 1)


                if 'Z' in data_dict:
                    zs = data_dict['Z']
                else:
                    zs = np.zeros(len(ys))

                for i in range(len(nodes)):

                    centroids[nodes[i]] = [int(zs[i]), int(ys[i]), int(xs[i])]

                my_network.node_centroids = centroids

                self.format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')

                print("Centroids succesfully set")

            except Exception as e:
                print(f"Error: {e}")

        elif property_name == 'Node Identities':

            try:

                if not add or my_network.node_identities is None:
                    identities = {}
                    max_val = 0
                else:
                    identities = my_network.node_identities
                    if my_network.node_centroids is not None:
                        max_val = max(list(my_network.node_centroids.keys()))
                    else:
                        max_val = max(list(my_network.node_identities.keys()))

                idens = data_dict['Identity Column']

                if 'Numerical IDs' in data_dict:
                    nodes = data_dict['Numerical IDs']
                    if add:
                        for i, node in enumerate(nodes):
                            nodes[i] = node + max_val

                else:
                    nodes = np.arange(max_val + 1, max_val + len(data_dict['Identity Column']) + 1)

                for i in range(len(nodes)):

                    identities[nodes[i]] = str(idens[i])

                my_network.node_identities = identities

                self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', title = 'Node Identities')

                print("Identities succesfully set")

            except Exception as e:
                print(f"Error: {e}")

        elif property_name == 'Node Communities':

            try:

                if not add or my_network.communities is None:
                    communities = {}
                    max_val = 0
                else:
                    communities = my_network.communities
                    max_val = max(list(my_network.communities.keys()))


                coms = data_dict['Community Identifier']

                if 'Numerical IDs' in data_dict:
                    nodes = data_dict['Numerical IDs']
                else:
                    nodes = np.arange(max_val + 1, max_val + len(data_dict['Community Identifier']) + 1)

                for i in range(len(nodes)):

                    communities[nodes[i]] = [str(coms[i])]

                my_network.communities = communities

                self.format_for_upperright_table(my_network.communities, 'NodeID', 'CommunityID', title = 'Community Partition')

                print("Communities succesfully set")

            except Exception as e:
                print(f"Error: {e}")


    def set_active_channel(self, index):
        """Set the active channel and update UI accordingly."""
        self.active_channel = index
        self.active_channel_combo.setCurrentIndex(index)
        # Update button appearances to show active channel
        for i, btn in enumerate(self.channel_buttons):
            if i == index and btn.isEnabled():
                btn.setStyleSheet("font-weight: bold; color: yellow;")
            else:
                btn.setStyleSheet("")

    def reduce_rgb_dimension(self, array, method='first'):
        """
        Reduces a 4D array (Z, Y, X, C) to 3D (Z, Y, X) by dropping the color dimension
        using the specified method.
        
        Parameters:
        -----------
        array : numpy.ndarray
            4D array with shape (Z, Y, X, C) where C is the color channel dimension
        method : str, optional
            Method to use for reduction:
            - 'first': takes the first color channel (default)
            - 'mean': averages across color channels
            - 'max': takes maximum value across color channels
            - 'min': takes minimum value across color channels
            - 'weight': takes weighted channel averages
        
        Returns:
        --------
        numpy.ndarray
            3D array with shape (Z, Y, X)
        
        Raises:
        -------
        ValueError
            If input array is not 4D or method is not recognized
        """
        if array.ndim != 4:
            raise ValueError(f"Expected 4D array, got {array.ndim}D array")
        
        if method not in ['first', 'mean', 'max', 'min', 'weight']:
            raise ValueError(f"Unknown method: {method}")
        
        if method == 'first':
            return array[..., 0]
        elif method == 'mean':
            return np.mean(array, axis=-1)
        elif method == 'max':
            return np.max(array, axis=-1)
        elif method == 'weight':
            # Apply the luminosity formula
            return (0.2989 * array[:,:,:,0] + 0.5870 * array[:,:,:,1] + 0.1140 * array[:,:,:,2])
        else:  # min
            return np.min(array, axis=-1)

    def confirm_rgb_dialog(self):
        """Shows a dialog asking user to confirm if image is 2D RGB"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Image Format Alert")
        msg.setInformativeText("Is this a 2D color (RGB/CMYK) image?")
        msg.setWindowTitle("Confirm Image Format")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def confirm_multichan_dialog(self):
        """Shows a dialog asking user to confirm if image is multichan"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Image Format Alert")
        msg.setInformativeText("Is this a Multi-Channel (4D) image?")
        msg.setWindowTitle("Confirm Image Format")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def confirm_resize_dialog(self, shapes):
        """Shows a dialog asking user to resize image"""
        old_shape = shapes[0]
        new_shape = shapes[1]
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Image Format Alert")
        msg.setInformativeText(f"This image is a different shape (New shape: {new_shape}) than the ones loaded into the viewer window (Current shape: {old_shape}). This program is not designed to accomodate loading of differently sized images.\nPress yes to resize the new image to the other images. Press no to go back.")
        msg.setWindowTitle("Resize")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def get_scaling_metadata_only(self, filename):
        # This only reads headers/metadata, not image data
        with tifffile.TiffFile(filename) as tif:
            x_scale = y_scale = z_scale = unit = None
            
            # ImageJ metadata (very lightweight)
            if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                metadata = tif.imagej_metadata
                z_scale = metadata.get('spacing')
                unit = metadata.get('unit')
            
            # TIFF tags (also lightweight - just header info)
            page = tif.pages[0]  # This doesn't load image data
            tags = page.tags
            
            if 'XResolution' in tags:
                x_res = tags['XResolution'].value
                x_scale = x_res[1] / x_res[0] if isinstance(x_res, tuple) else 1.0 / x_res
                
            if 'YResolution' in tags:
                y_res = tags['YResolution'].value
                y_scale = y_res[1] / y_res[0] if isinstance(y_res, tuple) else 1.0 / y_res

        if x_scale == None:
            x_scale = 1
        if z_scale == None:
            z_scale = 1
        if x_scale == 1 and z_scale == 1:
            return

        return x_scale, z_scale

    def load_channel(self, channel_index, channel_data=None, data=False, assign_shape = True, preserve_zoom = None, end_paint = False, begin_paint = False, color = False, load_highlight = False, filename = None):
        """Load a channel and enable active channel selection if needed."""

        try:

            if not data:  

                if not filename:
                    # For solo loading
                    filename, _ = QFileDialog.getOpenFileName(
                        self,
                        f"Load Channel {channel_index + 1}",
                        "",
                        "Image Files (*.tif *.tiff *.nii *.jpg *.jpeg *.png)"
                    )
                
                if not filename:
                    return
                
                file_extension = filename.lower().split('.')[-1]

                if channel_index == 0:
                    self.node_name = filename
                
                try:
                    if file_extension in ['tif', 'tiff']:
                        import tifffile
                        self.channel_data[channel_index] = None
                        if (self.channel_data[0] is None and self.channel_data[1] is None) and (channel_index == 0 or channel_index == 1):
                            try:
                                my_network.xy_scale, my_network.z_scale = self.get_scaling_metadata_only(filename)
                                print(f"xy_scale property set to {my_network.xy_scale}; z_scale property set to {my_network.z_scale}")
                                self.xy_scale_label.setText(f"xy_scale: {my_network.xy_scale:.2e}                   ")
                                self.z_scale_label.setText(f"z_scale: {my_network.z_scale:.2e}                   ")
                            except:
                                pass
                        test_channel_data = tifffile.imread(filename)
                        #test_channel_data = test_channel_data[:, :, 0] #DELETE THIS
                        if len(test_channel_data.shape) not in (2, 3, 4):
                            print("Invalid Shape")
                            return 
                        self.channel_data[channel_index] = test_channel_data

                    elif file_extension == 'nii':
                        try:
                            import nibabel as nib
                            nii_img = nib.load(filename)
                            # Get data and transpose to match TIFF orientation
                            # If X needs to become Z, we move axis 2 (X) to position 0 (Z)
                            arraydata = nii_img.get_fdata()
                            self.channel_data[channel_index] = np.transpose(arraydata, (2, 1, 0))
                        except:
                            return
                        
                    elif file_extension in ['jpg', 'jpeg', 'png']:
                        from PIL import Image
                        
                        with Image.open(filename) as img:
                            # Convert directly to numpy array, keeping color if present
                            self.channel_data[channel_index] = np.array(img)
                            
                            # Debug info to check shape
                            print(f"Loaded image shape: {self.channel_data[channel_index].shape}")
                            
                except ImportError as e:
                    QMessageBox.critical(self, "Error", f"Required library not installed: {str(e)}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")

            else:
                self.channel_data[channel_index] = channel_data
                if channel_data is None:
                    self.delete_channel(channel_index, called = False, update = True)
                    return

            try:
                #if len(self.channel_data[channel_index].shape) == 4:
                if 1 in self.channel_data[channel_index].shape:
                    #print("Removing singleton dimension (I am assuming this is a channel dimension?)")
                    self.channel_data[channel_index] = np.squeeze(self.channel_data[channel_index])
            except:
                pass

            if len(self.channel_data[channel_index].shape) == 2:  # handle 2d data
                self.channel_data[channel_index] = np.expand_dims(self.channel_data[channel_index], axis=0)

            if self.channel_data[channel_index].dtype == np.bool_: #Promote boolean arrays if they somehow get loaded
                self.channel_data[channel_index] = self.channel_data[channel_index].astype(np.uint8)

            try:
                if len(self.channel_data[channel_index].shape) == 3:  # potentially 2D RGB
                    if self.channel_data[channel_index].shape[-1] in (3, 4):  # last dim is 3 or 4
                        if not data:
                            if self.confirm_rgb_dialog():
                                # User confirmed it's 2D RGB, expand to 4D
                                self.channel_data[channel_index] = np.expand_dims(self.channel_data[channel_index], axis=0)
                        elif self.shape[0] == 1: # this can only be true if the user already loaded in a 2d image
                            self.channel_data[channel_index] = np.expand_dims(self.channel_data[channel_index], axis=0)

            except:
                pass

            if len(self.channel_data[channel_index].shape) == 4:
                if not self.channel_data[channel_index].shape[-1] in (3, 4):
                    if self.confirm_multichan_dialog(): # User is trying to load 4D channel stack:
                        my_data = copy.deepcopy(self.channel_data[channel_index])
                        self.channel_data[channel_index] = None 
                        self.show_multichan_dialog(data = my_data)
                        return
                elif not color and (channel_index == 0 or channel_index == 1):
                    try:
                        self.channel_data[channel_index] = self.reduce_rgb_dimension(self.channel_data[channel_index], 'weight')
                    except:
                        pass

            for i in range(4): #Try to ensure users don't load in different sized arrays
                if self.channel_data[i] is None or i == channel_index or data:
                    if self.highlight_overlay is not None: #Make sure highlight overlay is always the same shape as new images
                        try:
                            if self.channel_data[i].shape[:3] != self.highlight_overlay.shape:
                                self.highlight_overlay = None
                        except:
                            pass
                    #if not data:
                    self.original_xlim = None
                    self.original_ylim = None
                else:
                    old_shape = self.channel_data[i].shape[:3] #Ask user to resize images that are shaped differently
                    if old_shape != self.channel_data[channel_index].shape[:3]:
                        if self.confirm_resize_dialog([old_shape, self.channel_data[channel_index].shape[:3]]):
                            self.channel_data[channel_index] = n3d.upsample_with_padding(self.channel_data[channel_index], original_shape = old_shape)
                            break
                        else:
                            return

            if not begin_paint:
                if channel_index == 0:
                    my_network.nodes = self.channel_data[channel_index]
                elif channel_index == 1:
                    my_network.edges = self.channel_data[channel_index]
                elif channel_index == 2:
                    my_network.network_overlay = self.channel_data[channel_index]
                elif channel_index == 3:
                    my_network.id_overlay = self.channel_data[channel_index]
            
            # Enable the channel button
            if channel_index != 4:
                self.channel_buttons[channel_index].setEnabled(True)
                for button in self.delete_buttons:
                    button.setEnabled(True) 

            
                # Enable active channel selector if this is the first channel loaded
                #if not self.active_channel_combo.isEnabled():
                 #   self.active_channel_combo.setEnabled(True)
            
                # Update slider range if this is the first channel loaded
                try:
                    if len(self.channel_data[channel_index].shape) == 3 or len(self.channel_data[channel_index].shape) == 4:
                        if not self.slice_slider.isEnabled():
                            self.slice_slider.setEnabled(True)
                            self.slice_slider.setMinimum(0)
                            self.slice_slider.setMaximum(self.channel_data[channel_index].shape[0] - 1)
                            if self.slice_slider.value() < self.channel_data[channel_index].shape[0] - 1:
                                self.current_slice = self.slice_slider.value()
                            else:
                                self.slice_slider.setValue(0)
                                self.current_slice = 0
                        else:
                            self.slice_slider.setEnabled(True)
                            self.slice_slider.setMinimum(0)
                            self.slice_slider.setMaximum(self.channel_data[channel_index].shape[0] - 1)
                            if self.slice_slider.value() < self.channel_data[channel_index].shape[0] - 1:
                                self.current_slice = self.slice_slider.value()
                            else:
                                self.current_slice = 0
                                self.slice_slider.setValue(0)
                    else:
                        self.slice_slider.setEnabled(False)
                except:
                    pass

                
                # If this is the first channel loaded, make it active
                if all(not btn.isEnabled() for btn in self.channel_buttons[:channel_index]):
                    self.set_active_channel(channel_index)

                if not self.channel_buttons[channel_index].isChecked():
                    self.channel_buttons[channel_index].click()

                self.min_max[channel_index][0] = np.min(self.channel_data[channel_index])
                self.min_max[channel_index][1] = max(1, np.max(self.channel_data[channel_index]))
                self.volume_dict[channel_index] = None #reset volumes

            try:
                if assign_shape: #keep original shape tracked to undo resampling.
                    if self.original_shape is None:
                        self.original_shape = self.channel_data[channel_index].shape
                    elif self.original_shape[0] < self.channel_data[channel_index].shape[0] or self.original_shape[1] < self.channel_data[channel_index].shape[1] or self.original_shape[2] < self.channel_data[channel_index].shape[2]:
                        self.original_shape = self.channel_data[channel_index].shape
                    if len(self.original_shape) == 4:
                        self.original_shape = (self.original_shape[0], self.original_shape[1], self.original_shape[2])
            except:
                pass

            if self.shape == None:
                self.last_change = None
                self.shape = (self.channel_data[channel_index].shape[0], self.channel_data[channel_index].shape[1], self.channel_data[channel_index].shape[2])
                home = True
            elif self.shape[:3] == self.channel_data[channel_index].shape[:3]:
                self.shape = (self.channel_data[channel_index].shape[0], self.channel_data[channel_index].shape[1], self.channel_data[channel_index].shape[2])
                home = False
            else:
                self.last_change = None
                self.shape = (self.channel_data[channel_index].shape[0], self.channel_data[channel_index].shape[1], self.channel_data[channel_index].shape[2])
                home = True

            self.img_height, self.img_width = self.shape[1], self.shape[2]

            self.completed_paint_strokes = [] #Reset pending paint operations
            self.current_stroke_points = []
            self.current_stroke_type = None
            self.virtual_draw_operations = []
            self.virtual_erase_operations = []
            self.current_operation = []
            self.current_operation_type = None
            if self.shape[0] * self.shape[1] * self.shape[2] > self.mini_thresh:
                self.mini_overlay = True

            if load_highlight:
                self.highlight_overlay = n3d.binarize(self.channel_data[4].astype(np.uint8))
                self.mini_overlay_data = None
                self.mini_overlay = False
                self.channel_data[4] = None

            elif not end_paint:
                self.update_display(home = home)

                
        except Exception as e:

            #import traceback
            #traceback.print_exc()

            if not data:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Error Loading File",
                    f"Failed to load file: {str(e)}"
                )

    def delete_channel(self, channel_index, called = True, update = True):
        """Delete the specified channel and update the display."""
        if called:
            if self.channel_data[channel_index] is not None:
                # Confirm deletion
                reply = QMessageBox.question(
                    self,
                    'Delete Channel',
                    f'Are you sure you want to delete the {self.channel_names[channel_index]} channel?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
            else:
                # Confirm deletion
                reply = QMessageBox.question(
                    self,
                    'Populate Channel',
                    f'Would you like to populate the {self.channel_names[channel_index]} channel with an empty array (ie for drawing on)?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.load_channel(channel_index, np.zeros(self.shape, dtype = np.uint8), data = True)
                return
        else:
            reply = False

        
        if reply == QMessageBox.StandardButton.Yes or not called:
            # Set channel data to None
            self.channel_data[channel_index] = None
            
            # Update corresponding network property
            if channel_index == 0:
                my_network.nodes = None
                #my_network.node_centroids = None
                #my_network.node_identities = None
            elif channel_index == 1:
                my_network.edges = None
                my_network.edge_centroids = None
            elif channel_index == 2:
                my_network.network_overlay = None
            elif channel_index == 3:
                my_network.id_overlay = None
            
            # Disable buttons
            self.channel_buttons[channel_index].setEnabled(False)
            self.channel_buttons[channel_index].setChecked(False)
            all_none = True
            for chan in self.channel_data:
                if chan is not None:
                    all_none = False
            if all_none:
                for button in self.delete_buttons:
                    button.setEnabled(False)
            self.channel_visible[channel_index] = False
            self.reset_dicts(channel_index)
            
            # If this was the active channel, switch to the first available channel
            if self.active_channel == channel_index:
                for i in range(4):
                    if self.channel_data[i] is not None:
                        self.set_active_channel(i)
                        break
                else:
                    # If no channels are available, disable active channel selector
                    #self.active_channel_combo.setEnabled(False)
                    self.shape = None # Also there is not an active shape anymore
            
            if update:
                # Update display
                self.update_display(preserve_zoom = (self.ax.get_xlim(), self.ax.get_ylim()))

    def reset_dicts(self, index):
        self.volume_dict[index] = None
        self.radii_dict[index] = None
        self.surface_area_dict[index] = None
        self.sphericity_dict[index] = None
        self.branch_dict[index] = None

    def reset(self, nodes = False, network = False, xy_scale = 1, z_scale = 1, edges = False, search_region = False, network_overlay = False, id_overlay = False, update = True, node_identities = False):
        """Method to flexibly reset certain fields to free up the RAM as desired"""
        
        # Set scales first before any clearing operations
        my_network.xy_scale = xy_scale
        my_network.z_scale = z_scale
        self.xy_scale_label.setText(f"xy_scale: {my_network.xy_scale:.2e}                   ")
        self.z_scale_label.setText(f"z_scale: {my_network.z_scale:.2e}                   ")

        if network:
            my_network.network = None
            my_network.communities = None
            self.stats_dict = {}

            # Create empty DataFrame
            empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
            
            # Clear network table
            self.network_table.setModel(PandasModel(empty_df))
            
            # Clear selection table
            self.selection_table.setModel(PandasModel(empty_df))
            self.clear_subgraphs()

        if node_identities:
            my_network.node_identities = None

        if nodes:
            self.delete_channel(0, False, update = update)

        if edges:
            self.delete_channel(1, False, update = update)
        try:
            if search_region:
                my_network.search_region = None
        except:
            pass

        if network_overlay:
            self.delete_channel(2, False, update = update)

        if id_overlay:
            self.delete_channel(3, False, update = update)



    def save_network_3d(self, asbool=True):
        try:
            if asbool:  # Save As
                # Use getSaveFileName which allows non-existent paths
                # This lets users navigate to parent AND type the child folder name in one step
                full_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Select Location and Name for Network3D Output Folder",
                    "",
                    "Folder (*.folder)",  # Dummy filter, we'll ignore the extension
                    options=QFileDialog.Option.DontConfirmOverwrite  # Don't warn about overwriting
                )
                
                if not full_path:  # User canceled
                    return
                
                # Parse the result: extract parent directory and folder name
                parent_dir = os.path.dirname(full_path)
                new_folder_name = os.path.basename(full_path)
                
                # Remove any extension the user might have typed (from dummy filter)
                if new_folder_name.endswith('.folder'):
                    new_folder_name = new_folder_name[:-7]
                
                # Validate parent directory exists
                if not os.path.isdir(parent_dir):
                    QMessageBox.critical(
                        self,
                        "Invalid Location",
                        f"Parent directory does not exist: {parent_dir}"
                    )
                    return
                
                # Validate folder name is not empty
                if not new_folder_name:
                    QMessageBox.critical(
                        self,
                        "Invalid Name",
                        "Please enter a name for the output folder."
                    )
                    return

            else:  # Save
                if self.last_saved is None:
                    self.save_network_3d()
                    return
                else:
                    parent_dir = self.last_saved
                    new_folder_name = self.last_save_name
                    
            # Handle RGB dimension reduction before saving
            if len(self.channel_data[0].shape) == 4:
                try:
                    self.load_channel(0, self.reduce_rgb_dimension(self.channel_data[0], 'weight'), True)
                except:
                    pass

            # Call save method
            my_network.dump(parent_dir=parent_dir, name=new_folder_name)
            self.last_saved = parent_dir
            self.last_save_name = new_folder_name
            self.setWindowTitle(f"NetTracer3D - Session: {self.last_save_name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Error Saving File",
                f"Failed to save file: {str(e)}"
            )

    def save_pickled_net(self):

        # This lets users navigate to parent AND type the child folder name in one step
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Please Find your Active Session Folder; Will Make Network Files that Load Faster than CSV")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setOption(QFileDialog.Option.ReadOnly)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_path = dialog.directory().absolutePath()
        else:
            return  # User cancelled directory selection
        
        if not selected_path:  # User canceled
            return
        
        n3d.save_pickle(my_network.network, f'{selected_path}/quickload_network.pkl')
        n3d.save_pickle(my_network.network_lists, f'{selected_path}/quickload_network_lists.pkl')


    def save(self, ch_index, asbool=True):
        """Handle both Save and Save As operations."""
        try:
            if asbool:  # Save As
                # Open file dialog for saving
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    f"Save Image As",
                    "",  # Default directory
                    "TIFF Files (*.tif *.tiff);;All Files (*)"  # File type filter
                )
                
                if filename:  # Only proceed if user didn't cancel
                    # If user didn't type an extension, add .tif
                    if not filename.endswith(('.tif', '.tiff')):
                        filename += '.tif'
            else:  # Save
                if self.last_saved is None:
                    self.save(ch_index)
                    return
                else:
                    if ch_index == 0:
                        filename = self.last_save_name + "/labelled_nodes.tif"
                        print(filename)
                    elif ch_index == 1:
                        filename = self.last_save_name + "/labelled_edges.tif"
                    elif ch_index == 2:
                        filename = self.last_save_name + "/overlay_1.tif"
                    elif ch_index == 3:
                        filename = self.last_save_name + "/overlay_2.tif"
                    elif ch_index == 4:
                        filename = self.last_save_name + "/Highlighted_Element.tif"
            
            # Call appropriate save method
            if (filename is not None and filename != "") or not asbool:  # Proceed if we have a filename OR if it's a regular save
                if ch_index == 0:
                    my_network.save_nodes(filename=filename)
                elif ch_index == 1:
                    my_network.save_edges(filename=filename)
                elif ch_index == 2:
                    my_network.save_network_overlay(filename=filename)
                elif ch_index == 3:
                    my_network.save_id_overlay(filename=filename)
                elif ch_index == 4:
                    if self.mini_overlay == True:
                        self.create_highlight_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                    if filename == None:
                        filename = "Highlighted_Element.tif"
                    tifffile.imwrite(f"{filename}", self.highlight_overlay)
                
                #print(f"Saved {self.channel_names[ch_index]}" + (f" to: {filename}" if filename else ""))  # Debug print
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Error Saving File",
                f"Failed to save file: {str(e)}"
            )

    def toggle_channel(self, channel_index):
        """Toggle visibility of a channel."""
        # Store current zoom settings before toggling

        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

        self.channel_visible[channel_index] = self.channel_buttons[channel_index].isChecked()
        self.update_display(preserve_zoom=(current_xlim, current_ylim))


    
    def update_slice(self):
        """Queue a slice update when slider moves."""
        # Store current view settings
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

        
        # Store the pending slice and view settings
        self.pending_slice = (self.slice_slider.value(), (current_xlim, current_ylim))
        
        # Reset and restart timer
        self._slice_update_timer.start(1)  # 20ms delay
        
    def _do_slice_update(self):
        """Actually perform the slice update after debounce delay."""
        if self.pending_slice is not None:
            slice_value, view_settings = self.pending_slice
            if (hasattr(self, 'completed_paint_strokes') and self.completed_paint_strokes) or \
               (hasattr(self, 'current_stroke_points') and self.current_stroke_points) or \
               (hasattr(self, 'virtual_paint_items') and self.virtual_paint_items) or \
               (hasattr(self, 'current_paint_items') and self.current_paint_items):
                if hasattr(self, 'current_stroke_points') and self.current_stroke_points:
                    self.pm.finish_current_virtual_operation()
                self.pm.convert_virtual_strokes_to_data()
            self.current_slice = slice_value
            if self.preview:
                 self.highlight_overlay = None
                 self.mini_overlay_data = None
                 self.mini_overlay = False
                 self.create_highlight_overlay_slice(self.targs, bounds=self.bounds)
            elif self.mini_overlay == True: #If we are rendering the highlight overlay for selected values one at a time.
                self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])

            if self.resizing:
                self.highlight_overlay = None
                view_settings = ((-0.5, self.shape[2] - 0.5), (self.shape[1] - 0.5, -0.5))
                self.resizing = False
            self.z_label.setText(f"Slice {self.current_slice}")
            self.update_display(preserve_zoom=view_settings)
            self.pending_slice = None

    def update_brightness(self, channel_index, values):
        """Update brightness/contrast settings for a channel."""

        # Store current zoom settings before toggling
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
        # Convert slider values (0-100) to data values (0-1)
        min_val, max_val = values
        self.channel_brightness[channel_index]['min'] = min_val / 65535 
        self.channel_brightness[channel_index]['max'] = max_val / 65535
        self.update_display(preserve_zoom = (current_xlim, current_ylim))

    def update_display(self, preserve_zoom=None, dims=None, called=False, skip=False, quick_wheel_update=False, home = False, downsample = True, needs_mini = True):
        """Optimized display update with view-based cropping and downsampling."""
        try:
            # Initialize reusable components
            if not hasattr(self, 'channel_images'):
                self.channel_images = {}
                self.highlight_image = None
                self.measurement_artists = []
                self.view_initialized = False
                self.original_dims = None
                       
            if (hasattr(self, 'completed_paint_strokes') and self.completed_paint_strokes) or \
               (hasattr(self, 'current_stroke_points') and self.current_stroke_points) or \
               (hasattr(self, 'virtual_paint_items') and self.virtual_paint_items) or \
               (hasattr(self, 'current_paint_items') and self.current_paint_items):
                if hasattr(self, 'current_stroke_points') and self.current_stroke_points:
                    self.pm.finish_current_virtual_operation()
                self.pm.convert_virtual_strokes_to_data()
                self.channel_buttons[self.active_channel].setChecked(True)
            
            # Get dimensions
            active_channels = [i for i in range(4) if self.channel_data[i] is not None]
            if dims is None:
                if active_channels:
                    dims = [(self.channel_data[i].shape[1:3] if len(self.channel_data[i].shape) >= 3 else 
                            self.channel_data[i].shape) for i in active_channels]
                    min_height = min(d[0] for d in dims)
                    min_width = min(d[1] for d in dims)
                else:
                    min_height = min_width = 1
            else:
                min_height, min_width = dims[:2]
            
            self.original_dims = (min_height, min_width)
            if home:
                if self.original_xlim is None and self.original_dims is not None:
                    self.original_xlim = (0, self.original_dims[1])
                    self.original_ylim = (0, self.original_dims[0])
                self.view.setRange(xRange=self.original_xlim, yRange=self.original_ylim, padding=0)

            self.img_height, self.img_width = min_height, min_width

            # Initialize view only once
            if not self.view_initialized:
                self.view.setRange(xRange=(0, min_width), yRange=(0, min_height), padding=0)
                self.view.invertY(True)
                self.view_initialized = True

            # Determine cropping and downsampling based on update type
            if quick_wheel_update:
                # Quick update: render full image at low resolution
                self.x_min_padded = 0
                self.x_max_padded = min_width
                self.y_min_padded = 0
                self.y_max_padded = min_height
                
                # Use aggressive downsampling for speed
                total_pixels = min_width * min_height
                self.downsample_factor = max(1, int(np.sqrt(total_pixels / (1500 * 1500))))
            else:
                # Normal update: use cropping and smart downsampling
                # Get current visible region from pyqtgraph
                view_range = self.view.viewRange()
                x_range, y_range = view_range[0], view_range[1]
                
                # Calculate visible region in pixel coordinates
                x_min = max(0, int(np.floor(x_range[0])))
                x_max = min(min_width, int(np.ceil(x_range[1])))
                y_min = max(0, int(np.floor(y_range[0])))
                y_max = min(min_height, int(np.ceil(y_range[1])))
                
                if downsample:
                    # Calculate downsample factor based on visible area
                    visible_area = (x_max - x_min) * (y_max - y_min)
                    val = int(np.ceil(visible_area / (3000 * 3000)))
                    self.validate_downsample_input(text=val, update=False)
                    downsample_factor = self.downsample_factor
                else:
                    self.downsample_factor = 1
                
                # Determine padding/expansion based on pan mode
                if self.pan_mode:
                    # In pan mode, expand the render region to allow smooth panning
                    box_len = int(1.5 * (x_max - x_min))  # Full width
                    box_height = int(1.5 * (y_max - y_min))  # Full height
                    
                    # Expand the crop region (this is what gets rendered)
                    self.x_min_padded = max(0, x_min - box_len)
                    self.x_max_padded = min(min_width, x_max + box_len)
                    self.y_min_padded = max(0, y_min - box_height)
                    self.y_max_padded = min(min_height, y_max + box_height)
                else:
                    # In normal mode, just add minimal padding to avoid edge artifacts
                    padding = max(10, downsample_factor * 2)
                    self.x_min_padded = max(0, x_min - padding)
                    self.x_max_padded = min(min_width, x_max + padding)
                    self.y_min_padded = max(0, y_min - padding)
                    self.y_max_padded = min(min_height, y_max + padding)
            
            base_colors = self.base_colors
            
            # Helper function to crop and downsample
            def crop_and_downsample(image, y_start, y_end, x_start, x_end, factor):
                # Crop to visible region
                if len(image.shape) == 2:
                    cropped = image[y_start:y_end, x_start:x_end]
                elif len(image.shape) == 3:
                    cropped = image[y_start:y_end, x_start:x_end, :]
                else:
                    cropped = image
                
                # Downsample if needed
                if factor == 1:
                    return cropped
                
                if len(cropped.shape) == 2:
                    return cropped[::factor, ::factor]
                elif len(cropped.shape) == 3:
                    return cropped[::factor, ::factor, :]
                else:
                    return cropped
            # Clear old measurement artists
            for artist in self.measurement_artists:
                try:
                    self.view.removeItem(artist)
                except (TypeError, RuntimeError):
                    # Skip items that are matplotlib objects or already removed
                    pass
            self.measurement_artists = []
            # Update channel images
            for channel in range(4):
                if channel in self.channel_images and not self.channel_visible[channel]:
                    try:
                        self.view.removeItem(self.channel_images[channel])
                    except:
                        pass
                    del self.channel_images[channel]
                    continue
                    
                if self.channel_visible[channel] and self.channel_data[channel] is not None:
                    is_rgb = len(self.channel_data[channel].shape) == 4 and (
                        self.channel_data[channel].shape[-1] in [3, 4])
                    
                    if len(self.channel_data[channel].shape) == 3 and not is_rgb:
                        current_image = self.channel_data[channel][self.current_slice, :, :]
                    elif is_rgb:
                        current_image = self.channel_data[channel][self.current_slice]
                    else:
                        current_image = self.channel_data[channel]

                    # Crop to visible region and downsample
                    display_image = crop_and_downsample(
                        current_image, self.y_min_padded, self.y_max_padded,
                        self.x_min_padded, self.x_max_padded, self.downsample_factor)

                    # Create or reuse ImageItem
                    if channel not in self.channel_images:
                        self.channel_images[channel] = pg.ImageItem()
                        self.view.addItem(self.channel_images[channel])

                    if is_rgb and self.channel_data[channel].shape[-1] in [3, 4]:
                        brightness_min = self.channel_brightness[channel]['min']
                        brightness_max = self.channel_brightness[channel]['max']
                        alpha_range = brightness_max - brightness_min
                        base_alpha = brightness_min
                        final_alpha = np.clip(base_alpha + alpha_range, 0.0, 1.0)
                        
                        if display_image.shape[-1] == 4:
                            img_with_alpha = display_image.copy()
                            img_with_alpha[..., 3] = img_with_alpha[..., 3] * final_alpha
                            self.channel_images[channel].setImage(img_with_alpha.transpose(1, 0, 2))
                        else:
                            img_rgba = np.dstack([display_image, np.ones(display_image.shape[:2]) * final_alpha * 255])
                            self.channel_images[channel].setImage(img_rgba.transpose(1, 0, 2))
                        
                        # Position the cropped region correctly
                        self.channel_images[channel].setRect(
                            self.x_min_padded, self.y_min_padded,
                            self.x_max_padded - self.x_min_padded, self.y_max_padded - self.y_min_padded)
                    else:
                        # Regular channel processing
                        if self.min_max[channel][0] is None:
                            if self.channel_data[channel].size > 1000000:
                                sample = self.channel_data[channel][::max(1, self.channel_data[channel].shape[0]//100)]
                                self.min_max[channel] = [np.min(sample), np.max(sample)]
                            else:
                                self.min_max[channel] = [np.min(self.channel_data[channel]), 
                                                       np.max(self.channel_data[channel])]
                        
                        img_min, img_max = self.min_max[channel]
                        
                        if img_min == img_max:
                            normalized_image = np.zeros_like(display_image, dtype=np.float32)
                        else:
                            vmin = img_min + (img_max - img_min) * self.channel_brightness[channel]['min']
                            if self.channel_brightness[channel]['max'] < 2/65535:
                                vmax = 0.5
                            else:
                                vmax = img_min + (img_max - img_min) * self.channel_brightness[channel]['max']

                            if vmin == vmax:
                                normalized_image = np.zeros_like(display_image, dtype=np.float32)
                            else:
                                normalized_image = np.clip((display_image - vmin) / (vmax - vmin), 0, 1)
                        
                        if channel == 2 and self.machine_window is not None:
                            colors = np.array([
                                [0, 0, 0, 0],
                                [0.5, 1, 0.5, 1],
                                [1, 0.5, 0.5, 1]
                            ])
                            cmap = pg.ColorMap(pos=np.array([0, 1, 2]), color=colors * 255)
                            lut = cmap.getLookupTable(0, 2, 256)
                            self.channel_images[channel].setImage(display_image.T, levels=(0, 2))
                            self.channel_images[channel].setLookupTable(lut)
                            self.channel_images[channel].setOpacity(0.7)
                        else:
                            color = base_colors[channel]
                            colors = np.array([
                                [0, 0, 0, 0],
                                [*color, 1]
                            ])
                            cmap = pg.ColorMap(pos=np.array([0, 1]), color=colors * 255)
                            lut = cmap.getLookupTable(0, 1, 256)
                            
                            self.channel_images[channel].setImage(normalized_image.T, levels=(0, 1))
                            self.channel_images[channel].setLookupTable(lut)
                            self.channel_images[channel].setOpacity(0.7)
                        
                        # Position the cropped region correctly
                        self.channel_images[channel].setRect(
                            self.x_min_padded, self.y_min_padded,
                            self.x_max_padded - self.x_min_padded, self.y_max_padded - self.y_min_padded)

            # Handle overlays (with cropping)
            if not hasattr(self, 'overlay_image'):
                self.overlay_image = None
            if self.overlay_image is not None:
                if self.overlay_image.scene() == self.view.scene():
                    try:
                        self.view.removeItem(self.overlay_image)
                    except:
                        pass
                    self.overlay_image = None

            if self.mini_overlay and self.highlight and self.machine_window is None:
                if needs_mini and not self.preview:
                    self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'], update = False)
                elif needs_mini and self.preview:
                    self.create_highlight_overlay_slice(self.thresh_window_ref.targs, self.thresh_window_ref.bounds, update = False)

                display_overlay = self.mini_overlay_data
                colors = np.array([[0, 0, 0, 0], [1, 1, 0, 1]])
                cmap = pg.ColorMap(pos=np.array([0, 1]), color=colors * 255)
                lut = cmap.getLookupTable(0, 1, 256)

                self.overlay_image = pg.ImageItem()
                self.overlay_image.setImage(display_overlay.T)
                self.overlay_image.setLookupTable(lut)
                self.overlay_image.setOpacity(0.8)
                self.overlay_image.setRect(
                    self.x_min_padded, self.y_min_padded,
                    self.x_max_padded - self.x_min_padded, self.y_max_padded - self.y_min_padded)
                self.view.addItem(self.overlay_image)
                
            elif self.highlight_overlay is not None and self.highlight:
                highlight_slice = self.highlight_overlay[self.current_slice]
                display_highlight = crop_and_downsample(
                    highlight_slice, self.y_min_padded, self.y_max_padded,
                    self.x_min_padded, self.x_max_padded, self.downsample_factor)
                
                if self.machine_window is None:
                    colors = np.array([[0, 0, 0, 0], [1, 1, 0, 1]])
                    cmap = pg.ColorMap(pos=np.array([0, 1]), color=colors * 255)
                    lut = cmap.getLookupTable(0, 1, 256)
                    opacity = 0.8
                    levels = (0, 1)
                else:
                    colors = np.array([[0, 0, 0, 0], [1, 1, 0, 1], [0, 0.7, 1, 1]])
                    cmap = pg.ColorMap(pos=np.array([0, 1, 2]), color=colors * 255)
                    lut = cmap.getLookupTable(0, 2, 256)
                    opacity = 0.3
                    levels = (0, 2)
                
                self.overlay_image = pg.ImageItem()
                self.overlay_image.setImage(display_highlight.T, levels=levels)
                self.overlay_image.setLookupTable(lut)
                self.overlay_image.setOpacity(opacity)
                self.overlay_image.setRect(
                    self.x_min_padded, self.y_min_padded,
                    self.x_max_padded - self.x_min_padded, self.y_max_padded - self.y_min_padded)
                self.view.addItem(self.overlay_image)

            # Only draw measurement points on detailed updates, not quick wheel updates
            # Measurement points (unchanged - these are vectors)
            if hasattr(self, 'measurement_points') and self.measurement_points:
                for point in self.measurement_points:
                    x1, y1, z1 = point['point1']
                    x2, y2, z2 = point['point2']
                    pair_idx = point['pair_index']
                    point_type = point.get('type', 'distance')
                    
                    if point_type == 'angle':
                        marker_color = (0, 255, 0)
                        text_color = (0, 255, 0)
                        line_color = (0, 255, 0)
                    else:
                        marker_color = (255, 255, 0)
                        text_color = (255, 255, 0)
                        line_color = (255, 0, 0)
                    
                    if z1 == self.current_slice:
                        scatter1 = pg.ScatterPlotItem([x1], [y1], size=8, 
                                                      pen=pg.mkPen(marker_color), 
                                                      brush=pg.mkBrush(marker_color))
                        self.view.addItem(scatter1)
                        self.measurement_artists.append(scatter1)
                        
                        text1 = pg.TextItem(str(pair_idx), color=text_color, anchor=(0.5, 1))
                        text1.setPos(x1, y1 - 5)
                        self.view.addItem(text1)
                        self.measurement_artists.append(text1)
                        
                    if z2 == self.current_slice:
                        scatter2 = pg.ScatterPlotItem([x2], [y2], size=8,
                                                      pen=pg.mkPen(marker_color),
                                                      brush=pg.mkBrush(marker_color))
                        self.view.addItem(scatter2)
                        self.measurement_artists.append(scatter2)
                        
                        text2 = pg.TextItem(str(pair_idx), color=text_color, anchor=(0.5, 1))
                        text2.setPos(x2, y2 - 5)
                        self.view.addItem(text2)
                        self.measurement_artists.append(text2)
                        
                    if z1 == z2 == self.current_slice:
                        line_pen = pg.mkPen(color=line_color, style=pg.QtCore.Qt.PenStyle.DashLine, width=1)
                        line = pg.PlotDataItem([x1, x2], [y1, y2], pen=line_pen)
                        line.setOpacity(0.5)
                        self.view.addItem(line)
                        self.measurement_artists.append(line)

            # Current point in progress
            if hasattr(self, 'current_point') and self.current_point is not None:
                x, y, z = self.current_point
                if z == self.current_slice:
                    if hasattr(self, 'measurement_mode') and self.measurement_mode == "angle":
                        marker_color = (0, 255, 0)
                        text_color = (0, 255, 0)
                        label = f"A{self.current_trio_index}" if hasattr(self, 'current_trio_index') else "A"
                    else:
                        marker_color = (255, 255, 0)
                        text_color = (255, 255, 0)
                        label = f"D{self.current_pair_index}" if hasattr(self, 'current_pair_index') else "D"
                    
                    scatter = pg.ScatterPlotItem([x], [y], size=8,
                                                pen=pg.mkPen(marker_color),
                                                brush=pg.mkBrush(marker_color))
                    self.view.addItem(scatter)
                    self.measurement_artists.append(scatter)
                    
                    text = pg.TextItem(label, color=text_color, anchor=(0.5, 1))
                    text.setPos(x, y - 5)
                    self.view.addItem(text)
                    self.measurement_artists.append(text)

            # Current second point for angle
            if hasattr(self, 'current_second_point') and self.current_second_point is not None:
                x, y, z = self.current_second_point
                if z == self.current_slice:
                    label = f"B{self.current_trio_index}" if hasattr(self, 'current_trio_index') else "B"
                    
                    scatter = pg.ScatterPlotItem([x], [y], size=8,
                                                pen=pg.mkPen((0, 255, 0)),
                                                brush=pg.mkBrush((0, 255, 0)))
                    self.view.addItem(scatter)
                    self.measurement_artists.append(scatter)
                    
                    text = pg.TextItem(label, color=(0, 255, 0), anchor=(0.5, 1))
                    text.setPos(x, y - 5)
                    self.view.addItem(text)
                    self.measurement_artists.append(text)
                    
                    if (hasattr(self, 'current_point') and self.current_point is not None and 
                        self.current_point[2] == self.current_slice):
                        x1, y1, z1 = self.current_point
                        line_pen = pg.mkPen(color=(0, 255, 0), style=pg.QtCore.Qt.PenStyle.DashLine, width=1)
                        line = pg.PlotDataItem([x1, x], [y1, y], pen=line_pen)
                        line.setOpacity(0.7)
                        self.view.addItem(line)
                        self.measurement_artists.append(line)
            # Handle scalebar (skip on quick updates)
            if hasattr(self, 'scalebar_artists') and self.scalebar_artists:
                self._draw_scalebar()
            else:
                self._remove_scalebar()
        except Exception as e:
            pass
            #import traceback
            #print(traceback.format_exc())

    def get_channel_image(self, channel):
        """Find the matplotlib image object for a specific channel."""
        if not hasattr(self.ax, 'images'):
            return None
            
        for img in self.ax.images:
            if hasattr(img, 'cmap') and hasattr(img.cmap, 'name'):
                if img.cmap.name == f'custom_{channel}':
                    return img
        return None

    def show_netshow_dialog(self, called = False):
        dialog = NetShowDialog(self, called = called)
        dialog.exec()

    def handle_report(self):

        def invert_dict(d):
            inverted = {}
            for key, value in d.items():
                inverted.setdefault(value, []).append(key)
            return inverted

        stats = {}
        
        try:
            # Basic graph properties
            stats['num_nodes'] = my_network.network.number_of_nodes()
            stats['num_edges'] = len(my_network.network_lists[0])
        except:
            try:
                stats['num_nodes'] = len(np.unique(my_network.nodes)) - 1
            except:
                pass

        try:
            idens = invert_dict(my_network.node_identities)

            for iden, nodes in idens.items():
                stats[f'num_nodes_{iden}'] = len(nodes)
        except:
            pass

        try:

            coms = invert_dict(my_network.communities)

            for com, nodes in coms.items():
                stats[f'num_nodes_community_{com}'] = len(nodes)
        except:
            pass

        self.format_for_upperright_table(stats, title = 'Network Report')



    def show_partition_dialog(self):
        dialog = PartitionDialog(self)
        dialog.exec()

    def handle_com_id(self):

        dialog = ComIdDialog(self)
        dialog.exec()

    def handle_com_neighbor(self):

        dialog = ComNeighborDialog(self)
        dialog.exec()

    def handle_com_cell(self):

        dialog = ComCellDialog(self)
        dialog.exec()

    def handle_neigh_com(self):

        dialog = NeighComDialog(self)
        dialog.exec()

    def show_radial_dialog(self):
        dialog = RadialDialog(self)
        dialog.exec()

    def show_neighbor_id_dialog(self):
        dialog = NeighborIdentityDialog(self)
        dialog.exec()

    def show_ripley_dialog(self):
        dialog = RipleyDialog(self)
        dialog.exec()

    def show_heatmap_dialog(self):
        dialog = HeatmapDialog(self)
        dialog.exec()

    def show_netnear_dialog(self):
        dialog = NetNearDialog(self)
        dialog.show()

    def show_nearneigh_dialog(self):
        dialog = NearNeighDialog(self)
        dialog.exec()

    def show_random_dialog(self):
        dialog = RandomDialog(self)
        dialog.exec()

    def show_randnode_dialog(self):
        dialog = RandNodeDialog(self)
        dialog.exec()

    def show_rad_dialog(self):
        dialog = RadDialog(self)
        dialog.exec()

    def handle_sa(self):

        try:

            if self.shape[0] == 1:
                print("The image is 2D and therefore does not have surface areas")
                return

            surface_areas = n3d.get_surface_areas(self.channel_data[self.active_channel], xy_scale = my_network.xy_scale, z_scale = my_network.z_scale)

            if self.active_channel == 0:
               self.surface_area_dict[0] = surface_areas
            elif self.active_channel == 1:
               self.surface_area_dict[1] = surface_areas
            elif self.active_channel == 2:
               self.surface_area_dict[2] = surface_areas
            elif self.active_channel == 3:
               self.surface_area_dict[3] = surface_areas

            self.format_for_upperright_table(surface_areas, title = '~Surface Areas of Objects (Jagged Faces)', metric='ObjectID', value='~Surface Area (Scaled)')

        except Exception as e:
            print(f"Error: {e}")

    def handle_sphericity(self):

        try:

            if self.shape[0] == 1:
                print("The image is 2D and therefore does not have sphericities")
                return

            self.volumes()
            self.handle_sa()
            volumes = self.volume_dict[self.active_channel]
            surface_areas = self.surface_area_dict[self.active_channel]

            sphericities = {
                label: (np.pi**(1/3) * (6 * volumes[label])**(2/3)) / surface_areas[label]
                for label in volumes.keys()
                if label in surface_areas and volumes[label] > 0 and surface_areas[label] > 0
            }

            if self.active_channel == 0:
               self.sphericity_dict[0] = sphericities
            elif self.active_channel == 1:
               self.sphericity_dict[1] = sphericities
            elif self.active_channel == 2:
               self.sphericity_dict[2] = sphericities
            elif self.active_channel == 3:
               self.sphericity_dict[3] = sphericities

            self.format_for_upperright_table(sphericities, title = 'Sphericities of Objects', metric='ObjectID', value='Sphericity')

        except Exception as e:
            print(f"Error: {e}")

    def show_branchstat_dialog(self):
        dialog = BranchStatDialog(self)
        dialog.exec()

    def show_interaction_dialog(self):
        dialog = InteractionDialog(self)
        dialog.exec()

    def show_violin_dialog(self, called = False):
        dialog = ViolinDialog(self, called = called)
        dialog.show()

    def show_degree_dialog(self):
        dialog = DegreeDialog(self)
        dialog.exec()


    def show_hub_dialog(self):
        dialog = HubDialog(self)
        dialog.exec()

    def show_mother_dialog(self):
        dialog = MotherDialog(self)
        dialog.exec()

    def show_code_dialog(self, sort = 'Community'):
        dialog = CodeDialog(self, sort = sort)
        dialog.exec()

    def handle_centroid_umap(self):

        if my_network.node_centroids is None:
            self.show_centroid_dialog()

        my_network.centroid_umap()


    def closeEvent(self, event):
        """Override closeEvent to close all windows when main window closes"""
        
        # Close all Qt windows
        QApplication.closeAllWindows()
        
        # Close all matplotlib figures
        plt.close('all')
        
        # Accept the close event
        event.accept()
        
        # Force quit the application
        QCoreApplication.quit()

        exit()



#TABLE RELATED: 
class SearchWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.search_input)
        
        close_button = QPushButton("×")
        close_button.setFixedSize(20, 20)
        close_button.clicked.connect(self.hide)
        layout.addWidget(close_button)
        
        # Store the last searched text and matches
        self.last_search = None
        self.current_match_index = -1
        self.current_matches = []
        
    def on_text_changed(self, text):
        self.last_search = text if text else None
        self.current_match_index = -1
        self.current_matches = []
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.last_search is not None:
                table_view = self.parent()
                
                if table_view.is_top_table:
                    self.search_top_table(table_view)
                else:
                    # Use existing bottom table search logic
                    main_window = table_view.parent
                    if not hasattr(main_window, 'active_table'):
                        main_window.active_table = main_window.network_table
                    if table_view == main_window.active_table:
                        try:
                            value = int(self.last_search)
                            main_window.highlight_value_in_tables(value)
                        except ValueError:
                            pass
        else:
            super().keyPressEvent(event)

    def search_top_table(self, table_view):
        """Search function for top tables that handles varying formats"""

        if not table_view.model():
            return
            
        model = table_view.model()
        
        try:
            df = model._data
            
            # If this is a new search, find all matches
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    cell_value = str(df.iloc[row, col]).lower()
                    if self.last_search.lower() in cell_value:
                        self.current_matches.append((row, col))
                
                        
            if not self.current_matches:
                return
                
            # Increment current match index or wrap around
            self.current_match_index = (self.current_match_index + 1) % len(self.current_matches)
            row, col = self.current_matches[self.current_match_index]
            
            # Create index for the current match
            model_index = model.index(row, col)
            
            # Highlight the cell in the model
            model.highlight_cell(row, col)
            
            # Select and scroll to the match
            table_view.setCurrentIndex(model_index)
            table_view.scrollTo(model_index)
            
            # Clear previous selection and select the current cell
            table_view.clearSelection()
            table_view.setFocus()
            
        except Exception as e:
            print(f"Error during search: {str(e)}")

class CustomTableView(QTableView):
    def __init__(self, parent=None, is_top_table=False, subgraph = None):
        super().__init__(parent)
        self.search_widget = SearchWidget(self)
        self.search_widget.hide()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.parent = parent  # Store reference to parent window
        self.is_top_table = is_top_table  # Flag to distinguish top tables
        self.subgraph = subgraph
        
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_F:
            pos = self.rect().topRight()
            self.search_widget.move(self.mapToGlobal(pos) - QPoint(self.search_widget.width(), 0))
            self.search_widget.show()
            self.search_widget.search_input.setFocus()
        elif (event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter) and self.search_widget.isVisible():
            self.search_widget.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def show_context_menu(self, position):
        # Get the index at the clicked position
        index = self.indexAt(position)
        
        if index.isValid():  # Only show menu if we clicked on a valid cell
            # Create context menu
            context_menu = QMenu(self)
            
            # Add Sort submenu for all tables
            if self.model() and hasattr(self.model(), '_data'):
                sort_menu = context_menu.addMenu("Sort")
                
                # Get column names from the DataFrame
                columns = self.model()._data.columns.tolist()
                
                # Create submenus for each column
                for col in columns:
                    col_menu = sort_menu.addMenu("Sort by: " + str(col))
                    
                    # Add sorting options
                    asc_action = col_menu.addAction("Low to High")
                    desc_action = col_menu.addAction("High to Low")
                    
                    # Connect actions
                    asc_action.triggered.connect(lambda checked, c=col: self.sort_table(c, ascending=True))
                    desc_action.triggered.connect(lambda checked, c=col: self.sort_table(c, ascending=False))
            
            # Different menus for top and bottom tables
            if self.is_top_table:  # Use the flag instead of checking membership
                save_menu = context_menu.addMenu("Save As")
                save_csv = save_menu.addAction("CSV")
                save_excel = save_menu.addAction("Excel")
                
                if self.model() and len(self.model()._data.columns) == 2:
                    thresh_action = context_menu.addAction("Use to Threshold Nodes")
                    thresh_action.triggered.connect(lambda: self.thresh(self.create_threshold_dict()))
                
                close_action = context_menu.addAction("Close All")
                close_action.triggered.connect(self.close_all)
                
                # Connect the save actions
                save_csv.triggered.connect(lambda: self.save_table_as('csv'))
                save_excel.triggered.connect(lambda: self.save_table_as('xlsx'))
            else:  # Bottom tables
                # Add Find action
                find_menu = context_menu.addMenu("Find")
                find_action = find_menu.addAction("Find Node/Edge/")
                find_pair_action = find_menu.addAction("Find Pair")
                find_action.triggered.connect(lambda: self.handle_find_action(
                    index.row(), index.column(), 
                    self.model()._data.iloc[index.row(), index.column()]
                ))
                find_pair_action.triggered.connect(lambda: self.handle_find_action(
                    [index.row()], [0,1,2],
                    [self.model()._data.iloc[index.row(), 0], self.model()._data.iloc[index.row(), 1], self.model()._data.iloc[index.row(), 2]]
                    ))
                
                # Add separator
                context_menu.addSeparator()
                
                # Add Save As menu
                save_menu = context_menu.addMenu("Save As")
                save_csv = save_menu.addAction("CSV")
                save_excel = save_menu.addAction("Excel")
                save_gephi = save_menu.addAction("Gephi")
                save_graphml = save_menu.addAction("GraphML")
                save_pajek = save_menu.addAction("Pajek")
                save_quick_load = save_menu.addAction("Pickle (For Quickload)")
                
                # Connect the actions - ensure we're saving the active table
                save_csv.triggered.connect(lambda: self.parent.active_table.save_table_as('csv'))
                save_excel.triggered.connect(lambda: self.parent.active_table.save_table_as('xlsx'))
                save_gephi.triggered.connect(lambda: self.parent.active_table.save_table_as('gexf'))
                save_graphml.triggered.connect(lambda: self.parent.active_table.save_table_as('graphml'))
                save_pajek.triggered.connect(lambda: self.parent.active_table.save_table_as('net'))
                save_quick_load.triggered.connect(self.parent.save_pickled_net)


                if self == self.parent.selection_table:
                    set_action = context_menu.addAction("Swap with network table (also sets internal network properties - may affect related functions)")
                    set_action.triggered.connect(self.set_selection_to_active)
            
            # Show the menu at cursor position
            cursor_pos = QCursor.pos()
            context_menu.exec(cursor_pos)



    def thresh(self, special_dict):
        try:
            self.parent.special_dict = special_dict
            thresh_window = ThresholdWindow(self.parent, 4)
            thresh_window.show()
        except:
            pass

    def create_threshold_dict(self):
        try:
            """Create a dictionary from the 2-column table data."""
            if not self.model() or not hasattr(self.model(), '_data'):
                return {}
            
            df = self.model()._data
            if len(df.columns) != 2:
                return {}
            
            # Create dictionary: {column_0_value: column_1_value}
            threshold_dict = {}
            for index, row in df.iterrows():
                key = row.iloc[0]    # Column 0 value
                value = row.iloc[1]  # Column 1 value
                threshold_dict[int(key)] = float(value)
            
            return threshold_dict
        except:
            pass


    def sort_table(self, column, ascending=True):
        """Sort the table by the specified column."""
        try:
            # Get the current DataFrame
            df = self.model()._data
            
            # Create a copy of the DataFrame for sorting
            sorting_df = df.copy()
            
            # Check if column contains any numeric values
            has_numbers = pd.to_numeric(sorting_df[column], errors='coerce').notna().any()
            
            if has_numbers:
                # For columns with numbers, use numeric sorting
                sorted_index = sorting_df.sort_values(
                    by=column,
                    ascending=ascending,
                    na_position='last',
                    key=lambda x: pd.to_numeric(x, errors='coerce')
                ).index
            else:
                # For non-numeric columns, use regular sorting
                sorted_index = sorting_df.sort_values(
                    by=column,
                    ascending=ascending,
                    na_position='last'
                ).index
            
            # Use the sorted index on the original DataFrame
            sorted_df = df.loc[sorted_index]
            
            # Create new model with sorted DataFrame
            new_model = PandasModel(sorted_df)
            
            # Preserve any bold formatting from the old model
            if hasattr(self.model(), 'bold_cells'):
                new_model.bold_cells = self.model().bold_cells
            
            # Set the new model
            self.setModel(new_model)
            
            # Adjust column widths
            for col in range(new_model.columnCount(None)):
                self.resizeColumnToContents(col)
                
        except Exception as e:
            pass

    def save_table_as(self, file_type):
        """Save the table data as either CSV or Excel file."""
        if not self.model():
            return
            
        df = self.model()._data
        
        # Get table name for the file dialog title
        if self in self.parent.data_table:
            table_name = "Statistics"
        elif self == self.parent.network_table:
            table_name = "Network"
        else:
            table_name = "Selection"
        
        # Get save file name
        file_filter = ("CSV Files (*.csv)" if file_type == 'csv' else 
                      "Excel Files (*.xlsx)" if file_type == 'xlsx' else 
                      "Gephi Graph (*.gexf)" if file_type == 'gexf' else
                      "GraphML (*.graphml)" if file_type == 'graphml' else
                      "Pajek Network (*.net)")

        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {table_name} Table As",
            "",
            file_filter
        )

        if filename:
            try:
                if file_type == 'csv':
                    # If user didn't type extension, add .csv
                    if not filename.endswith('.csv'):
                        filename += '.csv'
                    df.to_csv(filename, index=False)
                elif file_type == 'xlsx':
                    # If user didn't type extension, add .xlsx
                    if not filename.endswith('.xlsx'):
                        filename += '.xlsx'
                    df.to_excel(filename, index=False)
                elif file_type == 'gexf':
                    # If user didn't type extension, add .gexf
                    if not filename.endswith('.gexf'):
                        filename += '.gexf'
                    #for node in my_network.network.nodes():
                        #my_network.network.nodes[node]['label'] = str(node)
                    nx.write_gexf(my_network.network, filename, encoding='utf-8', prettyprint=True)
                elif file_type == 'graphml':
                    # If user didn't type extension, add .graphml
                    if not filename.endswith('.graphml'):
                        filename += '.graphml'
                    nx.write_graphml(my_network.network, filename)
                elif file_type == 'net':
                    # If user didn't type extension, add .net
                    if not filename.endswith('.net'):
                        filename += '.net'
                    nx.write_pajek(my_network.network, filename)
                    
                QMessageBox.information(
                    self,
                    "Success",
                    f"{table_name} table successfully saved to {filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save file: {str(e)}"
                )

    def set_selection_to_active(self):
        """Set selection table to the active one"""

        try:

            # Confirm swap
            reply = QMessageBox.question(
                self,
                'Set Network',
                f'Are you sure you want to set the Selected Network as the Main Network? (Recommend Saving the Main Network first)',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )


            if reply == QMessageBox.StandardButton.Yes:

                df = self.model()._data
                old = self.parent.network_table.model()._data

                new_lists = [list(df.iloc[:, 0]), list(df.iloc[:, 1]), list(df.iloc[:, 2])]
                my_network.network_lists = new_lists
                self.parent.network_graph_widget.set_graph(my_network.network)

                model = PandasModel(my_network.network_lists)
                self.parent.network_table.setModel(model)
                # Adjust column widths to content
                for column in range(model.columnCount(None)):
                    self.parent.network_table.resizeColumnToContents(column)

                #move old model to selection
                new_lists = [list(old.iloc[:, 0]), list(old.iloc[:, 1]), list(old.iloc[:, 2])]
                temp_network = n3d.Network_3D()
                temp_network.network_lists = new_lists
                model = PandasModel(new_lists)
                self.parent.selection_table.setModel(model)
                for column in range(model.columnCount(None)):
                    self.parent.selection_table.resizeColumnToContents(column)
                self.parent.selection_graph_widget.set_graph(temp_network.network)

        except Exception as e:
            print(f"Error setting new network: {e}")

    def close_all(self):

        self.parent.tabbed_data.clear_all_tabs()

    def handle_find_action(self, row, column, value):
        """Handle the Find action for bottom tables."""
        try:

            if type(column) is not list: #If highlighting one element
                value = int(value)
                
                # Get the currently active table
                active_table = self.parent.active_table
                
                # Determine if we're looking for a node or edge based on column
                if column < 2:  # First two columns are nodes

                    if my_network.node_centroids is None:
                        self.parent.show_centroid_dialog()

                    if value in my_network.node_centroids:
                        # Get centroid coordinates (Z, Y, X)
                        centroid = my_network.node_centroids[value]
                        # Set the active channel to nodes (0)
                        self.parent.set_active_channel(0)
                        # Toggle on the nodes channel if it's not already visible
                        if not self.parent.channel_visible[0]:
                            self.parent.channel_buttons[0].setChecked(True)
                            self.parent.toggle_channel(0)
                        # Navigate to the Z-slice
                        self.parent.slice_slider.setValue(int(centroid[0]))
                        print(f"Found node {value} at Z-slice {centroid[0]}")
                        if self.parent.channel_data[0].shape[0] * self.parent.channel_data[0].shape[1] * self.parent.channel_data[0].shape[2] > self.parent.mini_thresh:
                            self.parent.mini_overlay = True
                            self.parent.create_mini_overlay(node_indices = [value])
                        else:
                            self.parent.create_highlight_overlay(node_indices=[value])
                        self.parent.clicked_values['nodes'] = []
                        self.parent.clicked_values['edges'] = []
                        self.parent.clicked_values['nodes'].append(value)
                        
                        try:
                            # Highlight the value in both tables if it exists
                            self.highlight_value_in_table(self.parent.network_table, value, column)
                            self.highlight_value_in_table(self.parent.selection_table, value, column)
                        except:
                            pass
                    else:
                        print(f"Node {value} not found in centroids dictionary")
                        
                elif column == 2:  # Third column is edges
                    if my_network.edge_centroids is None:
                        self.parent.show_centroid_dialog()

                    if value in my_network.edge_centroids:

                        # Get centroid coordinates (Z, Y, X)
                        centroid = my_network.edge_centroids[value]
                        # Set the active channel to edges (1)
                        self.parent.set_active_channel(1)
                        # Toggle on the edges channel if it's not already visible
                        if not self.parent.channel_visible[1]:
                            self.parent.channel_buttons[1].setChecked(True)
                            self.parent.toggle_channel(1)
                        # Navigate to the Z-slice
                        self.parent.slice_slider.setValue(int(centroid[0]))
                        print(f"Found edge {value} at Z-slice {centroid[0]}")
                        if self.parent.channel_data[1].shape[0] * self.parent.channel_data[1].shape[1] * self.parent.channel_data[1].shape[2] > self.parent.mini_thresh:
                            self.parent.mini_overlay = True
                            self.parent.create_mini_overlay(edge_indices = [value])
                        else:
                            self.parent.create_highlight_overlay(edge_indices=[value])
                        self.parent.clicked_values['nodes'] = []
                        self.parent.clicked_values['edges'] = []
                        self.parent.clicked_values['edges'].append(value)

                        try:
                            # Highlight the value in both tables if it exists
                            self.highlight_value_in_table(self.parent.network_table, value, column)
                            self.highlight_value_in_table(self.parent.selection_table, value, column)
                        except:
                            pass
                    else:
                        print(f"Edge {value} not found in centroids dictionary")
            else: #If highlighting paired elements
                if my_network.node_centroids is None:
                    self.parent.show_centroid_dialog()
                centroid1 = my_network.node_centroids[int(value[0])]
                centroid2 = my_network.node_centroids[int(value[1])]
                try:
                    centroid3 = my_network.edge_centroids[int(value[3])]
                except:
                    pass

                # Set the active channel to nodes (0)
                self.parent.set_active_channel(0)
                # Toggle on the nodes channel if it's not already visible
                if not self.parent.channel_visible[0]:
                    self.parent.channel_buttons[0].setChecked(True)
                    self.parent.toggle_channel(0)
                # Navigate to the Z-slice
                self.parent.slice_slider.setValue(int(centroid1[0]))
                print(f"Found node pair {value[0]} and {value[1]} at Z-slices {centroid1[0]} and {centroid2[0]}, respectively")
                try:
                    if self.parent.channel_data[0].shape[0] * self.parent.channel_data[0].shape[1] * self.parent.channel_data[0].shape[2] > self.parent.mini_thresh:
                        self.parent.mini_overlay = True
                        self.parent.create_mini_overlay(node_indices=[int(value[0]), int(value[1])], edge_indices = int(value[2]))
                    else:
                        self.parent.create_highlight_overlay(node_indices=[int(value[0]), int(value[1])], edge_indices = int(value[2]))
                    self.parent.clicked_values['nodes'] = []
                    self.parent.clicked_values['edges'] = []
                    self.parent.clicked_values['edges'].append(value[2])
                    self.parent.clicked_values['nodes'].append(value[0])
                    self.parent.clicked_values['nodes'].append(value[1])
                except:
                    if self.parent.channel_data[0].shape[0] * self.parent.channel_data[0].shape[1] * self.parent.channel_data[0].shape[2] > self.parent.mini_thresh:
                        self.parent.mini_overlay = True
                        self.parent.create_mini_overlay(node_indices=[int(value[0]), int(value[1])])
                    else:
                        self.parent.create_highlight_overlay(node_indices=[int(value[0]), int(value[1])])
                    self.parent.clicked_values['nodes'] = []
                    self.parent.clicked_values['edges'] = []
                    self.parent.clicked_values['nodes'].append(value[0])
                    self.parent.clicked_values['nodes'].append(value[1])

        except (ValueError, TypeError) as e:
            print(f"Error processing value: {str(e)}")
            return


    def highlight_value_in_table(self, table, value, column):
        """Helper method to find and highlight a value in a specific table."""
        if table.model() is None:
            return
            
        df = table.model()._data
        
        if column < 2:  # Node
            col1_matches = df[df.columns[0]] == value
            col2_matches = df[df.columns[1]] == value
            all_matches = col1_matches | col2_matches
        else:  # Edge
            all_matches = df[df.columns[2]] == value
        
        if all_matches.any():
            match_indices = all_matches[all_matches].index.tolist()
            row_idx = match_indices[0]
            
            # Only scroll and select if this is the active table
            if table == self.parent.active_table:
                # Create index and scroll to it
                model_index = table.model().index(row_idx, 0)
                table.scrollTo(model_index)
                
                # Select the row
                table.clearSelection()
                table.selectRow(row_idx)
                table.setCurrentIndex(model_index)
            
            # Update bold formatting
            table.model().set_bold_value(value, column < 2 and 0 or 1)


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        if data is None:
            # Create an empty DataFrame with default columns
            import pandas as pd
            data = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
        elif type(data) == list:
            data = self.lists_to_dataframe(data[0], data[1], data[2], column_names=['Node A', 'Node B', 'Edge C'])
        self._data = data
        self.bold_cells = set()
        self.highlighted_cells = set()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        elif role == Qt.ItemDataRole.FontRole:
            # Get the actual index from the DataFrame for this row
            df_index = self._data.index[index.row()]
            if (df_index, index.column()) in self.bold_cells or (index.row(), index.column()) in self.highlighted_cells:
                font = QFont()
                font.setBold(True)
                return font
        elif role == Qt.ItemDataRole.BackgroundRole:
            if (index.row(), index.column()) in self.highlighted_cells:
                return QColor(255, 255, 0, 70)  # Light yellow background
        return None

    def highlight_cell(self, row, col):
        """Highlight a specific cell"""
        self.highlighted_cells.clear()  # Clear previous highlights
        self.highlighted_cells.add((row, col))
        # Emit signal to refresh the view
        self.layoutChanged.emit()

    def set_bold_value(self, value, active_channel=0):
        """Set bold formatting for cells containing this value in relevant columns based on active channel"""
        # Clear previous bold cells
        self.bold_cells.clear()
        self.highlighted_cells.clear()  # Also clear highlighted cells
        
        if active_channel == 0:
            # For nodes, search first two columns
            for col in [0, 1]:
                matches = self._data.iloc[:, col] == value
                for idx in matches[matches].index:
                    self.bold_cells.add((idx, col))
        elif active_channel == 1:
            # For edges, only search third column
            matches = self._data.iloc[:, 2] == value
            for idx in matches[matches].index:
                self.bold_cells.add((idx, 2))
        
        # Emit signal to refresh the view
        self.layoutChanged.emit()

    @staticmethod
    def lists_to_dataframe(list1, list2, list3, column_names=['Column1', 'Column2', 'Column3']):
        """
        Convert three lists into a pandas DataFrame with specified column names.
        
        Parameters:
        list1, list2, list3: Lists of equal length
        column_names: List of column names (default provided)
        
        Returns:
        pandas.DataFrame: DataFrame with three columns
        """
        df = pd.DataFrame({
            column_names[0]: list1,
            column_names[1]: list2,
            column_names[2]: list3
        })
        return df

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None


# Tables related for the data tables:

class TabCornerWidget(QWidget):
    """Widget for the corner of the tab widget, can be used to add controls"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

class TabButton(QPushButton):
    """Custom close button for tabs"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.setText("×")
        self.setStyleSheet("""
            QPushButton {
                border: none;
                color: gray;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                color: red;
            }
        """)

class TabbedDataWidget(QTabWidget):
    """Widget that manages multiple data tables in tabs"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        
        # Store tables with their associated names
        self.tables = {}
        self.tabCloseRequested.connect(self.close_tab)
        
        # Set corner widget
        self.setCornerWidget(TabCornerWidget(self))
        
    def add_table(self, name, table_widget, switch_to=True):
        """Add a new table with the given name"""
        if name in self.tables:
            # If tab already exists, update its content
            old_table = self.tables[name]
            idx = self.indexOf(old_table)
            
            # Remove the old table reference from parent's data_table
            if self.parent_window and old_table in self.parent_window.data_table:
                self.parent_window.data_table.remove(old_table)
                
            self.removeTab(idx)
            
        # Create a new CustomTableView with is_top_table=True
        new_table = CustomTableView(self.parent_window, is_top_table=True)
        
        # If we received a model or table_widget, use its model
        if isinstance(table_widget, QAbstractTableModel):
            new_table.setModel(table_widget)
        elif isinstance(table_widget, QTableView):
            new_table.setModel(table_widget.model())
        
        self.tables[name] = new_table
        idx = self.addTab(new_table, name)
        
        if switch_to:
            self.setCurrentIndex(idx)
            
        # Update parent's data_table reference
        if self.parent_window:
            self.parent_window.data_table.append(new_table)
            
    def close_tab(self, index):
        """Close the tab at the given index"""
        widget = self.widget(index)
        # Find and remove the table name from our dictionary
        name_to_remove = None
        for name, table in self.tables.items():
            if table == widget:
                name_to_remove = name
                break
                
        if name_to_remove:
            del self.tables[name_to_remove]
            
        # Update parent's data_table reference by removing the widget
        if self.parent_window and widget in self.parent_window.data_table:
            self.parent_window.data_table.remove(widget)
            
        self.removeTab(index)
                
    def clear_all_tabs(self):
        """Remove all tabs"""
        while self.count() > 0:
            self.close_tab(0)
            
    def get_current_table(self):
        """Get the currently active table"""
        return self.currentWidget()


# IMAGE MENU RELATED

class PropertiesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Properties")
        self.setModal(False)
        
        layout = QFormLayout(self)

        self.xy_scale = QLineEdit(f"{my_network.xy_scale}")
        layout.addRow("xy_scale:", self.xy_scale)

        self.z_scale = QLineEdit(f"{my_network.z_scale}")
        layout.addRow("z_scale:", self.z_scale)

        layout.addRow("Note:", QLabel(f"The below properties reflect what properties are being held in RAM. \nDisabling their button will reset the property and clear them from RAM. \nEnabling their button when nothing was set beforehand will not do anything.\nPleaes use file -> load to load specific elements."))


        self.nodes = QPushButton("Nodes")
        self.nodes.setCheckable(True)
        self.nodes.setChecked(self.check_checked(my_network.nodes))
        layout.addRow("Nodes Status", self.nodes)

        self.edges = QPushButton("edges")
        self.edges.setCheckable(True)
        self.edges.setChecked(self.check_checked(my_network.edges))
        layout.addRow("Edges Status", self.edges)

        self.network_overlay = QPushButton("overlay 1")
        self.network_overlay.setCheckable(True)
        self.network_overlay.setChecked(self.check_checked(my_network.network_overlay))
        layout.addRow("Overlay 1 Status", self.network_overlay)

        self.id_overlay = QPushButton("overlay 2")
        self.id_overlay.setCheckable(True)
        self.id_overlay.setChecked(self.check_checked(my_network.id_overlay))
        layout.addRow("Overlay 2 Status", self.id_overlay)

        self.search_region = QPushButton("search region")
        self.search_region.setCheckable(True)
        self.search_region.setChecked(self.check_checked(my_network.search_region))
        layout.addRow("Node Search Region Status", self.search_region)

        self.network = QPushButton("Network")
        self.network.setCheckable(True)
        self.network.setChecked(self.check_checked(my_network.network))
        layout.addRow("Network Status", self.network)

        self.node_identities = QPushButton("Node Identities")
        self.node_identities.setCheckable(True)
        self.node_identities.setChecked(self.check_checked(my_network.node_identities))
        layout.addRow("Identities Status", self.node_identities)

        # Add Run button
        run_button = QPushButton("Enter (Erases Unchecked Properties)")
        run_button.clicked.connect(self.run_properties)
        layout.addWidget(run_button)

        self.report_button = QPushButton("Report Properties (Show in Top Right Tables)")
        self.report_button.clicked.connect(self.report)
        layout.addWidget(self.report_button)

    def check_checked(self, ques):

        if ques is None:
            return False
        else:
            return True


    def run_properties(self):

        try:
            
            # Get amount
            try:
                xy_scale = float(self.xy_scale.text()) if self.xy_scale.text() else 1
            except ValueError:
                xy_scale = 1

            try:
                z_scale = float(self.z_scale.text()) if self.z_scale.text() else 1
            except ValueError:
                z_scale = 1

            nodes = not self.nodes.isChecked()
            edges = not self.edges.isChecked()
            network_overlay = not self.network_overlay.isChecked()
            id_overlay = not self.id_overlay.isChecked()
            search_region = not self.search_region.isChecked()
            network = not self.network.isChecked()
            node_identities = not self.node_identities.isChecked()

            self.parent().reset(nodes = nodes, edges = edges, search_region = search_region, network_overlay = network_overlay, id_overlay = id_overlay, network = network, xy_scale = xy_scale, z_scale = z_scale, node_identities = node_identities)
            
            self.accept()

        except Exception as e:
            print(f"Error: {e}")

    def report(self):

        try:

            self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
        except:
            pass
        try:

            self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
        except:
            pass

        try:
            self.parent().format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
        except:
            pass
        try:
            self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'Community', 'Node Communities')
        except:
            pass



class BrightnessContrastDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Brightness/Contrast Controls")
        self.setModal(False)  # Allows interaction with main window while open
        
        layout = QVBoxLayout(self)
        
        # Create range sliders for each channel
        self.brightness_sliders = []
        self.min_inputs = []  # Store min value inputs
        self.max_inputs = []  # Store max value inputs
        
        for i in range(4):
            channel_widget = QWidget()
            channel_layout = QVBoxLayout(channel_widget)
            
            # Add label
            label = QLabel(f"Channel {i+1} Brightness/Contrast")
            channel_layout.addWidget(label)
            
            # Create slider control container
            slider_container = QWidget()
            slider_layout = QHBoxLayout(slider_container)
            slider_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create min value input
            min_input = QLineEdit()
            min_input.setFixedWidth(50)  # Make input fields compact
            min_input.setText("0")
            self.min_inputs.append(min_input)
            
            # Create range slider
            slider = QRangeSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(65535)
            slider.setValue((0, 65535))
            slider.setMinimumWidth(300)
            self.brightness_sliders.append(slider)
            
            # Create max value input
            max_input = QLineEdit()
            max_input.setFixedWidth(50)
            max_input.setText("65535")
            self.max_inputs.append(max_input)
            
            # Add all components to slider container
            slider_layout.addWidget(min_input)
            slider_layout.addWidget(slider, stretch=1)  # Give slider stretch priority
            slider_layout.addWidget(max_input)
            
            channel_layout.addWidget(slider_container)
            layout.addWidget(channel_widget)
            
            #debouncing
            self.debounce_timer = QTimer()
            self.debounce_timer.setSingleShot(True)
            self.debounce_timer.timeout.connect(self._apply_pending_updates)
            self.pending_updates = {}
            self.debounce_delay = 1  # 300ms delay

            # Connect signals
            slider.valueChanged.connect(lambda values, ch=i: self.on_slider_change(ch, values))
            min_input.editingFinished.connect(lambda ch=i: self.on_min_input_change(ch))
            max_input.editingFinished.connect(lambda ch=i: self.on_max_input_change(ch))
            
    def on_slider_change(self, channel, values):
        """Update text inputs when slider changes"""
        min_val, max_val = values
        self.min_inputs[channel].setText(str(min_val))
        self.max_inputs[channel].setText(str(max_val))
        
        # Store the pending update
        self.pending_updates[channel] = values
        
        # Restart the debounce timer
        self.debounce_timer.start(self.debounce_delay)
    
    def _apply_pending_updates(self):
        """Apply all pending brightness updates"""
        for channel, values in self.pending_updates.items():
            self.parent().update_brightness(channel, values)
        self.pending_updates.clear()
        
    def on_min_input_change(self, channel):
        """Handle changes to minimum value input"""
        try:
            min_val = self.parse_input_value(self.min_inputs[channel].text())
            current_min, current_max = self.brightness_sliders[channel].value()
            
            if min_val < 0:
                min_val = 0
            # Ensure min doesn't exceed max
            min_val = min(min_val, current_max - 1)
            
            # Update slider and text input
            self.brightness_sliders[channel].setValue((min_val, current_max))
            self.min_inputs[channel].setText(str(min_val))
            
        except ValueError:
            # Reset to current slider value if input is invalid
            current_min, _ = self.brightness_sliders[channel].value()
            self.min_inputs[channel].setText(str(current_min))
            
    def on_max_input_change(self, channel):
        """Handle changes to maximum value input"""
        try:
            max_val = self.parse_input_value(self.max_inputs[channel].text())
            current_min, current_max = self.brightness_sliders[channel].value()
            
            if max_val > 65535:
                max_val = 65535
            # Ensure max doesn't go below min
            max_val = max(max_val, current_min + 1)
            
            # Update slider and text input
            self.brightness_sliders[channel].setValue((current_min, max_val))
            self.max_inputs[channel].setText(str(max_val))
            
        except ValueError:
            # Reset to current slider value if input is invalid
            _, current_max = self.brightness_sliders[channel].value()
            self.max_inputs[channel].setText(str(current_max))
            
    def parse_input_value(self, text):
        """Parse and validate input value"""
        try:
            # Convert to float first to handle decimal inputs
            value = float(text)
            # Round to nearest integer
            value = int(round(value))
            # Clamp between 0 and 65535
            return max(0, min(65535, value))
        except ValueError:
            raise ValueError("Invalid input")

class ColorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Colors")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        # Store the combo boxes to access their values later
        self.color_combos = []
        
        # Create a dropdown for each channel
        for i in range(4):
            combo = QComboBox()
            # Add all color options from parent's color dictionary
            combo.addItems(self.parent().color_dictionary.keys())
            
            # Set current selection to match current color
            current_color = self.parent().base_colors[i]
            # Find the key for this color value in the dictionary
            current_key = [k for k, v in self.parent().color_dictionary.items() 
                         if v == current_color][0]
            combo.setCurrentText(current_key)
            
            # Add to layout with appropriate label
            layout.addRow(f"Channel {i+1} ({self.parent().channel_names[i]}):", combo)
            self.color_combos.append(combo)
        
        # Add Run button
        run_button = QPushButton("Apply Colors")
        run_button.clicked.connect(self.update_colors)
        layout.addWidget(run_button)

    def update_colors(self):
        """Update the colors in the parent class and refresh display"""
        # For each channel, check if color has changed
        for i, combo in enumerate(self.color_combos):
            new_color = self.parent().color_dictionary[combo.currentText()]
            if new_color != self.parent().base_colors[i]:
                self.parent().base_colors[i] = new_color
        
        # Update the display
        self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
        self.accept()

class ArbitraryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Arbitrary Selector")
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Form layout for inputs
        layout = QFormLayout()
        main_layout.addLayout(layout)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["nodes", "edges"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Type to select:", self.mode_selector)
        
        # Selection section
        excel_button = QPushButton("Import selection from spreadsheet (Col 1)")
        excel_button.clicked.connect(self.import_excel)
        layout.addWidget(excel_button)
        
        self.select = QLineEdit("")
        layout.addRow("Select the following? (Use this format - '1,2,3,4' etc:", self.select)
        
        # Deselection section
        deexcel_button = QPushButton("Import deselection from spreadsheet (Col 1)")
        deexcel_button.clicked.connect(self.import_deexcel)
        layout.addWidget(deexcel_button)
        
        self.deselect = QLineEdit("")
        layout.addRow("Deselect the following? (Use this format - '1,2,3,4' etc:", self.deselect)
        
        # Run button
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.process_selections)
        main_layout.addWidget(run_button)

    def import_excel(self):
        """Import selection from Excel/CSV and populate the select QLineEdit."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Spreadsheet Files (*.xlsx *.xls *.csv)"
        )
        
        if file_path:
            try:
                selection_list = self.read_selection_from_file(file_path)
                selection_string = ",".join(map(str, selection_list))
                self.select.setText(selection_string)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import: {str(e)}")
    
    def import_deexcel(self):
        """Import deselection from Excel/CSV and populate the deselect QLineEdit."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "Spreadsheet Files (*.xlsx *.xls *.csv)"
        )
        
        if file_path:
            try:
                deselection_list = self.read_selection_from_file(file_path)
                deselection_string = ",".join(map(str, deselection_list))
                self.deselect.setText(deselection_string)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import: {str(e)}")
    
    def read_selection_from_file(self, file_path):
        """Read selection IDs from Excel/CSV file and return as a list."""
        # Determine file type and read accordingly
        if file_path.lower().endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file_path, header=None)
        else:
            # Read Excel file
            df = pd.read_excel(file_path, header=None)
        
        # Check if first row looks like a header
        first_row = df.iloc[0]
        if all(isinstance(x, str) for x in first_row):
            # First row is likely a header, skip it
            values = df.iloc[1:, 0].dropna().tolist()
        else:
            # No header, use all rows
            values = df.iloc[:, 0].dropna().tolist()
        
        # Convert to integers when possible, keep floats when necessary
        processed_values = []
        for val in values:
            try:
                # Try to convert to int first
                processed_values.append(int(val))
            except ValueError:
                try:
                    # If int fails, try float
                    processed_values.append(float(val))
                except ValueError:
                    # Skip values that can't be converted to numbers
                    continue
        
        return processed_values

    def handle_find_action(self, mode, value):
        """Handle the Find action."""
                
        # Determine if we're looking for a node or edge 
        if mode == 0:

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()

            if value in my_network.node_centroids:
                # Get centroid coordinates (Z, Y, X)
                centroid = my_network.node_centroids[value]
                # Set the active channel to nodes (0)
                self.parent().set_active_channel(0)
                # Toggle on the nodes channel if it's not already visible
                if not self.parent().channel_visible[0]:
                    self.parent().channel_buttons[0].setChecked(True)
                    self.parent().toggle_channel(0)
                # Navigate to the Z-slice
                self.parent().slice_slider.setValue(int(centroid[0]))
                print(f"Found node {value} at Z-slice {centroid[0]}")
                
            else:
                print(f"Node {value} not found in centroids dictionary")
                
        else:  # edges
            if my_network.edge_centroids is None:
                self.parent().show_centroid_dialog()

            if value in my_network.edge_centroids:

                # Get centroid coordinates (Z, Y, X)
                centroid = my_network.edge_centroids[value]
                # Set the active channel to edges (1)
                self.parent().set_active_channel(1)
                # Toggle on the edges channel if it's not already visible
                if not self.parent().channel_visible[1]:
                    self.parent().channel_buttons[1].setChecked(True)
                    self.parent().toggle_channel(1)
                # Navigate to the Z-slice
                self.parent().slice_slider.setValue(int(centroid[0]))
                print(f"Found edge {value} at Z-slice {centroid[0]}")

            else:
                print(f"Edge {value} not found in centroids dictionary")
    
    def process_selections(self):
        """Process the selection and deselection inputs."""
        try:
            from ast import literal_eval
            # Get values from QLineEdit fields
            select_text = self.select.text()
            deselect_text = self.deselect.text()
            
            # Format text for literal_eval by adding brackets
            if select_text:
                select_list = literal_eval(f"[{select_text}]")
            else:
                select_list = []
                
            if deselect_text:
                deselect_list = literal_eval(f"[{deselect_text}]")
            else:
                deselect_list = []
            
            # Get the current mode
            mode = self.mode_selector.currentText()
            
            if mode == 'nodes':
                num = self.parent().channel_data[0].shape[0] * self.parent().channel_data[0].shape[1] * self.parent().channel_data[0].shape[2]
            else:
                num = self.parent().channel_data[1].shape[0] * self.parent().channel_data[1].shape[1] * self.parent().channel_data[1].shape[2]
            

            for item in deselect_list:
                try:
                    self.parent().clicked_values[mode].remove(item)
                except:
                    pass #Forgive mistakes

            select_list.reverse()

            self.parent().clicked_values[mode].extend(select_list)

            select_list.reverse()

            try:
                if mode == 'nodes':
                    self.handle_find_action(0, select_list[0])
                    self.parent().handle_info(sort = 'node')
                elif mode == 'edges':
                    self.handle_find_action(1, select_list[0])
                    self.parent().handle_info(sort = 'edge')
            except:
                pass

            self.parent().clicked_values[mode] = list(set(self.parent().clicked_values[mode]))

            if num > self.parent().mini_thresh:
                self.parent().mini_overlay = True
                self.parent().create_mini_overlay(node_indices = self.parent().clicked_values['nodes'], edge_indices = self.parent().clicked_values['edges'])
            else:
                self.parent().create_highlight_overlay(
                    node_indices=self.parent().clicked_values['nodes'], 
                    edge_indices=self.parent().clicked_values['edges']
                )


            
            # Close the dialog after processing
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing selections: {str(e)}")


class MergeNodeIdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Merge Node Identities From Folder")
        self.setModal(True)
        
        main_layout = QVBoxLayout(self)
        
        # Info label at top
        info_label = QLabel(
            "Note: You should prelabel [ie via cellpose or similar] or prewatershed your current node objects before doing this.\n"
            "(See Process → Image) This tool does not label them for you."
        )
        info_label.setWordWrap(True)
        #info_label.setStyleSheet("QLabel { background-color: #ffffcc; padding: 8px; border: 1px solid #ccccaa; }")
        main_layout.addWidget(info_label)
        
        # ===== GROUP 1: Step-out Parameters =====
        stepout_group = QGroupBox("Step-out Parameters")
        stepout_layout = QFormLayout()
        
        self.search = QLineEdit("")
        stepout_layout.addRow("Step-out distance:", self.search)
        stepout_layout.addRow("(Leave empty to skip dilation)", QLabel())
        
        self.xy_scale = QLineEdit(f"{my_network.xy_scale}")
        stepout_layout.addRow("xy_scale:", self.xy_scale)
        
        self.z_scale = QLineEdit(f"{my_network.z_scale}")
        stepout_layout.addRow("z_scale:", self.z_scale)
        
        self.search_mode = QComboBox()
        self.search_mode.addItems([
            "Standard", 
            "Fast (May be rougher - use for very large 3D images)"
        ])
        self.search_mode.setCurrentIndex(0)
        stepout_layout.addRow("Algorithm:", self.search_mode)
        
        stepout_group.setLayout(stepout_layout)
        main_layout.addWidget(stepout_group)
        
        # ===== GROUP 2: Binarization Strategy =====
        binarization_group = QGroupBox("Binarization Strategy")
        binarization_layout = QVBoxLayout()
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "Auto-Binarize (Otsu) / Presegmented", 
            "Manual (Interactive Thresholder)", 
            "Auto with ML Predictor (must load model first)",
            "Using Previous Thresholds (will prompt for spreadsheet)"
        ])
        self.mode_selector.setCurrentIndex(1)
        binarization_layout.addWidget(self.mode_selector)
        
        binarization_group.setLayout(binarization_layout)
        main_layout.addWidget(binarization_group)

        # ===== GROUP 3: Processing Options =====
        process_group = QGroupBox("Continue Processing Options")
        process_layout = QFormLayout()

        self.load_label = QLabel("Click below to load previous mean intensities spreadsheet; skipping initial computation:")
        self.load_label.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        process_layout.addWidget(self.load_label)
        self.load_dict = QPushButton("📁 Load Previous")
        self.load_dict.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        self.load_dict.setChecked(False)
        self.load_dict.clicked.connect(self.load_prev)
        process_layout.addWidget(self.load_dict)
        self.id_dicts = None
        self.all_list = None

        process_group.setLayout(process_layout)
        main_layout.addWidget(process_group)

        
        # ===== GROUP 4: ML Model Options =====
        ml_group = QGroupBox("ML Model Options")
        ml_layout = QVBoxLayout()
        
        self.train = QPushButton("Train ML Model")
        self.train.setCheckable(True)
        self.train.setChecked(False)
        train_label = QLabel("Enable to train/update a model during processing:")
        train_label.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        ml_layout.addWidget(train_label)
        ml_layout.addWidget(self.train)
        
        ml_layout.addSpacing(10)
        
        # DISTINCT: Load Model Button
        self.model_button = QPushButton("📁 Load ML Model")
        self.model_button.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        self.model_button.clicked.connect(self.get_model)
        ml_layout.addWidget(self.model_button)
        
        ml_group.setLayout(ml_layout)
        main_layout.addWidget(ml_group)
        
        # ===== OUTPUT OPTIONS =====
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()
        
        self.include = QPushButton("Include Negative IDs (mark nodes without signal as '-')")
        self.include.setCheckable(True)
        self.include.setChecked(False)
        output_layout.addWidget(self.include)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        main_layout.addStretch()
        
        # ===== DISTINCT: Run Button =====
        run_button = QPushButton("▶ Select Directory and Start Processing")
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #5CB85C;
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4CAF50;
            }
        """)
        run_button.clicked.connect(self.run)
        main_layout.addWidget(run_button)
        
        self.predictor = False
        
        # Set reasonable dialog size
        self.setMinimumWidth(500)

    def load_prev(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                f"Load Intensity Spreadsheet",
                "",
                "Spreadsheets (*.xlsx *.csv)"
            )
            # Read the CSV file
            if filename.endswith('csv'):
                df = pd.read_csv(filename)
            else:
                df = pd.read_excel(filename)
            # Get the headers (column names)
            headers = df.columns.tolist()
            del headers[0]
            nodes = df.iloc[:, 0].values  
            output = {}
            for col_idx in range(1, len(df.columns)):  # Changed variable name
                new_data = {}
                column_data = df.iloc[:, col_idx].values
                for row_idx in range(len(column_data)):  # Changed variable name
                    new_data[nodes[row_idx]] = column_data[row_idx]
                output[headers[col_idx - 1]] = new_data
            self.id_dicts = output
            self.all_list = headers
            self.load_dict.setText("Previous Data Loaded!")
            self.load_label.setText(f"Loaded: {filename}")

        except:
            return



    def wait_for_threshold_processing(self, nohigh = False, title = "", histo_dict = {}, img = "", automated = False):
        """
        Opens ThresholdWindow and waits for user to process the image.
        Returns True if completed, False if cancelled.
        The thresholded image will be available in the main window after completion.
        """
        # Create event loop to wait for user
        loop = QEventLoop()
        result = {'completed': False}
        
        # Create the threshold window
        thresh_window = ThresholdWindow(self.parent(), 4, nohigh = nohigh, title = title, has_model = self.predictor, automated = automated)
        
        # Connect signals
        def on_processing_complete():
            result['completed'] = True
            loop.quit()
            
        def on_processing_cancelled():
            result['completed'] = False
            loop.quit()
        
        thresh_window.processing_complete.connect(on_processing_complete)
        thresh_window.processing_cancelled.connect(on_processing_cancelled)
        
        # Show window and wait
        thresh_window.show()
        thresh_window.raise_()
        thresh_window.activateWindow()

        histo_dict[img] = [thresh_window.counts, thresh_window.bin_edges]
        
        # Block until user clicks "Apply Threshold & Continue" or "Cancel"
        loop.exec()

    
        # Clean up
        thresh_window.deleteLater()
        
        return result['completed']

    def run(self):
        try:

            search = float(self.search.text()) if self.search.text().strip() else 0
            xy_scale = float(self.xy_scale.text()) if self.xy_scale.text().strip() else 1
            z_scale = float(self.z_scale.text()) if self.z_scale.text().strip() else 1
            data = self.parent().channel_data[0]
            include = self.include.isChecked()
            umap = True
            search_mode = self.search_mode.currentIndex()
            train = self.train.isChecked()

            if search_mode == 0:
                fast_dil = False
            else:
                fast_dil = True
            
            if data is None:
                return

            automated = False

            if self.mode_selector.currentIndex() == 2:
                if not self.predictor:
                    print("Please load a model first (or train one if you don't have one)")
                    return
                else:
                    automated = True
                    self.parent().old_thresh = None
                    train = False # Don't let user make the model train itself
            
            if self.mode_selector.currentIndex() == 3:
                filename, _ = QFileDialog.getOpenFileName(
                    self,
                    f"Load Previous Thresholds (Formatted How this Program Saves Them)",
                    "",
                    "Spreadsheets (*.csv *.xlsx)"
                )

                if filename:

                    # Read the CSV file
                    df = pd.read_csv(filename)
                    
                    # Create dictionary using first column as keys
                    # and columns 2 and 3 as list values
                    self.parent().old_thresh = {
                        row[0]: [float(row[1]), float(row[2])]
                        for row in df.values
                    }
                    automated = True
                else:
                    return
                    

            dialog = QFileDialog(self)
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
            dialog.setOption(QFileDialog.Option.ReadOnly)
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setViewMode(QFileDialog.ViewMode.Detail)
            
            if dialog.exec() == QFileDialog.DialogCode.Accepted:
                selected_path = dialog.directory().absolutePath()
            else:
                return  # User cancelled directory selection
                
            if search > 0:
                data = sdl.smart_dilate(data, fast_dil=fast_dil, 
                                      use_dt_dil_amount=search, xy_scale=xy_scale, z_scale=z_scale)
            


            # Check if manual mode is selected
            if self.mode_selector.currentIndex() > 0:  # Manual mode

                if my_network.node_identities is None: # Prepare modular dict

                    my_network.node_identities = {}

                    nodes = list(np.unique(data))
                    if 0 in nodes:
                        del nodes[0]
                    for node in nodes:

                        my_network.node_identities[node] = [] # Assign to lists


                img_list = n3d.directory_info(selected_path)
                self.parent().load_channel(0, data, data = True)
                self.hide()
                self.parent().highlight_overlay = None


                real_list = []
                for img in img_list: # Filter out non images
                    if img.endswith('tiff') or img.endswith('.tif'):
                        real_list.append(img)
                img_list = real_list
                all_list = []

                for img_file in img_list:

                    if img_file.endswith('.tiff'):
                        base_name = img_file[:-5]
                    elif img_file.endswith('.tif'):
                        base_name = img_file[:-4]
                    else:
                        base_name = img_file
                    all_list.append(base_name)

                if not self.id_dicts:

                    id_dicts = my_network.get_merge_node_dictionaries(selected_path, data, img_list)
                    all_keys = list(id_dicts.values())[0].keys()
                    result = {key: np.array([d[key] for d in list(id_dicts.values())]) for key in all_keys}

                    print("Please save your identity table if desired for use with the violin plot and intensity communities function")
                    self.parent().format_for_upperright_table(result, 'NodeID', all_list, 'Mean Intensity (Save this Table for "Analyze -> Stats -> Show Violins")', save = True)

                else: # Preloaded condition
                    need_data = []
                    for img_file in img_list: # Add any new data

                        if img_file.endswith('.tiff'):
                            base_name = img_file[:-5]
                        elif img_file.endswith('.tif'):
                            base_name = img_file[:-4]
                        else:
                            base_name = img_file
                        if base_name not in self.id_dicts:
                            need_data.append(img_file)
                            #self.all_list.insert(0, base_name)
                    if need_data:
                        id_dicts = my_network.get_merge_node_dictionaries(selected_path, data, need_data)
                        id_dicts = id_dicts | self.id_dicts
                    else:
                        id_dicts = self.id_dicts
                    all_keys = list(id_dicts.values())[0].keys()
                    result = {key: np.array([d[key] for d in list(id_dicts.values())]) for key in all_keys}
                    if need_data:
                        print("Please save your identity table if desired for use with the violin plot and intensity communities function")
                        self.parent().format_for_upperright_table(result, 'NodeID', list(id_dicts.keys()), 'Mean Intensity (Save this Table for "Analyze -> Stats -> Show Violins")', save = True)


                histo_dict = {}
                thresh_dict = {}
                for i, img in enumerate(img_list):

                    print(f"Please threshold {img}")
                    self.parent().setWindowTitle(f"NetTracer3D: Please threshold {img}")

                    mask = tifffile.imread(f'{selected_path}/{img}')
                    if img.endswith('.tiff'):
                        base_name = img[:-5]
                    elif img.endswith('.tif'):
                        base_name = img[:-4]
                    else:
                        base_name = img
                    self.parent().load_channel(2, mask, data = True)

                    # Wait for user to threshold this data
                    self.parent().special_dict = id_dicts[base_name]
                    processing_completed = self.wait_for_threshold_processing(nohigh = True, title = img, histo_dict = histo_dict, img = img, automated = automated)
                    
                    if not processing_completed:
                        self.parent().thresh_min = None
                        self.parent().thresh_max = None
                        # User cancelled, ask if they want to continue
                        reply = QMessageBox.question(self, 'Continue?', 
                                                   f'Threshold cancelled for item {i+1}. Continue with remaining items?',
                                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        if reply == QMessageBox.StandardButton.No:
                            self.parent().old_thresh = None
                            break
                        continue
                    
                    thresh_dict[img] = [self.parent().thresh_min, self.parent().thresh_max]
                    # At this point, the thresholded image is in the main window's memory
                    # Get the processed/thresholded data from wherever ThresholdWindow stored it
                    thresholded_vals = self.parent().thresh_vals

                    assigned = {}
                    thresholded_set = set(thresholded_vals)

                    pos_marker = f'{base_name}+'
                    neg_marker = f'{base_name}-'

                    for node in my_network.node_identities.keys():
                        try:
                            node_int = int(node)
                        except (ValueError, TypeError):
                            continue
                        
                        node_ids = my_network.node_identities[node]
                        
                        # Remove if present
                        if pos_marker in node_ids:
                            node_ids.remove(pos_marker)
                        elif neg_marker in node_ids:
                            node_ids.remove(neg_marker)
                        
                        if node_int in thresholded_set:
                            node_ids.append(pos_marker)
                        elif include:
                            node_ids.append(neg_marker)
                    
                    # Process the thresholded data
                    self.parent().highlight_overlay = None


                my_network.node_identities = {node: iden for node, iden in my_network.node_identities.items() if iden != []}
                self.parent().old_thresh = None

                if my_network.node_identities == {}:
                    my_network.node_identities = None
                    return
                self.parent().setWindowTitle(f"NetTracer3D")

                self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity')
                self.parent().format_for_upperright_table(thresh_dict, 'Identity', ['Min Value', 'Max Value'], 'Threshold Information')


                if train:
                    try:
                        train_data = {}
                        for key, val in histo_dict.items():
                            if key in thresh_dict:
                                train_data[key] = val + [thresh_dict[key]]

                        filename, _ = QFileDialog.getSaveFileName(
                            self,
                            f"Save Model As",
                            "",  # Default directory
                            "PKL Files (*.pkl);;All Files (*)"  # File type filter
                        )
                        
                        if filename:  # Only proceed if user didn't cancel
                            # If user didn't type an extension, add .tif
                            if not filename.endswith('.pkl'):
                                filename += '.pkl'

                            if self.predictor:
                                self.predictor.train(
                                    training_data=train_data,
                                    epochs=100,
                                    from_scratch=False,  # Important: False means incremental!
                                    save_path=filename
                                )
                            else:
                                self.predictor = threshold_predictor.ThresholdPredictor(hidden_layers=(128, 64, 32))
                                self.predictor.train(
                                    training_data=train_data,
                                    epochs=100,              # Number of training iterations
                                    batch_size=4,            # Samples per batch
                                    learning_rate=0.001,     # Learning rate
                                    from_scratch=True,       # Start fresh (not incremental)
                                    save_path=filename
                                )
                    except Exception as e:
                        print(f"Error Training: {e}")


                QMessageBox.information(
                    self,
                    "Success",
                    "Node Identities Merged. New IDs represent presence of corresponding img foreground with +, absence with -. If desired, please save your new identities as csv, then use File -> Load -> Load From Excel Helper to bulk search and rename desired combinations. If desired, please save the outputted mean intensity table to use with 'Analyze -> Stats -> Show Violins'. (Press Help [above] for more info)"
                )


                """
                try:
                    self.parent().show_violin_dialog(called = True)
                    QMessageBox.information(
                        self,
                        "FYI",
                        "Here is the violin plot/intensity communities function control window for the aforementioned table. Feel free to close these windows if you do not desire to use this analysis, however you will need to reference the saved table to get back here."
                    )
                except:
                    pass
                """
                self.accept()
            else:
                my_network.merge_node_ids(selected_path, data, include)

                self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity')

                QMessageBox.information(
                    self,
                    "Success",
                    "Node Identities Merged. New IDs represent presence of corresponding img foreground with +, absence with -. Please save your new identities as csv, then use File -> Load -> Load From Excel Helper to bulk search and rename desired combinations. (Press Help [above] for more info)"
                )

                self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            #print(f"Error: {e}")

    def get_model(self):

        try:

            filename, _ = QFileDialog.getOpenFileName(
                self,
                f"Load Model",
                "",
                "Pickled_model (*.pkl)"
            )

            self.predictor = threshold_predictor.ThresholdPredictor.load(filename)
            self.model_button.setText("Load ML Model (Currently Has One Loaded!)")
        except:
            import traceback
            print(traceback.format_exc())
            self.model_button.setText("Load ML Model (For Auto-Predict)")
            self.predictor = False


class MultiChanDialog(QDialog):

    def __init__(self, parent=None, data = None):

        super().__init__(parent)
        self.setWindowTitle("Channel Loading")
        self.setModal(False)
        
        layout = QFormLayout(self)

        self.data = data

        self.nodes = QComboBox()
        self.edges = QComboBox()
        self.overlay1 = QComboBox()
        self.overlay2 = QComboBox()
        options = ["None"]
        for i in range(self.data.shape[0]):
            options.append(str(i))
        self.nodes.addItems(options)
        self.edges.addItems(options)
        self.overlay1.addItems(options)
        self.overlay2.addItems(options)
        self.nodes.setCurrentIndex(0)
        self.edges.setCurrentIndex(0)
        self.overlay1.setCurrentIndex(0)
        self.overlay2.setCurrentIndex(0)
        layout.addRow("Load this channel into nodes?", self.nodes)
        layout.addRow("Load this channel into edges?", self.edges)
        layout.addRow("Load this channel into overlay1?", self.overlay1)
        layout.addRow("Load this channel into overlay2?", self.overlay2)

        run_button = QPushButton("Load Channels")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

        run_button2 = QPushButton("Save Channels to Directory")
        run_button2.clicked.connect(self.run2)
        layout.addWidget(run_button2)


    def run(self):

        try:
            node_chan = int(self.nodes.currentText())
            self.parent().load_channel(0, self.data[node_chan, :, :, :], data = True)
        except:
            pass
        try:
            edge_chan = int(self.edges.currentText())
            self.parent().load_channel(1, self.data[edge_chan, :, :, :], data = True)
        except:
            pass
        try:
            overlay1_chan = int(self.overlay1.currentText())
            self.parent().load_channel(2, self.data[overlay1_chan, :, :, :], data = True)
        except:
            pass
        try:
            overlay2_chan = int(self.overlay2.currentText())
            self.parent().load_channel(3, self.data[overlay2_chan, :, :, :], data = True)
        except:
            pass

    def run2(self):

        try:
            # First let user select parent directory
            parent_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Location to Save Channels",
                "",
                QFileDialog.Option.ShowDirsOnly
            )

            for i in range(self.data.shape[0]):
                try:
                    tifffile.imwrite(f'{parent_dir}/C{i}.tif', self.data[i, :, :, :])
                except:
                    continue
        except:
            pass


class Show3dDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Parameters (Napari)")
        self.setModal(True)
        
        layout = QFormLayout(self)

        self.downsample = QLineEdit("")
        layout.addRow("Downsample Factor (Optional to speed up display):", self.downsample)

        # Network Overlay checkbox (default True)
        self.cubic = QPushButton("Cubic")
        self.cubic.setCheckable(True)
        self.cubic.setChecked(False)
        layout.addRow("Use cubic downsample (Slower but preserves visualization better potentially)?", self.cubic)

        self.box = QPushButton("Box")
        self.box.setCheckable(True)
        self.box.setChecked(False)
        layout.addRow("Include bounding box?", self.box)
        
        # Add Run button
        run_button = QPushButton("Show 3D")
        run_button.clicked.connect(self.show_3d)
        layout.addWidget(run_button)


    def show_3d(self):

        try:
            
            # Get amount
            try:
                downsample = float(self.downsample.text()) if self.downsample.text() else None
            except ValueError:
                downsample = None

            cubic = self.cubic.isChecked()
            box = self.box.isChecked()

            if cubic:
                order = 3
            else:
                order = 0

            arrays_3d = []
            arrays_4d = []

            color_template = ['red', 'green', 'white', 'cyan', 'yellow']  # color list
            colors = []


            for i, channel in enumerate(self.parent().channel_data):
                if channel is not None:

                    if len(channel.shape) == 3:
                        visible = self.parent().channel_buttons[i].isChecked()
                        if visible:
                            arrays_3d.append(channel)
                            colors.append(color_template[i])
                    elif len(channel.shape) == 4:
                        visible = self.parent().channel_buttons[i].isChecked()
                        if visible:
                            arrays_4d.append(channel)

            if self.parent().thresh_window_ref is not None:
                self.parent().thresh_window_ref.make_full_highlight()

            if self.parent().highlight_overlay is not None or self.parent().mini_overlay_data is not None:
                if self.parent().mini_overlay == True:
                    self.parent().create_highlight_overlay(node_indices = self.parent().clicked_values['nodes'], edge_indices = self.parent().clicked_values['edges'])
                arrays_3d.append(self.parent().highlight_overlay)
                colors.append(color_template[4])
        
            n3d.show_3d(arrays_3d, arrays_4d, down_factor = downsample, order = order, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale, colors = colors, box = box)
            
            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error showing 3D: {str(e)}\nNote: You may need to install napari first - in your environment, please call 'pip install napari'"
            )
            print(f"Error: {e}")
            import traceback
            print(traceback.format_exc())


class NetOverlayDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Generate Network Overlay?")
        self.setModal(True)

        layout = QFormLayout(self)

        self.downsample = QLineEdit("")
        layout.addRow("Downsample Factor While Drawing? (Int - Makes the outputted lines larger):", self.downsample)

        # Add Run button
        run_button = QPushButton("Generate (Will go to Overlay 1)")
        run_button.clicked.connect(self.netoverlay)
        layout.addWidget(run_button)

    def netoverlay(self):

        try:

            if my_network.node_centroids is None:

                self.parent().show_centroid_dialog()

            if my_network.node_centroids is None:
                return

            try:
                downsample = float(self.downsample.text()) if self.downsample.text() else None
            except ValueError:
                downsample = None

            my_network.network_overlay = my_network.draw_network(down_factor = downsample)

            if downsample is not None: 
                my_network.network_overlay = n3d.upsample_with_padding(my_network.network_overlay, original_shape = self.parent().shape)


            self.parent().load_channel(2, channel_data = my_network.network_overlay, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

            self.accept()

        except Exception as e:

            print(f"Error with Overlay Generation: {e}")


class IdOverlayDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Generate Node Number Overlay?")
        self.setModal(True)

        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        self.downsample = QLineEdit("")
        layout.addRow("Downsample Factor While Drawing? (Int - Makes the outputted numbers larger):", self.downsample)

        # Add Run button
        run_button = QPushButton("Generate (Will go to Overlay 2)")
        run_button.clicked.connect(self.idoverlay)
        layout.addWidget(run_button)

    def idoverlay(self):

        try:

            accepted_mode = self.mode_selector.currentIndex()

            try:
                downsample = float(self.downsample.text()) if self.downsample.text() else None
            except ValueError:
                downsample = None

            if accepted_mode == 0:

                if my_network.node_centroids is None:

                    self.parent().show_centroid_dialog()

                if my_network.node_centroids is None:
                    return

            elif accepted_mode == 1:

                if my_network.edge_centroids is None:

                    self.parent().show_centroid_dialog()

                if my_network.edge_centroids is None:
                    return

            if accepted_mode == 0:

                my_network.id_overlay = my_network.draw_node_indices(down_factor = downsample)

            elif accepted_mode == 1:

                my_network.id_overlay = my_network.draw_edge_indices(down_factor = downsample)

            if downsample is not None:
                my_network.id_overlay = n3d.upsample_with_padding(my_network.id_overlay, original_shape = self.parent().shape)

            self.parent().load_channel(3, channel_data = my_network.id_overlay, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

            self.accept()

        except:
            print(f"Error with Overlay Generation: {e}")


class ColorOverlayDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Generate Node (or Edge) -> Color Overlay?")
        self.setModal(True)

        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges"])
        if self.parent().active_channel == 0 and self.parent().channel_data[0] is not None:
            self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        else:
            self.mode_selector.setCurrentIndex(1)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        self.down_factor = QLineEdit("")
        layout.addRow("down_factor (int - for speeding up overlay generation - optional):", self.down_factor)

        # Add Run button
        run_button = QPushButton("Generate (Will go to Overlay 2)")
        run_button.clicked.connect(self.coloroverlay)
        layout.addWidget(run_button)

    def coloroverlay(self):

        try:

            down_factor = float(self.down_factor.text()) if self.down_factor.text().strip() else None

            mode = self.mode_selector.currentIndex()
            if mode == 0:
                self.sort = 'Node'
            else:
                self.sort = 'Edge'


            result, legend = my_network.node_to_color(down_factor = down_factor, mode = mode)

            self.parent().format_for_upperright_table(legend, f'{self.sort} Id', f'Encoding Val: {self.sort}', 'Legend')


            self.parent().load_channel(3, channel_data = result, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

            self.accept()

        except:
            pass


class ShuffleDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Shuffle Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        layout.addRow(QLabel("Swap: "))

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2", "Highlight Overlay"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Channel 1:", self.mode_selector)

        layout.addRow(QLabel("With: "))

        # Add mode selection dropdown
        self.target_selector = QComboBox()
        self.target_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2", 'Highlight Overlay'])
        self.target_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Channel 2:", self.target_selector)

        # Add Run button
        run_button = QPushButton("swap")
        run_button.clicked.connect(self.swap)
        layout.addWidget(run_button)

    def swap(self):

        try:

            accepted_mode = self.mode_selector.currentIndex()
            accepted_target = self.target_selector.currentIndex()

            if accepted_mode == 4:
                if self.parent().mini_overlay == True:
                    self.parent().create_highlight_overlay(node_indices = self.parent().clicked_values['nodes'], edge_indices = self.parent().clicked_values['edges'])
                active_data = self.parent().highlight_overlay
            else:
                active_data = self.parent().channel_data[accepted_mode]

            if accepted_target == 4:
                if self.parent().mini_overlay == True:
                    self.parent().create_highlight_overlay(node_indices = self.parent().clicked_values['nodes'], edge_indices = self.parent().clicked_values['edges'])
                target_data = self.parent().highlight_overlay
            else:
                target_data = self.parent().channel_data[accepted_target]


            try:
                if accepted_mode == 4:
                    try:
                        self.parent().highlight_overlay = n3d.binarize(target_data)
                    except:
                        self.parent().highlight_overay = None
                else:
                    self.parent().load_channel(accepted_mode, channel_data = target_data, data = True)
            except:
                pass


            try:
                if accepted_target == 4:
                    try:
                        self.parent().highlight_overlay = n3d.binarize(active_data)
                    except:
                        self.parent().highlight_overlay = None
                else:
                    self.parent().load_channel(accepted_target, channel_data = active_data, data = True)
            except:
                pass




            self.parent().update_display()

            self.accept()

        except Exception as e:
            print(f"Error swapping: {e}")









# ANALYZE MENU RELATED

class NetShowDialog(QDialog):
    def __init__(self, parent=None, called = None):
        super().__init__(parent)
        self.setWindowTitle("Display Parameters")
        self.setModal(True)
        
        main_layout = QVBoxLayout(self)

        self.called = called

        layout_group = QGroupBox("Layout")
        layout_layout = QFormLayout(layout_group)

        # Add mode selection dropdown
        self.render_mode = QComboBox()
        self.render_mode.addItems(["Spring Layout (Try to Logically Group Nodes)", "Centroid Layout (Place Nodes to Match Image)", "Component Layout Spring (Separate All Nontouching Components)", "Component Layout Shell (Centrally Places Important Nodes)"])
        self.render_mode.setCurrentIndex(0)  # Default to Mode 1
        layout_layout.addRow("Render Mode:", self.render_mode)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Default", "Community Coded", "Node ID Coded"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout_layout.addRow("Execution Mode:", self.mode_selector)

        main_layout.addWidget(layout_group)

        render_group = QGroupBox("Node/Edge Rendering")
        render_layout = QFormLayout(render_group)

        # Add mode selection dropdown
        self.edge_color = QComboBox()
        self.edge_color.addItems(["Translucent Gray", "Solid Black"])
        self.edge_color.setCurrentIndex(0)  # Default to Mode 1
        render_layout.addRow("Edge Color:", self.edge_color)

        self.node_size = QLineEdit("10")
        render_layout.addRow("Node Sizes", self.node_size)

        self.edge_size = QLineEdit("1")
        render_layout.addRow("Edge Sizes", self.edge_size)

        main_layout.addWidget(render_group)

        misc_group = QGroupBox("Misc")
        misc_layout = QFormLayout(misc_group)

        self.show_labels = QCheckBox("Show Node Numerical IDs?")
        self.show_labels.setChecked(True)
        misc_layout.addRow(self.show_labels)

        # weighted checkbox
        self.weighted = QCheckBox("Draw weighted edges (if applicable)")
        self.weighted.setChecked(True)
        misc_layout.addRow(self.weighted)

        # weighted checkbox (default True)
        self.z_size = QCheckBox("For Centroid Layout: Scale Node Sizes by Z?")
        self.z_size.setChecked(True)
        misc_layout.addRow(self.z_size)

        main_layout.addWidget(misc_group)

        
        # Add Run button
        run_button = QPushButton("Show Network")
        run_button.clicked.connect(self.show_network)
        main_layout.addWidget(run_button)
    
    def show_network(self):

        try:
            # Get parameters and run analysis

            show_labels = self.show_labels.isChecked()

            geo = False
            component = False
            shell = False

            render = self.render_mode.currentIndex()

            edge_color = self.edge_color.currentIndex()

            if render == 1:
                geo = True
            elif render == 2:
                component = True
            elif render == 3:
                shell = True
            if geo:
                if my_network.node_centroids is None:
                    self.parent().show_centroid_dialog()
            accepted_mode = self.mode_selector.currentIndex()  # Convert to 1-based index
            # Get directory (None if empty)
            directory = None

            try:
                node_size = min(abs(int(self.node_size.text())), 100)
            except:
                node_size = 10

            try:
                edge_size = min(abs(int(self.edge_size.text())), 100)
            except:
                edge_size = 1


            weighted = self.weighted.isChecked()
            z_size = self.z_size.isChecked()

            communities = False
            identities = False

            if accepted_mode == 1:
                communities = True
            elif accepted_mode == 2:
                identities = True

            if accepted_mode == 1:

                if my_network.communities is None:
                    self.parent().show_partition_dialog()
                    if my_network.communities is None:
                        return
            if edge_color == 0:
                black_edges = False
            else:
                black_edges = True

            if not self.called:
                # Create graph widgets
                temp_graph_widget = ngw.NetworkGraphWidget(
                    parent=self.parent(),
                    weight=weighted,
                    geometric=geo,
                    component = component,
                    centroids=my_network.node_centroids,
                    communities=communities,
                    community_dict=my_network.communities,
                    labels=show_labels,
                    identities = identities,
                    identity_dict = my_network.node_identities,
                    z_size = z_size,
                    shell = shell,
                    node_size = node_size,
                    black_edges = black_edges,
                    edge_size = edge_size
                )

                temp_graph_widget.set_graph(my_network.network)
                temp_graph_widget.show_in_window(title="Network Graph", width=1000, height=800)
                temp_graph_widget.load_graph()
                self.parent().temp_graph_widgets.append(temp_graph_widget)
                self.accept()
            else:
                self.called.weight, self.called.geometric, self.called.component, self.called.centroids, self.called.communities, self.called.community_dict, self.called.labels, self.called.identities, self.called.identity_dict, self.called.z_size, self.called.shell, self.called.node_size, self.called.black_edges, self.called.edge_size = edge_size = weighted, geo, component, my_network.node_centroids, communities, my_network.communities, show_labels, identities, my_network.node_identities, z_size, shell, node_size, black_edges, edge_size
                self.called._clear_graph()
                self.called.load_graph()
                self.accept()

        except Exception as e:
            print(f"Error: {e}")


class PartitionDialog(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Partition Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        # weighted checkbox (default True)
        self.weighted = QPushButton("weighted (Considers Duplicate Connections)")
        self.weighted.setCheckable(True)
        self.weighted.setChecked(True)
        layout.addRow("Use Weighted Network:", self.weighted)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Louvain", "Label Propogation", "Leiden"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # stats checkbox (default True)
        self.stats = QPushButton("Stats")
        self.stats.setCheckable(True)
        self.stats.setChecked(False)
        layout.addRow("Community Stats:", self.stats)

        self.seed = QLineEdit("")
        layout.addRow("Seed (int):", self.seed)

        # Add Run button
        run_button = QPushButton("Partition")
        run_button.clicked.connect(self.partition)
        layout.addWidget(run_button)

    def partition(self):

        self.parent().prev_coms = None

        accepted_mode = self.mode_selector.currentIndex()
        if accepted_mode == 0: #I switched where these are in the selection box
            accepted_mode = 1
        elif accepted_mode == 1:
            accepted_mode = 0
        if accepted_mode == 2:
            try:
                import igraph
                import leidenalg
            except:
                QMessageBox.critical(
                    self,
                    "Error:",
                    f"Leiden Clustering requires the igraph and leidenalg packages.\nPlease try 'pip install leidenalg' in your terminal."
                )

        weighted = self.weighted.isChecked()
        dostats = self.stats.isChecked()

        try:
            seed = int(self.seed.text()) if self.seed.text() else 42
        except:
            seed = None


        my_network.communities = None

        try:
            stats = my_network.community_partition(weighted = weighted, style = accepted_mode, dostats = dostats, seed = seed)
            #print(f"Discovered communities: {my_network.communities}")

            self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'CommunityID', title = 'Community Partition')

            if len(stats.keys()) > 0:
                self.parent().format_for_upperright_table(stats, title = 'Community Stats')

            self.accept()

        except Exception as e:
            print(f"Error creating communities: {e}")

class ComIdDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Select Mode")
        self.setModal(True)

        layout = QFormLayout(self)

        self.mode = QComboBox()
        self.mode.addItems(["Average Identities Per Community", "Weighted Average Identity of All Communities", ])
        self.mode.setCurrentIndex(0)
        layout.addRow("Mode", self.mode)

        # umap checkbox (default False)
        self.umap = QPushButton("UMAP")
        self.umap.setCheckable(True)
        self.umap.setChecked(False)
        layout.addRow("Generate UMAP?:", self.umap)

        # Heatmaps checkbox
        self.heatmaps = QPushButton("Heatmaps")
        self.heatmaps.setCheckable(True)
        self.heatmaps.setChecked(True)
        layout.addRow("Generate Heatmaps?:", self.heatmaps)

        self.style = QPushButton("Treat Multi-Identity Nodes as Unique Identities?")
        self.style.setChecked(False)
        self.style.setCheckable(True)
        layout.addRow("Heatmaps will be cluttered if too many permutations ^^:", self.style)

        self.label = QComboBox()
        self.label.addItems(["No Label", "By Community", "By Supercommunity (If already calculated via 'Analyze -> Network -> Convert Network Communities...')"])
        self.label.setCurrentIndex(0)
        layout.addRow("Label UMAP Points How?:", self.label)

        self.limit = QLineEdit("")
        layout.addRow("Min Community Size for UMAP (Smaller communities will be ignored in graph, does not apply if empty)", self.limit)

        # weighted checkbox (default True)
        self.proportional = QPushButton("Robust")
        self.proportional.setCheckable(True)
        self.proportional.setChecked(False)
        #layout.addRow("Return Node Type Distribution Robust UMAP (ie, communities will show how much they overrepresent a node type rather than just their proportional composition):", self.proportional)

        # Add Run button
        run_button = QPushButton("Get Community ID Info")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

    def run(self):

        try:

            if self.parent().prev_coms is not None:
                temp = my_network.communities
                my_network.communities = self.parent().prev_coms
            else:
                temp = None

            if my_network.node_identities is None:
                print("Node identities must be set")

            if my_network.communities is None:
                self.parent().show_partition_dialog()

                if my_network.communities is None:
                    return

            mode = self.mode.currentIndex()
            heatmaps = self.heatmaps.isChecked()
            umap = self.umap.isChecked()
            style = self.style.isChecked()

            label = self.label.currentIndex()

            proportional = self.proportional.isChecked()
            limit = int(self.limit.text()) if self.limit.text().strip() else 0


            if mode == 1:

                info = my_network.community_id_info()

                self.parent().format_for_upperright_table(info, 'Node Identity Type', 'Weighted Proportion in Communities', 'Weighted Average of Community Makeup')

            else:

                if not heatmaps: 
                    info, names = my_network.community_id_info_per_com(umap = umap, label = label, limit = limit, proportional = proportional, neighbors = temp, unique = style)
                    self.parent().format_for_upperright_table(info, 'Community', names, 'Average of Community Makeup')
                else:
                    if umap:
                        info, names = my_network.community_id_info_per_com(umap = umap, label = label, limit = limit, proportional = proportional, neighbors = temp, unique = style)
                    matrixes, id_set = my_network.community_heatmaps(unique = style)

                    for i, matrix in enumerate(matrixes):
                        self.parent().format_for_upperright_table(matrix, 'CommunityID', id_set, title = f'Community Heatmap {i + 1}')

            if self.parent().prev_coms is not None:
                my_network.communities = temp
 
            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error: {e}")



class ComNeighborDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Reassign Communities Based on Identity Similarity? (Note this alters communities outside of this function)")
        self.setModal(True)

        layout = QFormLayout(self)

        self.neighborcount = QLineEdit("")
        self.neighborcount.setPlaceholderText("Enter any number; KMeans Only. Empty = auto-predict (between 2 and 20)")
        layout.addRow("Num Supercommunities:", self.neighborcount)

        self.seed = QLineEdit("")
        layout.addRow("Clustering Seed:", self.seed)

        self.limit = QLineEdit("")
        layout.addRow("Min Community Size to be grouped (Smaller communities will be placed in supercommunity 0 - does not apply if empty)", self.limit)

        # weighted checkbox (default True)
        self.proportional = QPushButton("Robust")
        self.proportional.setCheckable(True)
        self.proportional.setChecked(True)
        #layout.addRow("Return Node Type Distribution Robust Heatmaps (ie, will give two more heatmaps that are not beholden to the total number of nodes of each type, representing which structures are overrepresented in a network):", self.proportional)

        self.heatmaps = QPushButton("Heatmaps")
        self.heatmaps.setCheckable(True)
        self.heatmaps.setChecked(True)
        layout.addRow("Return Heatmaps Showing Supercommunity Composition?:", self.heatmaps)

        self.style = QPushButton("Treat Multi-Identity Nodes as Unique Identities?")
        self.style.setChecked(False)
        self.style.setCheckable(True)
        layout.addRow("Heatmaps will be cluttered if too many permutations ^^:", self.style)

        self.mode = QComboBox()
        self.mode.addItems(["KMeans", "DBSCAN"])
        self.mode.setCurrentIndex(0)
        layout.addRow("Mode", self.mode)

        # Add Run button
        run_button = QPushButton("Get Supercommunities")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

    def run(self):

        try:

            if my_network.node_identities is None:
                print("Node identities must be set")

            if my_network.communities is None:
                self.parent().show_partition_dialog()

                if my_network.communities is None:
                    return

            mode = self.mode.currentIndex()

            seed = int(self.seed.text()) if self.seed.text().strip() else 42

            limit = int(self.limit.text()) if self.limit.text().strip() else None

            proportional = self.proportional.isChecked()

            neighborcount = int(self.neighborcount.text()) if self.neighborcount.text().strip() else None

            unique = self.style.isChecked()
            heatmaps = self.heatmaps.isChecked()

            if self.parent().prev_coms is None:

                self.parent().prev_coms = copy.deepcopy(my_network.communities)
                len_dict, matrixes, id_set = my_network.assign_neighborhoods(seed, neighborcount, limit = limit, proportional = proportional, mode = mode, unique = unique, heatmaps = heatmaps)
            else:
                len_dict, matrixes, id_set = my_network.assign_neighborhoods(seed, neighborcount, limit = limit, prev_coms = self.parent().prev_coms, proportional = proportional, mode = mode, unique = unique, heatmaps = heatmaps)

            if matrixes is not None:
                for i, matrix in enumerate(matrixes):
                    self.parent().format_for_upperright_table(matrix, 'SupercommunityID', id_set, title = f'Supercommunity Heatmap {i + 1}')


            self.parent().format_for_upperright_table(len_dict, 'SupercommunityID', ['Number of Communities', 'Proportion of Total Nodes'], title = 'Supercommunity Counts')
            self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'SupercommunityID', title = 'Supercommunity Partition')

            print("Supercommunities have been assigned to communities based on similarity")

            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error assigning supercommunities: {e}")

class ComCellDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Assign Communities Based on Proximity Within Cuboidal Cells?")
        self.setModal(True)

        layout = QFormLayout(self)

        self.size = QLineEdit("")
        layout.addRow("Cell Size:", self.size)

        self.xy_scale = QLineEdit(f"{my_network.xy_scale}")
        layout.addRow("xy scale:", self.xy_scale)

        self.z_scale = QLineEdit(f"{my_network.z_scale}")
        layout.addRow("z scale:", self.z_scale)

        # Add Run button
        run_button = QPushButton("Get Communities (Note this overwrites current communities - save your coms first)")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

    def run(self):

        try:

            self.parent().prev_coms = None

            size = float(self.size.text()) if self.size.text().strip() else None
            xy_scale = float(self.xy_scale.text()) if self.xy_scale.text().strip() else 1
            z_scale = float(self.z_scale.text()) if self.z_scale.text().strip() else 1

            if size is None:
                return

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()
            if my_network.node_centroids is None:
                return

            my_network.community_cells(size = size, xy_scale = xy_scale, z_scale = z_scale)

            self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'CommunityID')

            self.accept()

        except Exception as e:

            print(f"Error: {e}")


class NeighComDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Communities Based on Neighbor Identities")
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        # Description label
        desc = QLabel("Assigns communities based on kmeans clustering of each cell's neighbor distribution.\n"
                     "May first want to find proximity network for desired neighbor count/distance.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-style: italic;")
        main_layout.addWidget(desc)
        
        # === Clustering Parameters Group ===
        cluster_group = QGroupBox("Clustering Parameters")
        cluster_layout = QFormLayout()
        
        self.comcount = QLineEdit("")
        self.comcount.setPlaceholderText("Can pick any number. Empty = auto-predict (between 2 and 20)")
        cluster_layout.addRow("Num Communities:", self.comcount)
        
        self.seed = QLineEdit("")
        cluster_layout.addRow("Clustering Seed:", self.seed)
        
        self.limit = QLineEdit("")
        self.limit.setPlaceholderText("Empty = no minimum")
        cluster_layout.addRow("Min Neighbor Count:", self.limit)

        self.local = QCheckBox("Create Communities based on Inidividual Nodes' Identities Instead of Neighbors? (Does not use network; Needs many identities)")
        self.local.setChecked(False)
        cluster_layout.addRow(self.local)
        
        cluster_group.setLayout(cluster_layout)
        main_layout.addWidget(cluster_group)
        
        # === Quick Network Generation Group ===
        network_group = QGroupBox("Network Generation (Optional - More Proximity Network Options in Calculate Menu. Skip these to use current network.)")
        network_layout = QFormLayout()
        
        self.quicknet = QLineEdit("")
        self.quicknet.setPlaceholderText("e.g., 10")
        network_layout.addRow("N Nearest Neighbors:", self.quicknet)
        
        self.quicknet2 = QLineEdit("")
        self.quicknet2.setPlaceholderText("e.g., 50.0")
        network_layout.addRow("Max Distance (Uses scalings):", self.quicknet2)
        
        network_group.setLayout(network_layout)
        main_layout.addWidget(network_group)
        
        # === Visualization Options Group ===
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        
        self.umap = QPushButton("Generate UMAP")
        self.umap.setCheckable(True)
        self.umap.setChecked(False)
        viz_layout.addRow("UMAP:", self.umap)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Label UMAP By Community", "Label UMAP By Identity"])
        self.mode_selector.setCurrentIndex(0)
        viz_layout.addRow("Label UMAP How?", self.mode_selector)
        
        self.label = QPushButton("Label Points")
        self.label.setCheckable(True)
        self.label.setChecked(False)
        # Can add this row if needed: viz_layout.addRow("Label UMAP:", self.label)
        
        self.heatmaps = QPushButton("Generate Heatmaps")
        self.heatmaps.setCheckable(True)
        self.heatmaps.setChecked(False)
        viz_layout.addRow("Neighborhood Heatmaps (Slightly Different than actual communities that result):", self.heatmaps)
        
        viz_group.setLayout(viz_layout)
        main_layout.addWidget(viz_group)
        
        # === Style Option ===
        self.style = QPushButton("Treat Multi-Identity Nodes as Unique Identities")
        self.style.setCheckable(True)
        self.style.setChecked(False)
        main_layout.addWidget(self.style)
        
        # === Run Button ===
        run_button = QPushButton("Get Communities")
        run_button.clicked.connect(self.run)
        run_button.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        main_layout.addWidget(run_button)

    def run(self):

        try:

            try:
                comcount = int(self.comcount.text())
            except:
                comcount = None

            try:
                seed = int(self.seed.text())
            except:
                seed = 42

            try:
                limit = int(self.limit.text())
            except:
                limit = 0

            unique = self.style.isChecked()

            local = self.local.isChecked()


            try:
                quicknet = int(self.quicknet.text())
            except:
                quicknet = None
            try:
                quicknet2 = int(self.quicknet2.text())
            except:
                quicknet2 = None

            mode = self.mode_selector.currentIndex()

            if quicknet or quicknet2:
                if my_network.node_centroids is None:
                    self.parent().show_centroid_dialog()
                    if my_network.node_centroids is None:
                        return
                my_network.kd_network(distance = quicknet2, max_neighbors = quicknet)
            G = my_network.network
            nodes = list(G.nodes())
            umap = self.umap.isChecked()
            label = self.label.isChecked()
            heatmaps = self.heatmaps.isChecked()
            all_idens = my_network.node_identities.values()
            iden_set = set()
            for iden in all_idens:
                iden_set.update(iden)
            if unique:
                unique_set = list(set(str(iden) for iden in my_network.node_identities.values() if len(iden) > 1))
                iden_set = iden_set.union(unique_set)
                del unique_set
            iden_set = sorted(iden_set)
            iden_index = {}
            for i, iden in enumerate(iden_set): # Dict to assist indexing the identities to array
                iden_index[iden] = i

            kmean_dict = {} # Dict to hold arrays for clustering

            if local:
                try:
                    nodes = list(np.unique(my_network.nodes))
                    if 0 in nodes:
                        del nodes[0]
                except:
                    try:
                        nodes = list(my_network.node_centroids.keys())
                    except:
                        try:
                            nodes = list(my_network.node_identities.keys())
                        except:
                            return
                for node in nodes:
                    array = np.zeros(len(iden_set))
                    try:
                        for iden in my_network.node_identities[node]:
                            array[iden_index[iden]] = 1
                        kmean_dict[node] = array
                    except:
                        continue

            else:
                for node in nodes: 
                    neighbors = list(G.neighbors(node))
                    array = np.zeros(len(iden_set))
                    if len(neighbors) < limit:
                        continue
                    for neighbor in neighbors:
                        try:
                            iden = my_network.node_identities[neighbor]
                        except:
                            continue
                        if unique:
                            if len(iden) < 2:
                                array[iden_index[iden[0]]] += 1 # Add up totality of available neighbor identity counts and store in array
                            else:
                                array[iden_index[str(iden)]] += 1
                        else:
                            for sub_iden in iden:
                                array[iden_index[sub_iden]] += 1 # Check list for multi identities
                    kmean_dict[node] = array/len(neighbors)

            from . import neighborhoods # Do kmean clustering
            clusters = neighborhoods.cluster_arrays(kmean_dict, n_clusters=comcount, seed=seed) # This returns a list of lists where each is a community of nodes

            if heatmaps: # Graphs to characterize neighborhood specifically
                heatmap_dict = {}
                for i, cluster in enumerate(clusters):
                    cluster_avg = np.zeros(len(iden_set))
                    for node in cluster:
                        cluster_avg = cluster_avg + kmean_dict[node]
                    heatmap_dict[i + 1] = cluster_avg/len(cluster) # Just get average composition of neighborhood. Scale invariant because that's how the kmean cluster-er saw them in the first place. Nodes should have similar neighbor counts anyway.

            com_dict = {}
            for i, cluster in enumerate(clusters): #Unpack output
                com_dict[i + 1] = cluster
            my_network.communities = n3d.revert_dict(com_dict) # This just reverses the dict to be {node: com}
            old_to_new = my_network.com_by_size()

            if heatmaps:

                rearranged = {}
                for com, new_com in old_to_new.items():
                    rearranged[new_com] = heatmap_dict[com]
                heatmap_dict = rearranged
                del rearranged

                output, sorted_id_set = neighborhoods.plot_dict_heatmap(heatmap_dict, iden_set, title = "Neighborhood Heatmap by Proportional Composition of Nodes Neighboring Nodes in Community", sublabel = "Neighborhood of Community")

                try: # Image itself represents more accurate node counts
                    node_count = np.unique(my_network.nodes)
                    if 0 in node_count:
                        np.delete(node_count, 0)
                    node_count = len(node_count)
                except: # But if user is working with a pure network we can try referencing that instead. They should have centroids in this situation.
                    try:
                        node_count = len(list(my_network.node_centroids.keys()))
                    except: # No centroids, we can try the network itself
                        node_count = len(list(my_network.network.nodes()))

                if unique:
                    idens = n3d.invert_dict(my_network.node_identities)
                else:
                    idens = n3d.invert_dict_special(my_network.node_identities)
                iden_count = np.zeros(len(iden_set))

                for iden, nodes in idens.items(): # Indexed array counting total instance of all identities in image
                    iden_count[iden_index[iden]] = len(nodes)/node_count

                for neighborhood in heatmap_dict:
                    heatmap_dict[neighborhood] = heatmap_dict[neighborhood]/iden_count

                output2, sorted_id_set = neighborhoods.plot_dict_heatmap(heatmap_dict, iden_set, title = "Over/Underrepresentation of Node Identities per Neighborhood (val < 1 = underrepresented, val > 1 = overrepresented)", sublabel = "Neighborhood of Community", center_at_one = True)

                try:
                    self.parent().format_for_upperright_table(output, 'CommunityID', sorted_id_set, title = f'Community-Neighborhood Heatmap 1')
                    self.parent().format_for_upperright_table(output2, 'CommunityID', sorted_id_set, title = f'Community-Neighborhood Heatmap 3')
                except:
                    pass

            self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'Community', 'Node Communities')

            if umap:
                if mode == 1:
                    color_mode = 'identity'
                else:
                    color_mode = 'community'

                neighbor_classes = {}

                for item in kmean_dict.keys():
                    if item in my_network.node_identities:
                        idens = my_network.node_identities[item]
                        neighbor_classes[item] = random.choice(idens)

                umap = umw.UMAPGraphWidget(
                    parent=self.parent(),
                    labels=False,
                    color_mode=color_mode
                )

                self.parent().umap_widgets.append(umap)

                umap.set_data(
                    kmean_dict,
                    community_dict=my_network.communities,
                    identity_dict=neighbor_classes,
                    color_mode=color_mode
                )

                umap.show_in_window("UMAP Visualization")

            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error: {e}")


class UMAPDisplayDialog(QDialog):
    """Settings dialog for reconfiguring a live UMAP widget."""

    def __init__(self, umap_widget, parent_window=None):
        super().__init__(umap_widget)
        self.setWindowTitle("UMAP Display Parameters")
        self.setModal(True)
        self.setMinimumWidth(340)

        self._umap = umap_widget
        self._parent_window = parent_window

        # Identity planning state
        self._identity_panel_visible = False
        self._include_checks = {}   # {identity_label: QCheckBox}
        self._exclude_checks = {}   # {identity_label: QCheckBox}

        self._build_ui()

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        self._outer_layout = QHBoxLayout(self)
        self._outer_layout.setContentsMargins(6, 6, 6, 6)
        self._outer_layout.setSpacing(6)

        # ---- left side: main settings ----
        self._main_widget = QWidget()
        main_layout = QVBoxLayout(self._main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # -- Color mode --
        color_group = QGroupBox("Color Mode")
        color_layout = QFormLayout(color_group)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Colorless", "Community", "Identity"])
        # Set current index from widget state
        if self._umap.color_mode == 'community':
            self.mode_selector.setCurrentIndex(1)
        elif self._umap.color_mode == 'identity':
            self.mode_selector.setCurrentIndex(2)
        else:
            self.mode_selector.setCurrentIndex(0)
        color_layout.addRow("Execution Mode:", self.mode_selector)

        main_layout.addWidget(color_group)

        # -- Node rendering --
        render_group = QGroupBox("Node Rendering")
        render_layout = QFormLayout(render_group)

        self.node_size = QLineEdit(str(self._umap.node_size))
        #self.node_size = str(10)
        render_layout.addRow("Node Size:", self.node_size)

        main_layout.addWidget(render_group)

        # -- Misc --
        misc_group = QGroupBox("Misc")
        misc_layout = QFormLayout(misc_group)

        self.show_labels = QCheckBox("Show Node Labels")
        self.show_labels.setChecked(self._umap.labels)
        misc_layout.addRow(self.show_labels)

        main_layout.addWidget(misc_group)

        # -- Identity planning button --
        self.plan_identities_btn = QPushButton("Plan Identities ▶")
        self.plan_identities_btn.setToolTip(
            "Toggle panel to include/exclude specific identities")
        self.plan_identities_btn.clicked.connect(self._toggle_identity_panel)
        main_layout.addWidget(self.plan_identities_btn)

        # -- Apply button --
        run_button = QPushButton("Apply")
        run_button.clicked.connect(self._apply)
        main_layout.addWidget(run_button)

        main_layout.addStretch()
        self._outer_layout.addWidget(self._main_widget)

        # ---- right side: identity planning (hidden initially) ----
        self._identity_widget = QWidget()
        identity_layout = QVBoxLayout(self._identity_widget)
        identity_layout.setContentsMargins(0, 0, 0, 0)

        id_title = QLabel("Identity Planning")
        id_title.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 3px;")
        identity_layout.addWidget(id_title)

        columns_widget = QWidget()
        columns_layout = QHBoxLayout(columns_widget)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(8)

        # -- Include column --
        include_group = QGroupBox("Include Only (Overrides Exclude)")
        include_group.setToolTip(
            "Check identities to show ONLY these.\n"
            "Leave all unchecked to show everything (minus excludes).")
        self._include_layout = QVBoxLayout(include_group)
        self._include_layout.setContentsMargins(4, 4, 4, 4)
        self._include_layout.setSpacing(2)

        self._include_scroll = QScrollArea()
        self._include_scroll.setWidgetResizable(True)
        self._include_scroll.setMinimumWidth(140)
        self._include_scroll.setMinimumHeight(200)
        self._include_inner = QWidget()
        self._include_inner_layout = QVBoxLayout(self._include_inner)
        self._include_inner_layout.setContentsMargins(2, 2, 2, 2)
        self._include_inner_layout.setSpacing(1)
        self._include_inner_layout.addStretch()
        self._include_scroll.setWidget(self._include_inner)
        self._include_layout.addWidget(self._include_scroll)

        columns_layout.addWidget(include_group)

        # -- Exclude column --
        exclude_group = QGroupBox("Exclude")
        exclude_group.setToolTip(
            "Check identities to HIDE them.\n"
            "Excluded identities become 'Unassigned'.")
        self._exclude_layout = QVBoxLayout(exclude_group)
        self._exclude_layout.setContentsMargins(4, 4, 4, 4)
        self._exclude_layout.setSpacing(2)

        self._exclude_scroll = QScrollArea()
        self._exclude_scroll.setWidgetResizable(True)
        self._exclude_scroll.setMinimumWidth(140)
        self._exclude_scroll.setMinimumHeight(200)
        self._exclude_inner = QWidget()
        self._exclude_inner_layout = QVBoxLayout(self._exclude_inner)
        self._exclude_inner_layout.setContentsMargins(2, 2, 2, 2)
        self._exclude_inner_layout.setSpacing(1)
        self._exclude_inner_layout.addStretch()
        self._exclude_scroll.setWidget(self._exclude_inner)
        self._exclude_layout.addWidget(self._exclude_scroll)

        columns_layout.addWidget(exclude_group)

        identity_layout.addWidget(columns_widget, stretch=1)

        self._identity_widget.setVisible(False)
        self._outer_layout.addWidget(self._identity_widget)

    # -------------------------------------------------------- identity panel --
    def _toggle_identity_panel(self):
        self._identity_panel_visible = not self._identity_panel_visible

        if self._identity_panel_visible:
            self.plan_identities_btn.setText("Plan Identities ◀")
            self._populate_identity_lists()
            self._identity_widget.setVisible(True)
            # Widen the dialog to accommodate
            self.setMinimumWidth(660)
            self.resize(680, self.height())
        else:
            self.plan_identities_btn.setText("Plan Identities ▶")
            self._identity_widget.setVisible(False)
            self.setMinimumWidth(340)
            self.resize(340, self.height())

    def _populate_identity_lists(self):
        """Populate include/exclude checkboxes from available identities."""
        # Clear old checkboxes
        for cb in self._include_checks.values():
            cb.setParent(None)
        for cb in self._exclude_checks.values():
            cb.setParent(None)
        self._include_checks.clear()
        self._exclude_checks.clear()

        all_idens = list(my_network.node_identities.values())
        seen = set()
        for iden in all_idens:
            seen.update(iden)
        identity_labels = sorted(list(seen))

        identity_labels = self._get_identity_labels()

        for label in identity_labels:
            label_str = str(label)

            inc_cb = QCheckBox(label_str)
            self._include_checks[label] = inc_cb
            # Insert before the stretch
            self._include_inner_layout.insertWidget(
                self._include_inner_layout.count() - 1, inc_cb)

            exc_cb = QCheckBox(label_str)
            self._exclude_checks[label] = exc_cb
            self._exclude_inner_layout.insertWidget(
                self._exclude_inner_layout.count() - 1, exc_cb)

    def _get_identity_labels(self):
        """
        return a sorted list of unique identity labels.
        """
        if self._umap.identity_dict:
            try:
                return sorted(set(self._umap.identity_dict.values()), key=str)
            except TypeError:
                return sorted(set(str(v) for v in self._umap.identity_dict.values()))
        return []

    def _get_identity_filter(self):
        """
        Read the include/exclude checkboxes and return the filter sets.

        Returns
        -------
        included : set or None
            Set of identity labels the user explicitly included.
            None if no includes were checked (meaning "show all minus excludes").
        excluded : set
            Set of identity labels the user explicitly excluded.

        Semantics:
        - If included is not None  → only show those identities
          (include overrides exclude).
        - If included is None      → show everything EXCEPT excluded.
        """
        included = set()
        excluded = set()

        for label, cb in self._include_checks.items():
            if cb.isChecked():
                included.add(label)

        for label, cb in self._exclude_checks.items():
            if cb.isChecked():
                excluded.add(label)

        if len(included) == 0:
            included = None  # means "all"

        return included, excluded

    # ------------------------------------------------------------- apply ------
    def _apply(self):
        try:
            mode_idx = self.mode_selector.currentIndex()
            if mode_idx == 0:
                color_mode = 'colorless'
            elif mode_idx == 1:
                color_mode = 'community'
            else:
                color_mode = 'identity'

            # Parse node size
            try:
                new_node_size = min(abs(int(self.node_size.text())), 100)
            except:
                new_node_size = 10

            show_labels = self.show_labels.isChecked()

            # --- Handle identity filtering ---
            included, excluded = self._get_identity_filter()

            if color_mode == 'identity':

                filtered = {}
                for node, idens in my_network.node_identities.items():
                    if included is not None:
                        new_idens = []
                        for iden in included:
                            if iden in idens:
                                new_idens.append(iden)
                        if new_idens:
                            filtered[node] = random.choice(new_idens)
                    else:
                        new_idens = []
                        for iden in idens:
                            if iden not in excluded:
                                new_idens.append(iden)
                        if new_idens:
                            filtered[node] = random.choice(new_idens)
                self._umap.set_color_mode('identity', filtered)

            elif color_mode == 'community':
                if self._parent_window and hasattr(self._parent_window, 'request_umap_color_mode'):
                    self._parent_window.request_umap_color_mode('community', self._umap)
                else:
                    self._umap.set_color_mode('community')

            else:
                # Colorless — push an empty dict so everything goes grey
                self._umap.set_color_mode('colorless', {})

            # Push node size & labels
            self._umap.node_size = new_node_size
            self._umap.labels = show_labels

            # Re-render with new sizes if we have data
            if self._umap.rendered and self._umap.embedding is not None:
                self._umap._build_and_render()

            self.accept()

        except Exception as e:
            print(f"UMAP settings error: {e}")

class RadialDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Radial Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        self.distance = QLineEdit("50")
        layout.addRow("Bucket Distance for Searching For Node Neighbors (automatically scaled by xy and z scales):", self.distance)

        # Add Run button
        run_button = QPushButton("Get Radial Distribution")
        run_button.clicked.connect(self.radial)
        layout.addWidget(run_button)

    def radial(self):

        try:

            distance = float(self.distance.text()) if self.distance.text().strip() else 50

            directory = None

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()

            radial = my_network.radial_distribution(distance, directory = directory)

            self.parent().format_for_upperright_table(radial, 'Radial Distance From Any Node', 'Average Number of Neighboring Nodes', title = 'Radial Distribution Analysis')

            self.accept()

        except Exception as e:
            print(f"An error occurred: {e}")

class NetNearDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Finds the nodes the least number of steps in the network away from another group of nodes")
        self.setModal(False)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Identities group box (only if node_identities exists)
        identities_group = QGroupBox("Identities")
        identities_layout = QFormLayout(identities_group)
        
        if my_network.node_identities is not None:

            self.root = QComboBox()

            all_idens = list(my_network.node_identities.values())
            seen = set()
            for iden in all_idens:
                seen.update(iden)
            try:
                seen.remove('Edge')
            except:
                pass
            roots = list(seen)

            neighs = copy.copy(roots)
        else:
            roots = []

        roots.insert(0, "Highlighted Nodes")
        self.root.addItems(roots)  
        self.root.setCurrentIndex(0)
        identities_layout.addRow("Root Identity to Search for Neighbor's IDs?", self.root)
        
        self.targ = QComboBox()
        neighs.insert(0, "Highlighted Nodes")
        self.targ.addItems(neighs)  
        self.targ.setCurrentIndex(0)
        identities_layout.addRow("Neighbor Identities to Search For?", self.targ)

        main_layout.addWidget(identities_group)

        heatmap_group = QGroupBox("Optional Overlays")
        heatmap_layout = QFormLayout(heatmap_group)

        self.heatmap = QPushButton("Heatmap")
        self.heatmap.setChecked(False)
        self.heatmap.setCheckable(True)
        heatmap_layout.addRow("Generate Heatmap (Overlay2)?:", self.heatmap)

        self.shortpath = QPushButton("Shortest Paths")
        self.shortpath.setChecked(False)
        self.shortpath.setCheckable(True)
        heatmap_layout.addRow("Generate Shortest Path Overlay (Overlay1)?:", self.shortpath)

        main_layout.addWidget(heatmap_group)

        # Get Distribution group box - ENHANCED STYLING
        distribution_group = QGroupBox("Get Distribution")
        distribution_layout = QVBoxLayout(distribution_group)
        
        run_button = QPushButton("🔍 Get Nearest Network Neighbors")
        # Style for primary action - blue with larger font
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        run_button.clicked.connect(self.run)
        distribution_layout.addWidget(run_button)
        
        main_layout.addWidget(distribution_group)
        
        # Get All Averages group box - ENHANCED STYLING (only if node_identities exists)
        if my_network.node_identities is not None:
            averages_group = QGroupBox("Get All Averages")
            averages_layout = QVBoxLayout(averages_group)  # Changed to QVBoxLayout for better control
            
            # Add the button with full width
            run_button2 = QPushButton("📊 Get Average Nearest All ID Combinations (Returns Matrix Graph rather than Heatmap)")
            run_button2.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: 2px solid #45a049;
                    padding: 10px 16px;
                    font-size: 13px;
                    font-weight: normal;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                    border-color: #3d8b40;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            run_button2.clicked.connect(self.run2)
            averages_layout.addWidget(run_button2)
            
            main_layout.addWidget(averages_group)

    def run(self):

        root = self.root.currentIndex()
        targ = self.targ.currentIndex()
        heatmap = self.heatmap.isChecked()
        node_list = my_network.network.nodes()
        shortpath = self.shortpath.isChecked()
        missing = 0

        if root == 0:
            root = 'Highlighted'
            roots = self.parent().clicked_values['nodes']
        else:
            roots = []
            root = self.root.currentText()
            for node, idens in my_network.node_identities.items():
                if root in idens and node in node_list:
                    roots.append(node)
                elif node not in node_list:
                    missing += 1

        if targ == 0:
            targ = 'Highlighted'
            targs = self.parent().clicked_values['nodes']
        else:
            targs = []
            targ = self.targ.currentText()
            for node, idens in my_network.node_identities.items():
                if targ in idens and node in node_list:
                    targs.append(node)

        if roots == [] or targs == []:
            return

        if not shortpath:
            dist_dict = my_network.shortest_distances_to_targets(roots, targs)
        else:
            dist_dict, paths = my_network.shortest_distances_to_targets(roots, targs, return_path_edges = True)

            # Get the existing DataFrame from the model
            original_df = self.parent().network_table.model()._data
            
            # Create boolean mask
            mask = pd.Series([False] * len(original_df))
            
            for u, v in paths:
                # Check for both (u,v) and (v,u) orientations
                bridge_mask = (
                    ((original_df.iloc[:, 0] == u) & (original_df.iloc[:, 1] == v)) |
                    ((original_df.iloc[:, 0] == v) & (original_df.iloc[:, 1] == u))
                )
                mask |= bridge_mask

            # Filter the DataFrame to only include bridge connections
            filtered_df = original_df[mask].copy()
            old_clicked = copy.copy(self.parent().clicked_values)

            nodes = list(set(filtered_df.iloc[:, 0].to_list() + filtered_df.iloc[:, 1].to_list()))
            edges = filtered_df.iloc[:, 2].unique().tolist()
            self.parent().create_highlight_overlay(node_indices = nodes, edge_indices = edges)
            self.parent().load_channel(2, self.parent().highlight_overlay, True)
            self.parent().evaluate_mini(subgraph_push = True)


        if dist_dict == {}:
            return
        misses = len(roots) - len(dist_dict) # Which nodes can't find target neighbor

        avg = sum(dist_dict.values())/len(dist_dict)

        bonus_info = {f'Average Shortest Path from {root} nodes to {targ} nodes':avg, f'Number {root} in network that could not find target':misses, f'Number {root} not in network':missing}

        if heatmap:
            from . import neighborhoods
            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()
            pred_dict = my_network.shortest_distances_to_targets(list(my_network.network.nodes()), targs)
            pred_avg = sum(pred_dict.values())/len(pred_dict)
            bonus_info[f'Average Shortest Path any node to {targ} nodes'] = pred_avg
            node_intensity = {}
            node_intensity = {node:math.log(pred_avg/dist) for node, dist in dist_dict.items()}

            try:
                overlay = neighborhoods.create_node_heatmap(node_intensity, my_network.node_centroids, shape = self.parent().shape, is_3d=True, labeled_array = my_network.nodes)
                self.parent().load_channel(3, overlay, True)
            except:
                pass

        self.parent().format_for_upperright_table(dist_dict, 'Node', f'Shortest Path Length to {targ} Nodes', title = f'Shortest Path Length from {root} Nodes to {targ} Nodes')
        self.parent().format_for_upperright_table(bonus_info, 'Category', 'Value', f'Average Path Length from {root} Nodes to {targ} Nodes')

        self.accept()

    def run2(self):
        try:

            all_idens = list(my_network.node_identities.values())
            seen = set()
            for iden in all_idens:
                seen.update(iden)
            try:
                seen.remove('Edge')
            except:
                pass
            available = list(seen)
            node_list = my_network.network.nodes()

            output_dict = {}

            while len(available) > 0:
                root = available[0]
                roots = []
                for node, idens in my_network.node_identities.items():
                    if root in idens and node in node_list:
                        roots.append(node)

                for targ in available:
                    targs = []
                    for node, idens in my_network.node_identities.items():
                        if targ in idens and node in node_list:
                            targs.append(node)
                    if roots == [] or targs == []:
                        continue

                    dist_dict = my_network.shortest_distances_to_targets(roots, targs)
                    if dist_dict == {}:
                        continue

                    avg = sum(dist_dict.values())/len(dist_dict)

                    output_dict[f"Includes: {root} vs Neighbors: Includes {targ}"] = avg
                    if root != targ:
                        dist_dict = my_network.shortest_distances_to_targets(targs, roots)
                        avg = sum(dist_dict.values())/len(dist_dict)
                        output_dict[f"Includes: {targ} vs Neighbors: Includes {root}"] = avg

                del available[0]

            self.parent().format_for_upperright_table(output_dict, "ID Combo", f"Avg Shortest Path", title = "Average Distance to Nearest Neighbors for All ID Combos")

            from . import neighborhoods

            neighborhoods.create_neighbor_heatmap(output_dict, title = "Nearest Neighbor Network Matrix", subtitle = "Average shortest paths between node populations", y_label = "Num Steps")

            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error: {e}")


class NearNeighDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Nearest Neighborhood Averages (Using Centroids)")
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Identities group box (only if node_identities exists)
        identities_group = QGroupBox("Identities")
        identities_layout = QFormLayout(identities_group)
        
        if my_network.node_identities is not None:

            self.root = QComboBox()
            # Get unique identities <-- Commented this out because it is lag on a huge number of permutations. (The lag is adding to the QComboBox). Would need to reformat the architecture to use a lazy loading type of widget. Not sure how relevant these combos even are so i am just disabling this at the moment.
            #unique_idens = set(tuple(iden) for iden in my_network.node_identities.values())

            # Sort by length (number of elements)
            #sorted_idens = sorted(unique_idens, key=len)

            # Convert back to strings
            #roots = [str(list(iden)) for iden in sorted_idens]

            all_idens = list(my_network.node_identities.values())
            seen = set()
            for iden in all_idens:
                seen.update(iden)
            roots = list(seen)
            roots = ["['" + iden + "']" for iden in roots] # In the future can update downstream components to expect a pure string but would only minorly increase performance

            neighs = copy.copy(roots)

            roots.append("All (Excluding Targets)")
            self.root.addItems(roots)  
            self.root.setCurrentIndex(0)
            identities_layout.addRow("Root Identity to Search for Neighbor's IDs?", self.root)
            
            self.targ = QComboBox()
            neighs.append("All Others (Excluding Self)")
            self.targ.addItems(neighs)  
            self.targ.setCurrentIndex(0)
            identities_layout.addRow("Neighbor Identities to Search For?", self.targ)
        else:
            self.root = None
            self.targ = None

        self.num = QLineEdit("1")
        identities_layout.addRow("Number of Nearest Neighbors to Evaluate Per Node?:", self.num)

        self.centroids = QPushButton("Centroids")
        self.centroids.setCheckable(True)
        self.centroids.setChecked(True)
        identities_layout.addRow("Use Centroids? (Recommended for spheroids) Deselecting finds true nearest neighbors for mask but will be slower, and will only support a single nearest neighbor calculation for each root (rather than an avg)", self.centroids)

        main_layout.addWidget(identities_group)
        
        # Optional Heatmap group box
        heatmap_group = QGroupBox("Optional Heatmap")
        heatmap_layout = QFormLayout(heatmap_group)
        
        self.map = QPushButton("(If getting distribution): Generate Heatmap?")
        self.map.setCheckable(True)
        self.map.setChecked(False)
        heatmap_layout.addRow("Heatmap:", self.map)
        
        self.threed = QPushButton("(For above): Return 3D map? (uncheck for 2D): ")
        self.threed.setCheckable(True)
        self.threed.setChecked(True)
        heatmap_layout.addRow("3D:", self.threed)
        
        self.numpy = QPushButton("(For heatmap): Return image overlay instead of graph? (Goes in Overlay 2): ")
        self.numpy.setCheckable(True)
        self.numpy.setChecked(False)
        self.numpy.clicked.connect(self.toggle_map)
        heatmap_layout.addRow("Overlay:", self.numpy)

        self.mode = QComboBox()
        self.mode.addItems(["Anywhere", "Within Masked Bounds of Edges", "Within Masked Bounds of Overlay1", "Within Masked Bounds of Overlay2"])
        self.mode.setCurrentIndex(0)
        heatmap_layout.addRow("For heatmap, measure theoretical point distribution how?", self.mode)
        
        main_layout.addWidget(heatmap_group)

        quant_group = QGroupBox("Quantifiable Overlay")
        quant_layout = QFormLayout(quant_group)

        self.quant = QPushButton("Return quantifiable overlay? (Labels nodes by distance, good with intensity-thresholding to isolate targets. Requires labeled nodes image.)")
        self.quant.setCheckable(True)
        self.quant.setChecked(False)
        quant_layout.addRow("Overlay:", self.quant)

        main_layout.addWidget(quant_group)
        
        # Get Distribution group box - ENHANCED STYLING
        distribution_group = QGroupBox("Get Distribution")
        distribution_layout = QVBoxLayout(distribution_group)
        
        run_button = QPushButton("🔍 Get Average Nearest Neighbor (Plus Distribution)")
        # Style for primary action - blue with larger font
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        run_button.clicked.connect(self.run)
        distribution_layout.addWidget(run_button)
        
        main_layout.addWidget(distribution_group)
        
        # Get All Averages group box - ENHANCED STYLING (only if node_identities exists)
        if my_network.node_identities is not None:
            averages_group = QGroupBox("Get All Averages")
            averages_layout = QVBoxLayout(averages_group)  # Changed to QVBoxLayout for better control
            
            # Create a horizontal layout for the label and combobox
            mode_layout = QHBoxLayout()
            mode_label = QLabel("Include Multi-Identity Node Comparisons (Note - all multi-iden nodes contribute to the single identity instances in the output regardless)?")
            mode_label.setWordWrap(True)
            self.mode2 = QComboBox()
            self.mode2.addItems(["Include these", "Exclude these"])
            self.mode2.setCurrentIndex(1)
            self.mode2.setMinimumWidth(150)
            
            mode_layout.addWidget(mode_label, stretch=1)
            mode_layout.addWidget(self.mode2)
            averages_layout.addLayout(mode_layout)
            
            # Add some spacing
            averages_layout.addSpacing(10)
            
            # Add the button with full width
            run_button2 = QPushButton("📊 Get Average Nearest All ID Combinations (Returns Matrix Graph rather than Heatmap)")
            run_button2.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: 2px solid #45a049;
                    padding: 10px 16px;
                    font-size: 13px;
                    font-weight: normal;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                    border-color: #3d8b40;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            run_button2.clicked.connect(self.run2)
            averages_layout.addWidget(run_button2)
            
            main_layout.addWidget(averages_group)

    def toggle_map(self):
        if self.numpy.isChecked():
            if not self.map.isChecked():
                self.map.click()

    def run(self):
        try:
            try:
                root = self.root.currentText()
            except:
                root = None
            try:
                targ = self.targ.currentText()
            except:
                targ = None

            if root == "All (Excluding Targets)" and targ == 'All Others (Excluding Self)':
                root = None
                targ = None

            mode = self.mode.currentIndex()

            if mode == 0:
                mask = None
            else:
                try:
                    mask = self.parent().channel_data[mode] != 0
                except:
                    print("Could not binarize mask")
                    mask = None

            heatmap = self.map.isChecked()
            threed = self.threed.isChecked()
            numpy = self.numpy.isChecked()
            num = int(self.num.text()) if self.num.text().strip() else 1
            quant = self.quant.isChecked()
            centroids = self.centroids.isChecked()

            if not centroids:
                print("Using 1 nearest neighbor due to not using centroids")
                num = 1

            if root is not None and targ is not None:
                title = f"Nearest {num} Neighbor(s) Distance of includes: {targ} from includes: {root}"
                header = f"Avg Shortest Distance to Closest {num} {targ}(s)"
                header2 = f"Includes: {root} Node ID"
                header3 = f'Theoretical Uniform Distance to Closest {num} {targ}(s)'
            else:
                title = f"Nearest {num} Neighbor(s) Distance Between Nodes"
                header = f"Avg Shortest Distance to Closest {num} Nodes"
                header2 = "Root Node ID"
                header3 = f'Simulated Theoretical Uniform Distance to Closest {num} Nodes'

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()

            if not numpy:
                avg, output, quant_overlay, pred = my_network.nearest_neighbors_avg(root, targ, my_network.xy_scale, my_network.z_scale, num = num, heatmap = heatmap, threed = threed, quant = quant, centroids = centroids, mask = mask)
            else:
                avg, output, overlay, quant_overlay, pred = my_network.nearest_neighbors_avg(root, targ, my_network.xy_scale, my_network.z_scale, num = num, heatmap = heatmap, threed = threed, numpy = True, quant = quant, centroids = centroids, mask = mask)
                self.parent().load_channel(3, overlay, data = True)

            if quant_overlay is not None:
                self.parent().load_channel(2, quant_overlay, data = True)

            avg = {header:avg}

            if pred is not None:

                avg[header3] = pred

            
            self.parent().format_for_upperright_table(avg, 'Category', 'Value', title = f'Avg {title}')
            self.parent().format_for_upperright_table(output, header2, header, title = title)

            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error: {e}")

    def run2(self):
        try:
            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()
                if my_network.node_centroids is None:
                    return
            mode2 = self.mode2.currentIndex()
            if mode2 == 1:
                all_idens = list(my_network.node_identities.values())
                seen = set()
                for iden in all_idens:
                    seen.update(iden)
                available = list(seen)
            else:
                # Get unique identities
                unique_idens = set(tuple(iden) for iden in my_network.node_identities.values()) # This can stay because it's not tied to a widget

                # Sort by length (number of elements)
                sorted_idens = sorted(unique_idens, key=len)

                # Convert back to strings
                available = [str(list(iden)) for iden in sorted_idens]


            num = int(self.num.text()) if self.num.text().strip() else 1

            centroids = self.centroids.isChecked()
            if not centroids:
                num = 1

            output_dict = {}

            while len(available) > 0:
                root = available[0]

                for targ in available:
                    avg, _, _, _ = my_network.nearest_neighbors_avg(root, targ, my_network.xy_scale, my_network.z_scale, num = num, centroids = centroids)
                    if avg is not None:
                        output_dict[f"Includes: {root} vs Neighbors: Includes {targ}"] = avg
                    if root != targ:
                        avg, _, _, _ = my_network.nearest_neighbors_avg(targ, root, my_network.xy_scale, my_network.z_scale, num = num, centroids = centroids)
                        if avg is not None:
                            output_dict[f"Includes: {targ} vs Neighbors: Includes {root}"] = avg

                del available[0]

            self.parent().format_for_upperright_table(output_dict, "ID Combo", f"Avg Distance to Nearest {num}", title = "Average Distance to Nearest Neighbors for All ID Combos")

            from . import neighborhoods

            neighborhoods.create_neighbor_heatmap(output_dict)

            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error: {e}")


class NeighborIdentityDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle(f"Neighborhood Identity Distribution Parameters \n(Note - the same node is not included more than once as a neighbor even if it borders multiple nodes of the root ID)")
        self.setModal(True)

        layout = QFormLayout(self)

        if my_network.node_identities is not None:
            # Get unique identities <-- Commented this out because it is lag on a huge number of permutations. (The lag is adding to the QComboBox). Would need to reformat the architecture to use a lazy loading type of widget. Not sure how relevant these combos even are so i am just disabling this at the moment.
            #unique_idens = set(tuple(iden) for iden in my_network.node_identities.values())

            # Sort by length (number of elements)
            #sorted_idens = sorted(unique_idens, key=len)

            # Convert back to strings
            #all_idens = [str(list(iden)) for iden in sorted_idens]


            all_idens = list(my_network.node_identities.values())
            seen = set()
            for iden in all_idens:
                seen.update(iden)
            all_idens = list(seen)

            all_idens = ["['" + iden + "']" for iden in all_idens] # In the future can update downstream components to expect a pure string but would only minorly increase performance

            
            self.root = QComboBox()
            self.root.addItems(all_idens)  
            self.root.setCurrentIndex(0)
            layout.addRow("Root Identity to Search for Neighbor's IDs (search uses nodes of this ID, finds what IDs they connect to", self.root)
        else:
            self.root = None

        self.mode = QComboBox()
        self.mode.addItems(["From Network - Quantifies Neighbors Based on Adjacent Network Connections", "Use Labeled Nodes - Quantifies Neighbors Volume of Neighbor Within Search Region"])
        self.mode.setCurrentIndex(0)
        layout.addRow("Mode", self.mode)

        self.search = QLineEdit("")
        layout.addRow("Search Radius (Ignore if using network):", self.search)

        self.fastdil = QPushButton("Fast Dilate")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(True)
        layout.addRow("(If not using network) Use Fast Dilation (Parallelized):", self.fastdil)

        # Add Run button
        run_button = QPushButton("Get Neighborhood Identity Distribution")
        run_button.clicked.connect(self.neighborids)
        layout.addWidget(run_button)

    def neighborids(self):

        try:

            try:
                root = self.root.currentText()
            except:
                pass

            directory = None

            mode = self.mode.currentIndex()

            search = float(self.search.text()) if self.search.text().strip() else 0

            fastdil = self.fastdil.isChecked()


            result, result2, title1, title2, densities = my_network.neighborhood_identities(root = root, directory = directory, mode = mode, search = search, fastdil = fastdil)

            self.parent().format_for_upperright_table(result, 'Node Identity', 'Amount', title = title1)
            self.parent().format_for_upperright_table(result2, 'Node Identity', 'Proportion', title = title2)

            if mode == 1:

                self.parent().format_for_upperright_table(densities, 'Node Identity', 'Density in search/density total', title = f'Clustering Factor of Node Identities with {search} from nodes {root}')


            self.accept()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error: {e}")




class RipleyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Find Ripley's H Function From Centroids")
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Node Parameters Group (only if node_identities exist)
        if my_network.node_identities is not None:
            node_group = QGroupBox("Node Parameters")
            node_layout = QFormLayout(node_group)
            
            # Get unique identities <-- Commented this out because it is lag on a huge number of permutations. (The lag is adding to the QComboBox). Would need to reformat the architecture to use a lazy loading type of widget. Not sure how relevant these combos even are so i am just disabling this at the moment.
            #unique_idens = set(tuple(iden) for iden in my_network.node_identities.values())

            # Sort by length (number of elements)
            #sorted_idens = sorted(unique_idens, key=len)

            # Convert back to strings
            #all_idens = [str(list(iden)) for iden in sorted_idens]

            all_idens = list(my_network.node_identities.values())
            seen = set()
            for iden in all_idens:
                seen.update(iden)
            all_idens = list(seen)
            all_idens = ["['" + iden + "']" for iden in all_idens] # In the future can update downstream components to expect a pure string but would only minorly increase performance

            self.root = QComboBox()
            self.root.addItems(all_idens)  
            self.root.setCurrentIndex(0)
            node_layout.addRow("Root Identity to Search for Neighbors:", self.root)
            
            self.targ = QComboBox()
            self.targ.addItems(all_idens)  
            self.targ.setCurrentIndex(0)
            node_layout.addRow("Target Identity to be Searched For:", self.targ)
            
            main_layout.addWidget(node_group)
        else:
            self.root = None
            self.targ = None
        
        # Search Parameters Group
        search_group = QGroupBox("Search Parameters")
        search_layout = QFormLayout(search_group)
        
        self.distance = QLineEdit("5")
        search_layout.addRow("1. Bucket Distance for Searching For Clusters\n(automatically scaled by xy and z scales):", self.distance)
        
        self.proportion = QLineEdit("0.5")
        search_layout.addRow("2. Proportion of image to search?\n(0-1, high vals increase border artifacts):", self.proportion)
        
        main_layout.addWidget(search_group)
        
        # Border Safety Group
        border_group = QGroupBox("Border Safety")
        border_layout = QFormLayout(border_group)
        
        self.ignore = QPushButton("Ignore Border Roots")
        self.ignore.setCheckable(True)
        self.ignore.setChecked(True)
        border_layout.addRow("3. Exclude Root Nodes Near Borders?:", self.ignore)
        
        self.factor = QLineEdit("0.5")
        border_layout.addRow("4. (If param 3): Proportion of most internal nodes to use? (0 < n < 1) (Higher = more internal)?:", self.factor)
        
        self.mode = QComboBox()
        self.mode.addItems(["Boundaries of Entire Image", "Boundaries of Edge Image Mask", 
                           "Boundaries of Overlay1 Mask", "Boundaries of Overlay2 Mask"])
        self.mode.setCurrentIndex(0)
        border_layout.addRow("5. (If param 3): Define Boundaries How?:", self.mode)
        
        self.safe = QPushButton("Ignore Border Radii")
        self.safe.setCheckable(True)
        self.safe.setChecked(True)
        border_layout.addRow("6. (If param 3): Keep search radii within border (overrides Param 2, also assigns volume to that of mask)?:", self.safe)
        
        main_layout.addWidget(border_group)
        
        # Experimental Border Safety Group
        experimental_group = QGroupBox("Aggressive Border Safety (Creates duplicate centroids reflected across the image border - if you really need to search there for whatever reason - Not meant to be used if confining search to a masked object)")
        experimental_layout = QFormLayout(experimental_group)
        
        self.edgecorrect = QPushButton("Border Correction")
        self.edgecorrect.setCheckable(True)
        self.edgecorrect.setChecked(False)
        experimental_layout.addRow("7. Use Border Correction\n(Extrapolate for points beyond the border):", self.edgecorrect)
        
        main_layout.addWidget(experimental_group)
        
        # Add Run button
        run_button = QPushButton("Get Ripley's H")
        run_button.clicked.connect(self.ripley)
        main_layout.addWidget(run_button)

    def ripley(self):

        try:

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()

            try:
                root = self.root.currentText()
            except:
                root = None

            try:
                targ = self.targ.currentText()
            except:
                targ = None

            try:
                distance = float(self.distance.text())
            except:
                return


            try:
                proportion = abs(float(self.proportion.text()))
            except:
                proportion = 0.5

            try:
                factor = abs(float(self.factor.text()))

            except:
                factor = 0.25

            if factor > 1 or factor <= 0:
                print("Utilizing factor = 0.25")
                factor = 0.25

            if proportion > 1 or proportion <= 0:
                print("Utilizing proportion = 0.5")
                proportion = 0.5


            edgecorrect = self.edgecorrect.isChecked()

            ignore = self.ignore.isChecked()

            safe = self.safe.isChecked()

            mode = self.mode.currentIndex()

            if mode == 0:
                factor = factor/2 #The logic treats this as distance to border later, only if mode is 0, but its supposed to represent proportion internal.

            if my_network.nodes is not None:

                if my_network.nodes.shape[0] == 1:
                    bounds = (np.array([0, 0]), np.array([my_network.nodes.shape[2], my_network.nodes.shape[1]]))
                else:
                    bounds = (np.array([0, 0, 0]), np.array([my_network.nodes.shape[2], my_network.nodes.shape[1], my_network.nodes.shape[0]]))
            else:
                bounds = None

            r_vals, k_vals, h_vals = my_network.get_ripley(root, targ, distance, edgecorrect, bounds, ignore, proportion, mode, safe, factor)
            
            k_dict = dict(zip(r_vals, k_vals))
            h_dict = dict(zip(r_vals, h_vals))


            self.parent().format_for_upperright_table(k_dict, metric='Radius (scaled)', value='L Value', title="Ripley's K")
            self.parent().format_for_upperright_table(h_dict, metric='Radius (scaled)', value='L Normed', title="Ripley's H")


            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())

            QMessageBox.critical(
                self,
                "Error:",
                f"Failed to preform cluster analysis: {str(e)}"
            )

            print(f"Error: {e}")

class HeatmapDialog(QDialog):

    def __init__(self, parent = None):

        super().__init__(parent)
        self.setWindowTitle("Heatmap Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        self.nodecount = QLineEdit("")
        layout.addRow("(Optional) Total Number of Nodes?:", self.nodecount)


        # stats checkbox (default True)
        self.is3d = QPushButton("3D")
        self.is3d.setCheckable(True)
        self.is3d.setChecked(True)
        layout.addRow("Use 3D Plot (uncheck for 2D)?:", self.is3d)

        self.numpy = QPushButton("(For heatmap): Return image overlay instead of graph? (Goes in Overlay 2): ")
        self.numpy.setCheckable(True)
        self.numpy.setChecked(False)
        layout.addRow("Overlay:", self.numpy)


        # Add Run button
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

    def run(self):

        try:

            nodecount = int(self.nodecount.text()) if self.nodecount.text().strip() else None

            is3d = self.is3d.isChecked()


            if my_network.communities is None:
                if my_network.network is not None:
                    self.parent().show_partition_dialog()
                else:
                    self.parent().handle_com_cell()
                if my_network.communities is None:
                    return

            numpy = self.numpy.isChecked()

            if not numpy:

                heat_dict = my_network.community_heatmap(num_nodes = nodecount, is3d = is3d)

            else:

                heat_dict, overlay = my_network.community_heatmap(num_nodes = nodecount, is3d = is3d, numpy = True)
                self.parent().load_channel(3, overlay, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))


            self.parent().format_for_upperright_table(heat_dict, metric='Community', value='ln(Predicted Community Nodecount/Actual)', title="Community Heatmap")

            self.accept()

        except Exception as e:

            print(f"Error: {e}")







class RandomDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Random Parameters")
        self.setModal(True)

        layout = QFormLayout(self)


        # stats checkbox (default True)
        self.weighted = QPushButton("weighted")
        self.weighted.setCheckable(True)
        self.weighted.setChecked(True)
        layout.addRow("Allow Random Network to be weighted? (Whether or not edges can be repeatedly assigned between the same set of nodes to increase their weights, or if they must always find a new partner):", self.weighted)
        

        # Add Run button
        run_button = QPushButton("Get Random Network (Will go in Selection Table)")
        run_button.clicked.connect(self.random)
        layout.addWidget(run_button)

    def random(self):

        weighted = self.weighted.isChecked()

        _, df = my_network.assign_random(weighted = weighted)

        # Create new model with filtered DataFrame and update selection table
        self.parent().table_subgraph(self.parent().selection_table, df)

        # Switch to selection table
        self.parent().selection_button.click()

        self.accept()

class RandNodeDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Random Node Parameters")
        self.setModal(True)
        layout = QFormLayout(self)


        self.mode = QComboBox()
        self.mode.addItems(["Anywhere", "Within Dimensional Bounds of Nodes", "Within Masked Bounds of Edges", "Within Masked Bounds of Overlay1", "Within Masked Bounds of Overlay2"])
        self.mode.setCurrentIndex(0)
        layout.addRow("Mode", self.mode)

        # Add Run button
        run_button = QPushButton("Get Random Nodes (Will go in Nodes)")
        run_button.clicked.connect(self.random)
        layout.addWidget(run_button)

    def random(self):

        try:

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()

            bounds = None
            mask = None

            mode = self.mode.currentIndex()

            if mode == 0 and not (my_network.nodes is None and my_network.edges is None and my_network.network_overlay is None and my_network.id_overlay is None):
                pass
            elif mode == 1 or (my_network.nodes is None and my_network.edges is None and my_network.network_overlay is None and my_network.id_overlay is None):
                # Convert string labels to integers if necessary
                if any(isinstance(k, str) for k in my_network.node_centroids.keys()):
                    label_map = {label: idx for idx, label in enumerate(my_network.node_centroids.keys())}
                    my_network.node_centroids = {label_map[k]: v for k, v in my_network.node_centroids.items()}
                
                # Convert centroids to array and keep track of labels
                labels = np.array(list(my_network.node_centroids.keys()), dtype=np.uint32)
                centroid_points = np.array([my_network.node_centroids[label] for label in labels])
                
                # Calculate shape if not provided
                max_coords = centroid_points.max(axis=0)
                max_shape = tuple(max_coord for max_coord in max_coords)
                min_coords = centroid_points.min(axis=0)
                min_shape = tuple(min_coord for min_coord in min_coords)
                bounds = (min_shape, max_shape)
            else:
                mask = n3d.binarize(self.parent().channel_data[mode - 1])

            centroids, array = my_network.random_nodes(bounds = bounds, mask = mask)

            if my_network.nodes is not None:
                try:
                    self.parent().load_channel(0, array, data = True)
                except:
                    pass

            self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error:",
                f"Failed to randomize: {str(e)}"
            )
            print(f"Error: {e}")


class RadDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Obtain Radii of Active Image? (Returns Largest Radius for Each Labeled Object)")
        self.setModal(True)

        layout = QFormLayout(self)

        # GPU checkbox (default False)
        self.GPU = QPushButton("GPU")
        self.GPU.setCheckable(True)
        self.GPU.setChecked(False)
        #layout.addRow("Use GPU:", self.GPU)


        # Add Run button
        run_button = QPushButton("Calculate")
        run_button.clicked.connect(self.rads)
        layout.addWidget(run_button)

    def rads(self):

        try:
            GPU = self.GPU.isChecked() # <- I can never get these to be faster than parallel CPU *shrugs*

            active_data = self.parent().channel_data[self.parent().active_channel]

            radii = n3d.estimate_object_radii(active_data, gpu=False, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale)

            if self.parent().active_channel == 0:
                self.parent().radii_dict[0] = radii
            elif self.parent().active_channel == 1:
                self.parent().radii_dict[1] = radii
            elif self.parent().active_channel == 2:
                self.parent().radii_dict[2] = radii
            elif self.parent().active_channel == 3:
                self.parent().radii_dict[3] = radii

            self.parent().format_for_upperright_table(radii, title = 'Largest Radii of Objects', metric='ObjectID', value='Largest Radius (Scaled)')

            self.accept()

        except Exception as e:
            print(f"Error: {e}")


class InteractionDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Interaction Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        layout.addRow("Note:", QLabel(f"This is best done on original node/edge masks (nodes can be labeled first but edges will be significantly altered by labeling with Calculate All)\nConsider skeletonizing your edge mask first for increased standardization"))


        self.node_search = QLineEdit("0")
        layout.addRow("node_search:", self.node_search)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Include Regions Inside Node", "Exclude Regions Inside Node"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        self.length = QPushButton("Return Lengths")
        self.length.setCheckable(True)
        self.length.setChecked(False)
        layout.addRow("(Will Skeletonize the Edge Mirror and use that to calculate adjacent length of edges, as opposed to default volumes):", self.length)

        self.auto = QPushButton("Auto")
        self.auto.setCheckable(True)
        try:
            if self.parent().shape[0] == 1:
                self.auto.setChecked(False)
            else:
                self.auto.setChecked(True)
        except:
            self.auto.setChecked(False)
        layout.addRow("(If Above): Attempt to Auto Correct Skeleton Looping:", self.auto)

        self.fastdil = QPushButton("Fast Dilate")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(False)
        #layout.addRow("Use Fast Dilation (Higher speed, less accurate with search regions much larger than nodes):", self.fastdil)

        # Add Run button
        run_button = QPushButton("Calculate")
        run_button.clicked.connect(self.interaction)
        layout.addWidget(run_button)

    def interaction(self):

        try:

            accepted_mode = self.mode_selector.currentIndex()

            try:
                node_search = float(self.node_search.text()) if self.node_search.text() else 0
            except ValueError:
                node_search = 0
                

            fastdil = self.fastdil.isChecked()
            length = self.length.isChecked()
            auto = self.auto.isChecked()

            result = my_network.interactions(search = node_search, cores = accepted_mode, skele = length, length = length, auto = auto, fastdil = fastdil)

            if not length:
                self.parent().format_for_upperright_table(result, 'Node ID', ['Volume of Nearby Edge (Scaled)', 'Volume of Search Region (Scaled)'], title = 'Node/Edge Interactions')
            else:
                self.parent().format_for_upperright_table(result, 'Node ID', ['~Length of Nearby Edge (Scaled)', 'Volume of Search Region (Scaled)'], title = 'Node/Edge Interactions')


            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error finding interactions: {e}")


class ViolinDialog(QDialog):

    def __init__(self, parent=None, called = False):
        super().__init__(parent)
        if not called:
            QMessageBox.critical(
                self,
                "Notice",
                "Please select spreadsheet (Should be table output of 'File -> Images -> Node Identities -> Assign Node Identities from Overlap with Other Images'. Make sure to save that table as .csv/.xlsx and then load it here to use this.)"
            )
        try:
            if not called:
                try:
                    # Open file dialog
                    file_filter = "Spreadsheet Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)"
                    filename, _ = QFileDialog.getOpenFileName(
                        self,
                        "Load File",
                        "",
                        file_filter
                    )
                    
                    if not filename:
                        return
                    
                    # Read the file
                    if filename.endswith('.csv'):
                        self.df = pd.read_csv(filename)
                    elif filename.endswith('.xlsx'):
                        self.df = pd.read_excel(filename)
                    try: 
                        self.backup_df = copy.deepcopy(self.df)
                    except:
                        pass

                    try:
                        self.valid_nodes_list = list(np.unique(my_network.nodes))
                        if self.valid_nodes_list[0] == 0:
                            del self.valid_nodes_list[0]
                        print("Filtering out nodes that do not exist in current node image")                    
                        if self.valid_nodes_list:  # Only filter if list is not empty
                            first_col = self.df.columns[0]
                            self.df = self.df[self.df[first_col].isin(self.valid_nodes_list)]
                    except:
                        pass
                    
                except:
                    return

            try:
                # Get all identity lists and normalize the dataframe
                identity_lists = self.get_all_identity_lists()
                self.df = self.normalize_df_with_identity_centerpoints(self.df, identity_lists)
            except:
                pass
            self.setWindowTitle("Violin/Community Parameters")
            self.setModal(False)

            # ============================================================
            # Use an outer horizontal layout so we can attach the identity
            # planning panel on the right when the user expands it.
            # ============================================================
            self._outer_layout = QHBoxLayout(self)
            self._outer_layout.setContentsMargins(6, 6, 6, 6)
            self._outer_layout.setSpacing(6)

            # --- Left side: all existing controls ---
            self._main_widget = QWidget()
            layout = QFormLayout(self._main_widget)

            if my_network.node_identities is not None:
                self.idens = QComboBox()
                all_idens = list(my_network.node_identities.values())
                seen = set()
                for iden in all_idens:
                    seen.update(iden)
                idens = list(seen)
                idens.sort()
                idens.insert(0, "None")
                self.idens.addItems(idens)  
                self.idens.setCurrentIndex(0)
                layout.addRow("Return Identity Violin Plots?", self.idens)
            
            if my_network.communities is not None:
                self.coms = QComboBox()
                coms = list(set(my_network.communities.values()))
                coms.sort()
                coms.insert(0, "None")
                coms = [str(x) for x in coms]
                self.coms.addItems(coms)  
                self.coms.setCurrentIndex(0)
                layout.addRow("Return Community Violin Plots?", self.coms)
            
            # Add Run button
            run_button = QPushButton("Show Z-score-like Violin")
            run_button.clicked.connect(self.run)
            layout.addWidget(run_button)
            
            run_button2 = QPushButton("Show Z-score UMAP")
            run_button2.clicked.connect(self.run2)
            self.mode_selector = QComboBox()
            self.mode_selector.addItems(["Label UMAP By Identity", "Label UMAP By Community"])
            self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
            layout.addRow("Execution Mode:", self.mode_selector)
            layout.addRow(self.mode_selector, run_button2)

            # ============================================================
            # Identity Planning toggle button — sits between the upper
            # violin/UMAP controls and the lower clustering section.
            # ============================================================
            self._identity_panel_visible = False
            self._include_checks = {}
            self._exclude_checks = {}

            self.plan_identities_btn = QPushButton("Plan Identities ▶")
            self.plan_identities_btn.setToolTip(
                "Toggle panel to include/exclude specific identities for Violin and UMAP plots")
            self.plan_identities_btn.clicked.connect(self._toggle_identity_panel)
            layout.addWidget(self.plan_identities_btn)

            # Add separator to visually group the clustering options
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addRow(separator)
            
            # Clustering options section (visually grouped)
            clustering_label = QLabel("<b>Clustering Options (Based on Shared Channel Intensities):</b>")
            layout.addRow(clustering_label)
            
            # KMeans clustering
            run_button3 = QPushButton("Assign Communities via KMeans Clustering")
            run_button3.clicked.connect(self.run3)
            self.kmeans_num_input = QLineEdit()
            self.kmeans_num_input.setPlaceholderText("Auto (num communities)")
            self.kmeans_num_input.setMaximumWidth(150)
            from PyQt6.QtGui import QIntValidator
            self.kmeans_num_input.setValidator(QIntValidator(1, 1000))
            layout.addRow(run_button3, self.kmeans_num_input)

            self.mode_selector2 = QComboBox()
            self.mode_selector2.addItems(["Cluster Cells by Expression Profile", "Cluster Cells by Neighbor Expression Profile (Requires Network)"])
            self.mode_selector2.setCurrentIndex(0)  # Default to Mode 1
            layout.addRow("Execution Mode:", self.mode_selector2)
            
            # Reassign identities checkbox
            self.heatmap = QCheckBox("Generate Intensity Heatmap?")
            self.heatmap.setChecked(True)
            layout.addRow(self.heatmap)

            # Reassign identities checkbox
            self.reassign_identities_checkbox = QCheckBox("Reassign Identities Based on Clustering Results?")
            self.reassign_identities_checkbox.setChecked(False)
            layout.addRow(self.reassign_identities_checkbox)

            self._outer_layout.addWidget(self._main_widget)

            # ============================================================
            # Right side: Identity Planning panel (hidden by default)
            # ============================================================
            self._identity_widget = QWidget()
            identity_layout = QVBoxLayout(self._identity_widget)
            identity_layout.setContentsMargins(0, 0, 0, 0)

            id_title = QLabel("Identity Planning (For UMAPs and Violins)")
            id_title.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 3px;")
            identity_layout.addWidget(id_title)

            columns_widget = QWidget()
            columns_layout = QHBoxLayout(columns_widget)
            columns_layout.setContentsMargins(0, 0, 0, 0)
            columns_layout.setSpacing(8)

            # -- Include column --
            include_group = QGroupBox("Include Only (Overrides Exclude)")
            include_group.setToolTip(
                "Check identities to show ONLY these.\n"
                "Leave all unchecked to show everything (minus excludes).")
            self._include_layout = QVBoxLayout(include_group)
            self._include_layout.setContentsMargins(4, 4, 4, 4)
            self._include_layout.setSpacing(2)

            self._include_scroll = QScrollArea()
            self._include_scroll.setWidgetResizable(True)
            self._include_scroll.setMinimumWidth(140)
            self._include_scroll.setMinimumHeight(200)
            self._include_inner = QWidget()
            self._include_inner_layout = QVBoxLayout(self._include_inner)
            self._include_inner_layout.setContentsMargins(2, 2, 2, 2)
            self._include_inner_layout.setSpacing(1)
            self._include_inner_layout.addStretch()
            self._include_scroll.setWidget(self._include_inner)
            self._include_layout.addWidget(self._include_scroll)

            columns_layout.addWidget(include_group)

            # -- Exclude column --
            exclude_group = QGroupBox("Exclude")
            exclude_group.setToolTip(
                "Check identities to HIDE them.\n"
                "Excluded identities will be removed from results.")
            self._exclude_layout = QVBoxLayout(exclude_group)
            self._exclude_layout.setContentsMargins(4, 4, 4, 4)
            self._exclude_layout.setSpacing(2)

            self._exclude_scroll = QScrollArea()
            self._exclude_scroll.setWidgetResizable(True)
            self._exclude_scroll.setMinimumWidth(140)
            self._exclude_scroll.setMinimumHeight(200)
            self._exclude_inner = QWidget()
            self._exclude_inner_layout = QVBoxLayout(self._exclude_inner)
            self._exclude_inner_layout.setContentsMargins(2, 2, 2, 2)
            self._exclude_inner_layout.setSpacing(1)
            self._exclude_inner_layout.addStretch()
            self._exclude_scroll.setWidget(self._exclude_inner)
            self._exclude_layout.addWidget(self._exclude_scroll)

            columns_layout.addWidget(exclude_group)

            identity_layout.addWidget(columns_widget, stretch=1)

            self._identity_widget.setVisible(False)
            self._outer_layout.addWidget(self._identity_widget)

        except:
            import traceback
            print(traceback.format_exc())
            QTimer.singleShot(0, self.close)

    # ============================================================
    # Identity Planning Panel methods
    # ============================================================
    def _toggle_identity_panel(self):
        self._identity_panel_visible = not self._identity_panel_visible

        if self._identity_panel_visible:
            self.plan_identities_btn.setText("Plan Identities ◀")
            self._populate_identity_lists()
            self._identity_widget.setVisible(True)
            self.setMinimumWidth(660)
            self.resize(680, self.height())
        else:
            self.plan_identities_btn.setText("Plan Identities ▶")
            self._identity_widget.setVisible(False)
            self.setMinimumWidth(340)
            self.resize(340, self.height())

    def _populate_identity_lists(self):
        """Populate include/exclude checkboxes from available identities."""
        for cb in self._include_checks.values():
            cb.setParent(None)
        for cb in self._exclude_checks.values():
            cb.setParent(None)
        self._include_checks.clear()
        self._exclude_checks.clear()

        all_idens = list(my_network.node_identities.values())
        seen = set()
        for iden in all_idens:
            seen.update(iden)
        identity_labels = sorted(list(seen))

        for label in identity_labels:
            label_str = str(label)

            inc_cb = QCheckBox(label_str)
            self._include_checks[label] = inc_cb
            self._include_inner_layout.insertWidget(
                self._include_inner_layout.count() - 1, inc_cb)

            exc_cb = QCheckBox(label_str)
            self._exclude_checks[label] = exc_cb
            self._exclude_inner_layout.insertWidget(
                self._exclude_inner_layout.count() - 1, exc_cb)

    def _get_identity_filter(self):
        """
        Read the include/exclude checkboxes and return the filter sets.

        Returns
        -------
        included : set or None
            Set of identity labels the user explicitly included.
            None if no includes were checked (meaning "show all minus excludes").
        excluded : set
            Set of identity labels the user explicitly excluded.

        Semantics:
        - If included is not None  -> only show those identities
          (include overrides exclude).
        - If included is None      -> show everything EXCEPT excluded.
        """
        included = set()
        excluded = set()

        for label, cb in self._include_checks.items():
            if cb.isChecked():
                included.add(label)

        for label, cb in self._exclude_checks.items():
            if cb.isChecked():
                excluded.add(label)

        if len(included) == 0:
            included = None  # means "all"

        return included, excluded

    def _filter_iden_list_by_plan(self, iden_list):
        """
        Given a list of node IDs that belong to some identity group,
        filter it according to the identity planning panel.

        This is used by violin plots: if the user has checked includes/excludes,
        we remove nodes whose *only* identities are excluded (or not included).

        Returns the (possibly shortened) list of node IDs.
        """
        included, excluded = self._get_identity_filter()

        # No filtering needed
        if included is None and len(excluded) == 0:
            return iden_list

        filtered = []
        for node in iden_list:
            if node not in my_network.node_identities:
                continue
            node_idens = my_network.node_identities[node]

            if included is not None:
                # Keep node only if it has at least one included identity
                if any(iden in included for iden in node_idens):
                    filtered.append(node)
            else:
                # Keep node only if it has at least one non-excluded identity
                if any(iden not in excluded for iden in node_idens):
                    filtered.append(node)

        return filtered

    def _filter_violin_dict_columns_by_plan(self, violin_dict):
        """
        Filter the *columns* (keys) of a violin_dict to only those identities
        the user has included (or not excluded).

        This trims which markers/identities appear as separate violin columns.
        """
        included, excluded = self._get_identity_filter()

        if included is None and len(excluded) == 0:
            return violin_dict

        filtered = {}
        for key, val in violin_dict.items():
            if included is not None:
                # Keep column if key matches an included identity (with or without '+')
                if key in included or f"{key}+" in included:
                    filtered[key] = val
            else:
                # Remove column if key matches an excluded identity
                if key not in excluded and f"{key}+" not in excluded:
                    filtered[key] = val

        return filtered if filtered else violin_dict

    def _build_filtered_identity_dict_for_umap(self):
        """
        Build a filtered identity dict for UMAP, respecting include/exclude.
        Returns a dict of {node_id: single_identity_label} for coloring.
        """
        included, excluded = self._get_identity_filter()

        filtered = {}
        for node, idens in my_network.node_identities.items():
            if included is not None:
                new_idens = [iden for iden in idens if iden in included]
                if new_idens:
                    filtered[node] = random.choice(new_idens)
            else:
                new_idens = [iden for iden in idens if iden not in excluded]
                if new_idens:
                    filtered[node] = random.choice(new_idens)

        return filtered

    # ============================================================
    # Original methods (unchanged unless noted)
    # ============================================================

    def get_all_identity_lists(self):
        """
        Get all identity lists for normalization purposes.
        
        Returns:
        dict: Dictionary where keys are identity names and values are lists of node IDs
        """
        
        return n3d.invert_dict_special(my_network.node_identities)

    def prepare_data_for_umap(self, df, node_identities=None):
        """
        Prepare data for UMAP visualization by z-score normalizing columns.
        
        Args:
            df: DataFrame with first column as NodeID, rest as marker intensities
            node_identities: Optional dict mapping node_id (int) -> identity (string).
                            If provided, only nodes present as keys will be kept.
            
        Returns:
            dict: {node_id: [normalized_marker_values]}
        """
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Store marker names (column headers) before converting to numpy array
        marker_names = df.columns[1:].tolist()  # All columns except first (NodeID)
        node_id_col_name = df.columns[0]  # Store the first column name (e.g., "NodeID")
        
        # Extract node IDs from first column
        node_ids = df.iloc[:, 0].values
        # Extract marker data (all columns except first)
        X = df.iloc[:, 1:].values
        
        # Z-score normalization (column-wise)
        scaler = StandardScaler() # Ultimately decided to normalize with the entirety of the available data (even cells without identities) since those cells' low expression should represent something of a ground truth of background expression which is relevant for normalizing. 
        X_normalized = scaler.fit_transform(X)

        
        # Filter for actually present nodes
        mask = pd.Series(node_ids).isin(self.valid_nodes_list).values
        
        # Filter both node_ids and X_normalized using the mask
        node_ids = node_ids[mask]
        X_normalized = X_normalized[mask]
        
        if len(node_ids) == 0:
            raise ValueError("No matching nodes found between df and node_identities")
        
        # Reconstruct DataFrame with normalized values
        self.ref_df = pd.DataFrame(X_normalized, columns=marker_names)
        self.ref_df.insert(0, node_id_col_name, node_ids)  # Add NodeID column back as first column

        # Create dictionary mapping node_id -> normalized row
        result_dict = {
            int(node_ids[i]): X_normalized[i].tolist() 
            for i in range(len(node_ids))
        }
    
        return result_dict


    def normalize_df_with_identity_centerpoints(self, df, identity_lists):
        """
        Normalize the entire dataframe using identity-specific centerpoints.
        Uses Z-score-like normalization with identity centerpoint as the "mean".
        
        Parameters:
        df (pd.DataFrame): Original dataframe
        identity_lists (dict): Dictionary where keys are identity names and values are lists of node IDs
        
        Returns:
        pd.DataFrame: Normalized dataframe
        """
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Set the first column as the index (row headers)
        df_copy = df_copy.set_index(df_copy.columns[0])
        
        # Convert all remaining columns to float type (batch conversion)
        df_copy = df_copy.astype(float)
        
        # Convert index to set for O(1) lookup
        index_set = set(df_copy.index)
        
        # PRE-FILTER: Process identity_lists once and store valid nodes
        identity_filtered = {}
        for identity, node_list in identity_lists.items():
            valid_nodes = [node for node in node_list if node in index_set]
            if valid_nodes:
                identity_filtered[str(identity)] = valid_nodes
        
        # Calculate centerpoints for all columns
        column_centerpoints = {}
        
        for column in df_copy.columns:
            column_str = str(column)
            centerpoint = None
            
            # Try exact match first (e.g., column "CD4" matches identity "CD4")
            if column_str in identity_filtered:
                valid_nodes = identity_filtered[column_str]
                centerpoint = df_copy.loc[valid_nodes, column].min()
            # Then try with '+' appended (e.g., column "CD4" matches identity "CD4+")
            elif f'{column_str}+' in identity_filtered:
                valid_nodes = identity_filtered[f'{column_str}+']
                centerpoint = df_copy.loc[valid_nodes, column].min()
            
            if centerpoint is not None:
                column_centerpoints[column] = centerpoint
            else:
                # Fallback: use column median
                column_centerpoints[column] = df_copy[column].median()
        
        # Vectorized normalization: subtract all centerpoints at once
        centerpoints_series = pd.Series(column_centerpoints)
        
        # Calculate standard deviations for all columns at once
        std_devs = df_copy.std()
        
        # Perform vectorized normalization
        df_normalized = (df_copy - centerpoints_series) / std_devs
        
        # Handle columns with zero std_dev (where division would give NaN/inf)
        zero_std_cols = std_devs[std_devs == 0].index
        if len(zero_std_cols) > 0:
            df_normalized[zero_std_cols] = df_copy[zero_std_cols] - centerpoints_series[zero_std_cols]
        
        # Convert back to original format with first column as regular column
        df_normalized = df_normalized.reset_index()
        
        return df_normalized

    def show_in_table(self, df, metric, title):

        # Create new table
        table = CustomTableView(self.parent())
        table.setModel(PandasModel(df))

        try:
            first_column_name = table.model()._data.columns[0]
            table.sort_table(first_column_name, ascending=True)
        except:
             pass
        
        # Add to tabbed widget
        if title is None:
            self.parent().tabbed_data.add_table(f"{metric} Analysis", table)
        else:
            self.parent().tabbed_data.add_table(f"{title}", table)
        


        # Adjust column widths to content
        for column in range(table.model().columnCount(None)):
            table.resizeColumnToContents(column)

    def run(self, com = None):

        all_idens = list(my_network.node_identities.values())
        seen = set()
        for iden in all_idens:
            seen.update(iden)
        idens = list(seen)

        for i, iden in enumerate(idens):
            if iden.endswith('+'):
                idens[i] = iden[:-1]

        def df_to_dict_by_rows(df, row_indices, title):
            """
            Convert a pandas DataFrame to a dictionary by selecting specific rows.
            No normalization - dataframe is already normalized.
            
            Parameters:
            df (pd.DataFrame): DataFrame with first column as row headers, remaining columns contain floats
            row_indices (list): List of values from the first column representing rows to include
            
            Returns:
            dict: Dictionary where keys are column headers and values are lists of column values (as floats)
                  for the specified rows
            """
            # Make a copy to avoid modifying the original dataframe
            df_copy = df.copy()
            
            # Set the first column as the index (row headers)
            df_copy = df_copy.set_index(df_copy.columns[0])
            
            # Mask the dataframe to include only the specified rows
            masked_df = df_copy.loc[row_indices]
            
            # Create empty dictionary
            result_dict = {}
            
            # For each column, add the column header as key and column values as list
            for column in masked_df.columns:
                result_dict[column] = masked_df[column].tolist()
            
            masked_df.insert(0, "NodeIDs", row_indices)
            self.show_in_table(masked_df, metric = "NodeID", title = title)


            return result_dict

        from . import neighborhoods

        try:
            if com:

                self.ref_df = self.df

                com_dict = n3d.invert_dict(my_network.communities)

                com_list = com_dict[int(com)]

                # --- Apply identity plan filtering to community node list ---
                com_list = self._filter_iden_list_by_plan(com_list)

                violin_dict = df_to_dict_by_rows(self.df, com_list, f"Z-Score-like Channel Intensities of Community {com}, {len(com_list)} Nodes")

                try:
                    new_dict = {key:val for key, val in violin_dict.items() if key in idens}
                    if new_dict != {}:
                        violin_dict = new_dict
                except:
                    pass

                # --- Apply identity plan filtering to violin columns ---
                violin_dict = self._filter_violin_dict_columns_by_plan(violin_dict)

                neighborhoods.create_violin_plots(violin_dict, graph_title=f"Z-Score-like Channel Intensities of Community {com}, {len(com_list)} Nodes", idens = idens)

                return
        except:
            import traceback
            print(traceback.format_exc())
            pass

        try:

            if self.idens.currentIndex() != 0:

                iden = self.idens.currentText()
                iden_list = []

                for item in my_network.node_identities:

                    if iden in my_network.node_identities[item]:
                        iden_list.append(item)

                # --- Apply identity plan filtering to identity node list ---
                iden_list = self._filter_iden_list_by_plan(iden_list)

                violin_dict = df_to_dict_by_rows(self.df, iden_list, f"Z-Score-like Channel Intensities of Identity {iden}, {len(iden_list)} Nodes")

                try:
                    new_dict = {key:val for key, val in violin_dict.items() if key in idens}
                    if new_dict != {}:
                        violin_dict = new_dict
                except:
                    pass

                # --- Apply identity plan filtering to violin columns ---
                violin_dict = self._filter_violin_dict_columns_by_plan(violin_dict)

                neighborhoods.create_violin_plots(violin_dict, graph_title=f"Z-Score-like Channel Intensities of Identity {iden}, {len(iden_list)} Nodes", idens = idens)
        except:
            import traceback
            print(traceback.format_exc())
            pass

        try:
            if self.coms.currentIndex() != 0:

                com = self.coms.currentText()

                com_dict = n3d.invert_dict(my_network.communities)

                com_list = com_dict[int(com)]

                # --- Apply identity plan filtering to community node list ---
                com_list = self._filter_iden_list_by_plan(com_list)

                violin_dict = df_to_dict_by_rows(self.df, com_list, f"Z-Score-like Channel Intensities of Community {com}, {len(com_list)} Nodes")

                try:
                    new_dict = {key:val for key, val in violin_dict.items() if key in idens}
                    if new_dict != {}:
                        violin_dict = new_dict
                except:
                    pass

                # --- Apply identity plan filtering to violin columns ---
                violin_dict = self._filter_violin_dict_columns_by_plan(violin_dict)

                neighborhoods.create_violin_plots(violin_dict, graph_title=f"Z-Score-like Channel Intensities of Community {com}, {len(com_list)} Nodes", idens = idens)
        except:
            import traceback
            print(traceback.format_exc())
            pass

    def run2(self):
        
        try:
            umap_dict = self.prepare_data_for_umap(self.backup_df)
            mode = self.mode_selector.currentIndex()
            if mode == 0:
                color_mode = 'identity'
            else:
                color_mode = 'community'

            # --- Build identity dict using plan filtering ---
            neighbor_classes = self._build_filtered_identity_dict_for_umap()

            # If the plan resulted in no nodes and we had identities before,
            # fall back to unfiltered behavior so the UMAP isn't empty.
            if not neighbor_classes:
                for item in umap_dict.keys():
                    if item in my_network.node_identities:
                        idens = my_network.node_identities[item]
                        neighbor_classes[item] = random.choice(idens)

            umap = umw.UMAPGraphWidget(
                parent=self.parent(),
                labels=False,
                color_mode=color_mode
            )

            self.parent().umap_widgets.append(umap)

            umap.set_data(
                umap_dict,
                community_dict=my_network.communities,
                identity_dict=neighbor_classes,
                color_mode=color_mode
            )

            umap.show_in_window("UMAP Visualization")


            #my_network.identity_umap(umap_dict, mode)
        except:
            import traceback
            print(traceback.format_exc())
            pass

    def run3(self):
        num_clusters_text = self.kmeans_num_input.text()
        heatmap = self.heatmap.isChecked()
        mode = self.mode_selector2.currentIndex()
        
        if num_clusters_text:
            num_clusters = int(num_clusters_text)
            # Use specified number of clusters
            print(f"Using {num_clusters} clusters")
        else:
            num_clusters = None  # Auto-determine
            print("Auto-determining number of clusters")
        try:
            cluster_dict = self.prepare_data_for_umap(self.backup_df)

            if mode == 0:
                my_network.group_nodes_by_intensity(cluster_dict, count = num_clusters)
            else:
                new_dict = {}
                template = np.zeros(len(list(cluster_dict.values())[0]))
                G = my_network.network
                nodes = list(G.nodes())
                for node in nodes:
                    counter = 0
                    new_dict[node] = np.copy(template)
                    neighbors = list(G.neighbors(node))
                    for neighbor in neighbors:
                        new_dict[node] += cluster_dict[node]
                        counter += 1
                    new_dict[node] = new_dict[node]/counter
                cluster_dict = new_dict
                my_network.group_nodes_by_intensity(cluster_dict, count = num_clusters)


            if heatmap:
                marker_names = self.backup_df.columns[1:].tolist()
                #print(marker_names)
                heatmap_dict = {}
                com_dict = n3d.invert_dict(my_network.communities)
                for com in com_dict:
                    com_array = np.zeros(len(marker_names))
                    counter = 0
                    for node in com_dict[com]:
                        if node in cluster_dict:
                            counter += 1
                            com_array += cluster_dict[node]
                    com_array = np.copy(com_array)/counter
                    com_dict[com] = com_array
                from . import neighborhoods
                output, sorted_id_set = neighborhoods.plot_dict_heatmap(com_dict, marker_names, title = "Average Normalized Node Marker Expression per Community", center_at_zero = True, sublabel = "Community")
                self.parent().format_for_upperright_table(output, 'CommunityID', sorted_id_set, title = f'Community Intensity Heatmap')
            try:
                # Check if user wants to reassign identities
                if self.reassign_identities_checkbox.isChecked():
                    # Invert the dict to get {neighborhood_id: [node_ids]}
                    inverted_dict = n3d.invert_dict(my_network.communities)
                    
                    # Dictionary to store old -> new neighborhood names
                    neighborhood_rename_dict = {}
                    neighborhood_items = list(inverted_dict.items())
                    
                    def show_next_dialog(index=0):
                        if index >= len(neighborhood_items):
                            temp_dict = copy.deepcopy(neighborhood_rename_dict)
                            for item in temp_dict:
                                if temp_dict[item] == None:
                                    del neighborhood_rename_dict[item]
                            # All dialogs done, apply the renaming
                            for node_id, old_neighborhood_id in my_network.communities.items():
                                try:
                                    # Only update identity if this neighborhood was renamed
                                    if old_neighborhood_id in neighborhood_rename_dict:
                                        my_network.node_identities[node_id] = neighborhood_rename_dict[old_neighborhood_id]
                                    # Otherwise, keep the existing identity (do nothing)
                                except:
                                    pass
                            self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'Community', title = 'Node Communities')
                            self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', title = 'Node Identities')
                            self.accept()
                            return
                        
                        neighborhood_id, node_list = neighborhood_items[index]
                        
                        plt.close()
                        self.run(com = neighborhood_id)

                        # Filter self.ref_df to only nodes in this neighborhood
                        mask = self.ref_df.iloc[:, 0].isin(node_list)
                        filtered_df = self.ref_df[mask]
                        
                        # Calculate average for each marker (skip first column which is NodeID)
                        averages = filtered_df.iloc[:, 1:].mean()

                        # Show dialog to user
                        dialog = NeighborhoodRenameDialog(
                            neighborhood_id=neighborhood_id,
                            averages=averages,
                            node_count=len(node_list),
                            parent=self
                        )
                        
                        def on_dialog_finished(result):
                            if result == QDialog.DialogCode.Accepted:
                                new_name = dialog.get_new_name()
                                if new_name:  # If user provided a non-empty name
                                    neighborhood_rename_dict[neighborhood_id] = new_name
                                else:  # User clicked OK but left it blank
                                    neighborhood_rename_dict[neighborhood_id] = None
                            else:
                                # User cancelled or closed window
                                neighborhood_rename_dict[neighborhood_id] = None
                            
                            # Show next dialog
                            show_next_dialog(index + 1)
                        
                        dialog.finished.connect(on_dialog_finished)
                        dialog.show()
                    
                    # Start the chain
                    show_next_dialog(0)
                else:
                    # No renaming needed, proceed directly
                    self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'Community', title = 'Node Communities')
                    self.coms.setCurrentIndex(0)
                    self.coms.setEnabled(False)
                    #self.accept()
            except:
                self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'Community', title = 'Node Communities')
                self.coms.setCurrentIndex(0)
                self.coms.setEnabled(False)
                #self.accept()
        except:
            import traceback
            print(traceback.format_exc())
            pass

class NeighborhoodRenameDialog(QDialog):
    def __init__(self, neighborhood_id, averages, node_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Rename Community {neighborhood_id}")
        self.setModal(False)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            f"<b>Community {neighborhood_id}</b><br>"
            f"Contains {node_count} nodes<br><br>"
            f"Please review the normalized average marker intensities below and provide a name for this community:"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Create scrollable area for averages
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        
        averages_widget = QWidget()
        averages_layout = QVBoxLayout(averages_widget)
        
        # Display each marker average
        for marker_name, avg_value in averages.items():
            label = QLabel(f"{marker_name}: {avg_value:.4f}")
            averages_layout.addWidget(label)
        
        scroll.setWidget(averages_widget)
        layout.addWidget(scroll)
        
        # Text input for new name
        layout.addWidget(QLabel("<b>New Community Name:</b>"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(f"Leave blank to not overwrite node identities for this community'")
        layout.addWidget(self.name_input)
        
        # Buttons
        from PyQt6.QtWidgets import QDialogButtonBox
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.resize(400, 500)
    
    def get_new_name(self):
        """Return the new name entered by the user"""
        return self.name_input.text().strip()




class DegreeDialog(QDialog):


    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Degree Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        layout.addRow("Note:", QLabel(f"This operation will be executed on the image in 'Active Image', unless it is set to edges in which case it will use the nodes. \n (This is because you may want to run it on isolated nodes that have been placed in the Overlay channels)\nWe can draw optional overlays to Overlay 2 as described below:"))

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Just make table", "Draw degree of node as overlay (literally draws 1, 2, 3, etc... faster)", "Label nodes by degree (nodes will take on the value 1, 2, 3, etc, based on their degree, to export for array based analysis)", "Create Heatmap of Degrees"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        self.mask_limiter = QLineEdit("1")
        layout.addRow("Proportion of high degree nodes to keep (ignore if only returning degrees)", self.mask_limiter)

        self.down_factor = QLineEdit("1")
        layout.addRow("down_factor (for speeding up overlay generation - ignore if only returning degrees:", self.down_factor)

        # Add Run button
        run_button = QPushButton("Get Degrees")
        run_button.clicked.connect(self.degs)
        layout.addWidget(run_button)

    def degs(self):

        try:

            accepted_mode = self.mode_selector.currentIndex()

            if accepted_mode == 3:
                degree_dict, overlay = my_network.get_degrees(heatmap = True)
                self.parent().format_for_upperright_table(degree_dict, 'Node ID', 'Degree', title = 'Degrees of nodes')
                self.parent().load_channel(3, channel_data = overlay, data = True)
                self.accept()
                return


            try:
                down_factor = float(self.down_factor.text()) if self.down_factor.text() else 1
            except ValueError:
                down_factor = 1

            try:
                mask_limiter = float(self.mask_limiter.text()) if self.mask_limiter.text() else 1
            except ValueError:
                mask_limiter = 1

            if self.parent().active_channel == 1:
                active_data = self.parent().channel_data[0]
            else:
                # Get the active channel data from parent
                active_data = self.parent().channel_data[self.parent().active_channel]
                if active_data is None:
                    raise ValueError("No active image selected")

            if my_network.node_centroids is None and accepted_mode > 0:
                self.parent().show_centroid_dialog()
                if my_network.node_centroids is None:
                    accepted_mode == 0
                    print("Error retrieving centroids")

            original_shape = copy.deepcopy(active_data.shape)


            if mask_limiter < 1 and accepted_mode != 0:

                if len(np.unique(active_data)) < 3:
                    active_data, _ = n3d.label_objects(active_data)

                node_list = list(my_network.network.nodes)
                node_dict = {}

                for node in node_list:
                    node_dict[node] = (my_network.network.degree(node))

                # Calculate the number of top proportion% entries
                num_items = len(node_dict)
                num_top_10_percent = max(1, int(num_items * mask_limiter))  # Ensure at least one item

                # Sort the dictionary by values in descending order and get the top 10%
                sorted_items = sorted(node_dict.items(), key=lambda item: item[1], reverse=True)
                top_10_percent_items = sorted_items[:num_top_10_percent]

                # Extract the keys from the top proportion% items
                top_10_percent_keys = [key for key, value in top_10_percent_items]

                mask = np.isin(active_data, top_10_percent_keys)
                nodes = mask * active_data
                new_centroids = {}
                for node in my_network.node_centroids:
                    if node in top_10_percent_keys:
                        new_centroids[node] = my_network.node_centroids[node]
                del mask

                temp_network = n3d.Network_3D(nodes = nodes, node_centroids = new_centroids, network = my_network.network, network_lists = my_network.network_lists)

                result, nodes = temp_network.get_degrees(called = True, no_img = accepted_mode, down_factor = down_factor)

            else:
                temp_network = n3d.Network_3D(nodes = active_data, node_centroids = my_network.node_centroids, network = my_network.network, network_lists = my_network.network_lists)

                result, nodes = temp_network.get_degrees(called = True, no_img = accepted_mode, down_factor = down_factor)



            self.parent().format_for_upperright_table(result, 'Node ID', 'Degree', title = 'Degrees of nodes')

            if nodes.shape != original_shape:

                nodes = n3d.upsample_with_padding(nodes, down_factor, original_shape)

            if accepted_mode > 0:
                self.parent().load_channel(3, channel_data = nodes, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))


            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error finding degrees: {e}")


class HubDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Hub Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        layout.addRow("Note:", QLabel(f"Finds hubs, which are nodes in the network that have the shortest number of steps to the other nodes\nWe can draw optional overlays to Overlay 2 as described below:"))

        # Overlay checkbox (default True)
        self.overlay = QPushButton("Overlay")
        self.overlay.setCheckable(True)
        self.overlay.setChecked(True)
        layout.addRow("Make Overlay?:", self.overlay)


        self.proportion = QLineEdit("0.15")
        layout.addRow("Proportion of most connected hubs to keep (1 would imply returning entire network)", self.proportion)


        # Add Run button
        run_button = QPushButton("Get hubs")
        run_button.clicked.connect(self.hubs)
        layout.addWidget(run_button)

    def hubs(self):

        try:

            try:
                proportion = float(self.proportion.text()) if self.proportion.text() else 1
            except ValueError:
                proportion = 1

            overlay = self.overlay.isChecked()

            result, img = my_network.isolate_hubs(proportion = proportion, retimg = overlay)

            hub_dict = {}

            for node in result:
                hub_dict[node] = my_network.network.degree(node)

            self.parent().format_for_upperright_table(hub_dict, 'NodeID', 'Degree', title = f'Upper {proportion} Hub Nodes')

            if img is not None:

                self.parent().load_channel(3, channel_data = img, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))


            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error finding hubs: {e}")



class MotherDialog(QDialog):


    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Mother Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        layout.addRow("Note:", QLabel(f"Mother nodes are those that exist between communities. \nWe can draw optional overlays to Overlay 1 as described below:"))

        # Overlay checkbox (default False)
        self.overlay = QPushButton("Overlay")
        self.overlay.setCheckable(True)
        self.overlay.setChecked(False)
        layout.addRow("Make Overlay?:", self.overlay)

        # Add Run button
        run_button = QPushButton("Get Mothers")
        run_button.clicked.connect(self.mothers)
        layout.addWidget(run_button)

    def mothers(self):

        try:

            overlay = self.overlay.isChecked()

            if my_network.communities is None:
                self.parent().show_partition_dialog()
                if my_network.communities is None:
                    return

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()
                if my_network.node_centroids is None:
                    print("Error finding centroids")
                    overlay = False

            if not overlay:
                G = my_network.isolate_mothers(self, ret_nodes = True, called = True)
            else:
                G, result = my_network.isolate_mothers(self, ret_nodes = False, called = True)
                self.parent().load_channel(2, channel_data = result, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

            degree_dict = {}

            for node in G.nodes():
                degree_dict[node] = my_network.network.degree(node)

            self.parent().format_for_upperright_table(degree_dict, 'Mother ID', 'Degree', title = 'Mother Nodes')


            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error finding mothers: {e}")


class CodeDialog(QDialog):

    def __init__(self, parent=None, sort = 'Community'):

        super().__init__(parent)
        self.setWindowTitle(f"{sort} Code Parameters (Will go to Overlay2)")
        self.setModal(True)

        layout = QFormLayout(self)

        self.sort = sort

        self.down_factor = QLineEdit("")
        layout.addRow("down_factor (for speeding up overlay generation - optional):", self.down_factor)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Color Coded", "Grayscale Coded"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)


        # Add Run button
        run_button = QPushButton(f"{sort} Code")
        run_button.clicked.connect(self.code)
        layout.addWidget(run_button)

    def code(self):

        try:

            mode = self.mode_selector.currentIndex()

            down_factor = float(self.down_factor.text()) if self.down_factor.text().strip() else None


            if self.sort == 'Community':
                if my_network.communities is None:
                    self.parent().show_partition_dialog()
                    if my_network.communities is None:
                        return
            elif my_network.node_identities is None:
                print("Node identities are not set")
                return

            if self.sort == 'Community':
                if mode == 0:
                    image, output = my_network.extract_communities(down_factor = down_factor)
                elif mode == 1:
                    image, output = my_network.extract_communities(color_code = False, down_factor = down_factor)
            else:
                if mode == 0:
                    image, output = my_network.extract_communities(down_factor = down_factor, identities = True)
                elif mode == 1:
                    image, output = my_network.extract_communities(color_code = False, down_factor = down_factor, identities = True)

            self.parent().format_for_upperright_table(output, f'{self.sort} Id', f'Encoding Val: {self.sort}', 'Legend')

            self.parent().load_channel(3, image, True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
            self.accept()

        except Exception as e:
            print(f"An error has occurred: {e}")
            import traceback
            print(traceback.format_exc())





# PROCESS MENU RELATED:


class ResizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resize Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)
        self.resize = QLineEdit()
        self.resize.setPlaceholderText("Will Override Below")
        layout.addRow("Resize Factor (All Dimensions):", self.resize)
        self.zsize = QLineEdit("1")
        layout.addRow("Resize Z Factor:", self.zsize)
        self.ysize = QLineEdit("1")
        layout.addRow("Resize Y Factor:", self.ysize)
        self.xsize = QLineEdit("1")
        layout.addRow("Resize X Factor:", self.xsize)


        # cubic checkbox (default False)
        self.cubic = QPushButton("Use Cubic Resize? (For preserving visual characteristics, but not binary shape)")
        self.cubic.setCheckable(True)
        self.cubic.setChecked(False)
        layout.addRow("Use cubic algorithm:", self.cubic)
        
        if self.parent().original_shape is not None:
            undo_button = QPushButton(f"Resample to original shape: {self.parent().original_shape}")
            undo_button.clicked.connect(lambda: self.run_resize(undo = True))
            layout.addRow(undo_button)

        run_button = QPushButton("Run Resize")
        run_button.clicked.connect(self.run_resize)
        layout.addRow(run_button)

    def reset_fields(self):
        """Reset all input fields to default values"""
        self.resize.clear()
        self.zsize.setText("1")
        self.xsize.setText("1")
        self.ysize.setText("1")        

    def run_resize(self, undo = False, upsize = True, special = False):
        try:
            self.parent().resizing = True
            # Get parameters
            try:
                resize = float(self.resize.text()) if self.resize.text() else None
                zsize = float(self.zsize.text()) if self.zsize.text() else 1
                ysize = float(self.ysize.text()) if self.ysize.text() else 1
                xsize = float(self.xsize.text()) if self.xsize.text() else 1
            except ValueError as e:
                print(f"Invalid input value: {e}")
                self.reset_fields()
                return
            
            resize = resize if resize is not None else (zsize, ysize, xsize)

            if (self.parent().shape[1] * resize) < 1 or (self.parent().shape[2] * resize) < 1:
                print("Incompatible x/y dimensions")
                return
            elif (self.parent().shape[0] * resize) < 1:
                resize = (1, resize, resize)

            if special:
                if upsize:
                    if (my_network.z_scale > my_network.xy_scale):
                        # Z dimension needs to be stretched
                        resize = [my_network.z_scale/my_network.xy_scale, 1, 1]  # Scale factor for [z, y, x]
                        cardinal = my_network.xy_scale
                    elif (my_network.xy_scale > my_network.z_scale):
                        # XY dimensions need to be stretched
                        resize = [1, my_network.xy_scale/my_network.z_scale, my_network.xy_scale/my_network.z_scale]  # Scale factor for [z, y, x]
                        cardinal = my_network.z_scale
                else:
                    if (my_network.z_scale > my_network.xy_scale):
                        # XY dimension needs to be shrunk
                        resize = [1, my_network.xy_scale/my_network.z_scale, my_network.xy_scale/my_network.z_scale]  # Scale factor for [z, y, x]
                        cardinal = my_network.z_scale
                    elif (my_network.xy_scale > my_network.z_scale):
                        # Z dimensions need to be shrunk
                        resize = [my_network.z_scale/my_network.xy_scale, 1, 1]  # Scale factor for [z, y, x]
                        cardinal = my_network.xy_scale

            # Get the shape from whichever array exists
            array_shape = None
            if my_network.nodes is not None:
                array_shape = my_network.nodes.shape
            elif my_network.edges is not None:
                array_shape = my_network.edges.shape
            elif my_network.network_overlay is not None:
                array_shape = my_network.network_overlay.shape
            elif my_network.id_overlay is not None:
                array_shape = my_network.id_overlay.shape
                
            if array_shape is None:
                QMessageBox.critical(self, "Error", "No valid array found to resize")
                self.reset_fields()
                return
                
            # Check if resize would result in valid dimensions
            if isinstance(resize, (int, float)):
                new_shape = tuple(int(dim * resize) for dim in array_shape)
            else:
                new_shape = tuple(int(dim * factor) for dim, factor in zip(array_shape, resize))


            cubic = self.cubic.isChecked()
            order = 3 if cubic else 0
                
            # Reset slider before modifying data
            self.parent().slice_slider.setValue(0)
            self.parent().current_slice = 0
            
            if not undo:
                # Process each channel
                for channel in range(4):
                    if self.parent().channel_data[channel] is not None:
                        resized_data = n3d.resize(self.parent().channel_data[channel], resize, order)
                        self.parent().load_channel(channel, channel_data=resized_data, data=True)


                
                # Process highlight overlay if it exists
                if self.parent().mini_overlay_data is not None:
                    self.parent().create_highlight_overlay(self.parent().clicked_values['nodes'],  self.parent().clicked_values['edges'])

                if self.parent().highlight_overlay is not None:
                    self.parent().highlight_overlay = n3d.resize(self.parent().highlight_overlay, resize, order)
                if my_network.search_region is not None:
                    my_network.search_region = n3d.resize(my_network.search_region, resize, order)


            else:
                # Process each channel
                if array_shape == self.parent().original_shape:
                    return
                for channel in range(4):
                    if self.parent().channel_data[channel] is not None:
                        resized_data = n3d.upsample_with_padding(self.parent().channel_data[channel], original_shape = self.parent().original_shape)
                        self.parent().load_channel(channel, channel_data=resized_data, data=True)

                if self.parent().mini_overlay_data is not None:

                    self.parent().create_highlight_overlay(self.parent().clicked_values['nodes'],  self.parent().clicked_values['edges'])

                
                # Process highlight overlay if it exists
                if self.parent().highlight_overlay is not None:
                    self.parent().highlight_overlay = n3d.upsample_with_padding(self.parent().highlight_overlay, original_shape = self.parent().original_shape)
                if my_network.search_region is not None:
                    my_network.search_region = n3d.upsample_with_padding(my_network.search_region, original_shape = self.parent().original_shape)

            
            # Update slider range based on new z-dimension
            for channel in self.parent().channel_data:
                if channel is not None:
                    self.parent().slice_slider.setMinimum(0)
                    self.parent().slice_slider.setMaximum(channel.shape[0] - 1)
                    self.parent().shape = channel.shape
                    break

            if not special:
                if isinstance(resize, (int, float)):
                    my_network.xy_scale = my_network.xy_scale/resize
                    my_network.z_scale = my_network.z_scale/resize
                    print("xy_scales and z_scales have been adjusted per resample. Check image -> properties to manually reset them to 1 if desired.")
                else:
                    my_network.xy_scale = my_network.xy_scale/resize[1]
                    my_network.z_scale = my_network.z_scale/resize[0]
                    print("xy_scales and z_scales have been adjusted per resample. Check image -> properties to manually reset them to 1 if desired. Note that xy_scale will not correspond if you made your XY plane a non-square.")
            else:
                my_network.xy_scale = cardinal
                my_network.z_scale = cardinal
            self.parent().xy_scale_label.setText(f"xy_scale: {my_network.xy_scale:.2e}                   ")
            self.parent().z_scale_label.setText(f"z_scale: {my_network.z_scale:.2e}                   ")

            try:
                if my_network.node_centroids is not None:
                    centroids = copy.deepcopy(my_network.node_centroids)
                    if isinstance(resize, (int, float)):
                        for item in my_network.node_centroids:
                            try:
                                centroids[item] = np.round((my_network.node_centroids[item]) * resize)
                            except:
                                temp = np.array(my_network.node_centroids[item])
                                centroids[item] = np.round((temp) * resize)

                    else:
                        for item in my_network.node_centroids:
                            centroids[item][0] = int(np.round((my_network.node_centroids[item][0]) * resize[0]))
                            centroids[item][1] = int(np.round((my_network.node_centroids[item][1]) * resize[1]))
                            centroids[item][2] = int(np.round((my_network.node_centroids[item][2]) * resize[2]))

                    my_network.node_centroids = centroids
                    print("Node centroids resampled")
            except:
                print("Could not resample node centroids")
                import traceback
                print(traceback.format_exc())
            try:
                if my_network.edge_centroids is not None:
                    centroids = copy.deepcopy(my_network.edge_centroids)
                    if isinstance(resize, (int, float)):
                        for item in my_network.edge_centroids:
                            centroids[item] = np.round((my_network.edge_centroids[item]) * resize)
                    else:
                        for item in my_network.edge_centroids:
                            centroids[item][0] = int(np.round((my_network.edge_centroids[item][0]) * resize[0]))
                            centroids[item][1] = int(np.round((my_network.edge_centroids[item][1]) * resize[1]))
                            centroids[item][2] = int(np.round((my_network.edge_centroids[item][2]) * resize[2]))

                    my_network.edge_centroids = centroids
                    print("Edge centroids resampled")
            except:
                print("Could not resample edge centroids")
                import traceback
                print(traceback.format_exc())

            if hasattr(my_network, 'node_centroids') and my_network.node_centroids is not None:
                try:
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                except Exception as e:
                    print(f"Error loading node centroid table: {e}")

            if hasattr(my_network, 'edge_centroids') and my_network.edge_centroids is not None:
                try:
                    self.parent().format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
                except Exception as e:
                    print(f"Error loading edge centroid table: {e}")

            self.parent().update_display()
            self.accept()
            
        except Exception as e:
            print(f"Error during resize operation: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to resize: {str(e)}")

class CleanDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Some options for cleaning segmentation")
        self.setModal(False)

        layout = QFormLayout(self)

        # Add Run button
        run_button = QPushButton("Close")
        run_button.clicked.connect(self.close)
        layout.addRow("Close (Fill Small Gaps - Dilate then Erode by same amount):", run_button)

        # Add Run button
        run_button = QPushButton("Open")
        run_button.clicked.connect(self.open)
        layout.addRow("Open (Eliminate Noise, Jagged Borders, and Small Connections Between Objects - Erode then Dilate by same amount):", run_button)

        # Add Run button
        run_button = QPushButton("Fill Holes")
        run_button.clicked.connect(self.holes)
        layout.addRow("Call the fill holes function:", run_button)

        # Add Run button
        run_button = QPushButton("Connect Endpoints")
        run_button.clicked.connect(self.endpoints)
        layout.addRow("Connect Endpoints? (Unsupervised - Weak to noise):", run_button)

        # Add Run button
        run_button = QPushButton("Trace Filaments")
        run_button.clicked.connect(self.fils)
        layout.addRow("For Segmentations of Blood Vessels/Nerves:", run_button)

        # Add Run button
        run_button = QPushButton("Threshold Noise")
        run_button.clicked.connect(self.thresh)
        layout.addRow("Threshold Noise By Volume:", run_button)

    def endpoints(self):

        class CleanDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Rote Endpoint Joining - Connects ALL Detectable Endpoints Within Distance")
                self.setModal(True)

                layout = QFormLayout(self)

                self.amount = QLineEdit("10")
                layout.addRow("Voxel Distance to Connect Endpoints (Will be slow if large):", self.amount)

                self.spine = QLineEdit("0")
                layout.addRow("Skeleton Spine Removal Distance:", self.spine)

                run_button = QPushButton("Run")
                run_button.clicked.connect(self.run)
                layout.addRow(run_button)

            def run(self):
                try:
                    amount = float(self.amount.text())
                except:
                    return
                try:
                    spine = int(self.spine.text())
                except:
                    spine = 0

                try:
                    from . import endpoint_joiner
                    joined = endpoint_joiner.connect_endpoints(self.parent().channel_data[self.parent().active_channel], amount, spine)
                    self.parent().load_channel(3, joined, data = True)
                    self.accept()
                except Exception as e:
                    print(f"Error: {e}")

        dialog = CleanDialog(self.parent())
        dialog.exec()

    def close(self):

        try:
            self.parent().show_dilate_dialog(args = [1, 0], execute = True)
            self.parent().show_erode_dialog(args = self.parent().last_dil)
        except:
            import traceback
            print(traceback.format_exc())
            pass

    def open(self):

        try:
            self.parent().show_erode_dialog(args = [1, 0])
            self.parent().show_dilate_dialog(args = self.parent().last_ero, execute = True)
        except:
            import traceback
            print(traceback.format_exc())
            pass

    def holes(self):

        try:
            self.parent().show_hole_dialog()
        except:
            pass

    def fils(self):

        try:
            self.parent().show_filament_dialog()
        except:
            self.parent().show_filament_dialog()
            #pass

    def thresh(self):
        try:
            if len(np.unique(self.parent().channel_data[self.parent().active_channel])) < 3:
                self.parent().show_label_dialog()

            if self.parent().volume_dict[self.parent().active_channel] is None:
                self.parent().volumes()

            thresh_window = ThresholdWindow(self.parent(), 1)
            thresh_window.show()  # Non-modal window
            self.parent().highlight_overlay = None
            #self.mini_overlay = False
            self.parent().mini_overlay_data = None
        except:
            import traceback
            print(traceback.format_exc())
            pass






class OverrideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Override Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        layout.addRow(QLabel("Use Highlight Overlay to Place Data From: "))

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Overrider:", self.mode_selector)

        layout.addRow(QLabel("To Override Corresponding Data In: "))

        # Add mode selection dropdown
        self.target_selector = QComboBox()
        self.target_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2"])
        self.target_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("To be overwritten:", self.target_selector)

        layout.addRow(QLabel("Place output in: "))

        # Add mode selection dropdown
        self.output_selector = QComboBox()
        self.output_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2", "Highlight Overlay"])
        self.output_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Output Location:", self.output_selector)

        # Add Run button
        run_button = QPushButton("Override")
        run_button.clicked.connect(self.override)
        layout.addWidget(run_button)

    def override(self):

        try:

            accepted_mode = self.mode_selector.currentIndex()
            accepted_target = self.target_selector.currentIndex()
            output_target = self.output_selector.currentIndex()

            if accepted_mode == accepted_target:
                return

            active_data = self.parent().channel_data[accepted_mode]

            if accepted_mode == 0:
                self.parent().create_highlight_overlay(node_indices=self.parent().clicked_values['nodes'])
            else:
                self.parent().create_highlight_overlay(edge_indices=self.parent().clicked_values['edges'])

            target_data = self.parent().channel_data[accepted_target]

            if target_data is None:
                target_data = np.zeros_like(active_data)



            try:

                self.parent().highlight_overlay = self.parent().highlight_overlay > 0 #What we want in override image
                inv = n3d.invert_boolean(self.parent().highlight_overlay) #what we want to keep in target image

                target_data = target_data * inv #Cut out what we don't want in target image
                max_val = np.max(target_data) #Ensure non-val overlap
                other_max = np.max(active_data)
                true_max = max_val + other_max
                if true_max < 256:
                    dtype = np.uint8
                elif true_max < 65536:
                    dtype = np.uint16
                else:
                    dtype = np.uint32

                active_data = active_data.astype(dtype)

                active_data = active_data + max_val #Transpose override image

                active_data = self.parent().highlight_overlay * active_data #Cut out what we want from old image image

                target_data = target_data.astype(dtype)

                target_data = target_data + active_data #Insert new selection

                if output_target == 4:

                    self.parent().highlight_overlay = result

                else:


                    # Update both the display data and the network object
                    self.parent().load_channel(output_target, channel_data = target_data, data = True)

                self.parent().update_display()

                self.accept()

            except Exception as e:
                print(f"Error overriding: {e}")

        except Exception as e:
            print(f"Error overriding: {e}")



class BinarizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Binarize Active Channel?")
        self.setModal(True)
        
        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode = QComboBox()
        self.mode.addItems(["Total Binarize", "Predict Foreground"])
        self.mode.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Method:", self.mode)

       # Add Run button
        run_button = QPushButton("Run Binarize")
        run_button.clicked.connect(self.run_binarize)
        layout.addRow(run_button)

    def run_binarize(self):

        try:

            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")

            mode = self.mode.currentIndex()

            try:

                if mode == 0:
                    # Call binarize method with parameters
                    result = n3d.binarize(
                        active_data
                        )
                else:
                    result = n3d.otsu_binarize(
                        active_data, True
                        )

                # Update both the display data and the network object
                self.parent().load_channel(self.parent().active_channel, result, True)

                self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
                self.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error running binarize: {str(e)}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running binarize: {str(e)}"
            )

class LabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Active Channel?")
        self.setModal(True)
        
        layout = QFormLayout(self)

       # Add Run button
        run_button = QPushButton("Run Label")
        run_button.clicked.connect(self.run_label)
        layout.addRow(run_button)

    def run_label(self):

        try:

            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")

            try:
                # Call watershed method with parameters
                result, _ = n3d.label_objects(
                    active_data
                    )

                # Update both the display data and the network object
                self.parent().load_channel(self.parent().active_channel, result, True)

                self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
                self.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error running label: {str(e)}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running label: {str(e)}"
            )


class SLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label a binary image based on it's voxels proximity to labeled components of a second image?")
        self.setModal(True)
        
        layout = QFormLayout(self)


        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Prelabeled Array:", self.mode_selector)

        layout.addRow(QLabel("Will Label Binary Foreground Voxels in: "))

        # Add mode selection dropdown
        self.target_selector = QComboBox()
        self.target_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2"])
        self.target_selector.setCurrentIndex(1)  # Default to Mode 1
        layout.addRow("Binary Array:", self.target_selector)

        # GPU checkbox (default True)
        self.GPU = QPushButton("GPU")
        self.GPU.setCheckable(True)
        self.GPU.setChecked(False)
        #layout.addRow("Use GPU:", self.GPU)

        self.down_factor = QLineEdit("")
        #layout.addRow("Internal Downsample for GPU (if needed):", self.down_factor)

        self.label_mode = QComboBox()
        self.label_mode.addItems(["Label Individual Voxels based on Proximity", "Label Continuous Domains that Border Labels"])
        self.label_mode.setCurrentIndex(0)
        layout.addRow("Labeling Mode:", self.label_mode)

        self.fix = QPushButton("Correct")
        self.fix.setCheckable(True)
        self.fix.setChecked(False)
        layout.addRow("Correct Nontouching Labels in post (Causes non-contiguous labels to merge with neighbors except the largest instance of that label):", self.fix)

       # Add Run button
        run_button = QPushButton("Run Smart Label")
        run_button.clicked.connect(self.run_slabel)
        layout.addRow(run_button)

    def run_slabel(self):

        try:

            accepted_source = self.mode_selector.currentIndex()
            accepted_target = self.target_selector.currentIndex()
            label_mode = self.label_mode.currentIndex()
            GPU = self.GPU.isChecked()
            fix = self.fix.isChecked()


            if accepted_source == accepted_target:
                return

            binary_array = self.parent().channel_data[accepted_target]

            label_array = self.parent().channel_data[accepted_source]

            down_factor = float(self.down_factor.text()) if self.down_factor.text().strip() else None


            if label_mode == 1:

                label_mask = label_array == 0


                #if self.parent().shape[0] != 1:
                #    skele = n3d.skeletonize(binary_array)
                 #   skele = n3d.fill_holes_3d(skele)
                skele = n3d.skeletonize(binary_array)
                skele = label_mask * skele
                binary_array = label_mask * binary_array
                del label_mask
                skele, _ = n3d.label_objects(skele)
                skele = pxt.label_continuous(skele, label_array)
                skele = skele + label_array
                binary_array = sdl.smart_label(binary_array, skele, GPU = False, remove_template = False)
                binary_array = self.parent().separate_nontouching_objects(binary_array, max_val=np.max(binary_array), branches = True)
                #binary_array = binary_array + label_array
                self.parent().load_channel(accepted_target, binary_array, True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
                self.accept()

            else:

                try:

                    # Update both the display data and the network object
                    binary_array = sdl.smart_label(binary_array, label_array, directory = None, GPU = GPU, predownsample = down_factor, remove_template = True)
                    if fix:
                        binary_array = self.parent().separate_nontouching_objects(binary_array, max_val=np.max(binary_array), branches = True)

                    self.parent().load_channel(accepted_target, binary_array, True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

                    self.accept()
                    
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Error running smart label: {str(e)}"
                    )

        except Exception as e:
            import traceback
            traceback.print_exc()   

            QMessageBox.critical(
                self,
                "Error",
                f"Error running smart label: {str(e)}"
            )


class ThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Threshold Mode")
        self.setModal(False)
        
        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Using Label/Brightness", "Using Volumes", "Using Radii", "Using Node Degree"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # Add Run button
        run_button = QPushButton("Select")
        run_button.clicked.connect(self.thresh_mode)
        layout.addRow(run_button)

        # Add ML button
        ML = QPushButton("Machine Learning")
        ML.clicked.connect(lambda: self.start_ml(GPU = False))
        layout.addRow(ML)


    def thresh_mode(self):

        try:

            accepted_mode = self.mode_selector.currentIndex()

            if accepted_mode == 1:
                if len(np.unique(self.parent().channel_data[self.parent().active_channel])) < 3:
                    self.parent().show_label_dialog()

                if self.parent().volume_dict[self.parent().active_channel] is None:
                    self.parent().volumes()

            elif accepted_mode == 2:
                if len(np.unique(self.parent().channel_data[self.parent().active_channel])) < 3:
                    self.parent().show_label_dialog()

                if self.parent().radii_dict[self.parent().active_channel] is None:
                    self.parent().show_rad_dialog()

                    if self.parent().radii_dict[self.parent().active_channel] is None:
                        return

            elif accepted_mode == 3:

                if my_network.nodes is None or my_network.network is None:
                    print("Error - please calculate network first")
                    return

            thresh_window = ThresholdWindow(self.parent(), accepted_mode)
            thresh_window.show()  # Non-modal window
            self.highlight_overlay = None
            #self.mini_overlay = False
            self.mini_overlay_data = None
            self.accept()
        except:
            import traceback
            traceback.print_exc()   
            pass

    def start_ml(self, GPU = False):

        if self.parent().channel_data[0] is None:

            try:
                print("Please select image to load into nodes channel for segmentation or press X if you already have the one you want. Note that this load may permit a color image in the nodes channel for segmentation purposes only, which is otherwise not allowed.")
                self.parent().load_channel(0, color = True)
            except:
                pass


        if self.parent().channel_data[2] is not None or self.parent().channel_data[3] is not None or self.parent().highlight_overlay is not None:
            if self.confirm_machine_dialog():
                pass
            else:
                return
        elif self.parent().channel_data[0] is None and self.parent().channel_data[1] is None:
            QMessageBox.critical(
                self,
                "Alert",
                "Requires the channel for segmentation to be loaded into either the nodes or edges channels"
            )
            return

        try:
            import cupy as cp
        except:
            #print("Cupy import failed, using CPU version")
            GPU = False

        if self.parent().mini_overlay_data is not None:
            self.parent().mini_overlay_data = None

        self.parent().machine_window = MachineWindow(self.parent(), GPU = GPU)
        self.parent().machine_window.show()  # Non-modal window
        self.accept()

    def confirm_machine_dialog(self):
        """Shows a dialog asking user to confirm if they want to start the segmenter"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Alert")
        msg.setInformativeText("Use of this feature will require use of both overlay channels and the highlight overlay. Please save any data and return, or proceed if you do not need those overlays")
        msg.setWindowTitle("Proceed?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

class ExcelotronManager(QObject):
    # Signal to emit when data is received from Excelotron
    data_received = pyqtSignal(dict, str, bool)  # dictionary, property_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.excelotron_window = None
        self.last_data = None
        self.last_property = None
        self.last_add = None
    
    def launch(self):
        """Launch the Excelotron window"""
        
        if self.excelotron_window is None:
            ExcelGUIClass = excelotron.main(standalone=False)
            self.excelotron_window = ExcelGUIClass()
            self.excelotron_window.data_exported.connect(self._on_data_exported)
            # Connect to both close event and destroyed signal
            self.excelotron_window.destroyed.connect(self._on_window_destroyed)
            self.excelotron_window.closeEvent = self._create_close_handler(self.excelotron_window.closeEvent)
            self.excelotron_window.show()
        else:
            self.excelotron_window.raise_()
            self.excelotron_window.activateWindow()
    
    def _create_close_handler(self, original_close_event):
        """Create a close event handler that cleans up properly"""
        def close_handler(event):
            self._cleanup_window()
            original_close_event(event)
        return close_handler
    
    def close(self):
        """Close the Excelotron window"""
        if self.excelotron_window is not None:
            self.excelotron_window.close()
            self._cleanup_window()
    
    def _cleanup_window(self):
        """Properly cleanup the window reference"""
        if self.excelotron_window is not None:
            try:
                # Disconnect all signals to prevent issues
                self.excelotron_window.data_exported.disconnect()
                self.excelotron_window.destroyed.disconnect()
            except:
                pass  # Ignore if already disconnected
            
            # Schedule for deletion
            self.excelotron_window.deleteLater()
            self.excelotron_window = None
    
    def is_open(self):
        """Check if Excelotron window is open"""
        is_open = self.excelotron_window is not None
        return is_open
    
    def _on_data_exported(self, data_dict, property_name, add):
        """Internal slot to handle data from Excelotron"""
        self.last_data = data_dict
        self.last_property = property_name
        self.last_add = add
        # Re-emit the signal for parent to handle
        self.data_received.emit(data_dict, property_name, add)
    
    def _on_window_destroyed(self):
        """Handle when the Excelotron window is destroyed/closed"""
        self.excelotron_window = None
    
    def get_last_data(self):
        """Get the last exported data"""
        return self.last_data, self.last_property, self.last_add

class MachineWindow(QMainWindow):

    def __init__(self, parent=None, GPU = False, tutorial_example = False):
        super().__init__(parent)

        try:

            self.tutorial_example = tutorial_example

            if not tutorial_example:
                if self.parent().active_channel == 0:
                    if self.parent().channel_data[0] is not None:
                        try:
                            active_data = self.parent().channel_data[0]
                            act_channel = 0
                        except:
                            active_data = self.parent().channel_data[1]
                            act_channel = 1
                    else:
                        active_data = self.parent().channel_data[1]
                        act_channel = 1

                try:
                    if len(active_data.shape) == 3:
                        array1 = np.zeros_like(active_data).astype(np.uint8)
                    elif len(active_data.shape) == 4:
                        array1 = np.zeros_like(active_data)[:,:,:,0].astype(np.uint8)
                except:
                    print("No data in nodes channel")
                    return

            if not tutorial_example:
                self.setWindowTitle("Segmenter")
            else:
                self.setWindowTitle("Tutorial Segmenter View (This window will not actually segment)")

            
            # Create central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)


            # Create form layout for inputs
            form_layout = QFormLayout()

            layout.addLayout(form_layout)

            if self.parent().pen_button.isChecked(): #Disable the pen mode if the user is in it because the segmenter pen forks it
                self.parent().pen_button.click()
            self.parent().threed = False
            self.parent().can = False
            self.parent().last_change = None

            self.parent().pen_button.setEnabled(False)

            if not tutorial_example:
                array3 = np.zeros_like(array1).astype(np.uint8)
                self.parent().highlight_overlay = array3 #Clear this out for the segmenter to use

                self.parent().load_channel(2, array1, True)
                # Enable the channel button
                # Not exactly sure why we need all this but the channel buttons weren't loading like they normally do when load_channel() is called:
                if not self.parent().channel_buttons[2].isEnabled():
                    self.parent().channel_buttons[2].setEnabled(True)
                    self.parent().channel_buttons[2].click()
                self.parent().delete_buttons[2].setEnabled(True)

                if len(active_data.shape) == 3:
                    self.parent().base_colors[act_channel] = self.parent().color_dictionary['WHITE']
                self.parent().base_colors[2] = self.parent().color_dictionary['LIGHT_GREEN']

                self.parent().update_display()
            
            # Set a reasonable default size for the window
            self.setMinimumWidth(600)  # Increased to accommodate grouped buttons
            self.setMinimumHeight(500)

            # Create main layout container
            main_widget = QWidget()
            main_layout = QVBoxLayout(main_widget)

            # Group 1: Drawing tools (Brush + Foreground/Background)c
            drawing_group = QGroupBox("Drawing Tools (Left Click = Draw, Right Click = Erase, 'Ctrl + Mousewheel' = Resize Brush, 'Ctrl + Z = Undo')")
            drawing_layout = QHBoxLayout()

            # Brush button
            self.brush_button = QPushButton("🖌️")
            self.brush_button.setToolTip("Segmentation Brush")
            self.brush_button.setCheckable(True)
            self.brush_button.setFixedSize(40, 40)
            self.brush_button.clicked.connect(self.toggle_brush_mode)
            self.brush_button.click()

            # Foreground/Background buttons in their own horizontal layout
            fb_layout = QHBoxLayout()
            self.fore_button = QPushButton("Foreground ('A' = Toggle)")
            self.fore_button.setCheckable(True)
            self.fore_button.setChecked(True)
            self.fore_button.clicked.connect(self.toggle_foreground)

            self.back_button = QPushButton("Background")
            self.back_button.setCheckable(True)
            self.back_button.setChecked(False)
            self.back_button.clicked.connect(self.toggle_background)

            fb_layout.addWidget(self.fore_button)
            fb_layout.addWidget(self.back_button)

            drawing_layout.addWidget(self.brush_button)
            drawing_layout.addLayout(fb_layout)
            drawing_group.setLayout(drawing_layout)

            # Group 2: Processing Options (GPU)
            processing_group = QGroupBox("Processing Options")
            processing_layout = QHBoxLayout()

            self.use_gpu = GPU
            self.two = QPushButton("Train By 2D Slice Patterns")
            self.two.setCheckable(True)
            self.two.setChecked(False)
            self.two.clicked.connect(self.toggle_two)
            self.use_two = False
            self.three = QPushButton("Train by 3D Patterns")
            self.three.setCheckable(True)
            self.three.setChecked(True)
            self.three.clicked.connect(self.toggle_three)
            self.GPU = QPushButton("GPU")
            self.GPU.setCheckable(True)
            self.GPU.setChecked(False)
            self.GPU.clicked.connect(self.toggle_GPU)
            processing_layout.addWidget(self.GPU)
            processing_layout.addWidget(self.two)
            processing_layout.addWidget(self.three)
            processing_group.setLayout(processing_layout)

            # Group 3: Training Options
            self.speed = True
            training_group = QGroupBox("Training ('T' = Train with Previous Mode)")
            training_layout = QHBoxLayout()
            train_quick = QPushButton("Train Quick Model (When Good SNR)")
            train_quick.clicked.connect(self.train_quick)
            train_detailed = QPushButton("Train Detailed Model (For Morphology)")
            train_detailed.clicked.connect(self.train_det)
            training_layout.addWidget(train_quick)
            training_layout.addWidget(train_detailed)
            training_group.setLayout(training_layout)

            # Group 4: Segmentation Options
            segmentation_group = QGroupBox("Segmentation")
            segmentation_layout = QHBoxLayout()
            seg_button = QPushButton("Preview Segment")
            self.seg_button = seg_button
            seg_button.clicked.connect(self.start_segmentation)
            self.pause_button = QPushButton("▶/⏸️")
            self.pause_button.setFixedSize(40, 40)
            self.pause_button.clicked.connect(self.toggle_segment)
            self.lock_button = QPushButton("🔒 Memory lock - (Prioritize RAM)")
            self.lock_button.setCheckable(True)
            self.lock_button.setChecked(True)
            self.lock_button.clicked.connect(self.toggle_lock)
            self.mem_lock = True
            full_button = QPushButton("Segment All")
            full_button.clicked.connect(self.segment)
            segmentation_layout.addWidget(seg_button)
            segmentation_layout.addWidget(self.pause_button)
            segmentation_layout.addWidget(full_button)
            segmentation_group.setLayout(segmentation_layout)

            # Group 5: Loading Options
            loading_group = QGroupBox("Saving/Loading")
            loading_layout = QHBoxLayout()
            self.save = QPushButton("Save Model")
            self.save.clicked.connect(self.save_model)
            self.load = QPushButton("Load Model")
            self.load.clicked.connect(self.load_model)
            load_nodes = QPushButton("Load Image (For Seg - Supports Color Images)")
            load_nodes.clicked.connect(self.load_nodes)
            loading_layout.addWidget(self.save)
            loading_layout.addWidget(self.load)
            loading_layout.addWidget(load_nodes)
            loading_group.setLayout(loading_layout) 

            # Add all groups to main layout
            main_layout.addWidget(drawing_group)
            if not GPU:
                main_layout.addWidget(processing_group)
            main_layout.addWidget(training_group)
            main_layout.addWidget(segmentation_group)
            main_layout.addWidget(loading_group)

            # Set the main widget as the central widget
            self.setCentralWidget(main_widget)

            self.trained = False
            self.previewing = False

            if not GPU:
                self.segmenter = segmenter.InteractiveSegmenter(active_data, use_gpu=False)
            else:
                self.segmenter = seg_GPU.InteractiveSegmenter(active_data)

            self.segmentation_worker = None

            self.fore_button.click()
            self.fore_button.click()

            self.num_chunks = 0


        except:
            return

    def load_nodes(self):

        def confirm_machine_dialog():
            """Shows a dialog asking user to confirm if they want to start the segmenter"""
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setText("Alert")
            msg.setInformativeText("Use of this feature will require use of both overlay channels and the highlight overlay. Please save any data and return, or proceed if you do not need those overlays")
            msg.setWindowTitle("Proceed?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            return msg.exec() == QMessageBox.StandardButton.Yes

        if self.parent().channel_data[2] is not None or self.parent().channel_data[3] is not None or self.parent().highlight_overlay is not None:
            if confirm_machine_dialog():
                pass
            else:
                return

        try:

            try:
                print("Please select image to load into nodes channel for segmentation or press X if you already have the one you want. Note that this load may permit a color image in the nodes channel for segmentation purposes only, which is otherwise not allowed.")
                self.parent().reset(nodes = True, edges = True, search_region = True, network_overlay = True, id_overlay = True)
                self.parent().highlight_overlay = None
                self.parent().load_channel(0, color = True)
                if self.parent().active_channel == 0:
                    if self.parent().channel_data[0] is not None:
                        try:
                            active_data = self.parent().channel_data[0]
                            act_channel = 0
                        except:
                            active_data = self.parent().channel_data[1]
                            act_channel = 1
                            import traceback
                            traceback.print_exc()
                    else:
                        active_data = self.parent().channel_data[1]
                        act_channel = 1

                try:
                    if len(active_data.shape) == 3:
                        array1 = np.zeros_like(active_data).astype(np.uint8)
                    elif len(active_data.shape) == 4:
                        array1 = np.zeros_like(active_data)[:,:,:,0].astype(np.uint8)
                except:
                    print("No data in nodes channel")
                    import traceback
                    traceback.print_exc()
                    return
                array3 = np.zeros_like(array1).astype(np.uint8)
                self.parent().highlight_overlay = array3 #Clear this out for the segmenter to use

                self.parent().load_channel(2, array1, True)
                self.trained = False
                self.previewing = False

                self.segmenter = segmenter.InteractiveSegmenter(active_data, use_gpu=False)

                self.segmentation_worker = None

                self.fore_button.click()
                self.fore_button.click()

                self.num_chunks = 0
                self.parent().update_display()
            except:

                pass            

        except:
            pass

    def toggle_segment(self):

        if self.segmentation_worker is not None:
            if not self.segmentation_worker._paused:
                self.segmentation_worker.pause()
                print("Segmentation Worker Paused")
            elif self.segmentation_worker._paused:
                self.segmentation_worker.resume()
                print("Segmentation Worker Resuming")


    def toggle_lock(self):

        self.mem_lock = self.lock_button.isChecked()


    def save_model(self):

        try:

            filename, _ = QFileDialog.getSaveFileName(
                self,
                f"Save Model As",
                "",  # Default directory
                "numpy data (*.npz);;All Files (*)"  # File type filter
            )
            
            if filename:  # Only proceed if user didn't cancel
                # If user didn't type an extension, add .tif
                if not filename.endswith(('.npz')):
                    filename += '.npz'

            self.segmenter.save_model(filename, self.parent().channel_data[2])

        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()


    def load_model(self):

        try:

            filename, _ = QFileDialog.getOpenFileName(
                self,
                f"Load Model",
                "",
                "numpy data (*.npz)"
            )

            self.segmenter.load_model(filename)
            self.trained = True

        except Exception as e:
            print(f"Error loading model: {e}")

    def toggle_two(self):
        if self.two.isChecked():
            # If button two is checked, ensure button three is unchecked
            self.three.setChecked(False)
            self.use_two = True
        else:
            # If button three is checked, ensure button two is unchecked
            self.three.setChecked(True)
            self.use_two = False

    def toggle_three(self):
        if self.three.isChecked():
            # If button two is checked, ensure button three is unchecked
            self.two.setChecked(False)
            self.use_two = False
        else:
            # If button three is checked, ensure button two is unchecked
            self.two.setChecked(True)
            self.use_two = True

    def toggle_GPU(self):

        if self.parent().active_channel == 0:
            if self.parent().channel_data[0] is not None:
                try:
                    active_data = self.parent().channel_data[0]
                    act_channel = 0
                except:
                    active_data = self.parent().channel_data[1]
                    act_channel = 1
            else:
                active_data = self.parent().channel_data[1]
                act_channel = 1

        if self.GPU.isChecked():

            try:
                self.segmenter = seg_GPU.InteractiveSegmenter(active_data)
                print("Using GPU")
            except:
                self.GPU.setChecked(False)
                print("Could not detect GPU")
                import traceback
                traceback.print_exc()

        else:
            self.segmenter = segmenter.InteractiveSegmenter(active_data, use_gpu=False)
            print("Using CPU")



    def toggle_foreground(self):

        self.parent().foreground = self.fore_button.isChecked()

        if self.parent().foreground:
            self.back_button.setChecked(False)
        else:
            self.back_button.setChecked(True)

    def switch_foreground(self):

        self.fore_button.click()

    def toggle_background(self):

        self.parent().foreground = not self.back_button.isChecked()

        if not self.parent().foreground:
            self.fore_button.setChecked(False)
        else:
            self.fore_button.setChecked(True)


    def toggle_brush_mode(self):
        """Toggle brush mode on/off"""
        self.parent().brush_mode = self.brush_button.isChecked()
        
        #if self.parent().pan_mode:
         #   self.parent().update_display(preserve_zoom=(self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

        if self.parent().brush_mode:

            self.parent().pm = painting.PaintManager(parent = self.parent())
            self.parent().pan_button.setChecked(False)
            self.parent().zoom_button.setChecked(False)
            if self.parent().pan_mode:
                self.parent().update_display()
            self.parent().pan_mode = False
            self.parent().zoom_mode = False
            self.parent().update_brush_cursor()
        else:
            self.parent().threed = False
            self.parent().can = False
            self.parent().zoom_button.click()

    def silence_button(self):
        self.brush_button.setChecked(False)

    def toggle_brush_button(self):

        self.brush_button.click()

    def train_quick(self):
        self.speed = True
        self.train_model()

    def train_det(self):
        self.speed = False
        self.train_model()

    def train_model(self):

        self.kill_segmentation()
        # Wait a bit for cleanup
        time.sleep(0.1)

        if (hasattr(self.parent(), 'completed_paint_strokes') and self.parent().completed_paint_strokes) or \
           (hasattr(self.parent(), 'current_stroke_points') and self.parent().current_stroke_points) or \
           (hasattr(self.parent(), 'virtual_paint_items') and self.parent().virtual_paint_items) or \
           (hasattr(self.parent(), 'current_paint_items') and self.parent().current_paint_items):
            if hasattr(self.parent(), 'current_stroke_points') and self.parent().current_stroke_points:
                self.parent().pm.finish_current_virtual_operation()
            self.parent().pm.convert_virtual_strokes_to_data()

        self.previewing = True
        try:
            try:
                self.segmenter.train_batch(self.parent().channel_data[2], speed = self.speed, use_gpu = self.use_gpu, use_two = self.use_two, mem_lock = self.mem_lock)
                self.trained = True
                self.start_segmentation()
            except Exception as e:
                print("Error training. Perhaps you forgot both foreground and background markers? I need both!")
                import traceback
                traceback.print_exc()
        except MemoryError:
            QMessageBox.critical(
                self,
                "Alert",
                "Out of memory computing feature maps. Note these for 3D require 7x the RAM of the active image (or 9x for the detailed map).\n Please use 2D slice models or RAM lock if you do not have enough RAM."
            )



    def start_segmentation(self):

        if self.parent().pan_mode:
            self.parent().pan_button.click()


        self.parent().static_background = None

        self.kill_segmentation()
        time.sleep(0.1)

        if not self.trained:
            return

        if self.parent().channel_data[2] is not None:
            active_data = self.parent().channel_data[2]
        else:
            active_data = self.parent().channel_data[0]

        array3 = np.zeros_like(active_data).astype(np.uint8)
        self.parent().highlight_overlay = array3 #Clear this out for the segmenter to use

        self.segmentation_worker = SegmentationWorker(self.parent().highlight_overlay, self.segmenter, self.use_gpu, self.use_two, self.previewing, self, self.mem_lock)
        self.segmentation_worker.chunk_processed.connect(self.update_display)  # Just update display
        current_xlim = self.parent().ax.get_xlim()
        current_ylim = self.parent().ax.get_ylim()
        try:
            x, y = self.parent().get_current_mouse_position()
        except:
            x, y = 0, 0
        self.segmenter.update_position(self.parent().current_slice, x, y)
        self.segmentation_worker.start()

    def confirm_seg_dialog(self):
        """Shows a dialog asking user to confirm segment all"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Alert")
        msg.setInformativeText("Segment Entire Image? (Window will freeze for processing)")
        msg.setWindowTitle("Confirm")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def confirm_close_dialog(self):
        """Shows a dialog asking user to confirm segment all"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Alert")
        msg.setInformativeText("Close Window?")
        msg.setWindowTitle("Confirm")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes


    def update_display(self):
        if not hasattr(self, '_last_update'):
            self._last_update = 0
        
        current_z = self.parent().current_slice
        if not hasattr(self, '_last_z'):
            self._last_z = current_z

        self._last_z = current_z

        self.num_chunks += 1

        current_time = time.time()
        if current_time - self._last_update >= 1:  # Match worker's interval
            try:

                try:
                    x, y = self.parent().get_current_mouse_position()
                except:
                    x, y = 0, 0
                self.segmenter.update_position(self.parent().current_slice, x, y)

                if not self.parent().painting:
                    # Only update if view limits are valid
                    self.parent().update_display()
                    
                    self._last_update = current_time
            except Exception as e:
                print(f"Display update error: {e}")

    def poke_segmenter(self):
        try:
            # Clear any processing flags in the segmenter
            if hasattr(self.segmenter, '_currently_processing'):
                self.segmenter._currently_processing = None
                
            # Force regenerating the worker
            if self.segmentation_worker is not None:
                self.kill_segmentation()
                
            time.sleep(0.2)
            self.start_segmentation()
            
        except Exception as e:
            print(f"Error in poke_segmenter: {e}")
            import traceback
            traceback.print_exc()


    def kill_segmentation(self):
        if hasattr(self, 'segmentation_worker') and self.segmentation_worker is not None:
            # Signal the thread to stop
            self.segmentation_worker.stop()
            
            # Wait for the thread to finish
            if self.segmentation_worker.isRunning():
                self.segmentation_worker.wait(1000)  # Wait up to 1 second
                
                # If thread is still running after timeout, try to force termination
                if self.segmentation_worker.isRunning():
                    self.segmentation_worker.terminate()
                    self.segmentation_worker.wait()  # Wait for it to be terminated
            
            # Now safe to delete
            del self.segmentation_worker
            self.segmentation_worker = None


    def segment(self):

        if not self.trained:
            return
        elif not self.confirm_seg_dialog():
            return
        else:
            self.kill_segmentation()
            time.sleep(0.1)

            self.previewing = False

            if self.parent().channel_data[2] is not None:
                active_data = self.parent().channel_data[2]
            else:
                active_data = self.parent().channel_data[0]

            array3 = np.zeros_like(active_data).astype(np.uint8)
            self.parent().highlight_overlay = array3 #Clear this out for the segmenter to use

            print("Segmenting entire volume with model...")
            #foreground_coords, background_coords = self.segmenter.segment_volume(array = self.parent().highlight_overlay)
            try:
                self.parent().highlight_overlay = self.segmenter.segment_volume(array = self.parent().highlight_overlay)
            except Exception as e:
                print(f"Error segmenting (Perhaps retrain the model...): {e}")
                import traceback
                traceback.print_exc()
                return

            # Clean up when done
            self.segmenter.cleanup()

        self.parent().load_channel(3, self.parent().highlight_overlay, True)

        # Not exactly sure why we need all this but the channel buttons weren't loading like they normally do when load_channel() is called:
        self.parent().channel_buttons[3].setEnabled(True)
        self.parent().channel_buttons[3].click()
        self.parent().delete_buttons[3].setEnabled(True)

        self.parent().highlight_overlay = None

        self.parent().update_display()

        self.previewing = False

        print("Finished segmentation moved to Overlay 2. Use File -> Save(As) for disk saving.")

    def closeEvent(self, event):
        try:
            if not self.tutorial_example:
                if self.parent() and self.parent().isVisible():
                    if self.confirm_close_dialog():
                        # Clean up resources before closing
                        if self.brush_button.isChecked():
                            self.silence_button()
                            self.toggle_brush_mode()
                        
                        self.parent().pen_button.setEnabled(True)
                        self.parent().brush_mode = False
                        
                        # Kill the segmentation thread and wait for it to finish
                        self.kill_segmentation()
                        time.sleep(0.2)  # Give additional time for cleanup
                        try:
                            self.parent().load_channel(0, self.parent().reduce_rgb_dimension(self.parent().channel_data[0], 'weight'), True)
                        except:
                            pass
                        
                        self.parent().machine_window = None
                        self.parent().highlight_overlay = None
                        event.accept()
                    else:
                        event.ignore()  # User cancelled, ignore the close
                else:
                    # Parent doesn't exist or isn't visible, just close
                    if hasattr(self, 'parent') and self.parent():
                        self.parent().machine_window = None
                    event.accept()
            else:
                self.parent().machine_window = None
                if self.brush_button.isChecked():
                    self.silence_button()
                    self.toggle_brush_mode()
                self.parent().pen_button.setEnabled(True)
                self.parent().brush_mode = False

        except Exception as e:
            print(f"Error in closeEvent: {e}")
            # Even if there's an error, allow the window to close
            if hasattr(self, 'parent') and self.parent():
                self.parent().machine_window = None
            event.accept()




class SegmentationWorker(QThread):
    finished = pyqtSignal()
    chunk_processed = pyqtSignal()
    
    def __init__(self, highlight_overlay, segmenter, use_gpu, use_two, previewing, machine_window, mem_lock):
        super().__init__()
        self.overlay = highlight_overlay
        self.segmenter = segmenter
        self.use_gpu = use_gpu
        self.use_two = use_two
        self.previewing = previewing
        self.machine_window = machine_window
        self.mem_lock = mem_lock
        self._stop = False
        self._paused = False  # Add pause flag
        self.update_interval = 2  # Increased to 2s
        self.chunks_since_update = 0
        self.chunks_per_update = 5  # Only update every 5 chunks
        self.poked = False # If it should wake up or not
        self.last_update = time.time()
        
    def stop(self):
        self._stop = True

    def pause(self):
        """Pause the segmentation worker"""
        self._paused = True

    def resume(self):
        """Resume the segmentation worker"""
        self._paused = False

    def is_paused(self):
        """Check if the worker is currently paused"""
        return self._paused

    def _check_pause(self):
        """Check if paused and wait until resumed"""
        while self._paused and not self._stop:
            self.msleep(50)  # Sleep for 50ms while paused

    def get_poked(self):
        self.machine_window.poke_segmenter()
        
    def run(self):
        try:
            self.overlay.fill(False)
            
            # Remember the starting z position
            self.starting_z = self.segmenter.current_z

            # Original 3D approach
            for foreground_coords, background_coords in self.segmenter.segment_volume_realtime(gpu=self.use_gpu):
                # Check for pause/stop before processing each chunk
                self._check_pause()
                if self._stop:
                    break
                
                if foreground_coords:
                    fg_array = np.array(list(foreground_coords))
                    self.overlay[fg_array[:, 0], fg_array[:, 1], fg_array[:, 2]] = 1

                if background_coords:
                    bg_array = np.array(list(background_coords))
                    self.overlay[bg_array[:, 0], bg_array[:, 1], bg_array[:, 2]] = 2

                self.chunks_since_update += 1
                current_time = time.time()
                if (self.chunks_since_update >= self.chunks_per_update and 
                    current_time - self.last_update >= self.update_interval):
                    self.chunk_processed.emit()
                    self.chunks_since_update = 0
                    self.last_update = current_time 

            self.finished.emit()

            
        except Exception as e:
            print(f"Error in segmentation: {e}")
            import traceback
            traceback.print_exc()


class ThresholdWindow(QMainWindow):
    processing_complete = pyqtSignal()  # Emitted when user finishes and images are modified
    processing_cancelled = pyqtSignal()  # Emitted when user cancels

    def __init__(self, parent=None, accepted_mode=0, nohigh = False, title = None, has_model = False, automated = False):
        super().__init__(parent)
        self.parent().thresh_window_ref = self
        if title:
            self.setWindowTitle(f"Threshold - {title}")
            self.title = title
        else:
            self.setWindowTitle("Threshold")


        self.accepted_mode = accepted_mode
        self.preview = True
        self.nohigh = nohigh
        self.has_model = has_model
        self.automated = automated
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Get histogram data
        if accepted_mode == 1:
            self.histo_list = list(self.parent().volume_dict[self.parent().active_channel].values())
            self.bounds = False
            self.parent().bounds = False
        elif accepted_mode == 2:
            self.histo_list = list(self.parent().radii_dict[self.parent().active_channel].values())
            self.bounds = False
            self.parent().bounds = False
        elif accepted_mode == 3:
            self.parent().degree_dict = {}
            self.parent().set_active_channel(0)
            nodes = list(my_network.network.nodes())
            img_nodes = list(np.unique(my_network.nodes))
            from . import histos
            G = histos.convert_to_multigraph(my_network.network)
            if 0 in img_nodes:
                del img_nodes[0]
            for node in img_nodes:
                if node in nodes:
                    self.parent().degree_dict[int(node)] = G.degree(node)
                else:
                    self.parent().degree_dict[int(node)] = 0

            self.histo_list = list(self.parent().degree_dict.values())
            self.bounds = False
            self.parent().bounds = False
        elif accepted_mode == 4:
            self.histo_list = list(self.parent().special_dict.values())
            self.bounds = False
            self.parent().bounds = False

        elif accepted_mode == 0:
            data = self.parent().channel_data[self.parent().active_channel]
            nonzero_data = data[data != 0]

            MAX_SAMPLES_FOR_HISTOGRAM = 10_000_000  # Downsample data above this size
            MAX_HISTOGRAM_BINS = 512  # Maximum bins for smooth matplotlib interaction
            MIN_HISTOGRAM_BINS = 128  # Minimum bins for decent resolution

            # Always compute min/max first (before any downsampling)
            self.data_min = np.min(nonzero_data)
            self.data_max = np.max(nonzero_data)
            self.histo_list = [self.data_min, self.data_max]

            # Downsample data if too large
            if nonzero_data.size > MAX_SAMPLES_FOR_HISTOGRAM:
                downsample_factor = int(np.ceil(nonzero_data.size / MAX_SAMPLES_FOR_HISTOGRAM))
                nonzero_data_sampled = n3d.downsample(nonzero_data, downsample_factor)
            else:
                nonzero_data_sampled = nonzero_data

            # Calculate optimal bin count (capped for matplotlib performance)
            # Using Sturges' rule but capped to reasonable limits
            n_bins = int(np.ceil(np.log2(nonzero_data_sampled.size)) + 1)
            n_bins = np.clip(n_bins, MIN_HISTOGRAM_BINS, MAX_HISTOGRAM_BINS)

            self.counts, self.bin_edges = np.histogram(nonzero_data_sampled, bins=n_bins, density=False)
            
            self.bounds = True
            self.parent().bounds = True

        self.chan = self.parent().active_channel

            
        # Create matplotlib figure
        fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        
        # Pre-compute histogram with numpy
        if accepted_mode != 0:
            # First create initial histogram to analyze data distribution
            initial_counts, initial_edges = np.histogram(self.histo_list, bins=50)
            
            # Detect empty region on the right
            max_count = np.max(initial_counts)
            
            if max_count > 0:
                # Use a threshold - buckets with very few counts relative to max
                threshold = max(2, 0.01 * max_count)
                significant_buckets = np.where(initial_counts >= threshold)[0]
                
                if len(significant_buckets) > 0:
                    last_significant_idx = significant_buckets[-1]
                    empty_buckets = len(initial_counts) - last_significant_idx - 1
                    
                    if empty_buckets >= 3:
                        # There's a significant empty region, use special binning
                        visible_max = initial_edges[last_significant_idx + 1]
                        actual_max = max(self.histo_list)
                        data_min_val = min(self.histo_list)
                        
                        # Create 49 equal-width bins from min to visible_max
                        visible_bin_edges = np.linspace(data_min_val, visible_max, 50)
                        
                        # Create 21 equal-width bins from visible_max to actual_max
                        tail_bin_edges = np.linspace(visible_max, actual_max, 22)[1:]  # Skip first edge (duplicate)
                        
                        # Combine all edges
                        all_bin_edges = np.concatenate([visible_bin_edges, tail_bin_edges])
                        
                        # Compute histogram with these custom bins (49 + 21 = 70 bins total)
                        self.counts, self.bin_edges = np.histogram(self.histo_list, bins=all_bin_edges)
                        
                        # Store info for non-linear x-axis
                        self.has_nonuniform_bins = True
                        self.visible_max = visible_max
                        
                        # Visual positions: 70 total positions, proportional spacing
                        # First 49 bins: positions 0-49 (each gets width 1)
                        # Last 21 bins: positions 49-70 (each gets width 1)
                        visible_positions = np.arange(49)  # 0, 1, 2, ..., 48
                        tail_positions = np.arange(49, 70)  # 49, 50, 51, ..., 69
                        self.visual_positions = np.concatenate([visible_positions, tail_positions])
                    else:
                        self.counts, self.bin_edges = initial_counts, initial_edges
                        self.has_nonuniform_bins = False
                        self.visual_positions = None
                else:
                    self.counts, self.bin_edges = initial_counts, initial_edges
                    self.has_nonuniform_bins = False
                    self.visual_positions = None
            else:
                self.counts, self.bin_edges = initial_counts, initial_edges
                self.has_nonuniform_bins = False
                self.visual_positions = None
            
            # Store histogram bounds
            if self.bounds:
                self.data_min = 0
            else:
                self.data_min = min(self.histo_list)
            self.data_max = max(self.histo_list)
        else:
            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            bin_width = self.bin_edges[1] - self.bin_edges[0]
            self.has_nonuniform_bins = False
            self.visual_positions = None

        # Plot pre-computed histogram
        self.ax = fig.add_subplot(111)

        if self.has_nonuniform_bins:
            # Plot using visual positions - all bins have width 1.0 in visual space
            self.ax.bar(self.visual_positions, self.counts, width=1.0, alpha=0.5, align='center')
            
            # Set x-axis limits with padding (total visual range is 0-70)
            self.ax.set_xlim(-3.5, 73.5)
            
            # Create custom tick labels to show actual values
            self.ax.set_xticks([])  # Remove default ticks
            
            # Add text annotations - visible_max above, others below
            y_min, y_max = self.ax.get_ylim()
            text_y_below = y_min - 0.08 * (y_max - y_min)  # Below the axis
            text_y_above = y_max + 0.05 * (y_max - y_min)  # Above the histogram
            
            # Min label (below)
            self.ax.text(0, text_y_below, f'{self.data_min:.1f}', 
                        ha='center', va='top', fontsize=9, color='black')
            
            # Visible_max label (above, in white space) - positioned at boundary between regions
            self.ax.text(49, text_y_above, f'Long Tail Start: {self.visible_max:.1f}', 
                        ha='center', va='bottom', fontsize=9, color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8))
            
            # Max label (below)
            self.ax.text(69, text_y_below, f'{self.data_max:.1f}', 
                        ha='center', va='top', fontsize=9, color='black')
            
            # Add grid lines at these positions for clarity (subtle)
            for pos in [0, 49, 69]:
                self.ax.axvline(pos, color='lightgray', linestyle=':', alpha=0.5, linewidth=0.8)
            
            # Add vertical line for visible_max (green, prominent) - at position 49
            self.visible_max_line = self.ax.axvline(49, color='green', linestyle='--', 
                                                   linewidth=2.5, alpha=0.7, zorder=9, 
                                                   label='Tail Start')
            
            # Add vertical lines for thresholds (in visual space initially, more prominent)
            self.min_line = self.ax.axvline(self.value_to_visual(self.data_min), color='r', 
                                           linewidth=2.5, alpha=0.8, zorder=10, label='Min')
            self.max_line = self.ax.axvline(self.value_to_visual(self.data_max), color='b', 
                                           linewidth=2.5, alpha=0.8, zorder=10, label='Max')
            
            #self.ax.legend(loc='best', fontsize='small')
        else:
            # Standard linear plotting
            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
            self.ax.bar(bin_centers, self.counts, width=bin_widths, alpha=0.5, align='center')
            
            # Add padding (5% on each side)
            data_range = self.data_max - self.data_min
            padding = 0.05 * data_range
            self.ax.set_xlim(self.data_min - padding, self.data_max + padding)
            
            # Add vertical lines for thresholds (in actual value space, more prominent)
            self.min_line = self.ax.axvline(self.data_min, color='r', linewidth=2.5, alpha=0.8, 
                                           zorder=10, label='Min')
            self.max_line = self.ax.axvline(self.data_max, color='b', linewidth=2.5, alpha=0.8, 
                                           zorder=10, label='Max')
        
        # Connect events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.dragging = None

        # Create form layout for inputs
        form_layout = QFormLayout()
        
        self.min = QLineEdit(f"{self.data_min}")
        self.min.editingFinished.connect(self.min_value_changed)
        form_layout.addRow("Minimum Value to retain:", self.min)
        self.prev_min = self.data_min
        
        self.max = QLineEdit(f"{self.data_max}")
        self.max.editingFinished.connect(self.max_value_changed)
        form_layout.addRow("Maximum Value to retain:", self.max)
        self.prev_max = self.data_max

        self.targs = [self.prev_min, self.prev_max]
        
        # preview checkbox (default False)
        self.preview = QPushButton("Preview")
        self.preview.setCheckable(True)
        self.preview.setChecked(False)
        self.preview.clicked.connect(self.preview_mode)
        form_layout.addRow("Show Preview:", self.preview)

        button_layout = QHBoxLayout()


        # Keep your existing Apply Threshold button, but modify its behavior
        self.run_button = QPushButton("Apply Threshold/Continue")
        self.run_button.clicked.connect(self.apply_and_continue)  # New method
        button_layout.addWidget(self.run_button)

        if self.has_model:
            self.model_button = QPushButton("Predict Bounds with Current Model")
            self.model_button.clicked.connect(self.predict)
            button_layout.addWidget(self.model_button)
        
        # Add Cancel button for external dialog use
        cancel_button = QPushButton("Cancel/Skip (Retains Selection)")
        cancel_button.clicked.connect(self.cancel_processing)
        button_layout.addWidget(cancel_button)
        
        form_layout.addRow(button_layout)
        layout.addLayout(form_layout)
                
        # Set a reasonable default size
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    def value_to_visual(self, value):
        """Convert actual data value to visual x-axis position"""
        if not self.has_nonuniform_bins:
            return value
        
        # Clamp value to valid range
        value = np.clip(value, self.data_min, self.data_max)
        
        if value <= self.visible_max:
            # Linear mapping in the visible region (0-49)
            visual_pos = (value - self.data_min) / (self.visible_max - self.data_min) * 49
        else:
            # Map to the tail region (49-70)
            visual_pos = 49 + (value - self.visible_max) / (self.data_max - self.visible_max) * 21
        
        return visual_pos

    def visual_to_value(self, visual_pos):
        """Convert visual x-axis position to actual data value"""
        if not self.has_nonuniform_bins:
            return visual_pos
        
        # Clamp to valid visual range
        visual_pos = np.clip(visual_pos, 0, 70)
        
        if visual_pos <= 49:
            # Linear mapping from visible region
            value = self.data_min + (visual_pos / 49) * (self.visible_max - self.data_min)
        else:
            # Map from tail region
            value = self.visible_max + ((visual_pos - 49) / 21) * (self.data_max - self.visible_max)
        
        return value

    def showEvent(self, event):
        """Called when window is shown - trigger automation here"""
        super().showEvent(event)
        
        if self.automated:
            # Disable all user interaction
            self.setEnabled(False)
            
            # Optionally update title to show it's automated
            self.setWindowTitle(f"{self.windowTitle()} [AUTO-PROCESSING...]")
            
            # Use QTimer to execute after window is fully shown and event loop is running
            def auto_execute():
                if not self.parent().old_thresh:
                    if self.has_model:
                        self.predict()
                else:
                    self.parent().old_thresh[self.title]
                    min_thresh, max_thresh = self.parent().old_thresh[self.title]
                    self.max.setText(str(max_thresh))
                    self.min.setText(str(min_thresh))
                    self.max_value_changed()
                    self.min_value_changed()
                self.apply_and_continue()
            
            QTimer.singleShot(200, auto_execute)
        else:
            self.min.setText(str((self.data_min + self.data_max)/50))

            self.min_value_changed()

    def predict(self):
        try:
            data_dict = {'chan1': [self.counts, self.bin_edges]}
            bounds = self.has_model.predict(data_dict)
            min_thresh, max_thresh = bounds['chan1']
            self.max.setText(str(max_thresh))
            self.min.setText(str(min_thresh))
            self.max_value_changed()
            self.min_value_changed()
        except Exception as e:
            print(f"Error predicting: {e}")

    def apply_and_continue(self):
        """Apply threshold, modify main window images, then signal completion"""
        self.thresh()  # This should modify the main window images
        
        # Signal that processing is complete
        self.processing_complete.emit()
        self.close()
    
    def cancel_processing(self):
        """Cancel without applying changes"""
        self.processing_cancelled.emit()
        self.close()

    def make_full_highlight(self):

        try: # could probably be refactored but this just handles keeping the highlight elements if the user presses X
            if self.chan == 0:
                if not self.bounds:
                    self.parent().clicked_values['nodes'] = self.get_values_in_range_all_vols(self.chan, float(self.min.text()), float(self.max.text()))
                else:
                    vals = np.unique(self.parent().channel_data[self.chan])
                    self.parent().clicked_values['nodes'] = (vals[(vals >= float(self.min.text())) & (vals <= float(self.max.text()))]).tolist()

                if self.parent().channel_data[0].shape[0] * self.parent().channel_data[0].shape[1] * self.parent().channel_data[0].shape[2] > self.parent().mini_thresh:
                    self.parent().mini_overlay = True
                    self.parent().create_mini_overlay(node_indices = self.parent().clicked_values['nodes'])
                else:
                    self.parent().create_highlight_overlay(
                        node_indices=self.parent().clicked_values['nodes']
                    )
            elif self.chan == 1:
                if not self.bounds:
                    self.parent().clicked_values['edges'] = self.get_values_in_range_all_vols(self.chan, float(self.min.text()), float(self.max.text()))
                else:
                    vals = np.unique(self.parent().channel_data[self.chan])
                    self.parent().clicked_values['edges'] = (vals[(vals >= float(self.min.text())) & (vals <= float(self.max.text()))]).tolist()

                if self.parent().channel_data[1].shape[0] * self.parent().channel_data[1].shape[1] * self.parent().channel_data[1].shape[2] > self.parent().mini_thresh:
                    self.parent().mini_overlay = True
                    self.parent().create_mini_overlay(edge_indices = self.parent().clicked_values['edges'])
                else:
                    self.parent().create_highlight_overlay(
                        node_indices=self.parent().clicked_values['edges']
                    )
        except:
            pass


    def closeEvent(self, event):
        self.parent().preview = False
        self.parent().targs = None
        self.parent().bounds = False
        self.parent().thresh_window_ref = None
        self.counts = None
        self.bin_edges = None
        self.make_full_highlight()
        self.processing_cancelled.emit()


    def get_values_in_range_all_vols(self, chan, min_val, max_val):
        # Select the appropriate dictionary based on mode
        if self.accepted_mode == 1:
            data_dict = self.parent().volume_dict[chan]
        elif self.accepted_mode == 2:
            data_dict = self.parent().radii_dict[chan]
        elif self.accepted_mode == 3:
            data_dict = self.parent().degree_dict
        elif self.accepted_mode == 4:
            data_dict = self.parent().special_dict
        else:
            return []
        
        # Convert to numpy arrays once
        nodes = np.array(list(data_dict.keys()))
        values = np.array(list(data_dict.values()))
        
        # Vectorized range check (much faster than loop)
        mask = (values >= min_val) & (values <= max_val)
        # Return filtered nodes
        return nodes[mask].tolist()

    def get_values_in_range(self, lst, min_val, max_val):
        # Vectorized range filtering

        lst_array = np.array(lst)
        values = lst_array[(lst_array >= min_val) & (lst_array <= max_val)]
        
        # If no values in range, return early
        if len(values) == 0:
            return []
        
        # Select appropriate dictionary
        if self.accepted_mode == 1:
            data_dict = self.parent().volume_dict[self.parent().active_channel]
        elif self.accepted_mode == 2:
            data_dict = self.parent().radii_dict[self.parent().active_channel]
        elif self.accepted_mode == 3:
            data_dict = self.parent().degree_dict
        elif self.accepted_mode == 4:
            data_dict = self.parent().special_dict
        else:
            return []
        
        # Convert to arrays
        nodes = np.array(list(data_dict.keys()))
        dict_values = np.array(list(data_dict.values()))
        
        # Vectorized membership test (much faster than 'in values' loop)
        mask = np.isin(dict_values, values)
        
        return nodes[mask].tolist()


    def min_value_changed(self):
        try:
            if not self.preview.isChecked():
                self.preview.click()
            text = self.min.text()
            if not text:  # If empty, ignore
                return
            
            try:
                value = float(text)
                
                # Bound check against data limits
                value = max(self.data_min, value)

                # Check against max line
                max_val = float(self.max.text()) if self.max.text() else self.data_max
                if value > max_val:
                    # If min would exceed max, set max to its highest possible value
                    self.max.setText(str(self.data_max))
                    if self.has_nonuniform_bins:
                        self.max_line.set_xdata([self.value_to_visual(self.data_max), 
                                                self.value_to_visual(self.data_max)])
                    else:
                        self.max_line.set_xdata([self.data_max, self.data_max])
                    # And set min to the previous max value
                    value = max_val
                    self.min.setText(str(value))

                if value == self.prev_min:
                    return
                else:
                    self.prev_min = value
                    if self.bounds:
                        self.targs = [self.prev_min, self.prev_max]
                    else:
                        self.targs = self.get_values_in_range(self.histo_list, self.prev_min, self.prev_max)
                    self.parent().targs = self.targs
                    if self.preview.isChecked():
                        self.parent().highlight_overlay = None
                        self.parent().create_highlight_overlay_slice(self.targs, bounds = self.bounds)
                
                # Update the line (convert to visual coordinates if needed)
                if self.has_nonuniform_bins:
                    visual_pos = self.value_to_visual(value)
                    self.min_line.set_xdata([visual_pos, visual_pos])
                else:
                    self.min_line.set_xdata([value, value])
                
                self.canvas.draw()
                
            except ValueError:
                # If invalid number, reset to current line position
                current_x = self.min_line.get_xdata()[0]
                if self.has_nonuniform_bins:
                    actual_value = self.visual_to_value(current_x)
                else:
                    actual_value = current_x
                self.min.setText(str(actual_value))
        except:
            pass

    def max_value_changed(self):
        try:
            if not self.preview.isChecked():
                self.preview.click()
            text = self.max.text()
            if not text:  # If empty, ignore
                return
                
            try:
                value = float(text)
                
                # Bound check against data limits
                value = min(self.data_max, value)
                
                # Check against min line
                min_val = float(self.min.text()) if self.min.text() else self.data_min
                if value < min_val:
                    # If max would go below min, set min to its lowest possible value
                    self.min.setText(str(self.data_min))
                    if self.has_nonuniform_bins:
                        self.min_line.set_xdata([self.value_to_visual(self.data_min), 
                                                self.value_to_visual(self.data_min)])
                    else:
                        self.min_line.set_xdata([self.data_min, self.data_min])
                    # And set max to the previous min value
                    value = min_val
                    self.max.setText(str(value, 2))

                if value == self.prev_max:
                    return
                else:
                    self.prev_max = value
                    if self.bounds:
                        self.targs = [self.prev_min, self.prev_max]
                    else:
                        self.targs = self.get_values_in_range(self.histo_list, self.prev_min, self.prev_max)
                    self.parent().targs = self.targs
                    if self.preview.isChecked():
                        self.parent().highlight_overlay = None
                        self.parent().create_highlight_overlay_slice(self.targs, bounds = self.bounds)
                
                # Update the line (convert to visual coordinates if needed)
                if self.has_nonuniform_bins:
                    visual_pos = self.value_to_visual(value)
                    self.max_line.set_xdata([visual_pos, visual_pos])
                else:
                    self.max_line.set_xdata([value, value])
                
                self.canvas.draw()
            
            except ValueError:
                # If invalid number, reset to current line position
                current_x = self.max_line.get_xdata()[0]
                if self.has_nonuniform_bins:
                    actual_value = self.visual_to_value(current_x)
                else:
                    actual_value = current_x
                self.max.setText(str(actual_value))
        except:
            pass
        
    def on_press(self, event):
        try:
            if event.inaxes != self.ax:
                return
            
            # Left click controls left line
            if event.button == 1:  # Left click
                self.dragging = 'min'
            # Right click controls right line
            elif event.button == 3:  # Right click
                self.dragging = 'max'
        except:
            pass
                
    def on_motion(self, event):
        try:
            if not self.dragging or event.inaxes != self.ax:
                return
            
            # Convert visual position to actual value
            if self.has_nonuniform_bins:
                actual_value = self.visual_to_value(event.xdata)
            else:
                actual_value = event.xdata
                
            if self.dragging == 'min':
                max_val = float(self.max.text()) if self.max.text() else self.data_max
                if actual_value < max_val:
                    if self.has_nonuniform_bins:
                        visual_pos = self.value_to_visual(actual_value)
                        self.min_line.set_xdata([visual_pos, visual_pos])
                    else:
                        self.min_line.set_xdata([actual_value, actual_value])
                    self.min.setText(str(actual_value))
            else: 
                min_val = float(self.min.text()) if self.min.text() else self.data_min
                if actual_value > min_val:
                    if self.has_nonuniform_bins:
                        visual_pos = self.value_to_visual(actual_value)
                        self.max_line.set_xdata([visual_pos, visual_pos])
                    else:
                        self.max_line.set_xdata([actual_value, actual_value])
                    self.max.setText(str(actual_value))
                    
            self.canvas.draw()
        except:
            pass
            
    def on_release(self, event):
        self.min_value_changed()
        self.max_value_changed()
        self.dragging = None

    def preview_mode(self):
        try:
            preview = self.preview.isChecked()
            self.parent().preview = preview
            self.parent().targs = self.targs

            if preview and self.targs is not None:
                self.parent().create_highlight_overlay_slice(self.parent().targs, bounds = self.bounds)
        except:
            pass      

    def thresh(self):
        try:

            if not self.nohigh:

                if self.parent().active_channel == 0:
                    self.parent().create_highlight_overlay(node_indices = self.targs, bounds = self.bounds)
                elif self.parent().active_channel == 1:
                    self.parent().create_highlight_overlay(edge_indices = self.targs, bounds = self.bounds)
                elif self.parent().active_channel == 2:
                    self.parent().create_highlight_overlay(overlay1_indices = self.targs, bounds = self.bounds)
                elif self.parent().active_channel == 3:
                    self.parent().create_highlight_overlay(overlay2_indices = self.targs, bounds = self.bounds)
                channel_data = self.parent().channel_data[self.parent().active_channel]
                mask = self.parent().highlight_overlay > 0
                channel_data = channel_data * mask
                self.parent().load_channel(self.parent().active_channel, channel_data, True)
                self.parent().update_display()
            else:
                self.parent().thresh_vals = self.targs

            self.parent().thresh_min = self.prev_min
            self.parent().thresh_max = self.prev_max
            self.close()
            
        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                f"Error running threshold: {str(e)}"
            )

class SmartDilateDialog(QDialog):
    def __init__(self, parent, params):
        super().__init__(parent)
        self.setWindowTitle("Additional Smart Dilate Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        # dt checkbox (default False)
        self.predt = QPushButton("Fast Dilation")
        self.predt.setCheckable(True)
        self.predt.setChecked(False)
        layout.addRow("Use Fast Dilation? (Higher speed, may be rougher along adjacent boundaries):", self.predt)

        self.params = params

        # Add Run button
        run_button = QPushButton("Dilate")
        run_button.clicked.connect(self.smart_dilate)
        layout.addRow(run_button)

    def smart_dilate(self):

        predt = self.predt.isChecked()
        active_data, amount, xy_scale, z_scale = self.params

        result = sdl.smart_dilate(active_data, fast_dil = predt, use_dt_dil_amount = amount, xy_scale = xy_scale, z_scale = z_scale)

        self.parent().load_channel(self.parent().active_channel, result, True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
        self.accept()



class DilateDialog(QDialog):
    def __init__(self, parent=None, args = None):
        super().__init__(parent)
        self.setWindowTitle("Dilate Parameters")
        self.setModal(False)
        
        layout = QFormLayout(self)

        if args:
            self.parent().last_dil = args[0]
            if args[1] > 1:
                self.index = 1
            else:
                self.index = 0
        else:
            self.parent().last_dil = 1
            self.index = 0

        self.amount = QLineEdit(f"{self.parent().last_dil}")
        layout.addRow("Dilation Radius:", self.amount)

        self.xy_scale = QLineEdit("1")
        layout.addRow("xy_scale:", self.xy_scale)

        self.z_scale = QLineEdit("1")
        layout.addRow("z_scale:", self.z_scale)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Parallel Distance Transform-Based", "Preserve Labels (Distance Transform Based)", "Pseudo3D Binary Kernels (For Fast, small dilations for visualization purposes. Slightly inaccurate, moreso at large dilations)", "Distance Transform-Based (Non-Parallel Version)"])
        self.mode_selector.setCurrentIndex(self.index)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

       # Add Run button
        run_button = QPushButton("Run Dilate")
        run_button.clicked.connect(self.run_dilate)
        layout.addRow(run_button)

    def run_dilate(self):
        try:

            try: #for retaining zoom params
                current_xlim = self.parent().ax.get_xlim()
                current_ylim = self.parent().ax.get_ylim()
            except:
                current_xlim = None
                current_ylim = None


            accepted_mode = self.mode_selector.currentIndex()
            
            # Get amount
            try:
                amount = float(self.amount.text()) if self.amount.text() else 1
            except ValueError:
                amount = 1

            try:
                xy_scale = float(self.xy_scale.text()) if self.xy_scale.text() else 1
            except ValueError:
                xy_scale = 1

            try:
                z_scale = float(self.z_scale.text()) if self.z_scale.text() else 1
            except ValueError:
                z_scale = 1
            
            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")

            self.parent().last_dil = [amount, accepted_mode]

            if accepted_mode == 1:
                dialog = SmartDilateDialog(self.parent(), [active_data, amount, xy_scale, z_scale])
                dialog.exec()
                self.accept()
                return

            if accepted_mode == 0:
                result = n3d.dilate_3D_dt(active_data, amount, xy_scaling = xy_scale, z_scaling = z_scale, fast_dil = True)
            elif accepted_mode == 3:
                result = n3d.dilate_3D_dt(active_data, amount, xy_scaling = xy_scale, z_scaling = z_scale)
            else:

                # Call dilate method with parameters
                result = n3d.dilate(
                    active_data,
                    amount,
                    xy_scale = xy_scale,
                    z_scale = z_scale,
                    fast_dil = True)

            result = result * 255

            # Update both the display data and the network object
            self.parent().load_channel(self.parent().active_channel, result, True)

            self.parent().update_display(preserve_zoom=(current_xlim, current_ylim))
            self.accept()
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(
                self,
                "Error",
                f"Error running dilate: {str(e)}"
            )

class ErodeDialog(QDialog):
    def __init__(self, parent=None, args = None):
        super().__init__(parent)
        self.setWindowTitle("Erosion Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)

        if args:
            self.parent().last_ero = args[0]
            if args[1] == 1: #user opted to preserve labels
                self.index = 2 #this is where the labels option lives in the erode menu
            else:
                self.index = 0
        else:
            self.parent().last_ero = 1
            self.index = 0

        self.amount = QLineEdit(f"{self.parent().last_ero}")
        layout.addRow("Erosion Radius:", self.amount)

        self.xy_scale = QLineEdit("1")
        layout.addRow("xy_scale:", self.xy_scale)

        self.z_scale = QLineEdit("1")
        layout.addRow("z_scale:", self.z_scale)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Parallel Distance Transform Based", "Distance Transform-Based (Non-Parallel)", "Preserve Labels (Parallel)", "Preserve Labels (Non-Parallel)"])
        self.mode_selector.setCurrentIndex(self.index)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

       # Add Run button
        run_button = QPushButton("Run Erode")
        run_button.clicked.connect(self.run_erode)
        layout.addRow(run_button)

    def run_erode(self):
        try:

            try: #for retaining zoom params
                current_xlim = self.parent().ax.get_xlim()
                current_ylim = self.parent().ax.get_ylim()
            except:
                current_xlim = None
                current_ylim = None
            
            # Get amount
            try:
                amount = float(self.amount.text()) if self.amount.text() else 1
            except ValueError:
                amount = 1

            try:
                xy_scale = float(self.xy_scale.text()) if self.xy_scale.text() else 1
            except ValueError:
                xy_scale = 1

            try:
                z_scale = float(self.z_scale.text()) if self.z_scale.text() else 1
            except ValueError:
                z_scale = 1

            mode = self.mode_selector.currentIndex()

            if mode == 2 or mode == 3:
                preserve_labels = True
            else:
                preserve_labels = False
            
            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")
            
            # Call dilate method with parameters
            result = n3d.erode(
                active_data,
                amount,
                xy_scale = xy_scale,
                z_scale = z_scale,
                mode = mode,
                preserve_labels = preserve_labels
            )


            self.parent().load_channel(self.parent().active_channel, result, True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
            self.parent().last_ero = [amount, mode]
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running erode: {str(e)}"
            )

class HoleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fill Holes? (Active Image)")
        self.setModal(True)
        
        layout = QFormLayout(self)

        # auto checkbox (default True)
        self.headon = QPushButton("Head-on")
        self.headon.setCheckable(True)
        self.headon.setChecked(True)
        layout.addRow("Only Use 2D Slicing Dimension:", self.headon)

        # auto checkbox (default True)
        self.borders = QPushButton("Borders")
        self.borders.setCheckable(True)
        self.borders.setChecked(False)
        layout.addRow("Fill Small Holes Along Borders:", self.borders)

        self.preserve_labels = QPushButton("Preserve Labels")
        self.preserve_labels.setCheckable(True)
        self.preserve_labels.setChecked(False)
        layout.addRow("Preserve Labels (Slower):", self.preserve_labels)

        self.sep_holes = QPushButton("Seperate Hole Mask")
        self.sep_holes.setCheckable(True)
        self.sep_holes.setChecked(False)
        layout.addRow("Place Hole Mask in Overlay 2 (Instead of Filling):", self.sep_holes)

       # Add Run button
        run_button = QPushButton("Run Fill Holes")
        run_button.clicked.connect(self.run_holes)
        layout.addRow(run_button)

    def run_holes(self):
        try:
            
            
            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")

            borders = self.borders.isChecked()
            headon = self.headon.isChecked()
            sep_holes = self.sep_holes.isChecked()
            preserve_labels = self.preserve_labels.isChecked()
            if preserve_labels:
                label_copy = np.copy(active_data)

            if borders:
            
                # Call dilate method with parameters
                result = n3d.fill_holes_3d_old(
                    active_data,
                    head_on = headon,
                    fill_borders = borders
                )

            else:
                # Call dilate method with parameters
                result = n3d.fill_holes_3d(
                    active_data,
                    head_on = headon,
                    fill_borders = borders
                )


            if not sep_holes:
                if preserve_labels:
                    result = sdl.smart_label(result, label_copy, directory = None, GPU = False, remove_template = True)

                self.parent().load_channel(self.parent().active_channel, result, True)
            else:
                self.parent().load_channel(3, active_data - result, True)


            self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running fill holes: {str(e)}"
            )

class HexNodeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Hexagonal Nodes (Can serve as artificial cells to be labeled for channel identity). Dimensions match current image (requires any image to be loaded).")
        self.setModal(False)
        
        main_layout = QFormLayout(self)

        self.side_length = QLineEdit("")
        self.side_length.setPlaceholderText("Enter number...")
        main_layout.addRow("Hexagon Side Length?: ", self.side_length)

        self.xy_scale = QLineEdit(f"{my_network.xy_scale}")
        main_layout.addRow("xy scale for side length?: ", self.xy_scale)

        self.z_scale = QLineEdit(f"{my_network.z_scale}")
        main_layout.addRow("z scale for side depth?: ", self.z_scale)

        self.down_factor = QLineEdit("")
        main_layout.addRow("(Optional) - Internal Downsample Factor?: ", self.down_factor)

        # Add mode selection dropdown
        self.mode = QComboBox()
        self.mode.addItems(["Dodecahedrons (More 3D)", "Prisms (Faster)"])
        self.mode.setCurrentIndex(0)
        main_layout.addRow("For 3D Data:", self.mode)

        self.mask = QPushButton("Mask")
        self.mask.setCheckable(True)
        self.mask.setChecked(False)
        main_layout.addRow("Mask Hexagons by Binary Mask in Nodes Channel?: ", self.mask)

        run_button = QPushButton("Create Hexagons")
        run_button.clicked.connect(self.run)
        main_layout.addWidget(run_button)

    def run(self):

        try:
            side_length = float(self.side_length.text())
        except:
            return

        try:
            xy_scale = float(self.xy_scale)
        except:
            xy_scale = 1

        try:
            z_scale = float(self.z_scale)
        except:
            z_scale = 1

        try:
            down_factor = float(self.down_factor.text())
        except:
            down_factor = None

        shape = self.parent().shape

        if down_factor:
            shape = (max(1, int(shape[0]/down_factor)), max(1, int(shape[1]/down_factor)), max(1, int(shape[2]/down_factor)))
            #side_length = side_length/down_factor
            xy_scale = xy_scale * down_factor
            z_scale = z_scale * down_factor


        mask = self.mask.isChecked()

        if mask:
            nodes = my_network.nodes != 0
            if down_factor:
                nodes = n3d.downsample(nodes, down_factor)
                shape = nodes.shape
        else:
            nodes = None

        if self.mode.currentIndex() == 0:
            mode = 'dodecahedron'
        else:
            mode = 'prism'

        from . import hexagons

        overlay_2 = hexagons.generate_hexagonal_labels(side_length, shape, xy_scale = xy_scale, z_scale = z_scale, shape_3d = mode)

        if nodes is not None:
            overlay_2 = overlay_2 * nodes

        if down_factor:
            overlay_2 = n3d.upsample_with_padding(overlay_2, original_shape = self.parent().shape)


        self.parent().load_channel(3, overlay_2, True)

        self.accept()


class FilamentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameters for Vessel Tracer (Note none of these are scaled with xy or z scale properties)")
        self.setModal(False)
        
        main_layout = QVBoxLayout(self)
        
        # Speedup Group
        speedup_group = QGroupBox("Speedup")
        speedup_layout = QFormLayout()
        self.kernel_spacing = QLineEdit("3")
        speedup_layout.addRow("Kernel Spacing (lower is more sensitive to gaps, can increase to speed up or if too many gaps filled):", self.kernel_spacing)
        self.downsample_factor = QLineEdit("1")
        speedup_layout.addRow("Temporary Downsample Factor (Note that the below distances are not adjusted for this):", self.downsample_factor)
        speedup_group.setLayout(speedup_layout)
        main_layout.addWidget(speedup_group)
        
        # Reconnection Behavior Group
        reconnection_group = QGroupBox("Reconnection Behavior")
        reconnection_layout = QFormLayout()
        self.max_distance = QLineEdit("20")
        reconnection_layout.addRow("Max Distance to Consider Connecting Filaments (Will Slow Down a lot if Large):", self.max_distance)
        self.gap_tolerance = QLineEdit("6")
        reconnection_layout.addRow("Gap Tolerance. Higher Values Increase Likelihood of Connecting over Larger Gaps:", self.gap_tolerance)
        self.score_threshold = QLineEdit("2")
        reconnection_layout.addRow("Connection Quality Threshold. Lower Values Increase Likelihood of Connecting In General, can be Negative:", self.score_threshold)
        reconnection_group.setLayout(reconnection_layout)
        main_layout.addWidget(reconnection_group)
        
        # Artifact Removal Group
        artifact_group = QGroupBox("Artifact Removal")
        artifact_layout = QFormLayout()
        self.min_component = QLineEdit("20")
        artifact_layout.addRow("Minimum Component Size to Include:", self.min_component)
        self.blob_sphericity = QLineEdit("1.0")
        artifact_layout.addRow("Spherical Objects in the Output can Represent Noise. Enter a val 0 < x < 1 to consider removing spheroids. Larger vals are more spherical. 1.0 = a perfect sphere. 0.3 is usually the lower bound of a spheroid:", self.blob_sphericity)
        self.blob_volume = QLineEdit("200")
        artifact_layout.addRow("If filtering spheroids: Minimum Volume of Spheroid to Remove (Smaller spheroids may be real):", self.blob_volume)
        self.spine_removal = QLineEdit("0")
        artifact_layout.addRow("Remove Branch Spines Below this Length?", self.spine_removal)
        artifact_group.setLayout(artifact_layout)
        main_layout.addWidget(artifact_group)
        self.state = None
        self.first = True

    
        # Run Button
        run_button = QPushButton("Run Filament Tracer (Output Goes in Overlay 2)")
        run_button.clicked.connect(self.run)
        main_layout.addWidget(run_button)


    def run(self):

        try:

            from . import filaments


            kernel_spacing = int(self.kernel_spacing.text()) if self.kernel_spacing.text().strip() else 1
            max_distance = float(self.max_distance.text()) if self.max_distance.text().strip() else 20
            min_component = int(self.min_component.text()) if self.min_component.text().strip() else 20
            gap_tolerance = float(self.gap_tolerance.text()) if self.gap_tolerance.text().strip() else 5
            blob_sphericity = float(self.blob_sphericity.text()) if self.blob_sphericity.text().strip() else 1
            blob_volume = float(self.blob_volume.text()) if self.blob_volume.text().strip() else 200
            spine_removal = int(self.spine_removal.text()) if self.spine_removal.text().strip() else 0
            score_threshold = int(self.score_threshold.text()) if self.score_threshold.text().strip() else 0
            downsample_factor = int(self.downsample_factor.text()) if self.downsample_factor.text().strip() else None
            data = self.parent().channel_data[self.parent().active_channel]

            if downsample_factor and downsample_factor > 1:
                data = n3d.downsample(data, downsample_factor)
                self.state = None

            result, self.state = filaments.trace(data, kernel_spacing, max_distance, min_component, gap_tolerance, blob_sphericity, blob_volume, spine_removal, score_threshold, my_network.xy_scale, my_network.z_scale, cached_state = self.state)

            if downsample_factor and downsample_factor > 1:

                result = n3d.upsample_with_padding(result, original_shape = self.parent().shape)


            self.parent().load_channel(3, result, True)

            if self.first:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Filaments traced succesfully. Heavy computations are cached while this menu is open and so re-computing with different params will be fast. Feel free to try out other params. Altering kernel spacing, downsampling, or spine removal will reset this cache, however."
                )
                self.first = False

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"Error: {e}")

    def closeEvent(self, event):
        self.state = None
        event.accept()



class MaskDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Mask Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        layout.addRow(QLabel("Use: "))

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2", "Highlight Overlay"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Masker:", self.mode_selector)

        layout.addRow(QLabel("To mask: "))

        # Add mode selection dropdown
        self.target_selector = QComboBox()
        self.target_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2"])
        self.target_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("To be Masked:", self.target_selector)

        layout.addRow(QLabel("Place output in: "))

        # Add mode selection dropdown
        self.output_selector = QComboBox()
        self.output_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2", "Highlight Overlay"])
        self.output_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Output Location:", self.output_selector)

        # Add Run button
        run_button = QPushButton("Mask")
        run_button.clicked.connect(self.mask)
        layout.addWidget(run_button)

    def mask(self):

        try:

            accepted_mode = self.mode_selector.currentIndex()
            accepted_target = self.target_selector.currentIndex()
            output_target = self.output_selector.currentIndex()

            if accepted_mode == 4:
                if self.parent().mini_overlay == True:
                    self.parent().create_highlight_overlay(node_indices = self.parent().clicked_values['nodes'], edge_indices = self.parent().clicked_values['edges'])
                active_data = self.parent().highlight_overlay
            else:
                active_data = self.parent().channel_data[accepted_mode]

            target_data = self.parent().channel_data[accepted_target]


            try:
                result = n3d.mask(target_data, active_data)

                if output_target == 4:

                    self.parent().highlight_overlay = result

                else:


                    # Update both the display data and the network object
                    self.parent().load_channel(output_target, channel_data = result, data = True,)

                self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

                self.accept()

            except Exception as e:
                print(f"Error masking: {e}")

        except Exception as e:
            print(f"Error masking: {e}")

class CropDialog(QDialog):

    def __init__(self, parent=None, args = None):

        try:

            super().__init__(parent)
            self.setWindowTitle("Crop Image (Will transpose any centroids)?")
            self.setModal(True)

            if args is None:
                xmin = 0
                xmax = self.parent().shape[2]
                ymin = 0
                ymax = self.parent().shape[1]
            else:
                xmin, xmax, ymin, ymax = args

            layout = QFormLayout(self)

            self.xmin = QLineEdit(f"{xmin}")
            layout.addRow("X Min", self.xmin)

            self.xmax = QLineEdit(f"{xmax}")
            layout.addRow("X Max", self.xmax)

            self.ymin = QLineEdit(f"{ymin}")
            layout.addRow("Y Min", self.ymin)

            self.ymax = QLineEdit(f"{ymax}")
            layout.addRow("Y Max", self.ymax)

            self.zmin = QLineEdit("0")
            layout.addRow("Z Min", self.zmin)

            self.zmax = QLineEdit(f"{self.parent().shape[0]}")
            layout.addRow("Z Max", self.zmax)

            # Add Run button
            run_button = QPushButton("Run")
            run_button.clicked.connect(self.run)
            layout.addRow(run_button)

        except:
            pass

    def run(self):

        try:

            xmin = int(self.xmin.text()) if self.xmin.text() else 0
            ymin = int(self.ymin.text()) if self.ymin.text() else 0
            zmin = int(self.zmin.text()) if self.zmin.text() else 0
            xmax = int(self.xmax.text()) if self.xmax.text() else self.parent().shape[2]
            ymax = int(self.ymax.text()) if self.xmax.text() else self.parent().shape[1]
            zmax = int(self.zmax.text()) if self.xmax.text() else self.parent().shape[0]

            args = xmin, ymin, zmin, xmax, ymax, zmax

            for i, array in enumerate(self.parent().channel_data):

                if array is None:

                    continue

                else:

                    array = self.reslice_3d_array(array, args)

                    self.parent().load_channel(i, array, data = True)

            print("Transposing centroids...")

            try:

                if my_network.node_centroids is not None:
                    nodes = list(my_network.node_centroids.keys())
                    centroids = np.array(list(my_network.node_centroids.values()))
                    
                    # Transform all at once
                    transformed = centroids - np.array([zmin, ymin, xmin])
                    transformed = transformed.astype(int)
                    
                    # Create upper bounds array with same shape
                    upper_bounds = np.array([zmax - zmin, ymax - ymin, xmax - xmin])
                    
                    # Boolean mask for valid coordinates - check each dimension separately
                    z_valid = (transformed[:, 0] >= 0) & (transformed[:, 0] <= upper_bounds[0])
                    y_valid = (transformed[:, 1] >= 0) & (transformed[:, 1] <= upper_bounds[1])
                    x_valid = (transformed[:, 2] >= 0) & (transformed[:, 2] <= upper_bounds[2])
                    
                    valid_mask = z_valid & y_valid & x_valid

                    # Rebuild dictionary with only valid entries
                    my_network.node_centroids = {
                        nodes[int(i)]: [int(transformed[i, 0]), int(transformed[i, 1]), int(transformed[i, 2])]
                        for i in range(len(nodes)) if valid_mask[i]
                    }
                    
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')

                if my_network.node_identities is not None:
                    new_idens = {}
                    for node, iden in my_network.node_identities.items():
                        if node in my_network.node_centroids:
                            new_idens[node] = iden
                    my_network.node_identities = new_idens

                    self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')

            except Exception as e:

                print(f"Error transposing node centroids: {e}")

            try:

                if my_network.edge_centroids is not None:

                    if my_network.edge_centroids is not None:
                        nodes = list(my_network.edge_centroids.keys())
                        centroids = np.array(list(my_network.edge_centroids.values()))
                        
                        # Transform all at once
                        transformed = centroids - np.array([zmin, ymin, xmin])
                        transformed = transformed.astype(int)
                        
                        # Boolean mask for valid coordinates
                        valid_mask = ((transformed >= 0) & 
                                      (transformed <= np.array([zmax, ymax, xmax]))).all(axis=1)
                        
                        # Rebuild dictionary with only valid entries
                        my_network.edge_centroids = {
                            nodes[int(i)]: [int(transformed[i, 0]), int(transformed[i, 1]), int(transformed[i, 2])]
                            for i in range(len(nodes)) if valid_mask[i]
                        }
                        
                        self.parent().format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')

            except Exception as e:

                print(f"Error transposing edge centroids: {e}")


            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error cropping: {e}")








    def reslice_3d_array(self, array, args):
        """Internal method used for the secondary algorithm to reslice subarrays around nodes."""

        x_start, y_start, z_start, x_end, y_end, z_end = args
        
        # Reslice the array
        array = array[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
        
        return array


class TypeDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Active Channel dtype")
        self.setModal(True)

        layout = QFormLayout(self)

        self.active_chan = self.parent().active_channel

        active_data = self.parent().channel_data[self.active_chan]

        layout.addRow("Info:", QLabel(f"Active dtype (Channel {self.active_chan}): {active_data.dtype}"))

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["8bit uint", "16bit uint", "32bit uint", "32bit float", "64bit float"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Change to?:", self.mode_selector)

        # Add Run button
        run_button = QPushButton("Run")
        run_button.clicked.connect(lambda: self.run_type(active_data))
        layout.addRow(run_button)

    def run_type(self, active_data):

        try:

            mode = self.mode_selector.currentIndex()

            if mode == 0:

                active_data = active_data.astype(np.uint8)

            elif mode == 1:

                active_data = active_data.astype(np.uint16)

            elif mode == 2:

                active_data = active_data.astype(np.uint32)

            elif mode == 3:

                active_data = active_data.astype(np.float32)

            elif mode == 4:

                active_data = active_data.astype(np.float64)

            self.parent().load_channel(self.active_chan, active_data, True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))


            print(f"Channel {self.active_chan}) dtype now: {self.parent().channel_data[self.active_chan].dtype}")
            self.accept()

        except Exception as E:
            print(f"Error: {e}")




class SkeletonizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Skeletonize Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)

        self.remove = QLineEdit("0")
        layout.addRow("Remove Branches Pixel Length (int):", self.remove)

        # auto checkbox (default True)
        self.auto = QPushButton("Auto")
        self.auto.setCheckable(True)
        try:
            if self.shape[0] == 1:
                self.auto.setChecked(False)
            else:
                self.auto.setChecked(True)
        except:
            self.auto.setChecked(True)
        layout.addRow("Attempt to Auto Correct Skeleton Looping:", self.auto)

       # Add Run button
        run_button = QPushButton("Run Skeletonize")
        run_button.clicked.connect(self.run_skeletonize)
        layout.addRow(run_button)

    def run_skeletonize(self):
        try:
            
            # Get branch removal
            try:
                remove = int(self.remove.text()) if self.remove.text() else 0
            except ValueError:
                remove = 0

            auto = self.auto.isChecked()
            
            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")

            if auto:
                active_data = n3d.skeletonize(active_data)
                active_data = n3d.fill_holes_3d(active_data)
            
            # Call dilate method with parameters
            result = n3d.skeletonize(
                active_data
            )

            if remove > 0:
                result = n3d.remove_branches_new(result, remove)
                result = n3d.dilate_3D(result, 3, 3, 3)
                result = n3d.skeletonize(result)


            # Update both the display data and the network object
            self.parent().channel_data[self.parent().active_channel] = result


            # Update the corresponding property in my_network
            setattr(my_network, network_properties[self.parent().active_channel], result)

            self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running skeletonize: {str(e)}"
            )   


class BranchStatDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Make sure branches are labeled first (Image -> Generate -> Label Branches)")
        self.setModal(True)
        
        layout = QFormLayout(self)

        info_label = QLabel("Skeletonization Params for Getting Branch Stats, Make sure xy and z scale are set correctly in properties")
        layout.addRow(info_label)

        self.remove = QLineEdit("0")
        layout.addRow("Remove Branches Pixel Length (int):", self.remove)

        # auto checkbox (default True)
        self.auto = QPushButton("Auto")
        self.auto.setCheckable(True)
        try:
            if self.shape[0] == 1:
                self.auto.setChecked(False)
            else:
                self.auto.setChecked(True)
        except:
            self.auto.setChecked(True)
        layout.addRow("Attempt to Auto Correct Skeleton Looping:", self.auto)

       # Add Run button
        run_button = QPushButton("Get Branchstats (For Active Image)")
        run_button.clicked.connect(self.run)
        layout.addRow(run_button)

    def run(self):

        try:
            
            # Get branch removal
            try:
                remove = int(self.remove.text()) if self.remove.text() else 0
            except ValueError:
                remove = 0

            auto = self.auto.isChecked()
            
            # Get the active channel data from parent
            active_data = np.copy(self.parent().channel_data[self.parent().active_channel])
            if active_data is None:
                raise ValueError("No active image selected")

            if auto:
                active_data = n3d.skeletonize(active_data)
                active_data = n3d.fill_holes_3d(active_data)
            
            active_data = n3d.skeletonize(
                active_data
            )

            if remove > 0:
                active_data = n3d.remove_branches_new(active_data, remove)
                active_data = n3d.dilate_3D(active_data, 3, 3, 3)
                active_data = n3d.skeletonize(active_data)

            active_data = active_data * self.parent().channel_data[self.parent().active_channel]
            len_dict, tortuosity_dict, angle_dict = n3d.compute_optional_branchstats(None, active_data, None, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale)

            if self.parent().active_channel == 0:
                self.parent().branch_dict[0] = [len_dict, tortuosity_dict]
            elif self.parent().active_channel == 1:
                self.parent().branch_dict[1] = [len_dict, tortuosity_dict]

            self.parent().format_for_upperright_table(len_dict, 'BranchID', 'Length (Scaled)', 'Branch Lengths')
            self.parent().format_for_upperright_table(tortuosity_dict, 'BranchID', 'Tortuosity', 'Branch Tortuosities')


            self.accept()
            
        except Exception as e:
            print(f"Error: {e}")

class DistanceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compute Distance Transform (Applies xy and z scaling, set them to 1 if you want voxel correspondence)?")
        self.setModal(True)
        
        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Parallel (Faster)", "Non-Parallel (Uses less CPU resources)"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # Add Run button
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run)
        layout.addRow(run_button)

    def run(self):

        try:

            mode = self.mode_selector.currentIndex()
            if mode == 0:
                fast_dil = True
            else:
                fast_dil = False

            data = self.parent().channel_data[self.parent().active_channel]

            data = sdl.compute_distance_transform_distance(data, sampling = [my_network.z_scale, my_network.xy_scale, my_network.xy_scale], fast_dil = fast_dil)

            self.parent().load_channel(self.parent().active_channel, data, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

            self.accept()

        except Exception as e:

            print(f"Error: {e}")

class GrayWaterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Gray Watershed - Please segment out your background first (ie with intensity thresholding) or this will not work correctly. \nAt the moment, this is designed for similarly sized objects. Having mixed large/small objects may not work correctly.")
        self.setModal(True)
        
        layout = QFormLayout(self)

        self.min_peak_distance = QLineEdit("1")
        layout.addRow("Minimum Peak Distance (To any other peak - Recommended) (This is true voxel distance here)", self.min_peak_distance)

        # Minimum Intensity
        self.min_intensity = QLineEdit("")
        layout.addRow("Minimum Peak Intensity (Optional):", self.min_intensity)

        # Add Run button
        run_button = QPushButton("Run Watershed")
        run_button.clicked.connect(self.run_watershed)
        layout.addRow(run_button)

    def wait_for_threshold_processing(self):
        """
        Opens ThresholdWindow and waits for user to process the image.
        Returns True if completed, False if cancelled.
        The thresholded image will be available in the main window after completion.
        """
        # Create event loop to wait for user
        loop = QEventLoop()
        result = {'completed': False}
        
        # Create the threshold window
        thresh_window = ThresholdWindow(self.parent(), 0)

        
        # Connect signals
        def on_processing_complete():
            result['completed'] = True
            loop.quit()
            
        def on_processing_cancelled():
            result['completed'] = False
            loop.quit()
        
        thresh_window.processing_complete.connect(on_processing_complete)
        thresh_window.processing_cancelled.connect(on_processing_cancelled)
        
        # Show window and wait
        thresh_window.show()
        thresh_window.raise_()
        thresh_window.activateWindow()
        
        # Block until user clicks "Apply Threshold & Continue" or "Cancel"
        loop.exec()
        
        # Clean up
        thresh_window.deleteLater()
        
        return result['completed']

    def run_watershed(self):

        try:

            self.accept()
            print("Please threshold foreground, or press cancel/skip if not desired:")
            self.wait_for_threshold_processing()
            data = self.parent().channel_data[self.parent().active_channel]

            min_intensity = float(self.min_intensity.text()) if self.min_intensity.text().strip() else None

            min_peak_distance = int(self.min_peak_distance.text()) if self.min_peak_distance.text().strip() else 1

            data = n3d.gray_watershed(data, min_peak_distance, min_intensity)

            self.parent().load_channel(self.parent().active_channel, data, data = True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))

            self.accept()

        except Exception as e:
            print(f"Error: {e}")




class WatershedDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Watershed Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)

        try:

            active_shape = self.parent().channel_data[self.parent().active_channel].shape[0]

            if active_shape == 1:
                self.default = 0.2
            else:
                self.default = 0.05

        except:
            self.default = 0.05

        # Smallest radius (empty by default)
        self.smallest_rad = QLineEdit()
        self.smallest_rad.setPlaceholderText("Leave empty for None")
        layout.addRow(f"Smallest Radius (Objects any smaller may get thresholded out - this value always overrides below 'proportion' param). \n Somewhat more intuitive param then below, use a conservative value a bit smaller than your smallest object's radius:", self.smallest_rad)
        
        # Proportion (default 0.1)
        self.proportion = QLineEdit(f"{self.default}")
        layout.addRow(f"Proportion (0-1) of distance transform value set [ie unique elements] to exclude (ie 0.2 = 20% of the set of all values of the distance transform get excluded).\n Essentially, vals closer to 0 are less likely to split objects but also won't kick out small objects from the output, vals slightly further from 0 will split more aggressively, but vals closer to 1 become unstable, leading to objects being evicted or labelling errors. \nRecommend something between 0.05 and 0.4, but it depends on the data (Or just enter a smallest radius above to avoid using this). \nWill tell you in command window what equivalent 'smallest radius' this is):", self.proportion)
        

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Parallel (Faster)", "Non-Parallel (Uses less CPU resources)"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # GPU checkbox (default True)
        self.gpu = QPushButton("GPU")
        self.gpu.setCheckable(True)
        self.gpu.setChecked(False)
        #layout.addRow("Use GPU:", self.gpu)
        
        
        # Predownsample (empty by default)
        self.predownsample = QLineEdit()
        self.predownsample.setPlaceholderText("Leave empty for None")
        #layout.addRow("Kernel Obtainment GPU Downsample:", self.predownsample)
        
        # Predownsample2 (empty by default)
        #self.predownsample2 = QLineEdit()
        #self.predownsample2.setPlaceholderText("Leave empty for None")
        #layout.addRow("Smart Label GPU Downsample:", self.predownsample2)
        
        #layout.addRow("Note:", QLabel(f"If the optimal proportion watershed output is still labeling spatially seperated objects with the same label, try right placing the result in nodes or edges\nthen right click the image and choose 'select all', followed by right clicking and 'selection' -> 'split non-touching labels'."))


        # Add Run button
        run_button = QPushButton("Run Watershed")
        run_button.clicked.connect(self.run_watershed)
        layout.addRow(run_button)

    def run_watershed(self):
        try:

            mode = self.mode_selector.currentIndex()

            if mode == 0:
                fast_dil = True
            else:
                fast_dil = False

            # Get directory (None if empty)
            directory = None
            
            # Get proportion (0.1 if empty or invalid)
            try:
                proportion = float(self.proportion.text()) if self.proportion.text() else self.default
            except ValueError:
                proportion = self.default
            
            # Get GPU state
            gpu = self.gpu.isChecked()
            
            # Get smallest_rad (None if empty)
            try:
                smallest_rad = float(self.smallest_rad.text()) if self.smallest_rad.text() else None
            except ValueError:
                smallest_rad = None
            
            # Get predownsample (None if empty)
            try:
                predownsample = float(self.predownsample.text()) if self.predownsample.text() else None
            except ValueError:
                predownsample = None
            
            # Get predownsample2 (None if empty)
            try:
                predownsample2 = float(self.predownsample2.text()) if self.predownsample2.text() else None
            except:
                predownsample2 = None
            
            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")


            # Call watershed method with parameters
            result = n3d.watershed(
                active_data,
                directory=directory,
                proportion=proportion,
                GPU=gpu,
                smallest_rad=smallest_rad,
                fast_dil = fast_dil,
                predownsample=predownsample,
                predownsample2=predownsample2
            )

            # Update both the display data and the network object
            self.parent().channel_data[self.parent().active_channel] = result


            # Update the corresponding property in my_network
            setattr(my_network, network_properties[self.parent().active_channel], result)

            self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
            self.accept()
            
        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                f"Error running watershed: {str(e)}"
            )

class InvertDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Invert Active Channel?")
        self.setModal(True)
        
        layout = QFormLayout(self)

       # Add Run button
        run_button = QPushButton("Run Invert")
        run_button.clicked.connect(self.run_invert)
        layout.addRow(run_button)

    def run_invert(self):

        try:

            # Get the active channel data from parent
            active_data = self.parent().channel_data[self.parent().active_channel]
            if active_data is None:
                raise ValueError("No active image selected")

            try:
                # Call binarize method with parameters
                if active_data.dtype == 'uint8' or 'int8':
                    num = 255
                elif active_data.dtype == 'uint16' or 'int16':
                    num = 65535
                elif active_data.dtype == 'uint32' or 'int32':
                    num = 2147483647

                result = (num - active_data
                    )

                result[result < 0] = 0

                # Update both the display data and the network object
                self.parent().channel_data[self.parent().active_channel] = result


                # Update the corresponding property in my_network
                setattr(my_network, network_properties[self.parent().active_channel], result)

                self.parent().update_display()
                self.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error running invert: {str(e)}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running invert: {str(e)}"
            )

class ZDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Z Parameters (Save your network first - this will alter all channels into 2D versions)")
        self.setModal(True)
        
        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["max", "mean", "min", "sum", "std"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # Add Run button
        run_button = QPushButton("Run Z Project")
        run_button.clicked.connect(self.run_z)
        layout.addRow(run_button)

    def run_z(self):

        mode = self.mode_selector.currentText()

        for i in range(len(self.parent().channel_data)):
            try:
                self.parent().channel_data[i] = n3d.z_project(self.parent().channel_data[i], mode)
                self.parent().load_channel(i, self.parent().channel_data[i], True, preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
            except:
                pass

        self.accept()


class CentroidNodeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Nodes from Centroids")
        self.setModal(True)
        
        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Starting at 0", "Starting at Min Centroids (will transpose centroids)"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # Add Run button
        run_button = QPushButton("Run Node Generation? (Will override current nodes). Note it is presumed your nodes begin at 1, not 0.")
        run_button.clicked.connect(self.run_nodes)
        layout.addRow(run_button)

    def run_nodes(self):

        try:

            if my_network.node_centroids is None and my_network.nodes is not None:
                self.parent().show_centroid_dialog()

                if my_network.node_centroids is None:

                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Could not generate centroids from current nodes. Please load centroids in an Excel (.xlsx) or CSV (.csv) file with columns 'Node ID', 'Z', 'Y', and 'X' in that order. The first row should contain these column headers, followed by the numerical ID of each node and numeric values for each centroid. Note it is presumed your nodes begin at 1, not 0. Error"
                    )
                    return
            elif my_network.node_centroids is None:

                QMessageBox.critical(
                    self,
                    "Error",
                    f"Could not find centroids. Please load centroids in an Excel (.xlsx) or CSV (.csv) file with columns 'Node ID', 'Z', 'Y', and 'X' in that order. The first row should contain these column headers, followed by numeric values for each centroid. Note it is presumed your nodes begin at 1, not 0. Error:"
                )
                return

            mode = self.mode_selector.currentIndex()

            if mode == 0:

                try:
                    shape = my_network.nodes.shape

                except:
                    try:
                        shape = my_network.edges.shape
                    except:
                        try:
                            shape = my_network.network_overlay.shape
                        except:
                            try:
                                shape = my_network.id_overlay.shape
                            except:
                                shape = None

                my_network.nodes = my_network.centroid_array(shape = shape)

            else:

                my_network.nodes, my_network.node_centroids = my_network.centroid_array(clip = True)

                self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')


            self.parent().load_channel(0, channel_data = my_network.nodes, data = True)

            self.accept()

        except Exception as e:

            print(f"Error generating centroids: {e}")




class GenNodesDialog(QDialog):

    def __init__(self, parent=None, down_factor=None, called=False):
        super().__init__(parent)
        self.setWindowTitle("Create Nodes from Edge Vertices")
        self.setModal(False)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        self.called = called
        
        # Set down_factor
        if not down_factor:
            down_factor = None
        
        # --- Recommended Corrections Group ---
        rec_group = QGroupBox("Recommended Corrections")
        rec_layout = QGridLayout()
        
        # Branch removal
        self.branch_removal = QLineEdit("0")
        rec_layout.addWidget(QLabel("Skeleton Voxel Branch Length to Remove (Compensates for spines):"), 0, 0)
        rec_layout.addWidget(self.branch_removal, 0, 1)
        
        # Auto checkbox
        self.auto = QPushButton("Auto")
        self.auto.setCheckable(True)
        try:
            if my_network.edges.shape[0] == 1:
                self.auto.setChecked(False)
            else:
                self.auto.setChecked(True)
        except:
            self.auto.setChecked(True)
        #rec_layout.addWidget(QLabel("Attempt to Auto Correct Skeleton Looping:"), 1, 0)
        #rec_layout.addWidget(self.auto, 1, 1)
        
        rec_group.setLayout(rec_layout)
        main_layout.addWidget(rec_group)
        
        # --- Optional Corrections Group ---
        opt_group = QGroupBox("Optional Corrections")
        opt_layout = QGridLayout()
        
        # Max volume
        self.max_vol = QLineEdit("0")
        #opt_layout.addWidget(QLabel("Maximum Voxel Volume to Retain (Compensates for skeleton looping):"), 0, 0)
        #opt_layout.addWidget(self.max_vol, 0, 1)
        
        # Component dilation
        self.comp_dil = QLineEdit("0")
        opt_layout.addWidget(QLabel("Amount to expand nodes (Merges nearby nodes, say if they are overassigned, good for broader branch breaking):"), 1, 0)
        opt_layout.addWidget(self.comp_dil, 1, 1)
        
        opt_group.setLayout(opt_layout)
        main_layout.addWidget(opt_group)

        # --- Processing Options Group ---
        process_group = QGroupBox("Processing Options")
        process_layout = QGridLayout()
        
        if not called:


            # Fast dilation checkbox
            self.fast_dil = QPushButton("Fast-Dil")
            self.fast_dil.setCheckable(True)
            self.fast_dil.setChecked(True)
            process_layout.addWidget(QLabel("Use Fast Dilation if merging nodes (Parallelized):"), 0, 0)
            process_layout.addWidget(self.fast_dil, 0, 1)
            
            # Downsample factor
            self.down_factor = QLineEdit("0")
            process_layout.addWidget(QLabel("Downsample Factor (Speeds up calculation at the cost of fidelity):"), 1, 0)
            process_layout.addWidget(self.down_factor, 1, 1)
                        
            process_group.setLayout(process_layout)
            main_layout.addWidget(process_group)
        else:
            self.down_factor = down_factor
            
            self.fast_dil = QPushButton("Fast-Dil")
            self.fast_dil.setCheckable(True)
            self.fast_dil.setChecked(True)
            process_layout.addWidget(QLabel("Use Fast Dilation if merging nodes (Parallelized):"), 0, 0)
            process_layout.addWidget(self.fast_dil, 0, 1)
            
            process_group.setLayout(process_layout)
            main_layout.addWidget(process_group)
        
        # Set retain variable but don't add to layout
        if not called:
            self.retain = QPushButton("Retain")
            self.retain.setCheckable(True)
            self.retain.setChecked(True)
        else:
            self.retain = False
        
        # Add Run button
        run_button = QPushButton("Run Node Generation")
        run_button.clicked.connect(self.run_gennodes)
        main_layout.addWidget(run_button)

    def run_gennodes(self):

        try:

            if my_network.edges is None and my_network.nodes is not None:
                self.parent().load_channel(1, my_network.nodes, data = True)
                self.parent().delete_channel(0, False)
            # Get directory (None if empty)
            #directory = self.directory.text() if self.directory.text() else None
            
            # Get branch_removal
            try:
                branch_removal = int(self.branch_removal.text()) if self.branch_removal.text() else 0
            except ValueError:
                branch_removal = 0
                
            # Get max_vol
            try:
                max_vol = int(self.max_vol.text()) if self.max_vol.text() else 0
            except ValueError:
                max_vol = 0
            
            # Get comp_dil
            try:
                comp_dil = int(self.comp_dil.text()) if self.comp_dil.text() else 0
            except ValueError:
                comp_dil = 0
                
            # Get down_factor
            if type(self.down_factor) is int or self.down_factor is None:
                down_factor = self.down_factor
            else:
                try:
                    down_factor = int(self.down_factor.text()) if self.down_factor.text() else None
                    if down_factor == 0:
                        down_factor = None
                except ValueError:
                    down_factor = None
                
            try:
                retain = self.retain.isChecked()
            except:
                retain = True

            auto = self.auto.isChecked()

            fastdil = self.fast_dil.isChecked()

            if down_factor is not None:
                my_network.edges = n3d.downsample(my_network.edges, down_factor)

            if auto:
                my_network.edges = n3d.skeletonize(my_network.edges)
                my_network.edges = n3d.fill_holes_3d(my_network.edges)
            print(auto)
            
            result, skele = n3d.label_vertices(
                my_network.edges,
                max_vol=max_vol,
                branch_removal=branch_removal,
                comp_dil=comp_dil,
                order = 0,
                return_skele = True,
                fastdil = fastdil
            )

            if down_factor is not None and not self.called:
                self.parent().resizing = True

                my_network.edges = n3d.downsample(my_network.edges, down_factor, order = 0)
                my_network.xy_scale = my_network.xy_scale * down_factor
                my_network.z_scale = my_network.z_scale * down_factor
                print("xy_scales and z_scales have been adjusted per downsample. Check image -> properties to manually reset them to 1 if desired.")
                self.parent().xy_scale_label.setText(f"xy_scale: {my_network.xy_scale:.2e}                   ")
                self.parent().z_scale_label.setText(f"z_scale: {my_network.z_scale:.2e}                   ")

            try: #Resets centroid fields
                if my_network.node_centroids is not None:
                    my_network.node_centroids = None
            except:
                pass
            try:
                if my_network.edge_centroids is not None:
                    my_network.edge_centroids = None
            except:
                pass

            self.parent().load_channel(1, channel_data = skele, data = True)

            self.parent().load_channel(0, channel_data = result, data = True)

            if retain and self.called:
                self.parent().load_channel(3, channel_data = my_network.edges, data = True)



            self.parent().update_display()
            self.parent().resizing = False
            self.accept()
            
        except Exception as e:

            import traceback
            print(traceback.format_exc())


            QMessageBox.critical(
                self,
                "Error",
                f"Error running generate nodes: {str(e)}"
            )



class BranchDialog(QDialog):

    def __init__(self, parent=None, called = False, tutorial_example = False):
        super().__init__(parent)
        self.setWindowTitle("Label Branches (of edges)")
        self.setModal(False)

        # Main layout
        main_layout = QVBoxLayout(self)
        
        # --- Correction Options Group ---
        correction_group = QGroupBox("Correction Options")
        correction_layout = QGridLayout()
        
        # Branch Fix checkbox
        self.fix = QPushButton("Auto-Correct 1")
        self.fix.setCheckable(True)
        self.fix.setChecked(False)
        #correction_layout.addWidget(QLabel("Auto-Correct Branches by Collapsing Busy Neighbors: "), 0, 0)
        #correction_layout.addWidget(self.fix, 0, 1)
        
        # Fix value
        self.fix_val = QLineEdit('4')
        #correction_layout.addWidget(QLabel("(For Auto-Correct 1) Avg Degree of Nearby Branch Communities to Merge (4-6 recommended):"), 1, 0)
        #correction_layout.addWidget(self.fix_val, 1, 1)
        
        # Seed
        self.seed = QLineEdit('')
        #correction_layout.addWidget(QLabel("Random seed for auto correction (int - optional):"), 2, 0)
        #correction_layout.addWidget(self.seed, 2, 1)

        # Add mode selection dropdown
        self.fix2 = QComboBox()
        self.fix2.addItems(["Skip This Step", "Merge Internal Labels With All External Neighbors", "Merge Internal Labels With Non-Branch-Like External Neighbors"])
        self.fix2.setCurrentIndex(1)
        correction_layout.addWidget(QLabel("Auto-Correct Internal Branches Mode:"), 3, 0)
        correction_layout.addWidget(self.fix2, 3, 1)

        self.fix3 = QPushButton("Auto-Correct Nontouching Branches")
        self.fix3.setCheckable(True)
        self.fix3.setChecked(True)
        correction_layout.addWidget(QLabel("Auto-Correct Nontouching Branches?: "), 4, 0)
        correction_layout.addWidget(self.fix3, 4, 1)

        self.fix4 = QPushButton("Auto-Attempt to Reunify Main Branches?")
        self.fix4.setCheckable(True)
        self.fix4.setChecked(False)
        correction_layout.addWidget(QLabel("Reunify Main Branches: "), 5, 0)
        correction_layout.addWidget(self.fix4, 5, 1)

        self.fix4_val = QLineEdit('10')
        correction_layout.addWidget(QLabel("(For Reunify) Minimum Score to Merge? (Lower vals = More mergers, can be negative):"), 6, 0)
        correction_layout.addWidget(self.fix4_val, 6, 1)
        
        correction_group.setLayout(correction_layout)
        main_layout.addWidget(correction_group)
        
        # --- Processing Options Group ---
        processing_group = QGroupBox("Processing Options")
        processing_layout = QGridLayout()
        
        # Downsample factor
        self.down_factor = QLineEdit("0")
        processing_layout.addWidget(QLabel("Internal downsample factor (will recompute nodes):"), 0, 0)
        processing_layout.addWidget(self.down_factor, 0, 1)

        # Add mode selection dropdown
        self.mode = QComboBox()
        self.mode.addItems(["Standard", "Fast (May be a little rougher along adjacent labels)"])
        self.mode.setCurrentIndex(0)
        processing_layout.addWidget(QLabel("Algorithm (Standard or Fast?):"), 1, 0)
        processing_layout.addWidget(self.mode, 1, 1)

        
        processing_group.setLayout(processing_layout)
        main_layout.addWidget(processing_group)
        
        # --- Misc Options Group ---
        misc_group = QGroupBox("Misc Options")
        misc_layout = QGridLayout()

        # optional computation checkbox
        self.compute = QPushButton("Branch Stats")
        self.compute.setCheckable(True)
        self.compute.setChecked(True)
        misc_layout.addWidget(QLabel("Compute Branch Stats (Branch Lengths, Tortuosity. Set xy_scale and z_scale in properties first if real distances are desired.):"), 0, 0)
        misc_layout.addWidget(self.compute, 0, 1)
        
        # Nodes checkbox
        self.nodes = QPushButton("Generate Nodes")
        self.nodes.setCheckable(True)
        self.nodes.setChecked(True)
        misc_layout.addWidget(QLabel("Generate nodes from edges? (Skip if already completed):"), 1, 0)
        misc_layout.addWidget(self.nodes, 1, 1)
        
        # GPU checkbox
        self.GPU = QPushButton("GPU")
        self.GPU.setCheckable(True)
        self.GPU.setChecked(False)
        #misc_layout.addWidget(QLabel("Use GPU (May downsample large images):"), 2, 0)
        #misc_layout.addWidget(self.GPU, 2, 1)
        
        misc_group.setLayout(misc_layout)
        main_layout.addWidget(misc_group)
        
        # Add Run button
        run_button = QPushButton("Run Branch Label")
        run_button.clicked.connect(self.branch_label)
        main_layout.addWidget(run_button)

        if (self.parent().channel_data[0] is not None or self.parent().channel_data[3] is not None) and not tutorial_example:
            QMessageBox.critical(
                self,
                "Alert",
                "The nodes and overlay 2 channels will be intermittently overwritten when running this method"
            )

    def branch_label(self):

        try:

            try:
                down_factor = int(self.down_factor.text()) if self.down_factor.text() else None
            except ValueError:
                down_factor = None

            if down_factor == 0:
                down_factor = None

            nodes = self.nodes.isChecked()
            GPU = self.GPU.isChecked()
            fix = self.fix.isChecked()
            fix2 = self.fix2.currentIndex()
            if fix2 == 0:
                fix2 == None
            else:
                if fix2 == 1:
                    consider_prop = False
                else:
                    consider_prop = True

            fix3 = self.fix3.isChecked()
            fix4 = self.fix4.isChecked()
            mode = self.mode.currentIndex()
            fix_val = float(self.fix_val.text()) if self.fix_val.text() else None
            fix4_val = float(self.fix4_val.text()) if self.fix4_val.text() else 10
            seed = int(self.seed.text()) if self.seed.text() else None
            compute = self.compute.isChecked()

            if my_network.edges is None and my_network.nodes is not None:
                self.parent().load_channel(1, my_network.nodes, data = True)
                self.parent().delete_channel(0, False)

            original_shape = my_network.edges.shape
            original_array = copy.deepcopy(my_network.edges)

            self.parent().show_gennodes_dialog(down_factor = down_factor, called = True)

            if my_network.edges is not None and my_network.nodes is not None and my_network.id_overlay is not None:

                if fix4:
                    unify = True
                else:
                    unify = False

                output, verts, skeleton, endpoints = n3d.label_branches(my_network.edges, nodes = my_network.nodes, bonus_array = original_array, GPU = GPU, down_factor = down_factor, arrayshape = original_shape, compute = compute, unify = unify, union_val = fix4_val, mode = mode, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale)

                if fix2:

                    print("Correcting Internal Branches...")

                    temp_network = n3d.Network_3D(nodes = output)

                    max_val = np.max(temp_network.nodes)

                    background = temp_network.nodes == 0

                    background = background * max_val

                    temp_network.nodes = temp_network.nodes + background

                    del background

                    temp_network.morph_proximity(search = [3,3], fastdil = True) #Detect network of nearby branches

                    output = n3d.fix_branches(output, temp_network.network, max_val, consider_prop = consider_prop)


                if fix:

                    temp_network = n3d.Network_3D(nodes = output)

                    temp_network.morph_proximity(search = [3,3], fastdil = True) #Detect network of nearby branches

                    temp_network.community_partition(weighted = False, style = 1, dostats = False, seed = seed) #Find communities with louvain, unweighted params

                    targs = n3d.fix_branches_network(temp_network.nodes, temp_network.network, temp_network.communities, fix_val)

                    temp_network.com_to_node(targs)

                    output = temp_network.nodes

                if fix3:

                    output = self.parent().separate_nontouching_objects(output, max_val=np.max(output), branches = True)

                if compute:
                    if skeleton.shape != output.shape:
                        print("Since downsampling was applied, skipping branchstats. Please use 'Analyze -> Stats -> Calculate Branch Stats' after this to find branch stats.")
                    else:
                        labeled_image = (skeleton != 0) * output
                        len_dict, tortuosity_dict, angle_dict = n3d.compute_optional_branchstats(verts, labeled_image, endpoints, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale)
                        self.parent().branch_dict[1] = [len_dict, tortuosity_dict]
                        #max_length = max(len(v) for v in angle_dict.values())
                        #title = [str(i+1) if i < 2 else i+1 for i in range(max_length)]

                        #del labeled_image

                        self.parent().format_for_upperright_table(len_dict, 'BranchID', 'Length (Scaled)', 'Branch Lengths')
                        self.parent().format_for_upperright_table(tortuosity_dict, 'BranchID', 'Tortuosity', 'Branch Tortuosities')
                        #self.parent().format_for_upperright_table(angle_dict, 'Vertex ID', title, 'Branch Angles')

                scalings = my_network.xy_scale, my_network.z_scale

                if down_factor is not None:

                    self.parent().reset(nodes = True, id_overlay = True, edges = True)

                else:
                    self.parent().reset(id_overlay = True)
                self.parent().update_display(dims = (output.shape[1], output.shape[2]))

                my_network.xy_scale, my_network.z_scale = scalings


                self.parent().load_channel(1, channel_data = output, data = True)

            self.parent().update_display(preserve_zoom = (self.parent().ax.get_xlim(), self.parent().ax.get_ylim()))
            self.accept()

        except Exception as e:
            print(f"Error labeling branches: {e}")
            import traceback
            print(traceback.format_exc())



class IsolateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Node types to isolate")
        self.setModal(True)
        layout = QFormLayout(self)
        
        self.combo1 = QComboBox()
        self.combo1.addItems(list(set(my_network.node_identities.values())))  
        self.combo1.setCurrentIndex(0)
        layout.addRow("ID 1:", self.combo1)
        
        self.combo2 = QComboBox()
        self.combo2.addItems(list(set(my_network.node_identities.values())))      
        self.combo2.setCurrentIndex(1)
        layout.addRow("ID 2:", self.combo2)
        
        # Add submit button
        sub_button = QPushButton("Submit")
        sub_button.clicked.connect(self.submit_ids)
        layout.addRow(sub_button)

    def submit_ids(self):
        try:
            id1 = self.combo1.currentText()
            id2 = self.combo2.currentText()
            if id1 == id2:
                print("Please select different identities")
                self.parent().show_isolate_dialog()
                return
            else:
                my_network.isolate_internode_connections(id1, id2)
                self.accept()
        except Exception as e:
            print(f"An error occurred: {e}")

class AlterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Node/Edge groups to add/remove")
        self.setModal(False)
        layout = QFormLayout(self)
        
        # Node 1
        self.node1 = QLineEdit()
        self.node1.setPlaceholderText("Enter integer")
        layout.addRow("Node1:", self.node1)
        
        # Node 2
        self.node2 = QLineEdit()
        self.node2.setPlaceholderText("Enter integer")
        layout.addRow("Node2:", self.node2)
        
        # Edge
        self.edge = QLineEdit()
        self.edge.setPlaceholderText("Optional - Enter integer")
        layout.addRow("Edge:", self.edge)
        
        # Add add button
        addbutton = QPushButton("Add pair")
        addbutton.clicked.connect(self.add)
        layout.addRow(addbutton)
        
        # Add remove button
        removebutton = QPushButton("Remove pair")
        removebutton.clicked.connect(self.remove)
        layout.addRow(removebutton)

    def add(self):
        try:
            node1 = int(self.node1.text()) if self.node1.text().strip() else None
            node2 = int(self.node2.text()) if self.node2.text().strip() else None
            edge = int(self.edge.text()) if self.edge.text().strip() else None
            
            # Check if we have valid node pairs
            if node1 is not None and node2 is not None:
                # Add the node pair and its reverse
                my_network.network_lists[0].append(node1)
                my_network.network_lists[1].append(node2)
                # Add edge value (0 if none provided)
                my_network.network_lists[2].append(edge if edge is not None else 0)
                
                my_network.network_lists = my_network.network_lists
            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.parent().network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.parent().network_table.resizeColumnToContents(column)
                    self.parent().clear_subgraphs()
                    self.parent().network_graph_widget.set_graph(my_network.network)
            except Exception as e:
                print(f"Error showing network table: {e}")
        except ValueError:
            import traceback
            print(traceback.format_exc())
            pass  # Invalid input - do nothing

    def remove(self):
        try:
            node1 = int(self.node1.text()) if self.node1.text().strip() else None
            node2 = int(self.node2.text()) if self.node2.text().strip() else None
            edge = int(self.edge.text()) if self.edge.text().strip() else None
            
            # Check if we have valid node pairs
            if node1 is not None and node2 is not None:
                # Create lists for indices to remove
                indices_to_remove = []
                
                # Loop through the lists to find matching pairs
                for i in range(len(my_network.network_lists[0])):
                    forward_match = (my_network.network_lists[0][i] == node1 and 
                                   my_network.network_lists[1][i] == node2)
                    reverse_match = (my_network.network_lists[0][i] == node2 and 
                                   my_network.network_lists[1][i] == node1)
                    
                    if forward_match or reverse_match:
                        # If edge value specified, only remove if edge matches
                        if edge is not None:
                            if my_network.network_lists[2][i] == edge:
                                indices_to_remove.append(i)
                        else:
                            # If no edge specified, remove all matching pairs
                            indices_to_remove.append(i)
                
                # Remove elements in reverse order to maintain correct indices
                for i in sorted(indices_to_remove, reverse=True):
                    my_network.network_lists[0].pop(i)
                    my_network.network_lists[1].pop(i)
                    my_network.network_lists[2].pop(i)
                my_network.network_lists = my_network.network_lists

            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.parent().network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.parent().network_table.resizeColumnToContents(column)
                    self.parent().clear_subgraphs()
                    self.parent().network_graph_widget.set_graph(my_network.network)
            except Exception as e:
                print(f"Error showing network table: {e}")
                    
        except ValueError:
            import traceback
            print(traceback.format_exc())
            pass  # Invalid input - do nothing


class ModifyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modify Network Qualities")
        self.setModal(False)
        layout = QFormLayout(self)

        self.revid = QPushButton("Remove Unassigned")
        self.revid.setCheckable(True)
        self.revid.setChecked(False)
        layout.addRow("Remove Unassigned IDs from Centroid List?:", self.revid)

        self.revdupeid = QPushButton("Make Singleton IDs")
        self.revdupeid.setCheckable(True)
        self.revdupeid.setChecked(False)
        layout.addRow("Force Any Multiple IDs to Pick a Random Single ID?:", self.revdupeid)

        self.revnegid = QPushButton("Remove Negative IDs")
        self.revnegid.setCheckable(True)
        self.revnegid.setChecked(False)
        layout.addRow("Remove All Negative Identities (those that end with a -)?:", self.revnegid)


        self.remove = QPushButton("Remove Missing")
        self.remove.setCheckable(True)
        self.remove.setChecked(False)
        layout.addRow("Remove Any Nodes Not in Nodes Channel From Properties?:", self.remove)
        
        # trunk checkbox (default false)
        self.trunk = QPushButton("Remove Trunk")
        self.trunk.setCheckable(True)
        self.trunk.setChecked(False)
        layout.addRow("Remove Trunk? (Most connected edge basis - overrides below):", self.trunk)
        
        # trunk checkbox (default false)
        self.trunknode = QPushButton("Trunk -> Node")
        self.trunknode.setCheckable(True)
        self.trunknode.setChecked(False)
        layout.addRow("Convert Trunk to Node? (Most connected edge basis):", self.trunknode)
        
        # edgenode checkbox (default false)
        self.edgenode = QPushButton("Edges -> Nodes")
        self.edgenode.setCheckable(True)
        self.edgenode.setChecked(False)
        layout.addRow("Convert 'Edges (Labeled objects)' to node objects?:", self.edgenode)
        
        # edgeweight checkbox (default false)
        self.edgeweight = QPushButton("Remove weights")
        self.edgeweight.setCheckable(True)
        self.edgeweight.setChecked(False)
        layout.addRow("Remove network weights (Represent Duplicate Connections)?:", self.edgeweight)
        
        # prune checkbox (default false)
        self.prune = QPushButton("Prune Same Type")
        self.prune.setCheckable(True)
        self.prune.setChecked(False)
        layout.addRow("Prune connections between nodes of the same type (if assigned)?:", self.prune)
        
        # isolate checkbox (default false)
        self.isolate = QPushButton("Isolate Two Types")
        self.isolate.setCheckable(True)
        self.isolate.setChecked(False)
        layout.addRow("Isolate connections between two specific node types (if assigned)?:", self.isolate)

        # isolate checkbox (default false)
        self.com_sizes = QPushButton("Communities By Size")
        self.com_sizes.setCheckable(True)
        self.com_sizes.setChecked(False)
        layout.addRow("Rearrange Community IDs by size?:", self.com_sizes)

        # Community collapse checkbox (default False)
        self.comcollapse = QPushButton("Communities -> nodes")
        self.comcollapse.setCheckable(True)
        self.comcollapse.setChecked(False)
        layout.addRow("Convert communities to nodes?:", self.comcollapse)

        iden_button = QPushButton("Change/Remove Identities")
        iden_button.clicked.connect(lambda: self.show_alter_iden_dialog(parent = self.parent()))
        layout.addRow(iden_button)

        #change button
        change_button = QPushButton("Add/Remove Network Pairs")
        change_button.clicked.connect(self.show_alter_dialog)
        layout.addRow(change_button)
                
        # Add Run button
        run_button = QPushButton("Make Changes")
        run_button.clicked.connect(self.run_changes)
        layout.addRow(run_button)

    def show_isolate_dialog(self):

        dialog = IsolateDialog(self)
        dialog.exec()

    def show_alter_dialog(self):

        dialog = AlterDialog(self.parent())
        dialog.show()

    def run_changes(self):

        try:

            revid = self.revid.isChecked()
            revdupeid = self.revdupeid.isChecked()
            revnegid = self.revnegid.isChecked()
            trunk = self.trunk.isChecked()
            if not trunk:
                trunknode = self.trunknode.isChecked()
            else:
                trunknode = False
            edgenode = self.edgenode.isChecked()
            edgeweight = self.edgeweight.isChecked()
            prune = self.prune.isChecked()
            isolate = self.isolate.isChecked()
            comcollapse = self.comcollapse.isChecked()
            remove = self.remove.isChecked()
            com_size = self.com_sizes.isChecked()


            if isolate and my_network.node_identities is not None:
                self.show_isolate_dialog()

            if revid:
                try:
                    my_network.remove_ids()
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                except:
                    pass

            if revdupeid:
                try:
                    for node, iden in my_network.node_identities.items():
                        try:
                            my_network.node_identities[node] = [random.choice(iden)]
                        except:
                            pass
                    self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                except:
                    pass

            if revnegid:
                new_idens = {}
                for node, idens in my_network.node_identities.items():
                        new_subidens = []
                        for subiden in idens:
                            if not subiden.endswith('-'):
                                new_subidens.append(subiden)
                        if new_subidens != []:
                            new_idens[node] = new_subidens
                my_network.node_identities = new_idens
                self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')


            if remove:
                my_network.purge_properties()
                try:
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                except:
                    pass
                try:
                    self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                except:
                    pass
                try:
                    self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'Community', 'Node Communities')
                except:
                    pass
                try:
                    if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                        empty_df = pd.DataFrame(columns=['Node A', 'Node B', 'Edge C'])
                        model = PandasModel(empty_df)
                        self.parent().network_table.setModel(model)
                    else:
                        model = PandasModel(my_network.network_lists)
                        self.parent().network_table.setModel(model)
                        # Adjust column widths to content
                        for column in range(model.columnCount(None)):
                            self.parent().network_table.resizeColumnToContents(column)
                except:
                    pass


            if edgeweight:
                my_network.remove_edge_weights()
            if prune and my_network.node_identities is not None:
                my_network.prune_samenode_connections()
            if trunk:
                my_network.remove_trunk_post()
            if trunknode:
                if my_network.node_centroids is None or my_network.edge_centroids is None:
                    self.parent().show_centroid_dialog()
                my_network.trunk_to_node()
                self.parent().load_channel(0, my_network.nodes, True)
            if edgenode:
                if my_network.node_centroids is None or my_network.edge_centroids is None:
                    self.parent().show_centroid_dialog()
                my_network.edge_to_node()
                self.parent().load_channel(0, my_network.nodes, True)
                self.parent().load_channel(1, my_network.edges, True)
                try:
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                except:
                    pass
            if com_size:
                if my_network.communities is None:
                    self.parent().show_partition_dialog()
                    if my_network.communities is None:
                        return
                my_network.com_by_size()
                self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'Community', 'Node Communities')

            if comcollapse:
                if my_network.communities is None:
                    self.parent().show_partition_dialog()
                    if my_network.communities is None:
                        return
                my_network.com_to_node()
                self.parent().load_channel(0, my_network.nodes, True)
                my_network.communities = None

            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.parent().network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.parent().network_table.resizeColumnToContents(column)
                    self.parent().clear_subgraphs()
                    self.parent().network_graph_widget.set_graph(my_network.network)
            except Exception as e:
                print(f"Error showing network table: {e}")

            if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                try:
                    self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                except Exception as e:
                    print(f"Error loading node identity table: {e}")

            self.parent().clear_subgraphs()

            self.parent().network_graph_widget.set_graph(my_network.network)
            self.parent().update_graph_fields()

            self.parent().update_display()
            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"An error occurred: {e}")


    def show_alter_iden_dialog(self, parent=None):
        """
        Enhanced dialog to remove or rename identities
        
        Args:
            parent: Optional parent widget for the dialog
        
        Returns:
            tuple: (accepted: bool, classifiers: list of dicts)
        """
        all_idens = my_network.node_identities.values()
        seen = set()
        for iden in all_idens:
            seen.update(iden)
        string_list = sorted(seen)
        
        dialog = QDialog(parent)
        dialog.setWindowTitle("Manage Identities")
        dialog.setMinimumWidth(900)
        dialog.setMinimumHeight(600)
        
        main_layout = QVBoxLayout()
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # ============= DELETE TAB =============
        delete_tab = QWidget()
        delete_layout = QVBoxLayout()
        
        delete_label = QLabel(f"Manage items ({len(string_list)} unique items):")
        delete_layout.addWidget(delete_label)
        
        # Instructions
        instructions = QLabel(
            "Left: Select items to REMOVE | Right: Select items to KEEP (overrides remove)\n"
            "Tip: Click and drag to select multiple items, or use Ctrl/Shift+Click"
        )
        instructions.setStyleSheet("color: #666; font-style: italic;")
        delete_layout.addWidget(instructions)
        
        # Two-column layout for remove/keep
        columns_layout = QHBoxLayout()
        
        # Remove column
        remove_column = QVBoxLayout()
        remove_label = QLabel("Items to REMOVE:")
        remove_label.setStyleSheet("font-weight: bold;")
        remove_column.addWidget(remove_label)
        
        remove_list = QListWidget()
        remove_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for item in sorted(string_list):
            list_item = QListWidgetItem(item)
            list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            list_item.setCheckState(Qt.CheckState.Unchecked)
            remove_list.addItem(list_item)
        remove_column.addWidget(remove_list)
        
        # Quick select buttons for remove
        remove_buttons = QHBoxLayout()
        check_selected_remove = QPushButton("Check Selected")
        uncheck_selected_remove = QPushButton("Uncheck Selected")
        check_all_remove = QPushButton("Check All")
        uncheck_all_remove = QPushButton("Uncheck All")
        
        remove_buttons.addWidget(check_selected_remove)
        remove_buttons.addWidget(uncheck_selected_remove)
        remove_buttons.addWidget(check_all_remove)
        remove_buttons.addWidget(uncheck_all_remove)
        remove_column.addLayout(remove_buttons)
        
        columns_layout.addLayout(remove_column)
        
        # Keep column
        keep_column = QVBoxLayout()
        keep_label = QLabel("Items to KEEP (priority):")
        keep_label.setStyleSheet("font-weight: bold; color: green;")
        keep_column.addWidget(keep_label)
        
        keep_list = QListWidget()
        keep_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for item in sorted(string_list):
            list_item = QListWidgetItem(item)
            list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            list_item.setCheckState(Qt.CheckState.Unchecked)
            keep_list.addItem(list_item)
        keep_column.addWidget(keep_list)
        
        # Quick select buttons for keep
        keep_buttons = QHBoxLayout()
        check_selected_keep = QPushButton("Check Selected")
        uncheck_selected_keep = QPushButton("Uncheck Selected")
        check_all_keep = QPushButton("Check All")
        uncheck_all_keep = QPushButton("Uncheck All")
        
        keep_buttons.addWidget(check_selected_keep)
        keep_buttons.addWidget(uncheck_selected_keep)
        keep_buttons.addWidget(check_all_keep)
        keep_buttons.addWidget(uncheck_all_keep)
        keep_column.addLayout(keep_buttons)
        
        columns_layout.addLayout(keep_column)
        delete_layout.addLayout(columns_layout)
        
        # Connect quick select buttons
        def check_selected_items(list_widget):
            for item in list_widget.selectedItems():
                item.setCheckState(Qt.CheckState.Checked)
        
        def uncheck_selected_items(list_widget):
            for item in list_widget.selectedItems():
                item.setCheckState(Qt.CheckState.Unchecked)
        
        def check_all_items(list_widget):
            for i in range(list_widget.count()):
                list_widget.item(i).setCheckState(Qt.CheckState.Checked)
        
        def uncheck_all_items(list_widget):
            for i in range(list_widget.count()):
                list_widget.item(i).setCheckState(Qt.CheckState.Unchecked)
        
        check_selected_remove.clicked.connect(lambda: check_selected_items(remove_list))
        uncheck_selected_remove.clicked.connect(lambda: uncheck_selected_items(remove_list))
        check_all_remove.clicked.connect(lambda: check_all_items(remove_list))
        uncheck_all_remove.clicked.connect(lambda: uncheck_all_items(remove_list))
        
        check_selected_keep.clicked.connect(lambda: check_selected_items(keep_list))
        uncheck_selected_keep.clicked.connect(lambda: uncheck_selected_items(keep_list))
        check_all_keep.clicked.connect(lambda: check_all_items(keep_list))
        uncheck_all_keep.clicked.connect(lambda: uncheck_all_items(keep_list))
        
        delete_tab.setLayout(delete_layout)
        tab_widget.addTab(delete_tab, "Delete Identities")
        
        # ============= RENAME TAB =============
        rename_tab = QWidget()
        rename_layout = QVBoxLayout()
        
        rename_label = QLabel("Create classifiers to rename identities:")
        rename_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        rename_layout.addWidget(rename_label)
        
        # Instructions for rename
        rename_instructions = QLabel(
            "• Drag identities from left into classifier rows\n"
            "• Toggle 'Include/Exclude' before dragging to set gate type\n"
            "• Green gates = nodes WITH these identities | Red gates = nodes WITHOUT these identities\n"
            "• Combine multiple gates in one classifier for complex conditions\n"
            "• Classifiers are applied sequentially from top to bottom"
        )
        rename_instructions.setStyleSheet(
            "color: #666; font-style: italic; background-color: #f0f0f0; "
            "padding: 8px; border-radius: 4px; font-size: 10px;"
        )
        rename_layout.addWidget(rename_instructions)
        
        # Three-column layout
        three_column_layout = QHBoxLayout()
        
        # Left: Available identities
        left_layout = QVBoxLayout()
        left_label = QLabel("Available Identities:")
        left_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(left_label)
        
        identity_list = DraggableListWidget()
        for item in sorted(string_list):
            identity_list.addItem(item)
        left_layout.addWidget(identity_list)
        
        three_column_layout.addLayout(left_layout, stretch=1)
        
        # Right: Classifier rows in scrollable area
        classifier_container = QWidget()
        classifier_layout = QVBoxLayout()
        classifier_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        classifier_layout.setSpacing(10)
        
        classifier_rows = []
        
        def add_classifier_row():
            row = ClassifierRow()
            row.remove_requested.connect(remove_classifier_row)
            classifier_rows.append(row)
            # Insert before the add button
            classifier_layout.insertWidget(len(classifier_rows) - 1, row)
        
        def remove_classifier_row(row):
            if row in classifier_rows:
                classifier_rows.remove(row)
                classifier_layout.removeWidget(row)
                row.deleteLater()
        
        # Add initial classifier row
        add_classifier_row()
        
        # Add classifier button
        add_classifier_btn = QPushButton("+ Add Classifier Row")
        add_classifier_btn.setStyleSheet(
            "QPushButton { background-color: #007bff; color: white; "
            "padding: 8px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0056b3; }"
        )
        add_classifier_btn.clicked.connect(add_classifier_row)
        classifier_layout.addWidget(add_classifier_btn)
        
        classifier_container.setLayout(classifier_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(classifier_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(600)
        
        three_column_layout.addWidget(scroll_area, stretch=4)
        
        rename_layout.addLayout(three_column_layout)
        rename_tab.setLayout(rename_layout)
        tab_widget.addTab(rename_tab, "Rename Identities")
        
        main_layout.addWidget(tab_widget)
        
        # ============= DIALOG BUTTONS =============
        button_layout = QHBoxLayout()
        run_button = QPushButton("Run")
        run_button.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; "
            "padding: 10px 20px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("padding: 10px 20px;")
        
        button_layout.addStretch()
        button_layout.addWidget(run_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)
        
        dialog.setLayout(main_layout)
        
        # Store results
        result = {'accepted': False, 'classifiers': []}
        
        def on_run():
            result['accepted'] = True
            current_tab = tab_widget.currentIndex()
            
            if current_tab == 0:  # Delete tab
                # Get items to keep (priority)
                items_to_keep = set()
                for i in range(keep_list.count()):
                    item = keep_list.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        items_to_keep.add(item.text())
                
                # Get items to remove
                items_to_remove = set()
                for i in range(remove_list.count()):
                    item = remove_list.item(i)
                    if item.checkState() == Qt.CheckState.Checked:
                        items_to_remove.add(item.text())
                
                # Apply keep priority: if keep has items, remove everything not in keep
                if items_to_keep:
                    all_items = set(string_list)
                    items_to_remove = all_items - items_to_keep
                
                # Remove identities
                if items_to_remove:
                    string_list[:] = [s for s in string_list if s not in items_to_remove]
                    new_identities = {}
                    for node, idens in my_network.node_identities.items():
                        new_idens = [iden for iden in idens if iden in string_list]
                        if new_idens:
                            new_identities[node] = new_idens
                    my_network.node_identities = new_identities
                    self.parent().format_for_upperright_table(
                        my_network.node_identities, 'NodeID', 'Identity', 'Node Identities'
                    )
            
            elif current_tab == 1:  # Rename tab
                # Collect classifier data
                classifiers = []
                for row in classifier_rows:
                    data = row.get_classifier_data()
                    # Only include rows that have at least one gate and a rename value
                    if (data['positive'] or data['negative']) and data['rename'].strip():
                        classifiers.append(data)
                
                result['classifiers'] = classifiers
                
                if classifiers:
                    print("Classifiers to apply (in order):")
                    for i, classifier in enumerate(classifiers, 1):
                        print(f"  {i}. Rename to '{classifier['rename']}':")
                        if classifier['positive']:
                            print(f"     ✓ WITH: {classifier['positive']}")
                        if classifier['negative']:
                            print(f"     ✗ WITHOUT: {classifier['negative']}")
                    for classifier in classifiers:
                        appender = classifier['appender']
                        if not classifier['positive'] and not classifier['negative']:
                            continue
                        for node, idens in my_network.node_identities.items():
                            pos_gate = all(item in idens for item in classifier['positive'])
                            neg_gate = all(item not in idens for item in classifier['negative'])
                            if pos_gate and neg_gate:
                                if not appender:
                                    my_network.node_identities[node] = [classifier['rename']]
                                else:
                                    my_network.node_identities[node].append(classifier['rename'])
                    self.parent().format_for_upperright_table(
                        my_network.node_identities, 'NodeID', 'Identity', 'Node Identities'
                    )
            
            dialog.accept()
        
        run_button.clicked.connect(on_run)
        cancel_button.clicked.connect(dialog.reject)
        
        dialog.exec()
        
        return result['accepted'], result.get('classifiers', [])

class IdentityGate(QWidget):
    """Widget representing a single identity gate (positive or negative)"""
    remove_requested = pyqtSignal(object)
    
    def __init__(self, identity, is_positive=True, parent=None):
        super().__init__(parent)
        self.identity = identity
        self.is_positive = is_positive
        
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Identity label with color coding
        self.label = QLabel(identity)
        self.update_style()
        
        layout.addWidget(self.label)
        
        # Small remove button
        remove_btn = QToolButton()
        remove_btn.setText("×")
        remove_btn.setMaximumWidth(20)
        remove_btn.setMaximumHeight(20)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(remove_btn)
        
        self.setLayout(layout)
        self.setMaximumHeight(30)
    
    def update_style(self):
        if self.is_positive:
            self.label.setStyleSheet(
                "background-color: #d4edda; color: #155724; "
                "border: 1px solid #c3e6cb; border-radius: 3px; padding: 2px 6px;"
            )
        else:
            self.label.setStyleSheet(
                "background-color: #f8d7da; color: #721c24; "
                "border: 1px solid #f5c6cb; border-radius: 3px; padding: 2px 6px;"
            )


class ClassifierRow(QFrame):  # Changed from QWidget to QFrame
    """Widget representing a classifier with multiple identity gates and rename field"""
    remove_requested = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gates = {}  # {identity: is_positive}
        self.current_mode_positive = True
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Left side: Mode toggle
        toggle_layout = QVBoxLayout()
        toggle_label = QLabel("Add as:")
        toggle_label.setStyleSheet("font-size: 10px; color: #666;")
        toggle_layout.addWidget(toggle_label)
        
        self.positive_radio = QRadioButton("Include")
        self.negative_radio = QRadioButton("Exclude")
        self.appender = QCheckBox("Add Identity instead of Overriding?")
        self.appender.setChecked(False)
        self.positive_radio.setChecked(True)
        self.positive_radio.setToolTip("Next dragged identity will be a positive gate (WITH)")
        self.negative_radio.setToolTip("Next dragged identity will be a negative gate (WITHOUT)")
        
        button_group = QButtonGroup(self)
        button_group.addButton(self.positive_radio)
        button_group.addButton(self.negative_radio)
        
        self.positive_radio.toggled.connect(
            lambda checked: setattr(self, 'current_mode_positive', checked)
        )
        
        toggle_layout.addWidget(self.positive_radio)
        toggle_layout.addWidget(self.negative_radio)
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.appender)
        
        main_layout.addLayout(toggle_layout)
        #main_layout.addWidget(self.appender, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        #main_layout.addStretch()
        
        # Middle: Drop zone with gates display
        drop_zone_container = QVBoxLayout()
        
        # Drop instruction
        drop_instruction = QLabel("Drop identities here →")
        drop_instruction.setStyleSheet("color: #999; font-style: italic; font-size: 10px;")
        drop_zone_container.addWidget(drop_instruction)
        
        # Gates container with flow layout
        self.gates_widget = QWidget()
        self.gates_layout = QVBoxLayout()
        self.gates_layout.setContentsMargins(5, 5, 5, 5)
        self.gates_layout.setSpacing(3)
        self.gates_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.gates_widget.setLayout(self.gates_layout)
        
        # Scroll area for gates
        gates_scroll = QScrollArea()
        gates_scroll.setWidget(self.gates_widget)
        gates_scroll.setWidgetResizable(True)
        gates_scroll.setMinimumWidth(250)
        gates_scroll.setMaximumHeight(100)
        gates_scroll.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        gates_scroll.setAcceptDrops(True)
        gates_scroll.setStyleSheet("QScrollArea { background-color: #f8f9fa; }")
        
        drop_zone_container.addWidget(gates_scroll)
        
        # Legend
        legend = QLabel("🟢 Include (WITH) | 🔴 Exclude (WITHOUT)")
        legend.setStyleSheet("font-size: 9px; color: #666;")
        drop_zone_container.addWidget(legend)
        
        main_layout.addLayout(drop_zone_container, stretch=3)
        
        # Right side: Rename field
        rename_layout = QVBoxLayout()
        rename_label = QLabel("Rename to:")
        rename_label.setStyleSheet("font-size: 10px; color: #666;")
        rename_layout.addWidget(rename_label)
        
        self.rename_field = QLineEdit()
        self.rename_field.setPlaceholderText("New identity name...")
        self.rename_field.setMinimumWidth(150)
        rename_layout.addWidget(self.rename_field)
        rename_layout.addStretch()
        
        main_layout.addLayout(rename_layout, stretch=1)
        
        # Remove row button
        remove_btn = QPushButton("✕")
        remove_btn.setMaximumWidth(30)
        remove_btn.setToolTip("Remove this classifier")
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        main_layout.addWidget(remove_btn)
        
        self.setLayout(main_layout)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        identity = event.mimeData().text()
        if identity and identity not in self.gates:
            self.add_gate(identity, self.current_mode_positive)
        event.acceptProposedAction()
    
    def add_gate(self, identity, is_positive):
        """Add a new identity gate to this classifier"""
        self.gates[identity] = is_positive
        
        gate_widget = IdentityGate(identity, is_positive)
        gate_widget.remove_requested.connect(self.remove_gate)
        self.gates_layout.addWidget(gate_widget)
    
    def remove_gate(self, gate_widget):
        """Remove an identity gate from this classifier"""
        if gate_widget.identity in self.gates:
            del self.gates[gate_widget.identity]
        self.gates_layout.removeWidget(gate_widget)
        gate_widget.deleteLater()
    
    def get_classifier_data(self):
        """Return dict with positive, negative, and rename keys"""
        positive_gates = [iden for iden, is_pos in self.gates.items() if is_pos]
        negative_gates = [iden for iden, is_pos in self.gates.items() if not is_pos]
        
        return {
            'positive': positive_gates,
            'negative': negative_gates,
            'rename': self.rename_field.text(),
            'appender': self.appender.isChecked()
        }

class IdentitySelectionDialog(QDialog):
    """Simplified dialog for selecting identities with positive/negative gates"""
    
    def __init__(self, available_identities, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Identities to Show")
        self.setMinimumWidth(700)
        self.setMinimumHeight(400)
        
        self.gates = {}  # {identity: is_positive}
        self.current_mode_positive = True
        self.available_identities = sorted(available_identities)
        
        main_layout = QVBoxLayout()
        
        # Title and instructions
        title = QLabel("Select which nodes to display:")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        main_layout.addWidget(title)
        
        instructions = QLabel(
            "• Drag identities from left to center to add gates\n"
            "• Toggle 'Include/Exclude' before dragging\n"
            "• Green = nodes WITH identity | Red = nodes WITHOUT identity\n"
            "• Combine multiple gates for complex filtering"
        )
        instructions.setStyleSheet(
            "color: #666; font-style: italic; background-color: #f0f0f0; "
            "padding: 8px; border-radius: 4px; font-size: 10px;"
        )
        main_layout.addWidget(instructions)
        
        # Two-column layout
        content_layout = QHBoxLayout()
        
        # Left: Available identities
        left_layout = QVBoxLayout()
        left_label = QLabel("Available Identities:")
        left_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(left_label)
        
        self.identity_list = DraggableListWidget()
        for item in self.available_identities:
            self.identity_list.addItem(item)
        left_layout.addWidget(self.identity_list)
        
        content_layout.addLayout(left_layout, stretch=1)
        
        # Right: Classifier section
        right_layout = QVBoxLayout()
        
        # Mode toggle
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Add gates as:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        self.positive_radio = QRadioButton("Include")
        self.negative_radio = QRadioButton("Exclude")
        self.positive_radio.setChecked(True)
        self.positive_radio.setToolTip("Drag identities to add positive gates (WITH)")
        self.negative_radio.setToolTip("Drag identities to add negative gates (WITHOUT)")
        
        button_group = QButtonGroup(self)
        button_group.addButton(self.positive_radio)
        button_group.addButton(self.negative_radio)
        
        self.positive_radio.toggled.connect(
            lambda checked: setattr(self, 'current_mode_positive', checked)
        )
        
        mode_layout.addWidget(self.positive_radio)
        mode_layout.addWidget(self.negative_radio)
        mode_layout.addStretch()
        right_layout.addLayout(mode_layout)
        
        # Drop zone label
        drop_label = QLabel("Gates:")
        drop_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        right_layout.addWidget(drop_label)
        
        # Gates container
        self.gates_widget = QWidget()
        self.gates_layout = QVBoxLayout()
        self.gates_layout.setContentsMargins(5, 5, 5, 5)
        self.gates_layout.setSpacing(3)
        self.gates_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.gates_widget.setLayout(self.gates_layout)
        
        # Scroll area for gates
        gates_scroll = QScrollArea()
        gates_scroll.setWidget(self.gates_widget)
        gates_scroll.setWidgetResizable(True)
        gates_scroll.setMinimumWidth(300)
        gates_scroll.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        gates_scroll.setAcceptDrops(True)
        gates_scroll.setStyleSheet("QScrollArea { background-color: #f8f9fa; }")
        
        # Make the scroll area accept drops
        gates_scroll.dragEnterEvent = self.dragEnterEvent
        gates_scroll.dropEvent = self.dropEvent
        
        right_layout.addWidget(gates_scroll)
        
        # Legend
        legend = QLabel("🟢 Include (WITH) | 🔴 Exclude (WITHOUT)")
        legend.setStyleSheet("font-size: 9px; color: #666;")
        right_layout.addWidget(legend)
        
        # Clear all button
        clear_btn = QPushButton("Clear All Gates")
        clear_btn.clicked.connect(self.clear_all_gates)
        right_layout.addWidget(clear_btn)
        
        content_layout.addLayout(right_layout, stretch=2)
        
        main_layout.addLayout(content_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select")
        self.select_button.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; "
            "padding: 10px 20px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("padding: 10px 20px;")
        
        button_layout.addStretch()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        # Connect buttons
        self.select_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        identity = event.mimeData().text()
        if identity and identity not in self.gates:
            self.add_gate(identity, self.current_mode_positive)
        event.acceptProposedAction()
    
    def add_gate(self, identity, is_positive):
        """Add a new identity gate"""
        self.gates[identity] = is_positive
        
        gate_widget = IdentityGate(identity, is_positive)
        gate_widget.remove_requested.connect(self.remove_gate)
        self.gates_layout.addWidget(gate_widget)
    
    def remove_gate(self, gate_widget):
        """Remove an identity gate"""
        if gate_widget.identity in self.gates:
            del self.gates[gate_widget.identity]
        self.gates_layout.removeWidget(gate_widget)
        gate_widget.deleteLater()
    
    def clear_all_gates(self):
        """Clear all gates"""
        # Remove all gate widgets
        while self.gates_layout.count():
            item = self.gates_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.gates = {}
    
    def get_classifier_data(self):
        """Return dict with positive and negative gate lists"""
        positive_gates = [iden for iden, is_pos in self.gates.items() if is_pos]
        negative_gates = [iden for iden, is_pos in self.gates.items() if not is_pos]
        
        return {
            'positive': positive_gates,
            'negative': negative_gates
        }

class DraggableListWidget(QListWidget):
    """List widget that supports dragging items"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
    
    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(item.text())
            drag.setMimeData(mime_data)
            drag.exec(Qt.DropAction.CopyAction)




class CentroidDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Centroids")
        self.setModal(True)

        layout = QFormLayout(self)

        self.downsample = QLineEdit("1")
        layout.addRow("Downsample Factor:", self.downsample)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes and Edges", "Nodes", "Edges"])
        self.mode_selector.setCurrentIndex(1)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        self.ignore_empty = QPushButton("Skip ID-less?")
        self.ignore_empty.setCheckable(True)
        self.ignore_empty.setChecked(False)
        layout.addRow("Skip Node Centroids Without Identity Property?:", self.ignore_empty)

        # Add Run button
        run_button = QPushButton("Run Calculate Centroids")
        run_button.clicked.connect(self.run_centroids)
        layout.addRow(run_button)

    def run_centroids(self):

        try:

            print("Calculating centroids...")

            chan = self.mode_selector.currentIndex()
            ignore_empty = self.ignore_empty.isChecked()

            # Get directory (None if empty)
            directory = None
            
            # Get downsample
            try:
                downsample = float(self.downsample.text()) if self.downsample.text() else 1
            except ValueError:
                downsample = 1

            if chan == 0 and my_network.edges is None: #if we don't have edges, just do nodes by default
                chan = 1

            if chan == 1:
                my_network.calculate_node_centroids(
                    down_factor = downsample
                )
                self.parent().network_graph_widget.centroids = my_network.node_centroids
                self.parent().selection_graph_widget.centroids = my_network.node_centroids


            elif chan == 2:
                my_network.calculate_edge_centroids(
                    down_factor = downsample
                )

            elif chan == 0:
                try:
                    my_network.calculate_node_centroids(
                        down_factor = downsample
                    )
                    self.parent().network_graph_widget.centroids = my_network.node_centroids
                    self.parent().selection_graph_widget.centroids = my_network.node_centroids

                except:
                    pass

                try:

                    my_network.calculate_edge_centroids(
                        down_factor = downsample
                    )

                except:
                    pass

            if hasattr(my_network, 'node_centroids') and my_network.node_centroids is not None:
                try:
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                except Exception as e:
                    print(f"Error loading node centroid table: {e}")

            if hasattr(my_network, 'edge_centroids') and my_network.edge_centroids is not None:
                try:
                    self.parent().format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
                except Exception as e:
                    print(f"Error loading edge centroid table: {e}")

            if ignore_empty:
                try:
                    my_network.remove_ids()
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                except:
                    pass

            self.parent().update_display()
            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error finding centroids: {str(e)}"
            )





class CalcAllDialog(QDialog):
    # Class variables to store previous settings
    prev_search = ""
    prev_diledge = ""
    prev_down_factor = ""
    prev_GPU_downsample = ""
    prev_other_nodes = ""
    prev_remove_trunk = ""
    prev_gpu = False
    prev_label_nodes = True
    prev_inners = True
    prev_fastdil = True
    prev_overlays = False
    prev_updates = True
    prev_vor = False
    prev_label_branch = False
    prev_edge_node = False
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Connectivity Network Parameters")
        self.setModal(False)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Important Parameters Group
        important_group = QGroupBox("Important Parameters")
        important_layout = QFormLayout(important_group)
        
        self.xy_scale = QLineEdit(f'{my_network.xy_scale}')
        important_layout.addRow("xy_scale:", self.xy_scale)
        
        self.z_scale = QLineEdit(f'{my_network.z_scale}')
        important_layout.addRow("z_scale:", self.z_scale)
        
        self.search = QLineEdit(self.prev_search)
        self.search.setPlaceholderText("Leave empty for None")
        important_layout.addRow("Node Search (float - Does not merge nodes):", self.search)
        
        self.diledge = QLineEdit(self.prev_diledge)
        self.diledge.setPlaceholderText("Leave empty for None")
        important_layout.addRow("Edge Search (float - Note that edges that find each other will merge):", self.diledge)
        
        self.label_nodes = QPushButton("Label")
        self.label_nodes.setCheckable(True)
        self.label_nodes.setChecked(self.prev_label_nodes)
        important_layout.addRow("Re-Label Nodes (WARNING - OVERRIDES ANY CURRENT LABELS):", self.label_nodes)
        
        main_layout.addWidget(important_group)
        
        # Optional Parameters Group
        optional_group = QGroupBox("Optional Parameters")
        optional_layout = QFormLayout(optional_group)
        
        self.other_nodes = QLineEdit(self.prev_other_nodes)
        self.other_nodes.setPlaceholderText("Leave empty for None")
        #optional_layout.addRow("Filepath or directory containing additional node images:", self.other_nodes)
        
        self.remove_trunk = QLineEdit(self.prev_remove_trunk)
        self.remove_trunk.setPlaceholderText("Leave empty for 0")
        optional_layout.addRow("Times to remove edge trunks (int - Volumetric basis):", self.remove_trunk)
        
        self.inners = QPushButton("Inner Edges")
        self.inners.setCheckable(True)
        self.inners.setChecked(self.prev_inners)
        #optional_layout.addRow("Use Inner Edges:", self.inners)

        self.voronoi_safe = QPushButton("Auto-Trunk")
        self.voronoi_safe.setCheckable(True)
        self.voronoi_safe.setChecked(self.prev_vor)
        optional_layout.addRow("Auto-Simplify Trunk Elements (Can be slow):", self.voronoi_safe)

        self.labeled_branches = QPushButton("Use Pre-Labeled Edges")
        self.labeled_branches.setCheckable(True)
        self.labeled_branches.setChecked(self.prev_label_branch)
        optional_layout.addRow("Use pre-labeled edges (edges must be labeled first; ie using label branches):", self.labeled_branches)

        self.edge_node = QPushButton("Convert Edges to Nodes?")
        self.edge_node.setCheckable(True)
        self.edge_node.setChecked(self.prev_edge_node)
        optional_layout.addRow("Edge -> Node:", self.edge_node)
        
        main_layout.addWidget(optional_group)
        
        # Speed Up Options Group
        speedup_group = QGroupBox("Speed Up Options")
        speedup_layout = QFormLayout(speedup_group)
        
        self.down_factor = QLineEdit(self.prev_down_factor)
        self.down_factor.setPlaceholderText("Leave empty for None")
        speedup_layout.addRow("Downsample for Centroids/Overlays (int):", self.down_factor)
        
        self.GPU_downsample = QLineEdit(self.prev_GPU_downsample)
        self.GPU_downsample.setPlaceholderText("Leave empty for None")
        #speedup_layout.addRow("Downsample for Distance Transform (GPU) (int):", self.GPU_downsample)
        
        self.gpu = QPushButton("GPU")
        self.gpu.setCheckable(True)
        self.gpu.setChecked(self.prev_gpu)
        #speedup_layout.addRow("Use GPU:", self.gpu)
        
        self.fastdil = QPushButton("Fast Search")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(self.prev_fastdil)
        speedup_layout.addRow("Use Fast Search (Parallelized searching, search regions may be a tad rougher along adjacent boundaries):", self.fastdil)
        
        main_layout.addWidget(speedup_group)
        
        # Output Options Group
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)
        
        self.overlays = QPushButton("Overlays")
        self.overlays.setCheckable(True)
        self.overlays.setChecked(self.prev_overlays)
        output_layout.addRow("Generate Overlays:", self.overlays)
        
        self.update = QPushButton("Update")
        self.update.setCheckable(True)
        self.update.setChecked(self.prev_updates)
        output_layout.addRow("Update Node/Edge in NetTracer3D:", self.update)
        
        main_layout.addWidget(output_group)
        
        # Add Run button
        run_button = QPushButton("Run Calculate All")
        run_button.clicked.connect(self.run_calc_all)
        main_layout.addWidget(run_button)

    def run_calc_all(self):

        try:
            # Get directory (None if empty)
            directory = None
            
            # Get xy_scale and z_scale (1 if empty or invalid)
            try:
                xy_scale = float(self.xy_scale.text()) if self.xy_scale.text() else 1
            except ValueError:
                xy_scale = 1
                
            try:
                z_scale = float(self.z_scale.text()) if self.z_scale.text() else 1
            except ValueError:
                z_scale = 1
            
            # Get search value (None if empty)
            try:
                search = float(self.search.text()) if self.search.text() else None
            except ValueError:
                search = None
                
            # Get diledge value (None if empty)
            try:
                diledge = int(self.diledge.text()) if self.diledge.text() else None
            except ValueError:
                diledge = None
                
            # Get down_factor value (None if empty)
            try:
                down_factor = int(self.down_factor.text()) if self.down_factor.text() else None
            except ValueError:
                down_factor = None
                
            # Get GPU_downsample value (None if empty)
            try:
                GPU_downsample = int(self.GPU_downsample.text()) if self.GPU_downsample.text() else None
            except ValueError:
                GPU_downsample = None
                
            # Get other_nodes path (None if empty)
            other_nodes = self.other_nodes.text() if self.other_nodes.text() else None
            
            # Get remove_trunk value (0 if empty)
            try:
                remove_trunk = int(self.remove_trunk.text()) if self.remove_trunk.text() else 0
            except ValueError:
                remove_trunk = 0

            voronoi_safe = self.voronoi_safe.isChecked()
            labeled_branches = self.labeled_branches.isChecked()
            edge_node = self.edge_node.isChecked()
                
            # Get button states
            gpu = self.gpu.isChecked()
            label_nodes = self.label_nodes.isChecked()
            inners = self.inners.isChecked()
            fastdil = self.fastdil.isChecked()
            overlays = self.overlays.isChecked()
            update = self.update.isChecked()

            if voronoi_safe or not update:
                temp_nodes = my_network.nodes.copy()
                temp_edges = my_network.edges.copy()
            
            if labeled_branches:
                prev_idens = copy.copy(my_network.node_identities)
                my_network.node_identities = {}
                if label_nodes:
                    my_network.nodes, num_nodes = n3d.label_objects(my_network.nodes)
                if search:
                    my_network.nodes = sdl.smart_dilate(my_network.nodes, fast_dil = fastdil, use_dt_dil_amount = search, xy_scale = xy_scale, z_scale = z_scale)
                if diledge:
                    my_network.edges = sdl.smart_dilate(my_network.edges, fast_dil = fastdil, use_dt_dil_amount = diledge, xy_scale = xy_scale, z_scale = z_scale)

                temp_array = np.copy(my_network.nodes)
                node_list = list(np.unique(temp_array))
                if 0 in node_list:
                    node_list.remove(0)
                for node in node_list:
                    my_network.node_identities[node] = ['Node']

                my_network.nodes = my_network.edges # Note - i put the edge array in the node array for the merge. Why? Not sure.


                # Calculate the maximum value that will exist in the output
                max_root = np.max(temp_array)
                max_other = np.max(my_network.nodes)
                max_output = max_root + max_other  # Worst case: all other_nodes shifted by max_root
                
                # Determine the minimum dtype needed
                if max_output <= 255:
                    target_dtype = np.uint8
                elif max_output <= 65535:
                    target_dtype = np.uint16
                else:
                    target_dtype = np.uint32
                temp_array = temp_array.astype(target_dtype)
                my_network.nodes = my_network.nodes.astype(target_dtype)

                mask = my_network.nodes > 0 # Combine edge masks and nodes masks giving edges priority so we can follow their branches over nodespace
                if np.any(mask):
                    edges_shifted = np.where(my_network.nodes > 0, my_network.nodes + max_root, 0)
                    edge_list = list(np.unique(edges_shifted))
                    if 0 in edge_list:
                        edge_list.remove(0)
                    for edge in edge_list:
                        my_network.node_identities[edge] = ['Edge']
                    from . import network_analysis
                    new_dict = network_analysis._find_centroids(edges_shifted)
                    my_network.node_centroids.update(new_dict)
                    my_network.nodes = np.where(mask, edges_shifted, 0)

                my_network.nodes = np.where(my_network.nodes > 0, my_network.nodes, temp_array)

                my_network.morph_proximity(search = [3,3], fastdil = True)
                my_network.xy_scale = xy_scale
                my_network.z_scale = z_scale
                my_network.calculate_node_centroids(down_factor)
                my_network.prune_samenode_connections(target = 'Node')

                components_to_keep = [
                    comp for comp in nx.connected_components(my_network.network)
                    if any('Node' in (my_network.node_identities.get(node) or []) for node in comp)
                ]
                # Flatten the components to keep
                nodes_to_keep = set().union(*components_to_keep)
                # Remove all other nodes
                nodes_to_remove = set(my_network.network.nodes()) - nodes_to_keep
                my_network.network.remove_nodes_from(nodes_to_remove)
                my_network.network = my_network.network
                for node, iden in prev_idens.items():
                    my_network.node_identities[node] = iden

            else:
                my_network.calculate_all(
                my_network.nodes,
                my_network.edges,
                directory=directory,
                xy_scale=xy_scale,
                z_scale=z_scale,
                search=search,
                diledge=diledge,
                down_factor=down_factor,
                GPU_downsample=GPU_downsample,
                other_nodes=other_nodes,
                remove_trunk=remove_trunk,
                GPU=gpu,
                label_nodes=label_nodes,
                inners=inners,
                fast_dil=fastdil
            )

            # Store current values as previous values
            CalcAllDialog.prev_search = self.search.text()
            CalcAllDialog.prev_diledge = self.diledge.text()
            CalcAllDialog.prev_down_factor = self.down_factor.text()
            CalcAllDialog.prev_GPU_downsample = self.GPU_downsample.text()
            CalcAllDialog.prev_other_nodes = self.other_nodes.text()
            CalcAllDialog.prev_remove_trunk = self.remove_trunk.text()
            CalcAllDialog.prev_gpu = self.gpu.isChecked()
            CalcAllDialog.prev_label_nodes = self.label_nodes.isChecked()
            CalcAllDialog.prev_inners = self.inners.isChecked()
            CalcAllDialog.prev_fastdil = self.fastdil.isChecked()
            CalcAllDialog.prev_overlays = self.overlays.isChecked()
            CalcAllDialog.prev_updates = self.update.isChecked()
            CalcAllDialog.prev_vor = voronoi_safe
            CalcAllDialog.prev_label_branch = labeled_branches

            if voronoi_safe and not labeled_branches:
                print("Auto-handling trunk elements by blocking connections beyond voronoi cells... (will have to compute a second network with maxed out search regions without using parallel search)")
                temp_network = n3d.Network_3D()
                temp_network.calculate_all(temp_nodes, temp_edges, search=999999999999999, label_nodes=label_nodes, fast_dil=False)

                from . import network_analysis

                new_lists = network_analysis.combine_lists_to_sublists_no_edges([temp_network.network_lists[0], temp_network.network_lists[1]])
                old_lists = network_analysis.combine_lists_to_sublists_no_edges([my_network.network_lists[0], my_network.network_lists[1]])
                ref_lists = network_analysis.combine_lists_to_sublists(my_network.network_lists)
                old_dict = {}
                for i, pair in enumerate(old_lists):
                    # Store both orientations of the pair
                    old_dict[tuple(pair)] = i
                    old_dict[tuple(reversed(pair))] = i

                output_lists = []
                used_indices = set()

                for pair in new_lists:
                    pair_tuple = tuple(pair)
                    if pair_tuple in old_dict:
                        idx = old_dict[pair_tuple]
                        if idx not in used_indices:
                            output_lists.append(ref_lists[idx])
                            used_indices.add(idx)

                # Clean up old_lists and ref_lists by removing used items
                # Delete in reverse order to maintain indices
                for idx in sorted(used_indices, reverse=True):
                    del ref_lists[idx]
                    del old_lists[idx]

                list1, list2, list3 = zip(*output_lists)

                # Convert them back to lists (zip returns tuples by default)
                output_lists = [list(list1), list(list2), list(list3)]

                my_network.network_lists = output_lists
                del temp_network

            if edge_node and not labeled_branches:
                my_network.edge_to_node()
                my_network.calculate_node_centroids(down_factor)

            # Update both the display data and the network object
            if update:
                self.parent().load_channel(0, my_network.nodes, True)
                self.parent().load_channel(1, my_network.edges, True)
            else:
                my_network.nodes = temp_nodes.copy()
                del temp_nodes
                my_network.edges = temp_edges.copy()
                del temp_edges
                self.parent().load_channel(0, my_network.nodes, True)
                self.parent().load_channel(1, my_network.edges, True)

            self.parent().clear_subgraphs()
            self.parent().network_graph_widget.set_graph(my_network.network)
            self.parent().xy_scale_label.setText(f"xy_scale: {my_network.xy_scale:.2e}                   ")
            self.parent().z_scale_label.setText(f"z_scale: {my_network.z_scale:.2e}                   ")
            # Then handle overlays
            if overlays:
                if directory is None:
                    directory = 'my_network'
                
                # Generate and update overlays
                my_network.network_overlay = my_network.draw_network(directory=directory, down_factor = down_factor)
                my_network.id_overlay = my_network.draw_node_indices(directory=directory, down_factor = down_factor)

                if down_factor is not None:
                    my_network.id_overlay = n3d.upsample_with_padding(my_network.id_overlay, original_shape = self.parent().shape)
                    my_network.network_overlay = n3d.upsample_with_padding(my_network.network_overlay, original_shape = self.parent().shape)
                
                # Update channel data
                self.parent().load_channel(2, my_network.network_overlay, True)
                self.parent().load_channel(3, my_network.id_overlay, True)
                
                # Enable the overlay channel buttons
                self.parent().channel_buttons[2].setEnabled(True)
                self.parent().channel_buttons[3].setEnabled(True)


            self.parent().update_display()
            self.accept()

            # Display network_lists in the network table
            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.parent().network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.parent().network_table.resizeColumnToContents(column)
            except Exception as e:
                print(f"Error loading network_lists: {e}")

            #Display the other things if they exist
            try:

                if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                    try:
                        self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                    except Exception as e:
                        print(f"Error loading node identity table: {e}")

                if hasattr(my_network, 'node_centroids') and my_network.node_centroids is not None:
                    try:
                        self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                    except Exception as e:
                        print(f"Error loading node centroid table: {e}")


                if hasattr(my_network, 'edge_centroids') and my_network.edge_centroids is not None:
                    try:
                        self.parent().format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
                    except Exception as e:
                        print(f"Error loading edge centroid table: {e}")


            except Exception as e:
                print(f"An error has occured: {e}")

            
        except Exception as e:
            import traceback
            print(traceback.format_exc())

            QMessageBox.critical(
                self,
                "Error",
                f"Error running calculate all: {str(e)}"
            )



class ProxDialog(QDialog):
    def __init__(self, parent=None, tutorial_example = False):
        super().__init__(parent)
        self.setWindowTitle("Calculate Proximity Network")
        self.setModal(False)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Important Parameters Group
        important_group = QGroupBox("Important Parameters")
        important_layout = QFormLayout(important_group)
        
        self.search = QLineEdit()
        self.search.setPlaceholderText("search")
        important_layout.addRow("Search Region Distance? (Enter true value corresponding to scaling, ie in microns. You can set nothing if using centroids; see Speed Up Options below):", self.search)
        
        self.xy_scale = QLineEdit(f"{my_network.xy_scale}")
        important_layout.addRow("xy_scale:", self.xy_scale)
        
        self.z_scale = QLineEdit(f"{my_network.z_scale}")
        important_layout.addRow("z_scale:", self.z_scale)
        
        main_layout.addWidget(important_group)
        
        # Mode Group
        mode_group = QGroupBox("Mode")
        mode_layout = QFormLayout(mode_group)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["From Centroids (fast but ignores shape - use for small or spherical objects - search STARTS at centroid)", "From Morphological Shape (slower but preserves shape - use for oddly shaped objects - search STARTS at object border)", "Morphological - Using Distance Transform (Does not let search regions extend beyond immediate neighbors. Use to simplify morphological search neighbor counts if desired.)"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        mode_layout.addRow("Execution Mode:", self.mode_selector)
        
        if my_network.node_identities is not None:
            self.id_selector = QComboBox()
            # Get unique identities <-- Commented this out because it is lag on a huge number of permutations. (The lag is adding to the QComboBox). Would need to reformat the architecture to use a lazy loading type of widget. Not sure how relevant these combos even are so i am just disabling this at the moment.
            #unique_idens = set(tuple(iden) for iden in my_network.node_identities.values())

            # Sort by length (number of elements)
            #sorted_idens = sorted(unique_idens, key=len)

            # Convert back to strings
            #all_idens = [str(list(iden)) for iden in sorted_idens]

            all_idens = list(my_network.node_identities.values())
            seen = set()
            for iden in all_idens:
                seen.update(iden)
            all_idens = list(seen)
            all_idens = ["['" + iden + "']" for iden in all_idens] # In the future can update downstream components to expect a pure string but would only minorly increase performance

            self.id_selector.addItems(['None'] + all_idens)
            self.id_selector.setCurrentIndex(0)  # Default to Mode 1
            mode_layout.addRow("Create Networks only from a specific node identity?:", self.id_selector)
        elif tutorial_example:
            self.id_selector = QComboBox()
            self.id_selector.addItems(['None'] + ['Example Identity A', 'Example Identity B', 'Example Identity C', 'etc...'])
            self.id_selector.setCurrentIndex(0)  # Default to Mode 1
            mode_layout.addRow("Create Networks only from a specific node identity?:", self.id_selector)
        else:
            self.id_selector = None
        
        main_layout.addWidget(mode_group)
        
        # Output Options Group
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)
        
        self.overlays = QPushButton("Overlays")
        self.overlays.setCheckable(True)
        self.overlays.setChecked(False)
        output_layout.addRow("Generate Overlays:", self.overlays)

        self.downsample = QLineEdit()
        output_layout.addRow("(If above): Downsample factor for drawing overlays (Int - Makes Overlay Elements Larger):", self.downsample)
        
        self.populate = QPushButton("Populate Nodes from Centroids?")
        self.populate.setCheckable(True)
        self.populate.setChecked(False)
        output_layout.addRow("If using centroid search:", self.populate)
        
        main_layout.addWidget(output_group)
        
        # Speed Up Options Group
        speedup_group = QGroupBox("Speed Up Options")
        speedup_layout = QFormLayout(speedup_group)

        self.max_neighbors = QLineEdit("")
        speedup_layout.addRow("(If using centroids): Max number of closest neighbors each node can connect to? (Will find only the n closest neighbors in search region, or just n closest neighbors if no search value set)", self.max_neighbors)
    
        self.fastdil = QPushButton("Fast Search")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(True)
        speedup_layout.addRow("(If using distance transform morphological) Use Fast Search? (Requires edt package, adjacent search region borders may be a bit rougher, is not faster for huge searches)?:", self.fastdil)
        
        main_layout.addWidget(speedup_group)
        
        # Add Run button
        run_button = QPushButton("Run Proximity Network")
        run_button.clicked.connect(self.prox)
        main_layout.addWidget(run_button)

    def prox(self):

        try:

            populate = self.populate.isChecked()

            mode = self.mode_selector.currentIndex()

            if self.id_selector is not None and self.id_selector.currentText() != 'None':
                import ast
                target_str = self.id_selector.currentText()
                target_list = ast.literal_eval(target_str)  # Parse "['a']" back to ['a']
                
                if len(target_list) == 1:
                    # Single identity: find all nodes containing this identity
                    identity = target_list[0]
                    targets = [
                        int(node) 
                        for node, identity_list in my_network.node_identities.items() 
                        if identity in identity_list
                    ]
                else:
                    # Multiple identities: exact match only
                    targets = [
                        int(node) 
                        for node, identity_list in my_network.node_identities.items() 
                        if identity_list == target_list
                    ]
            else:
                targets = None

            directory = None


            # Get xy_scale and z_scale (1 if empty or invalid)
            try:
                xy_scale = float(self.xy_scale.text()) if self.xy_scale.text() else my_network.xy_scale 
            except ValueError:
                xy_scale = my_network.xy_scale
                
            try:
                z_scale = float(self.z_scale.text()) if self.z_scale.text() else my_network.z_scale
            except ValueError:
                z_scale = my_network.z_scale

            # Get search value (None if empty)
            try:
                search = float(self.search.text()) if self.search.text() else None
            except ValueError:
                search = None

            try:
                max_neighbors = int(self.max_neighbors.text()) if self.max_neighbors.text() else None
            except:
                max_neighbors = None


            try:
                downsample = int(self.downsample.text()) if self.downsample.text() else None
            except:
                downsample = None

            overlays = self.overlays.isChecked()  
            fastdil = self.fastdil.isChecked()

            my_network.xy_scale = xy_scale
            my_network.z_scale = z_scale
            self.parent().xy_scale_label.setText(f"xy_scale: {my_network.xy_scale:.2e}                   ")
            self.parent().z_scale_label.setText(f"z_scale: {my_network.z_scale:.2e}                   ")

            if mode == 1: # Morphological search
                if len(np.unique(my_network.nodes)) < 3:
                    my_network.nodes, _ = n3d.label_objects(my_network.nodes)
                if my_network.node_centroids is None:
                    self.parent().show_centroid_dialog()
                my_network.morph_proximity(search = search, targets = targets, fastdil = fastdil)

                self.parent().load_channel(0, channel_data = my_network.nodes, data = True)
            elif mode == 0: # Centroid based search

                if my_network.node_centroids is None and my_network.nodes is not None:
                    self.parent().show_centroid_dialog()

                    if my_network.node_centroids is None:

                        QMessageBox.critical(
                            self,
                            "Error",
                            f"Could not generate centroids from current nodes. Please load centroids in an Excel (.xlsx) or CSV (.csv) file with columns 'Node ID', 'Z', 'Y', and 'X' in that order. The first row should contain these column headers, followed by the numerical ID of each node and numeric values for each centroid. Note it is presumed your nodes begin at 1, not 0. Error"
                        )
                        return
                elif my_network.node_centroids is None:

                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Could not find centroids. Please load centroids in an Excel (.xlsx) or CSV (.csv) file with columns 'Node ID', 'Z', 'Y', and 'X' in that order. The first row should contain these column headers, followed by numeric values for each centroid. Note it is presumed your nodes begin at 1, not 0. Error:"
                    )
                    return
                    
                if populate:
                    my_network.nodes = my_network.kd_network(distance = search, targets = targets, make_array = True, max_neighbors = max_neighbors)
                    self.parent().load_channel(0, channel_data = my_network.nodes, data = True)
                else:
                    my_network.kd_network(distance = search, targets = targets, max_neighbors = max_neighbors)
            elif mode == 2: # DT based
                if my_network.node_centroids is None:
                    self.parent().show_centroid_dialog()

                temp_network = n3d.Network_3D(nodes = sdl.smart_dilate(self.parent().channel_data[0], use_dt_dil_amount = round(search/2), fast_dil = fastdil, xy_scale = xy_scale, z_scale = z_scale), xy_scale = xy_scale, z_scale = z_scale)
                temp_network.morph_proximity(search = [3,3], fastdil = True)
                my_network.network_lists = temp_network.network_lists
                del temp_network

            if directory is not None:
                my_network.dump(directory = directory)


            # Then handle overlays
            if overlays:

                if my_network.node_centroids is not None:
                    if directory is None:
                        directory = 'my_network'
                    
                    # Generate and update overlays
                    my_network.network_overlay = my_network.draw_network(directory=directory, down_factor = downsample)
                    my_network.id_overlay = my_network.draw_node_indices(directory=directory, down_factor = downsample)

                    if downsample is not None:
                        my_network.id_overlay = n3d.upsample_with_padding(my_network.id_overlay, original_shape = self.parent().shape)
                        my_network.network_overlay = n3d.upsample_with_padding(my_network.network_overlay, original_shape = self.parent().shape)
                    
                    # Update channel data
                    self.parent().load_channel(2, channel_data = my_network.network_overlay, data = True)
                    self.parent().load_channel(3, channel_data = my_network.id_overlay, data = True)
                    
            self.parent().update_display()
            self.accept()
            self.parent().clear_subgraphs()
            self.parent().network_graph_widget.set_graph(my_network.network)

            # Display network_lists in the network table
            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.parent().network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.parent().network_table.resizeColumnToContents(column)
            except Exception as e:
                print(f"Error loading network_lists: {e}")

            #Display the other things if they exist
            try:

                if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                    try:
                        self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                    except Exception as e:
                        print(f"Error loading node identity table: {e}")

                if hasattr(my_network, 'node_centroids') and my_network.node_centroids is not None:
                    try:
                        self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')
                    except Exception as e:
                        print(f"Error loading node centroid table: {e}")


                if hasattr(my_network, 'edge_centroids') and my_network.edge_centroids is not None:
                    try:
                        self.parent().format_for_upperright_table(my_network.edge_centroids, 'EdgeID', ['Z', 'Y', 'X'], 'Edge Centroids')
                    except Exception as e:
                        print(f"Error loading edge centroid table: {e}")
            except:
                pass

            if my_network.network is None:
                my_network.network = my_network.network_lists

        except Exception as e:
            print(f"Error running proximity network: {str(e)}")
            import traceback
            print(traceback.format_exc())

class TutorialSelectionDialog(QWidget):
    """Dialog for selecting which tutorial to run"""
    
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.setWindowTitle("NetTracer3D Tutorials")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setGeometry(200, 200, 400, 300)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Select a Tutorial")
        title_font = QFont("Arial", 16, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Choose a tutorial to learn about different features of NetTracer3D:")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)
        
        layout.addSpacing(20)
        
        # Tutorial buttons

        intro_btn = QPushButton("Intro")
        intro_btn.setMinimumHeight(50)
        intro_btn.clicked.connect(self.start_intro)
        layout.addWidget(intro_btn)

        basics_btn = QPushButton("Basic Interface Tour")
        basics_btn.setMinimumHeight(50)
        basics_btn.clicked.connect(self.start_basics_tutorial)
        layout.addWidget(basics_btn)

        image_btn = QPushButton("Visualization Control Overview")
        image_btn.setMinimumHeight(50)
        image_btn.clicked.connect(self.start_image_tutorial)
        layout.addWidget(image_btn)
        
        file_btn = QPushButton("Saving/Loading Data and Assigning Node Identities")
        file_btn.setMinimumHeight(50)
        file_btn.clicked.connect(self.start_file)
        layout.addWidget(file_btn)

        seg_btn = QPushButton("Segmenting Data")
        seg_btn.setMinimumHeight(50)
        seg_btn.clicked.connect(self.start_segment)
        layout.addWidget(seg_btn)

        con_btn = QPushButton("1. Creating 'Connectivity Networks'")
        con_btn.setMinimumHeight(50)
        con_btn.clicked.connect(self.start_connectivity)
        layout.addWidget(con_btn)

        branch_btn = QPushButton("2. Creating 'Branch Networks'")
        branch_btn.setMinimumHeight(50)
        branch_btn.clicked.connect(self.start_branch)
        layout.addWidget(branch_btn)

        prox_btn = QPushButton("3. Creating 'Proximity Networks'")
        prox_btn.setMinimumHeight(50)
        prox_btn.clicked.connect(self.start_prox)
        layout.addWidget(prox_btn)

        analysis_btn = QPushButton("Network and Image Analysis")
        analysis_btn.setMinimumHeight(50)
        analysis_btn.clicked.connect(self.start_analysis)
        layout.addWidget(analysis_btn)
        
        processing_btn = QPushButton("Image Processing")
        processing_btn.setMinimumHeight(50)
        processing_btn.clicked.connect(self.start_process_tutorial)
        layout.addWidget(processing_btn)
        
        layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def start_intro(self):
        """Start the basic interface tutorial"""
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'start_tutorial_manager'):
            self.window.start_tutorial_manager = tutorial.setup_start_tutorial(self.window)
        
        self.window.start_tutorial_manager.start()
    
    def start_basics_tutorial(self):
        """Start the basic interface tutorial"""
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'basics_tutorial_manager'):
            self.window.basics_tutorial_manager = tutorial.setup_basics_tutorial(self.window)
        
        self.window.basics_tutorial_manager.start()

    def start_file(self):
        """Start the basic interface tutorial"""
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'file_tutorial_manager'):
            self.window.file_tutorial_manager = tutorial.setup_file_tutorial(self.window)
        
        self.window.file_tutorial_manager.start()


    def start_segment(self):
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'seg_tutorial_manager'):
            self.window.seg_tutorial_manager = tutorial.setup_seg_tutorial(self.window)
        self.window.seg_tutorial_manager.start()

    def start_connectivity(self):
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'connectivity_tutorial_manager'):
            self.window.connectivity_tutorial_manager = tutorial.setup_connectivity_tutorial(self.window)
        
        self.window.connectivity_tutorial_manager.start()

    def start_branch(self):
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'branch_tutorial_manager'):
            self.window.branch_tutorial_manager = tutorial.setup_branch_tutorial(self.window)
        
        self.window.branch_tutorial_manager.start()

    def start_prox(self):
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'prox_tutorial_manager'):
            self.window.prox_tutorial_manager = tutorial.setup_prox_tutorial(self.window)
        
        self.window.prox_tutorial_manager.start()

    def start_analysis(self):
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'analysis_tutorial_manager'):
            self.window.analysis_tutorial_manager = tutorial.setup_analysis_tutorial(self.window)
        
        self.window.analysis_tutorial_manager.start()

    def start_process_tutorial(self):
        """Start the image processing tutorial"""
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'process_tutorial_manager'):
            self.window.process_tutorial_manager = tutorial.setup_process_tutorial(self.window)
        
        self.window.process_tutorial_manager.start()

    def start_image_tutorial(self):
        """Start the image tutorial"""
        self.close()
        from . import tutorial
        
        if not hasattr(self.window, 'image_tutorial_manager'):
            self.window.image_tutorial_manager = tutorial.setup_image_tutorial(self.window)
        
        self.window.image_tutorial_manager.start()

# Initiating this program from the script line:

def run_gui():
    global my_network
    my_network = n3d.Network_3D()
    global network_properties
    # Update the corresponding network property based on active channel
    network_properties = {
        0: 'nodes',
        1: 'edges',
        2: 'network_overlay',
        3: 'id_overlay'
    }

    app = QApplication(sys.argv)
    window = ImageViewerWindow()
    window.show()
    sys.exit(app.exec())




if __name__ == '__main__':
    global my_network
    my_network = n3d.Network_3D()
    global network_properties
    # Update the corresponding network property based on active channel
    network_properties = {
        0: 'nodes',
        1: 'edges',
        2: 'network_overlay',
        3: 'id_overlay'
    }

    app = QApplication(sys.argv)
    window = ImageViewerWindow()
    window.show()
    sys.exit(app.exec())

    #import traceback
    #print(traceback.format_exc())