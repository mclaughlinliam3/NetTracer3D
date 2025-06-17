import sys
import networkx as nx
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, 
                            QHBoxLayout, QSlider, QMenuBar, QMenu, QDialog, 
                            QFormLayout, QLineEdit, QPushButton, QFileDialog,
                            QLabel, QComboBox, QMessageBox, QTableView, QInputDialog,
                            QMenu, QTabWidget, QGroupBox)
from PyQt6.QtCore import (QPoint, Qt, QAbstractTableModel, QTimer,  QThread, pyqtSignal, QObject, QCoreApplication)
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
from PyQt6.QtGui import (QFont, QCursor, QColor, QPixmap, QFontMetrics, QPainter, QPen)
import tifffile
import copy
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
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



class ImageViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NetTracer3D")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize channel data and states
        self.channel_data = [None] * 4
        self.channel_visible = [False] * 4
        self.current_slice = 0
        self.active_channel = 0  # Initialize active channel
        self.node_name = "Root_Nodes"

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
        "DARK_GRAY": (0.2, 0.2, 0.2)
        }

        self.base_colors = [ #Channel colors
            self.color_dictionary['LIGHT_RED'],    # Lighter red
            self.color_dictionary['LIGHT_GREEN'],    # Lighter green
            self.color_dictionary['WHITE'],        # White
            self.color_dictionary['WHITE']         # White
        ]
        
        
        # Initialize selection state
        self.selecting = False
        self.selection_start = None
        self.selection_rect = None
        self.click_start_time = None  # Add this to track when click started
        self.selection_threshold = 1.0  # Time in seconds before starting rectangle selection
        self.background = None
        self.last_update_time = 0
        self.update_interval = 0.016  # 60 FPS
        
        # Initialize zoom mode state
        self.zoom_mode = False
        self.original_xlim = None
        self.original_ylim = None
        self.zoom_changed = False

        # Pan mode state
        self.pan_mode = False
        self.panning = False
        self.pan_start = None
        self.img_width = None
        self.img_height = None
        self.pre_pan_channel_state = None  # Store which channels were visible before pan
        self.pan_background = None         # Store the static background image during pan
        self.is_pan_preview = False        # Track if we're in pan preview mode
        self.pre_pan_channel_state = None    # Store which channels were visible before pan
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
        } for _ in range(4)]
        
        # Create the brightness dialog but don't show it yet
        self.brightness_dialog = BrightnessContrastDialog(self)
        
        self.min_max = {
            0: [0,0],
            1: [0,0],
            2: [0,0],
            3: [0,0]
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
        self.active_channel_combo.setEnabled(False)
        active_channel_layout.addWidget(self.active_channel_combo)
        
        control_layout.addWidget(active_channel_widget)

        # Create zoom button and pan button
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)

        # Create zoom button
        self.zoom_button = QPushButton("🔍")
        self.zoom_button.setCheckable(True)
        self.zoom_button.setFixedSize(40, 40)
        self.zoom_button.clicked.connect(self.toggle_zoom_mode)
        buttons_layout.addWidget(self.zoom_button)
        self.resizing = False

        self.pan_button = QPushButton("✋")
        self.pan_button.setCheckable(True)
        self.pan_button.setFixedSize(40, 40)
        self.pan_button.clicked.connect(self.toggle_pan_mode)
        buttons_layout.addWidget(self.pan_button)

        self.high_button = QPushButton("👁️")
        self.high_button.setCheckable(True)
        self.high_button.setFixedSize(40, 40)
        self.high_button.clicked.connect(self.toggle_highlight)
        self.high_button.setChecked(True)
        buttons_layout.addWidget(self.high_button)
        self.highlight = True
        self.needs_mini = False

        self.pen_button = QPushButton("🖊️")
        self.pen_button.setCheckable(True)
        self.pen_button.setFixedSize(40, 40)
        self.pen_button.clicked.connect(self.toggle_brush_mode)
        buttons_layout.addWidget(self.pen_button)

        self.thresh_button = QPushButton("✏️")
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
        left_layout = QVBoxLayout(left_panel)
        
        # Create matplotlib canvas for image display
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        left_layout.addWidget(self.canvas)

        self.canvas.mpl_connect('scroll_event', self.on_mpl_scroll)

        
        left_layout.addWidget(control_panel)

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
        
        left_layout.addWidget(slider_container)

        
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
        self.network_button = QPushButton("Network")
        self.network_button.setCheckable(True)
        self.network_button.setChecked(True)
        self.network_button.clicked.connect(self.show_network_table)
        
        self.selection_button = QPushButton("Selection")
        self.selection_button.setCheckable(True)
        self.selection_button.clicked.connect(self.show_selection_table)
        
        # Add buttons to control layout
        table_control_layout.addWidget(self.network_button)
        table_control_layout.addWidget(self.selection_button)
        
        # Add control panel to right layout
        right_layout.addWidget(table_control)
        
        # Create both table views
        self.network_table = CustomTableView(self)
        self.selection_table = CustomTableView(self)
        empty_df = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
        self.selection_table.setModel(PandasModel(empty_df))
        self.network_table.setAlternatingRowColors(True)
        self.selection_table.setAlternatingRowColors(True)
        self.network_table.setSortingEnabled(True)
        self.selection_table.setSortingEnabled(True)
        
        # Initially show network table and hide selection table
        right_layout.addWidget(self.network_table)
        right_layout.addWidget(self.selection_table)
        self.selection_table.hide()
        
        # Store reference to currently active table
        self.active_table = self.network_table
        
        main_layout.addWidget(right_panel)
        
        # Create menu bar
        self.create_menu_bar()

        # Initialize clicked values dictionary
        self.clicked_values = {
            'nodes': [],
            'edges': []
        }
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        #self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

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
        self.mini_overlay_data = None #Actual data for mini overlay
        self.mini_thresh = (500*500*500) # Array volume to start using mini overlays for
        self.shape = None

        self.excel_manager = ExcelotronManager(self)
        self.excel_manager.data_received.connect(self.handle_excel_data)
        self.prev_coms = None

        self.paint_timer = QTimer()
        self.paint_timer.timeout.connect(self.flush_paint_updates)
        self.paint_timer.setSingleShot(True)
        self.pending_paint_update = False
        
        # Threading for paint operations
        self.paint_queue = queue.Queue()
        self.paint_lock = Lock()
        self.paint_worker = threading.Thread(target=self.paint_worker_loop, daemon=True)
        self.paint_worker.start()
        
        # Background caching for blitting
        self.paint_background = None
        self.paint_session_active = False
        
        # Batch paint operations
        self.paint_batch = []
        self.last_paint_pos = None

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
        

    def create_highlight_overlay(self, node_indices=None, edge_indices=None, overlay1_indices = None, overlay2_indices = None, bounds = False):
        """
        Create a binary overlay highlighting specific nodes and/or edges using parallel processing.
        
        Args:
            node_indices (list): List of node indices to highlight
            edge_indices (list): List of edge indices to highlight
        """

        self.mini_overlay = False #If this method is ever being called, it means we are rendering the entire overlay so mini overlay needs to reset.
        self.mini_overlay_data = None

        def process_chunk(chunk_data, indices_to_check):
            """Process a single chunk of the array to create highlight mask"""
            mask = np.isin(chunk_data, indices_to_check)
            return mask * 255

        def process_chunk_bounds(chunk_data, indices_to_check):
            """Process a single chunk of the array to create highlight mask"""

            mask = (chunk_data >= indices_to_check[0]) & (chunk_data <= indices_to_check[1])
            return mask * 255

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
        
        if not node_indices and not edge_indices and not overlay1_indices and not overlay2_indices:
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
                process_func = partial(process_chunk, indices_to_check=indices)
            else:
                if len(indices) == 1:
                    indices.insert(0, 0)
                process_func = partial(process_chunk_bounds, indices_to_check=indices)

            
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
            self.highlight_overlay = np.maximum(self.highlight_overlay, node_overlay)
        if edge_overlay is not None:
            self.highlight_overlay = np.maximum(self.highlight_overlay, edge_overlay)
        if overlay1_overlay is not None:
            self.highlight_overlay = np.maximum(self.highlight_overlay, overlay1_overlay)
        if overlay2_overlay is not None:
            self.highlight_overlay = np.maximum(self.highlight_overlay, overlay2_overlay)
                
        # Update display
        self.update_display(preserve_zoom=(current_xlim, current_ylim))

    def create_highlight_overlay_slice(self, indices, bounds = False):

        """Highlight overlay generation method specific for the segmenter interactive mode"""


        def process_chunk_bounds(chunk_data, indices_to_check):
            """Process a single chunk of the array to create highlight mask"""
            mask = (chunk_data >= indices_to_check[0]) & (chunk_data <= indices_to_check[1])
            return mask * 255

        def process_chunk(chunk_data, indices_to_check):
            """Process a single chunk of the array to create highlight mask"""

            mask = np.isin(chunk_data, indices_to_check)
            return mask * 255

        array = self.channel_data[self.active_channel]
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

        current_slice = array[self.current_slice, :, :]
        full_shape = array.shape
        slice_shape = current_slice.shape

        if self.highlight_overlay is None:

            self.highlight_overlay = np.zeros(full_shape, dtype=np.uint8)

        # Get number of CPU cores
        num_cores = mp.cpu_count()
        
        # Calculate chunk size along y-axis
        chunk_size = slice_shape[0] // num_cores
        if chunk_size < 1:
            chunk_size = 1
        
        def process_channel(channel_data, indices, array_shape):
            if channel_data is None or not indices:
                return None
                
            # Create chunks
            chunks = []
            for i in range(0, array_shape[0], chunk_size):
                end = min(i + chunk_size, array_shape[0])
                chunks.append(channel_data[i:end])
                
            # Process chunks in parallel using ThreadPoolExecutor
            if not bounds:
                process_func = partial(process_chunk, indices_to_check=indices)
            else:
                if len(indices) == 1:
                    indices.insert(0, 0)
                process_func = partial(process_chunk_bounds, indices_to_check=indices)

            
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                chunk_results = list(executor.map(process_func, chunks))
                
            # Reassemble the chunks
            return np.vstack(chunk_results)
        
        # Process nodes and edges in parallel using multiprocessing
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            future_highlight = executor.submit(process_channel, current_slice, indices, slice_shape)
            
            # Get results
            overlay = future_highlight.result()

        try:

            self.highlight_overlay[self.current_slice, :, :] = overlay
        except:
            pass

        # Update display
        self.update_display(preserve_zoom=(current_xlim, current_ylim), called = True)

    def create_mini_overlay(self, node_indices = None, edge_indices = None):

        """
        Create a highlight overlay one slice at a time.
        
        Args:
            node_indices (list): List of node indices to highlight
            edge_indices (list): List of edge indices to highlight
        """

        if not self.high_button.isChecked():

            if len(self.clicked_values['edges']) > 0:
                self.format_for_upperright_table(self.clicked_values['edges'], title = 'Selected Edges')
                self.needs_mini = True
            if len(self.clicked_values['nodes']) > 0:
                self.format_for_upperright_table(self.clicked_values['nodes'], title = 'Selected Nodes')
                self.needs_mini = True

            return


        def process_chunk(chunk_data, indices_to_check):
            """Process a single chunk of the array to create highlight mask"""
            mask = np.isin(chunk_data, indices_to_check)
            return mask * 255


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

            
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
        
        if not node_indices and not edge_indices: #Theoretically this can't be called because it uses full highlight overlay method for empty clicks
            self.mini_overlay_data = None
            self.mini_overlay = False
            self.update_display(preserve_zoom=(current_xlim, current_ylim))
            return
            
        # Get the shape of the mini array from any existing channel
        for channel in self.channel_data:
            if channel is not None:
                full_shape = channel.shape
                full_shape = (full_shape[1], full_shape[2]) #Just get (Y, X) shape
                break
        else:
            return  # No valid channels to get shape from
            
        # Initialize full-size overlay
        self.mini_overlay_data = np.zeros(full_shape, dtype=np.uint8)
        
        # Get number of CPU cores
        num_cores = mp.cpu_count()
        
        # Calculate chunk size along y-axis
        chunk_size = full_shape[0] // num_cores
        if chunk_size < 1:
            chunk_size = 1
        
        def process_channel(channel_data, indices, array_shape):
            if channel_data is None or not indices:
                return None
                
            # Create chunks
            chunks = []
            for i in range(0, array_shape[0], chunk_size):
                end = min(i + chunk_size, array_shape[0])
                chunks.append(channel_data[i:end, :])
                
            # Process chunks in parallel using ThreadPoolExecutor
            process_func = partial(process_chunk, indices_to_check=indices)

            
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                chunk_results = list(executor.map(process_func, chunks))
                
            # Reassemble the chunks
            return np.concatenate(chunk_results, axis=0)
        
        # Process nodes and edges in parallel using multiprocessing
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            try:
                slice_node = self.channel_data[0][self.current_slice, :, :] #This is the only major difference to the big highlight... we are only looking at this
                future_nodes = executor.submit(process_channel, slice_node, node_indices, full_shape)
                node_overlay = future_nodes.result()
            except:
                node_overlay = None
            try:
                slice_edge = self.channel_data[1][self.current_slice, :, :]
                future_edges = executor.submit(process_channel, slice_edge, edge_indices, full_shape)
                edge_overlay = future_edges.result()
            except:
                edge_overlay = None
                        
        # Combine results
        if node_overlay is not None:
            self.mini_overlay_data = np.maximum(self.mini_overlay_data, node_overlay)
        if edge_overlay is not None:
            self.mini_overlay_data = np.maximum(self.mini_overlay_data, edge_overlay)

                
        # Update display
        self.update_display(preserve_zoom=(current_xlim, current_ylim))



        



#METHODS RELATED TO RIGHT CLICK:
    
    def create_context_menu(self, event):
        """Create and show context menu at mouse position."""
        if self.channel_data[self.active_channel] is not None:
            x_idx = int(round(event.xdata))
            y_idx = int(round(event.ydata))
            
            try:
                # Create context menu
                context_menu = QMenu(self)

                find = context_menu.addAction("Find Node/Edge")
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

                community_menu = QMenu("Show Community(s)", self)
                show_community_nodes = community_menu.addAction("Just nodes")
                show_community_edges = community_menu.addAction("Nodes + Edges")
                context_menu.addMenu(community_menu)

                if my_network.node_identities is not None:
                    identity_menu = QMenu("Show Identity", self)
                    for item in set(my_network.node_identities.values()):
                        show_identity = identity_menu.addAction(f"ID: {item}")
                        show_identity.triggered.connect(lambda checked, item=item: self.handle_show_identities(sort=item))
                    context_menu.addMenu(identity_menu)

                select_all_menu = QMenu("Select All", self)
                select_nodes = select_all_menu.addAction("Nodes")
                select_both = select_all_menu.addAction("Nodes + Edges")
                select_edges = select_all_menu.addAction("Edges")
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
                    override_obj = highlight_menu.addAction("Override Channel with Selection")
                    override_obj.triggered.connect(self.handle_override)
                    context_menu.addMenu(highlight_menu)

                # Create measurement submenu
                measure_menu = context_menu.addMenu("Measurements")
                
                # Distance measurement options
                distance_menu = measure_menu.addMenu("Distance")
                if self.current_point is None:
                    show_point_menu = distance_menu.addAction("Place First Point")
                    show_point_menu.triggered.connect(
                        lambda: self.place_distance_point(x_idx, y_idx, self.current_slice))
                else:
                    show_point_menu = distance_menu.addAction("Place Second Point")
                    show_point_menu.triggered.connect(
                        lambda: self.place_distance_point(x_idx, y_idx, self.current_slice))
                
                # Angle measurement options
                angle_menu = measure_menu.addMenu("Angle")
                if self.current_point is None:
                    angle_first = angle_menu.addAction("Place First Point (A)")
                    angle_first.triggered.connect(
                        lambda: self.place_angle_point(x_idx, y_idx, self.current_slice))
                elif self.current_second_point is None:
                    angle_second = angle_menu.addAction("Place Second Point (B - Vertex)")
                    angle_second.triggered.connect(
                        lambda: self.place_angle_point(x_idx, y_idx, self.current_slice))
                else:
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
            self.ax.plot(x, y, 'yo', markersize=8)
            self.ax.text(x, y+5, f"D{self.current_pair_index}", 
                        color='yellow', ha='center', va='bottom')
            self.canvas.draw()
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
            
            # Store the point pair
            self.measurement_points.append({
                'pair_index': self.current_pair_index,
                'point1': self.current_point,
                'point2': (x2, y2, z2),
                'distance': distance,
                'distance2': distance2
            })
            
            # Draw second point and line
            self.ax.plot(x2, y2, 'yo', markersize=8)
            self.ax.text(x2, y2+5, f"D{self.current_pair_index}", 
                        color='yellow', ha='center', va='bottom')
            if z1 == z2:  # Only draw line if points are on same slice
                self.ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.5)
            self.canvas.draw()
            
            # Update measurement display
            self.update_measurement_display()
            
            # Reset for next pair
            self.current_point = None
            self.current_pair_index += 1
            self.measurement_mode = "distance"

    def place_angle_point(self, x, y, z):
        """Place a measurement point for angle measurement."""
        if self.current_point is None:
            # First point (A)
            self.current_point = (x, y, z)
            self.ax.plot(x, y, 'go', markersize=8)
            self.ax.text(x, y+5, f"A{self.current_trio_index}", 
                        color='green', ha='center', va='bottom')
            self.canvas.draw()
            self.measurement_mode = "angle"
            
        elif self.current_second_point is None:
            # Second point (B - vertex)
            self.current_second_point = (x, y, z)
            x1, y1, z1 = self.current_point
            
            self.ax.plot(x, y, 'go', markersize=8)
            self.ax.text(x, y+5, f"B{self.current_trio_index}", 
                        color='green', ha='center', va='bottom')
            
            # Draw line from A to B
            if z1 == z:
                self.ax.plot([x1, x], [y1, y], 'g--', alpha=0.7)
            self.canvas.draw()
            
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
            
            # Also add the two distances as separate pairs
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
                    'distance2': dist_ab_voxel
                },
                {
                    'pair_index': f"B{self.current_trio_index}-C{self.current_trio_index}",
                    'point1': (x2, y2, z2),
                    'point2': (x3, y3, z3),
                    'distance': dist_bc,
                    'distance2': dist_bc_voxel
                }
            ])
            
            # Draw third point and line
            self.ax.plot(x3, y3, 'go', markersize=8)
            self.ax.text(x3, y3+5, f"C{self.current_trio_index}", 
                        color='green', ha='center', va='bottom')
            
            if z2 == z3:  # Draw line from B to C if on same slice
                self.ax.plot([x2, x3], [y2, y3], 'g--', alpha=0.7)
            self.canvas.draw()
            
            # Update measurement display
            self.update_measurement_display()
            
            # Reset for next trio
            self.current_point = None
            self.current_second_point = None
            self.current_trio_index += 1
            self.measurement_mode = "angle"

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


    def show_network_table(self):
        """Switch to display the main network table."""
        if not self.network_button.isChecked():
            self.network_button.setChecked(True)
            return
        self.selection_button.setChecked(False)
        self.network_table.show()
        self.selection_table.hide()
        self.active_table = self.network_table

    def show_selection_table(self):
        """Switch to display the selection table."""
        if not self.selection_button.isChecked():
            self.selection_button.setChecked(True)
            return
        self.network_button.setChecked(False)
        self.network_table.hide()
        self.selection_table.show()
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
                new_model = PandasModel(filtered_df)
                self.selection_table.setModel(new_model)
                
                # Switch to selection table
                self.selection_button.click()

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

                    if self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2] > self.mini_thresh:
                        self.mini_overlay = True
                        self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                    else:
                        self.create_highlight_overlay(
                            node_indices=self.clicked_values['nodes'], 
                            edge_indices=self.clicked_values['edges']
                        )
                else:
                    if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                        self.mini_overlay = True
                        self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                    else:
                        self.create_highlight_overlay(
                            node_indices=self.clicked_values['nodes'],
                            edge_indices = self.clicked_values['edges']
                    )
            
                
        except Exception as e:
            print(f"Error processing neighbors: {e}")

    
    def handle_show_component(self, edges = False, nodes = True):
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
                new_model = PandasModel(filtered_df)
                self.selection_table.setModel(new_model)
                
                # Switch to selection table
                self.selection_button.click()


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
                new_model = PandasModel(filtered_df)
                self.selection_table.setModel(new_model)
                
                # Switch to selection table
                self.selection_button.click()

            if not nodes:
                self.clicked_values['nodes'] = old_nodes
            else:
                self.clicked_values['nodes'] = G.nodes()

            if edges:
                edge_indices = filtered_df.iloc[:, 2].unique().tolist()
                self.clicked_values['edges'] = edge_indices
                if self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2] > self.mini_thresh:
                    self.mini_overlay = True
                    self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                else:
                    self.create_highlight_overlay(
                        node_indices=self.clicked_values['nodes'],
                        edge_indices=edge_indices
                    )
            else:
                if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                    self.mini_overlay = True
                    self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                else:
                    self.create_highlight_overlay(
                        node_indices = self.clicked_values['nodes'],
                        edge_indices = self.clicked_values['edges']
                )

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

                # Get the existing DataFrame from the model
                original_df = self.network_table.model()._data

                # Create mask for rows for nodes in question
                mask = (
                    (original_df.iloc[:, 0].isin(nodes) & original_df.iloc[:, 1].isin(nodes))
                    )
                
                # Filter the DataFrame to only include direct connections
                filtered_df = original_df[mask].copy()
                
                # Create new model with filtered DataFrame and update selection table
                new_model = PandasModel(filtered_df)
                self.selection_table.setModel(new_model)
                
                # Switch to selection table
                self.selection_button.click()

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

            for node in my_network.node_identities:
                if sort == my_network.node_identities[node]:
                    nodes.append(node)

            neighbors = set()  # Use a set from the start to avoid duplicates
            nodes += self.clicked_values['nodes']

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
                new_model = PandasModel(filtered_df)
                self.selection_table.setModel(new_model)
                
                # Switch to selection table
                self.selection_button.click()
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
                self.setWindowTitle("Find Node (or edge?)")
                self.setModal(True)
                
                layout = QFormLayout(self)

                self.targ = QLineEdit("")
                layout.addRow("Node/Edge ID:", self.targ)

                run_button = QPushButton("Enter")
                run_button.clicked.connect(self.run)
                layout.addWidget(run_button)

            def run(self):

                try:

                    value = int(self.targ.text()) if self.targ.text().strip() else None

                    if value is None:
                        return

                    if self.parent().active_channel == 1:

                        mode = 'edges'

                        if my_network.edge_centroids is None:
                            self.parent().show_centroid_dialog()

                        num = (self.parent().channel_data[1].shape[0] * self.parent().channel_data[1].shape[1] * self.parent().channel_data[1].shape[2])

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


                    else:

                        mode = 'nodes'

                        if my_network.node_centroids is None:
                            self.parent().show_centroid_dialog()

                        num = (self.parent().channel_data[0].shape[0] * self.parent().channel_data[0].shape[1] * self.parent().channel_data[0].shape[2])

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

                    self.parent().clicked_values[mode] = [value]

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




    def handle_select_all(self, nodes = True, edges = False):

        try:

            if nodes:
                nodes = list(np.unique(my_network.nodes))
                if nodes[0] == 0:
                    del nodes[0]
                num = (self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2])
                print(f"Found {len(nodes)} node objects")
            else:
                nodes = []
            if edges:
                edges = list(np.unique(my_network.edges))
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

                if my_network.communities is not None:
                    try:
                        info_dict['Community'] = my_network.communities[label]
                    except:
                        pass

                if my_network.node_centroids is not None:
                    try:
                        info_dict['Centroid'] = my_network.node_centroids[label]
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


            elif sort == 'edge':

                label = self.clicked_values['edges'][-1]

                info_dict['Label'] = label

                info_dict['Object Class'] = 'Edge'

                if my_network.edge_centroids is not None:
                    try:
                        info_dict['Centroid'] = my_network.edge_centroids[label]
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

            self.format_for_upperright_table(info_dict, title = f'Info on Object')

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
                    empty_df = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
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

    def separate_nontouching_objects(self, input_array, max_val=0):
        """
        optimized version using advanced indexing.
        """

        print("Splitting nontouching objects")

        binary_mask = input_array > 0
        labeled_array, _ = n3d.label_objects(binary_mask)
        
        # Create a compound key for each (original_label, connected_component) pair
        # This avoids the need for explicit mapping
        mask = binary_mask
        compound_key = input_array[mask] * (labeled_array.max() + 1) + labeled_array[mask]
        
        # Get unique compound keys and create new labels
        unique_keys, inverse_indices = np.unique(compound_key, return_inverse=True)
        new_labels = np.arange(max_val + 1, max_val + 1 + len(unique_keys))
        
        # Create output array
        output_array = np.zeros_like(input_array)
        output_array[mask] = new_labels[inverse_indices]
        
        return output_array

    def handle_seperate(self):

        try:
            # Handle nodes
            if len(self.clicked_values['nodes']) > 0:
                self.create_highlight_overlay(node_indices=self.clicked_values['nodes'])
                
                # Create a boolean mask for highlighted values
                self.highlight_overlay = self.highlight_overlay != 0
                
                # Create array with just the highlighted values
                highlighted_nodes = self.highlight_overlay * my_network.nodes
                
                # Get non-highlighted part of the array
                non_highlighted = my_network.nodes * (~self.highlight_overlay)

                if (highlighted_nodes==non_highlighted).all():
                    max_val = 0
                else:
                    max_val = np.max(non_highlighted)
                
                # Process highlighted part
                processed_highlights = self.separate_nontouching_objects(highlighted_nodes, max_val)
                
                # Combine back with non-highlighted parts
                my_network.nodes = non_highlighted + processed_highlights
                
                self.load_channel(0, my_network.nodes, True)
            
            # Handle edges
            if len(self.clicked_values['edges']) > 0:
 
                self.create_highlight_overlay(edge_indices=self.clicked_values['edges'])
                
                # Create a boolean mask for highlighted values
                self.highlight_overlay = self.highlight_overlay != 0
                
                # Create array with just the highlighted values
                highlighted_edges = self.highlight_overlay * my_network.edges
                
                # Get non-highlighted part of the array
                non_highlighted = my_network.edges * (~self.highlight_overlay)

                if (highlighted_edges==non_highlighted).all():
                    max_val = 0
                else:
                    max_val = np.max(non_highlighted)
                
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
                self.create_highlight_overlay(node_indices = self.clicked_values['nodes'])
                mask = self.highlight_overlay == 0
                my_network.nodes = my_network.nodes * mask
                self.load_channel(0, my_network.nodes, True)

                if my_network.network_lists is not None:
                    for i in range(len(my_network.network_lists[0]) - 1, -1, -1):
                        if my_network.network_lists[0][i] in self.clicked_values['nodes'] or my_network.network_lists[0][i] in self.clicked_values['nodes']:
                            del my_network.network_lists[0][i]
                            del my_network.network_lists[1][i]
                            del my_network.network_lists[2][i]



            if len(self.clicked_values['edges']) > 0:
                self.create_highlight_overlay(edge_indices = self.clicked_values['edges'])
                mask = self.highlight_overlay == 0
                my_network.edges = my_network.edges * mask
                self.load_channel(1, my_network.edges, True)

                if my_network.network_lists is not None:
                    for i in range(len(my_network.network_lists[1]) - 1, -1, -1):
                        if my_network.network_lists[2][i] in self.clicked_values['edges']:
                            del my_network.network_lists[0][i]
                            del my_network.network_lists[1][i]
                            del my_network.network_lists[2][i]

            my_network.network_lists = my_network.network_lists


            if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                empty_df = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
                model = PandasModel(empty_df)
                self.network_table.setModel(model)
            else:
                model = PandasModel(my_network.network_lists)
                self.network_table.setModel(model)
                # Adjust column widths to content
                for column in range(model.columnCount(None)):
                    self.network_table.resizeColumnToContents(column)

            self.show_centroid_dialog()
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
                empty_df = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
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
                empty_df = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
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
            new_model = PandasModel(filtered_df)
            self.selection_table.setModel(new_model)
            
            # Switch to selection table
            self.selection_button.click()

            print("Selected nodes + edges have been isolated in the selection table, alongside their neighbors")

        except Exception as e:
            print(f"Error: {e}")



    def toggle_highlight(self):
        self.highlight = self.high_button.isChecked()
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

        if self.high_button.isChecked():
            if self.highlight_overlay is None and ((len(self.clicked_values['nodes']) + len(self.clicked_values['edges'])) > 0):
                if self.needs_mini:
                    self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                    self.needs_mini = False
        
        self.update_display(preserve_zoom=(current_xlim, current_ylim))

        
    def toggle_zoom_mode(self):
        """Toggle zoom mode on/off."""
        self.zoom_mode = self.zoom_button.isChecked()
        if self.zoom_mode:
            self.pan_button.setChecked(False)
            self.pen_button.setChecked(False)
            self.pan_mode = False
            self.brush_mode = False
            self.can = False
            self.threed = False
            self.last_change = None
            if self.machine_window is not None:
                self.machine_window.silence_button()
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        else:
            if self.machine_window is None:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
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
            if self.machine_window is not None:
                self.machine_window.silence_button()
            self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            if self.machine_window is None:
                self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.machine_window.toggle_brush_button()

    def toggle_brush_mode(self):
        """Toggle brush mode on/off"""
        self.brush_mode = self.pen_button.isChecked()
        if self.brush_mode:
            self.pan_button.setChecked(False)
            self.zoom_button.setChecked(False)
            self.pan_mode = False
            self.zoom_mode = False
            self.update_brush_cursor()
        else:
            self.last_change = None
            self.can = False
            self.threed = False
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

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
            self.threed = True
            self.threedthresh = 5
            self.update_brush_cursor()
        else:
            self.threed = False
            self.update_brush_cursor()


    def on_mpl_scroll(self, event):
        """Handle matplotlib canvas scroll events"""
        #Wheel events
        if self.brush_mode and event.inaxes == self.ax:

            # Get modifiers
            modifiers = event.guiEvent.modifiers()
            ctrl_pressed = bool(modifiers & Qt.ControlModifier)
            shift_pressed = bool(modifiers & Qt.ShiftModifier)
            alt_pressed = bool(modifiers & Qt.AltModifier)
            
            # Check if threed is enabled and ONLY if no specific modifiers are pressed
            if self.threed and not ctrl_pressed and not shift_pressed and not alt_pressed:
                import math
                step = 1 if event.button == 'up' else -1
                self.threedthresh += step
                
                # Round to appropriate odd integer based on scroll direction
                if event.button == 'up':
                    # Round up to nearest odd
                    self.threedthresh = math.ceil(self.threedthresh)
                    if self.threedthresh % 2 == 0:
                        self.threedthresh += 1
                else:  # event.button == 'down'
                    # Round down to nearest odd, but not below 1
                    self.threedthresh = math.floor(self.threedthresh)
                    if self.threedthresh % 2 == 0:
                        self.threedthresh -= 1
                    # Ensure not below minimum value of 1
                    self.threedthresh = max(1, self.threedthresh)
                
                # Update the brush cursor to show the new threshold
                self.update_brush_cursor()
                return
                
            # Check if Ctrl is pressed

            if event.guiEvent.modifiers() & Qt.ShiftModifier:
                pass

            elif event.guiEvent.modifiers() & Qt.ControlModifier:
                # Brush size adjustment code...
                step = 1 if event.button == 'up' else -1
                new_size = self.brush_size + step
                    
                if new_size < self.min_brush_size:
                    new_size = self.min_brush_size
                elif new_size > self.max_brush_size:
                    new_size = self.max_brush_size
                    
                self.brush_size = new_size
                self.update_brush_cursor()
                event.guiEvent.accept()
                return

        # General scrolling code outside the brush mode condition
        step = 1 if event.button == 'up' else -1

        if event.guiEvent.modifiers() & Qt.ShiftModifier:
            if event.guiEvent.modifiers() & Qt.ControlModifier:
                step = step * 3
            if (self.current_slice + step) < 0 or (self.current_slice + step) > self.slice_slider.maximum():
                return
                
            self.current_slice = self.current_slice + step
            self.slice_slider.setValue(self.current_slice + step)

            current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
            current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
            
            self.update_display(preserve_zoom=(current_xlim, current_ylim))


    def keyPressEvent(self, event):

        """Key press shortcuts for main class"""

        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            try:
                self.load_channel(self.last_change[1], self.last_change[0], True)
            except:
                pass

            return  # Return to prevent triggering the regular Z key action below

        if event.key() == Qt.Key_Z:
            self.zoom_button.click()
        if self.machine_window is not None:
            if event.key() == Qt.Key_A:
                self.machine_window.switch_foreground()
        if event.key() == Qt.Key_X:
            self.high_button.click()
        if event.key() == Qt.Key_F and event.modifiers() == Qt.ShiftModifier:
            self.handle_find()
        if self.brush_mode and self.machine_window is None:
            if event.key() == Qt.Key_F:
                self.toggle_can()
            elif event.key() == Qt.Key_D:
                self.toggle_threed()



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
        self.canvas.setCursor(cursor)
        
        painter.end()

    def get_line_points(self, x0, y0, x1, y1):
        """Get all points in a line between (x0,y0) and (x1,y1) using Bresenham's algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append((x, y))
        return points

    def get_current_mouse_position(self):
        # Get the main application's current mouse position
        cursor_pos = QCursor.pos()
        
        # Convert global screen coordinates to canvas widget coordinates
        canvas_pos = self.canvas.mapFromGlobal(cursor_pos)
        
        # Check if the position is within the canvas bounds
        if not (0 <= canvas_pos.x() < self.canvas.width() and 
                0 <= canvas_pos.y() < self.canvas.height()):
            return 0, 0  # Mouse is outside of the matplotlib canvas
        
        # Convert from canvas widget coordinates to matplotlib data coordinates
        x = canvas_pos.x()
        y = canvas_pos.y()
        
        # Transform display coordinates to data coordinates
        inv = self.ax.transData.inverted()
        data_coords = inv.transform((x, y))
        
        return data_coords[0], data_coords[1]

    def on_mouse_press(self, event):
        """Handle mouse press events - OPTIMIZED VERSION."""
        if event.inaxes != self.ax:
            return

        if event.button == 2:
            self.pan_button.click()
            return
                
        if self.zoom_mode:
            # Handle zoom mode press
            if self.original_xlim is None:
                self.original_xlim = self.ax.get_xlim()
                self.original_ylim = self.ax.get_ylim()
            
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            
            if event.button == 1:  # Left click - zoom in
                x_range = (current_xlim[1] - current_xlim[0]) / 4
                y_range = (current_ylim[1] - current_ylim[0]) / 4
                
                self.ax.set_xlim([xdata - x_range, xdata + x_range])
                self.ax.set_ylim([ydata - y_range, ydata + y_range])

                self.zoom_changed = True  # Flag that zoom has changed
                
                if not hasattr(self, 'zoom_changed'):
                    self.zoom_changed = False
                
            elif event.button == 3:  # Right click - zoom out
                x_range = (current_xlim[1] - current_xlim[0])
                y_range = (current_ylim[1] - current_ylim[0])
                
                new_xlim = [xdata - x_range, xdata + x_range]
                new_ylim = [ydata - y_range, ydata + y_range]
                
                if (new_xlim[0] <= self.original_xlim[0] or 
                    new_xlim[1] >= self.original_xlim[1] or
                    new_ylim[0] <= self.original_ylim[0] or
                    new_ylim[1] >= self.original_ylim[1]):
                    self.ax.set_xlim(self.original_xlim)
                    self.ax.set_ylim(self.original_ylim)
                else:
                    self.ax.set_xlim(new_xlim)
                    self.ax.set_ylim(new_ylim)

                    self.zoom_changed = False  # Flag that zoom has changed
                    
                    if not hasattr(self, 'zoom_changed'):
                        self.zoom_changed = False
            
            self.canvas.draw()
                
        elif self.pan_mode:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)
            
            # Store current channel visibility state
            self.pre_pan_channel_state = self.channel_visible.copy()
            
            # Create static background from currently visible channels
            self.create_pan_background()
            
            # Hide all channels and show only the background
            self.channel_visible = [False] * 4
            self.is_pan_preview = True
            
            # Update display to show only background
            self.update_display_pan_mode()

        elif self.brush_mode:
            if event.inaxes != self.ax:
                return
            
            if event.button == 1 or event.button == 3:
                x, y = int(event.xdata), int(event.ydata)

                if event.button == 1 and self.can:
                    self.handle_can(x, y)
                    return

                if event.button == 3:
                    self.erase = True
                else:
                    self.erase = False

                self.painting = True
                self.last_paint_pos = (x, y)

                if self.pen_button.isChecked():
                    channel = self.active_channel
                else:
                    channel = 2

                # Get current zoom to preserve it
                current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
                current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None

                # Paint at initial position
                self.paint_at_position(x, y, self.erase, channel)
                                
                self.canvas.draw()

                self.restore_channels = []
                if not self.channel_visible[channel]:
                    self.channel_visible[channel] = True

                # No need to hide other channels or track restore_channels
                self.restore_channels = []
                        
                self.update_display(preserve_zoom=(current_xlim, current_ylim), begin_paint=True)
                self.update_display_slice_optimized(channel, preserve_zoom=(current_xlim, current_ylim))
        
        elif event.button == 3:  # Right click (for context menu)
            self.create_context_menu(event)
        
        elif event.button == 1:  # Left click
            # Store initial click position but don't start selection yet
            self.selection_start = (event.xdata, event.ydata)
            self.selecting = False  # Will be set to True if the mouse moves while button is held

    def paint_at_position(self, center_x, center_y, erase = False, channel = 2):
        """Paint pixels within brush radius at given position"""
        if self.channel_data[channel] is None:
            return

        if erase:
            val = 0
        elif self.machine_window is None:
            val = 255
        elif self.foreground:
            val = 1
        else:
            val = 2
        height, width = self.channel_data[channel][self.current_slice].shape
        radius = self.brush_size // 2

        # Calculate brush area
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                # Check if point is within circular brush area
                if (x - center_x) * 2 + (y - center_y) * 2 <= radius ** 2:
                    if self.threed and self.threedthresh > 1:
                        amount = (self.threedthresh - 1) / 2
                        low = max(0, self.current_slice - amount)
                        high = min(self.channel_data[channel].shape[0] - 1, self.current_slice + amount)
                        for i in range(int(low), int(high + 1)):
                            self.channel_data[channel][i][y, x] = val
                    else:
                        self.channel_data[channel][self.current_slice][y, x] = val

    def paint_at_position_vectorized(self, center_x, center_y, erase=False, channel=2, 
                                   slice_idx=None, brush_size=None, threed=False, 
                                   threedthresh=1, foreground=True, machine_window=None):
        """Vectorized paint operation for better performance."""
        if self.channel_data[channel] is None:
            return
        
        # Use provided parameters or fall back to instance variables
        slice_idx = slice_idx if slice_idx is not None else self.current_slice
        brush_size = brush_size if brush_size is not None else getattr(self, 'brush_size', 5)
        
        # Determine paint value
        if erase:
            val = 0
        elif machine_window is None:
            val = 255
        elif foreground:
            val = 1
        else:
            val = 2
        
        height, width = self.channel_data[channel][slice_idx].shape
        radius = brush_size // 2
        
        # Calculate affected region bounds
        y_min = max(0, center_y - radius)
        y_max = min(height, center_y + radius + 1)
        x_min = max(0, center_x - radius)
        x_max = min(width, center_x + radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            return  # No valid region to paint
        
        # Create coordinate grids for the affected region
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Calculate distances squared (avoid sqrt for performance)
        distances_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        mask = distances_sq <= radius ** 2
        
        # Apply paint to affected slices
        if threed and threedthresh > 1:
            amount = (threedthresh - 1) / 2
            low = max(0, int(slice_idx - amount))
            high = min(self.channel_data[channel].shape[0] - 1, int(slice_idx + amount))
            
            for i in range(low, high + 1):
                self.channel_data[channel][i][y_min:y_max, x_min:x_max][mask] = val
        else:
            self.channel_data[channel][slice_idx][y_min:y_max, x_min:x_max][mask] = val

    def handle_can(self, x, y):


        if self.threed:
            ref = copy.deepcopy(self.channel_data[self.active_channel])
            the_slice = self.channel_data[self.active_channel]
            
            # First invert the boolean array
            inv = n3d.invert_boolean(the_slice)
            
            # Label the connected components in the inverted array
            labeled_array, num_features = n3d.label_objects(inv)
            
            # Get the target label at the clicked point
            target_label = labeled_array[self.current_slice][y][x]
            
            # Only fill if we clicked on a valid region (target_label > 0)
            if target_label > 0:
                # Create a mask of the connected component we clicked on
                fill_mask = (labeled_array == target_label) * 255

                self.last_change = [ref, self.active_channel]

                # Add this mask to the original slice
                the_slice = the_slice | fill_mask  # Use logical OR to add the filled region
            
            # Update the channel data
            self.load_channel(self.active_channel, the_slice, True)
        else:

            ref = copy.deepcopy(self.channel_data[self.active_channel])

            the_slice = self.channel_data[self.active_channel][self.current_slice]

            # First invert the boolean array
            inv = n3d.invert_boolean(the_slice)
            
            # Label the connected components in the inverted array
            labeled_array, num_features = n3d.label_objects(inv)
            
            # Get the target label at the clicked point
            target_label = labeled_array[y][x]
            
            # Only fill if we clicked on a valid region (target_label > 0)
            if target_label > 0:
                # Create a mask of the connected component we clicked on
                fill_mask = (labeled_array == target_label) * 255

                self.last_change = [ref, self.active_channel]
                
                # Add this mask to the original slice
                the_slice = the_slice | fill_mask  # Use logical OR to add the filled region
            
            # Update the channel data
            self.channel_data[self.active_channel][self.current_slice] = the_slice
            self.load_channel(self.active_channel, self.channel_data[self.active_channel], True)





    def on_mouse_move(self, event):
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return
            
        current_time = time.time()
        
        if self.selection_start and not self.selecting and not self.pan_mode and not self.zoom_mode and not self.brush_mode:
            if (abs(event.xdata - self.selection_start[0]) > 1 or 
                abs(event.ydata - self.selection_start[1]) > 1):
                self.selecting = True
                self.background = self.canvas.copy_from_bbox(self.ax.bbox)
                
                self.selection_rect = plt.Rectangle(
                    (self.selection_start[0], self.selection_start[1]), 0, 0,
                    fill=False, color='white', linestyle='--', animated=True
                )
                self.ax.add_patch(self.selection_rect)
                
        if self.selecting and self.selection_rect is not None:
            # Throttle updates
            if current_time - self.last_update_time < self.update_interval:
                return
            self.last_update_time = current_time
            
            # Use blitting
            self.canvas.restore_region(self.background)
            
            x_min = min(self.selection_start[0], event.xdata)
            y_min = min(self.selection_start[1], event.ydata)
            width = abs(event.xdata - self.selection_start[0])
            height = abs(event.ydata - self.selection_start[1])
            
            self.selection_rect.set_bounds(x_min, y_min, width, height)
            self.ax.draw_artist(self.selection_rect)
            self.canvas.blit(self.ax.bbox)

        elif self.panning and self.pan_start is not None:
            # Calculate the movement
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            # Get current view limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Calculate new limits
            new_xlim = [xlim[0] - dx, xlim[1] - dx]
            new_ylim = [ylim[0] - dy, ylim[1] - dy]
            
            # Get image bounds using cached dimensions
            if self.img_width is not None:  # Changed from self.channel_data[0] check
                # Ensure new limits don't go beyond image bounds
                if new_xlim[0] < 0:
                    new_xlim = [0, xlim[1] - xlim[0]]
                elif new_xlim[1] > self.img_width:  # Changed from img_width variable lookup
                    new_xlim = [self.img_width - (xlim[1] - xlim[0]), self.img_width]
                    
                if new_ylim[0] < 0:
                    new_ylim = [0, ylim[1] - ylim[0]]
                elif new_ylim[1] > self.img_height:  # Changed from img_height variable lookup
                    new_ylim = [self.img_height - (ylim[1] - ylim[0]), self.img_height]
            
            # Apply new limits
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw_idle()  # Changed from draw() to draw_idle()
            
            # Update pan start position
            self.pan_start = (event.xdata, event.ydata)

        elif self.painting and self.brush_mode:
            if event.inaxes != self.ax:
                return
            
            # OPTIMIZED: Queue paint operation instead of immediate execution
            self.queue_paint_operation(event)
            
            # OPTIMIZED: Schedule display update at controlled frequency
            if not self.pending_paint_update:
                self.pending_paint_update = True
                self.paint_timer.start(16)  # ~60fps max update rate

    def queue_paint_operation(self, event):
        """Queue a paint operation for background processing."""
        x, y = int(event.xdata), int(event.ydata)

        if self.pen_button.isChecked():
            channel = self.active_channel
        else:
            channel = 2

        if self.channel_data[channel] is not None:
            # Prepare paint session if needed
            if not self.paint_session_active:
                self.prepare_paint_session(channel)
            
            # Create paint operation
            paint_op = {
                'type': 'stroke',
                'x': x,
                'y': y,
                'last_pos': getattr(self, 'last_paint_pos', None),
                'brush_size': self.brush_size,
                'erase': self.erase,
                'channel': channel,
                'slice': self.current_slice,
                'threed': getattr(self, 'threed', False),
                'threedthresh': getattr(self, 'threedthresh', 1),
                'foreground': getattr(self, 'foreground', True),
                'machine_window': getattr(self, 'machine_window', None)
            }
            
            # Queue the operation
            try:
                self.paint_queue.put_nowait(paint_op)
            except queue.Full:
                pass  # Skip if queue is full to avoid blocking
            
            self.last_paint_pos = (x, y)

    def prepare_paint_session(self, channel):
        """Prepare optimized background for blitting during paint session."""
        if self.paint_session_active:
            return
            
        # IMPORTANT: Don't capture background here - let the main display update handle it
        # We'll capture the background after the proper channel visibility setup
        self.paint_session_active = True

    def end_paint_session(self):
        """Clean up after paint session."""
        self.paint_session_active = False
        self.paint_background = None
        self.last_paint_pos = None

    def paint_worker_loop(self):
        """Background thread for processing paint operations."""
        while True:
            try:
                paint_op = self.paint_queue.get(timeout=1.0)
                if paint_op is None:  # Shutdown signal
                    break
                
                with self.paint_lock:
                    self.execute_paint_operation(paint_op)
                    
            except queue.Empty:
                continue

    def shutdown(self):
        """Clean shutdown of worker thread."""
        self.paint_queue.put(None)  # Signal worker to stop
        if hasattr(self, 'paint_worker'):
            self.paint_worker.join(timeout=1.0)

    def execute_paint_operation(self, paint_op):
        """Execute a single paint operation on the data arrays."""
        if paint_op['type'] == 'stroke':
            channel = paint_op['channel']
            x, y = paint_op['x'], paint_op['y']
            last_pos = paint_op['last_pos']
            
            if last_pos is not None:
                # Paint line from last position to current
                points = self.get_line_points(last_pos[0], last_pos[1], x, y)
                for px, py in points:
                    height, width = self.channel_data[channel][paint_op['slice']].shape
                    if 0 <= px < width and 0 <= py < height:
                        self.paint_at_position_vectorized(
                            px, py, paint_op['erase'], paint_op['channel'],
                            paint_op['slice'], paint_op['brush_size'],
                            paint_op['threed'], paint_op['threedthresh'],
                            paint_op['foreground'], paint_op['machine_window']
                        )
            else:
                # Single point paint
                height, width = self.channel_data[channel][paint_op['slice']].shape
                if 0 <= x < width and 0 <= y < height:
                    self.paint_at_position_vectorized(
                        x, y, paint_op['erase'], paint_op['channel'],
                        paint_op['slice'], paint_op['brush_size'],
                        paint_op['threed'], paint_op['threedthresh'],
                        paint_op['foreground'], paint_op['machine_window']
                    )

    def flush_paint_updates(self):
        """Update the display with batched paint changes."""
        self.pending_paint_update = False
        
        # Determine which channel to update
        channel = self.active_channel if hasattr(self, 'pen_button') and self.pen_button.isChecked() else 2
        
        # Get current zoom to preserve it
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
        
        # Update display
        self.update_display_slice_optimized(channel, preserve_zoom=(current_xlim, current_ylim))

    def create_pan_background(self):
        """Create a static background image from currently visible channels with proper rendering"""
        # Store current zoom state
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Render all visible channels with proper colors/brightness into a single composite
        self.pan_background_image = self.create_composite_for_pan()
        self.pan_zoom_state = (current_xlim, current_ylim)

    def create_composite_for_pan(self):
        """Create a properly rendered composite image for panning"""
        # Get active channels and dimensions (copied from update_display)
        active_channels = [i for i in range(4) if self.channel_data[i] is not None]
        if active_channels:
            dims = [(self.channel_data[i].shape[1:3] if len(self.channel_data[i].shape) >= 3 else 
                    self.channel_data[i].shape) for i in active_channels]
            min_height = min(d[0] for d in dims)
            min_width = min(d[1] for d in dims)
        else:
            return None
        
        # Create a blank RGBA composite to accumulate all channels
        composite = np.zeros((min_height, min_width, 4), dtype=np.float32)
        
        # Process each visible channel exactly like update_display does
        for channel in range(4):
            if (self.channel_visible[channel] and 
                self.channel_data[channel] is not None):
                
                # Get current slice data (same logic as update_display)
                is_rgb = len(self.channel_data[channel].shape) == 4 and (self.channel_data[channel].shape[-1] == 3 or self.channel_data[channel].shape[-1] == 4)
                
                if len(self.channel_data[channel].shape) == 3 and not is_rgb:
                    current_image = self.channel_data[channel][self.current_slice, :, :]
                elif is_rgb:
                    current_image = self.channel_data[channel][self.current_slice]
                else:
                    current_image = self.channel_data[channel]

                if is_rgb and self.channel_data[channel].shape[-1] == 3:
                    # RGB image - convert to RGBA and blend
                    rgb_alpha = np.ones((*current_image.shape[:2], 4), dtype=np.float32)
                    rgb_alpha[:, :, :3] = current_image.astype(np.float32) / 255.0
                    rgb_alpha[:, :, 3] = 0.7  # Same alpha as update_display
                    composite = self.blend_layers(composite, rgb_alpha)
                    
                elif is_rgb and self.channel_data[channel].shape[-1] == 4:
                    # RGBA image - blend directly
                    rgba_image = current_image.astype(np.float32) / 255.0
                    composite = self.blend_layers(composite, rgba_image)
                    
                else:
                    # Regular channel processing (same logic as update_display)
                    if self.min_max[channel][0] == None:
                        self.min_max[channel][0] = np.min(current_image)
                    if self.min_max[channel][1] == None:
                        self.min_max[channel][1] = np.max(current_image)

                    img_min = self.min_max[channel][0]
                    img_max = self.min_max[channel][1]
                    
                    if img_min == img_max:
                        vmin = img_min
                        vmax = img_min + 1
                    else:
                        vmin = img_min + (img_max - img_min) * self.channel_brightness[channel]['min']
                        vmax = img_min + (img_max - img_min) * self.channel_brightness[channel]['max']
                    
                    # Normalize the image
                    if vmin == vmax:
                        normalized_image = np.zeros_like(current_image)
                    else:
                        normalized_image = np.clip((current_image - vmin) / (vmax - vmin), 0, 1)
                    
                    # Apply channel color and alpha
                    if channel == 2 and self.machine_window is not None:
                        # Special case for machine window channel 2
                        channel_rgba = self.apply_machine_colormap(current_image)
                    else:
                        # Regular channel with custom color
                        color = self.base_colors[channel]
                        channel_rgba = np.zeros((*normalized_image.shape, 4), dtype=np.float32)
                        channel_rgba[:, :, 0] = normalized_image * color[0]  # R
                        channel_rgba[:, :, 1] = normalized_image * color[1]  # G  
                        channel_rgba[:, :, 2] = normalized_image * color[2]  # B
                        channel_rgba[:, :, 3] = normalized_image * 0.7       # A (same alpha as update_display)
                    
                    # Blend this channel into the composite
                    composite = self.blend_layers(composite, channel_rgba)
        
        # Add highlight overlays if they exist (same logic as update_display)
        if self.mini_overlay and self.highlight and self.machine_window is None:
            highlight_rgba = self.create_highlight_rgba(self.mini_overlay_data, yellow=True)
            composite = self.blend_layers(composite, highlight_rgba)
        elif self.highlight_overlay is not None and self.highlight:
            highlight_slice = self.highlight_overlay[self.current_slice]
            if self.machine_window is None:
                highlight_rgba = self.create_highlight_rgba(highlight_slice, yellow=True)
            else:
                highlight_rgba = self.create_highlight_rgba(highlight_slice, yellow=False)
            composite = self.blend_layers(composite, highlight_rgba)
        
        # Convert to 0-255 range for display
        return (composite * 255).astype(np.uint8)

    def apply_machine_colormap(self, image):
        """Apply the special machine window colormap for channel 2"""
        rgba = np.zeros((*image.shape, 4), dtype=np.float32)
        
        # Transparent for 0
        mask_0 = (image == 0)
        rgba[mask_0] = [0, 0, 0, 0]
        
        # Light green for 1
        mask_1 = (image == 1)
        rgba[mask_1] = [0.5, 1, 0.5, 0.7]
        
        # Light red for 2
        mask_2 = (image == 2)
        rgba[mask_2] = [1, 0.5, 0.5, 0.7]
        
        return rgba

    def create_highlight_rgba(self, highlight_data, yellow=True):
        """Create RGBA highlight overlay"""
        rgba = np.zeros((*highlight_data.shape, 4), dtype=np.float32)
        
        if yellow:
            # Yellow highlight
            mask = highlight_data > 0
            rgba[mask] = [1, 1, 0, 0.5]  # Yellow with alpha 0.5
        else:
            # Multi-color highlight for machine window
            mask_1 = (highlight_data == 1)
            mask_2 = (highlight_data == 2)
            rgba[mask_1] = [1, 1, 0, 0.5]      # Yellow for 1
            rgba[mask_2] = [0, 0.7, 1, 0.5]    # Blue for 2
        
        return rgba

    def blend_layers(self, base, overlay):
        """Alpha blend two RGBA layers"""
        # Standard alpha blending formula
        alpha_overlay = overlay[:, :, 3:4]
        alpha_base = base[:, :, 3:4]
        
        # Calculate output alpha
        alpha_out = alpha_overlay + alpha_base * (1 - alpha_overlay)
        
        # Calculate output RGB
        rgb_out = np.zeros_like(base[:, :, :3])
        mask = alpha_out[:, :, 0] > 0
        
        rgb_out[mask] = (overlay[mask, :3] * alpha_overlay[mask] + 
                         base[mask, :3] * alpha_base[mask] * (1 - alpha_overlay[mask])) / alpha_out[mask]
        
        # Combine RGB and alpha
        result = np.zeros_like(base)
        result[:, :, :3] = rgb_out
        result[:, :, 3:4] = alpha_out
        
        return result

    def update_display_pan_mode(self):
        """Lightweight display update for pan preview mode"""
        if self.is_pan_preview and self.pan_background_image is not None:
            # Clear and setup axes
            self.ax.clear()
            self.ax.set_facecolor('black')
            
            # Get dimensions
            height, width = self.pan_background_image.shape[:2]
            
            # Display the composite background with preserved zoom
            self.ax.imshow(self.pan_background_image, 
                          extent=(-0.5, width-0.5, height-0.5, -0.5),
                          aspect='equal')
            
            # Restore the zoom state from when pan began
            if hasattr(self, 'pan_zoom_state'):
                self.ax.set_xlim(self.pan_zoom_state[0])
                self.ax.set_ylim(self.pan_zoom_state[1])
            
            # Style the axes (same as update_display)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y') 
            self.ax.set_title(f'Slice {self.current_slice}')
            self.ax.xaxis.label.set_color('black')
            self.ax.yaxis.label.set_color('black')
            self.ax.title.set_color('black')
            self.ax.tick_params(colors='black')
            for spine in self.ax.spines.values():
                spine.set_color('black')
                
            # Add measurement points if they exist (same as update_display)
            for point in self.measurement_points:
                x1, y1, z1 = point['point1']
                x2, y2, z2 = point['point2']
                pair_idx = point['pair_index']
                
                if z1 == self.current_slice:
                    self.ax.plot(x1, y1, 'yo', markersize=8)
                    self.ax.text(x1, y1+5, str(pair_idx), 
                                color='white', ha='center', va='bottom')
                if z2 == self.current_slice:
                    self.ax.plot(x2, y2, 'yo', markersize=8)
                    self.ax.text(x2, y2+5, str(pair_idx), 
                                color='white', ha='center', va='bottom')
                        
                if z1 == z2 == self.current_slice:
                    self.ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.5)
            
            self.canvas.draw_idle()

    def on_mouse_release(self, event):
        """Handle mouse release events - OPTIMIZED VERSION."""
        if self.pan_mode:
            # Get current view limits before restoring channels
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            
            self.panning = False
            self.pan_start = None
            self.canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            
            # Restore previously visible channels
            self.channel_visible = self.pre_pan_channel_state.copy()
            self.is_pan_preview = False
            self.pan_background = None
            
            # Update display with preserved zoom at new position
            self.update_display(preserve_zoom=(current_xlim, current_ylim))
        elif event.button == 1:  # Left button release
            if self.selecting and self.selection_rect is not None:
                # Get the rectangle bounds
                x0 = min(self.selection_start[0], event.xdata)
                y0 = min(self.selection_start[1], event.ydata)
                width = abs(event.xdata - self.selection_start[0])
                height = abs(event.ydata - self.selection_start[1])
                
                # Get current slice data for active channel
                if self.channel_data[self.active_channel] is not None:
                    data = self.channel_data[self.active_channel][self.current_slice]
                    
                    # Convert coordinates to array indices
                    x_min = max(0, int(x0))
                    y_min = max(0, int(y0))
                    x_max = min(data.shape[1], int(x0 + width))
                    y_max = min(data.shape[0], int(y0 + height))
                    
                    # Extract unique non-zero values in selection rectangle
                    selected_region = data[y_min:y_max, x_min:x_max]
                    selected_values = np.unique(selected_region)
                    selected_values = selected_values[selected_values != 0]  # Remove background
                    
                    # Check if ctrl is pressed
                    ctrl_pressed = 'ctrl' in event.modifiers
                    
                    # Update clicked_values based on active channel
                    if self.active_channel == 0:  # Nodes
                        if not ctrl_pressed:
                            self.clicked_values['nodes'] = []  # Clear existing selection if ctrl not pressed
                        self.clicked_values['nodes'].extend(selected_values)
                        # Remove duplicates while preserving order
                        self.clicked_values['nodes'] = list(dict.fromkeys(self.clicked_values['nodes']))
                        if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                            self.mini_overlay = True
                            self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                        else:
                            self.create_highlight_overlay(node_indices=self.clicked_values['nodes'])
                        
                        # Try to highlight the last selected value in tables
                        if self.clicked_values['nodes']:
                            self.highlight_value_in_tables(self.clicked_values['nodes'][-1])
                            
                    elif self.active_channel == 1:  # Edges
                        if not ctrl_pressed:
                            self.clicked_values['edges'] = []  # Clear existing selection if ctrl not pressed
                        self.clicked_values['edges'].extend(selected_values)
                        # Remove duplicates while preserving order
                        self.clicked_values['edges'] = list(dict.fromkeys(self.clicked_values['edges']))
                        if self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2] > self.mini_thresh:
                            self.mini_overlay = True
                            self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                        else:
                            self.create_highlight_overlay(edge_indices=self.clicked_values['edges'])
                        
                        # Try to highlight the last selected value in tables
                        if self.clicked_values['edges']:
                            self.highlight_value_in_tables(self.clicked_values['edges'][-1])
            
            elif not self.selecting and self.selection_start:  # If we had a click but never started selection
                # Handle as a normal click
                self.on_mouse_click(event)
            
            # Clean up selection
            self.selection_start = None
            self.selecting = False

            if self.selection_rect is not None:
                try:
                    self.selection_rect.remove()
                except:
                    pass
                self.selection_rect = None
                self.canvas.draw()

        # OPTIMIZED: Handle brush mode cleanup with paint session management
        if self.brush_mode and hasattr(self, 'painting') and self.painting:
            self.painting = False
            
            # Restore hidden channels
            try:
                for i in self.restore_channels:
                    self.channel_visible[i] = True
                self.restore_channels = []
            except:
                pass
            
            # OPTIMIZED: End paint session and ensure all operations complete
            self.end_paint_session()
            
            # OPTIMIZED: Stop timer and process any pending paint operations
            if hasattr(self, 'paint_timer'):
                self.paint_timer.stop()
            if hasattr(self, 'pending_paint_update') and self.pending_paint_update:
                self.flush_paint_updates()
            
            # Get current zoom and do final display update
            current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
            current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
            self.update_display(preserve_zoom=(current_xlim, current_ylim))


    def highlight_value_in_tables(self, clicked_value):
        """Helper method to find and highlight a value in both tables."""
        
        if not self.network_table.model() and not self.selection_table.model():
            return False

        found = False
        tables_to_check = [self.network_table, self.selection_table]
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

    def on_mouse_click(self, event):
        """Handle mouse clicks for zooming and data inspection."""
        if event.inaxes != self.ax:
            return
            
        if self.zoom_mode:
            # Existing zoom functionality
            if self.original_xlim is None:
                self.original_xlim = self.ax.get_xlim()
                self.original_ylim = self.ax.get_ylim()
            
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            
            if event.button == 1:  # Left click - zoom in
                x_range = (current_xlim[1] - current_xlim[0]) / 4
                y_range = (current_ylim[1] - current_ylim[0]) / 4
                
                self.ax.set_xlim([xdata - x_range, xdata + x_range])
                self.ax.set_ylim([ydata - y_range, ydata + y_range])

                self.zoom_changed = True  # Flag that zoom has changed
                
                if not hasattr(self, 'zoom_changed'):
                    self.zoom_changed = False
                
            elif event.button == 3:  # Right click - zoom out
                x_range = (current_xlim[1] - current_xlim[0])
                y_range = (current_ylim[1] - current_ylim[0])
                
                new_xlim = [xdata - x_range, xdata + x_range]
                new_ylim = [ydata - y_range, ydata + y_range]
                
                if (new_xlim[0] <= self.original_xlim[0] or 
                    new_xlim[1] >= self.original_xlim[1] or
                    new_ylim[0] <= self.original_ylim[0] or
                    new_ylim[1] >= self.original_ylim[1]):
                    self.ax.set_xlim(self.original_xlim)
                    self.ax.set_ylim(self.original_ylim)
                else:
                    self.ax.set_xlim(new_xlim)
                    self.ax.set_ylim(new_ylim)


                self.zoom_changed = False  # Flag that zoom has changed
                
                
            
            self.canvas.draw()
        
        elif event.button == 3:  # Right click
            self.create_context_menu(event)

        else:  # Not in zoom mode - handle value inspection
            if self.channel_data[self.active_channel] is not None:
                try:
                    # Get clicked value
                    x_idx = int(round(event.xdata))
                    y_idx = int(round(event.ydata))
                    # Check if Ctrl key is pressed (using matplotlib's key_press system)
                    ctrl_pressed = 'ctrl' in event.modifiers
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
                                # Remove value if it's already selected
                                self.clicked_values['nodes'].remove(clicked_value)
                            else:
                                # Add value if it's not already selected
                                self.clicked_values['nodes'].append(clicked_value)
                        else:
                            # Reset both lists and start new selection
                            self.clicked_values = {'nodes': [clicked_value], 'edges': []}
                        # Get latest value (or the last remaining one if we just removed an item)
                        latest_value = self.clicked_values['nodes'][-1] if self.clicked_values['nodes'] else None
                        self.handle_info('node')
                    elif self.active_channel == 1:
                        if ctrl_pressed:
                            if clicked_value in self.clicked_values['edges']:
                                # Remove value if it's already selected
                                self.clicked_values['edges'].remove(clicked_value)
                            else:
                                # Add value if it's not already selected
                                self.clicked_values['edges'].append(clicked_value)
                        else:
                            # Reset both lists and start new selection
                            self.clicked_values = {'nodes': [], 'edges': [clicked_value]}
                        # Get latest value (or the last remaining one if we just removed an item)
                        latest_value = self.clicked_values['edges'][-1] if self.clicked_values['edges'] else None
                        self.handle_info('edge')

                    
                    # Try to find and highlight the latest value in the current table
                    try:
                        found = self.highlight_value_in_tables(latest_value)
                    except:
                        return
                    
                    # If not found in current table but it exists in the other table, offer to switch
                    if not found:
                        other_table = self.selection_table if self.active_table == self.network_table else self.network_table
                        if other_table.model() is not None:
                            df = other_table.model()._data
                            if self.active_channel == 0:
                                exists_in_other = (df[df.columns[0]] == latest_value).any() or (df[df.columns[1]] == latest_value).any()
                            else:
                                exists_in_other = (df[df.columns[2]] == latest_value).any()
                                
                            if exists_in_other:
                                # Switch to the other table
                                if other_table == self.network_table:
                                    self.network_button.click()
                                else:
                                    self.selection_button.click()
                                # Now highlight in the newly active table
                                self.highlight_value_in_tables(latest_value)

                    # Highlight the clicked element in the image using the stored lists                
                    if self.active_channel == 0 and (starting_vals['nodes']) != (self.clicked_values['nodes']):
                        if self.channel_data[0].shape[0] * self.channel_data[0].shape[1] * self.channel_data[0].shape[2] > self.mini_thresh:
                            self.mini_overlay = True
                            self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                        else:
                            self.create_highlight_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])
                    elif self.active_channel == 1 and starting_vals['edges'] != self.clicked_values['edges']:
                        if self.channel_data[1].shape[0] * self.channel_data[1].shape[1] * self.channel_data[1].shape[2] > self.mini_thresh:
                            self.mini_overlay = True
                            self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
                        else:
                            self.create_highlight_overlay(node_indices=self.clicked_values['nodes'], edge_indices=self.clicked_values['edges'])

                                
                except IndexError:
                    pass  # Clicked outside image boundaries
                
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")

        # Create Save submenu
        save_menu = file_menu.addMenu("Save")
        network_save = save_menu.addAction("Save Network3D Object")
        network_save.triggered.connect(lambda: self.save_network_3d(False))
        for i in range(4):
            save_action = save_menu.addAction(f"Save {self.channel_names[i]}")
            save_action.triggered.connect(lambda checked, ch=i: self.save(ch, False))
        highlight_save = save_menu.addAction("Save Highlight Overlay")
        highlight_save.triggered.connect(lambda checked, ch=4: self.save(ch, False))

        # Create Save As submenu
        save_as_menu = file_menu.addMenu("Save As")
        network_saveas = save_as_menu.addAction("Save Network3D Object As")
        network_saveas.triggered.connect(lambda: self.save_network_3d(True))
        for i in range(4):
            save_action = save_as_menu.addAction(f"Save {self.channel_names[i]} As")
            save_action.triggered.connect(lambda checked, ch=i: self.save(ch))
        highlight_save = save_as_menu.addAction("Save Highlight Overlay As")
        highlight_save.triggered.connect(lambda checked, ch=4: self.save(ch))
        
        # Create Load submenu
        load_menu = file_menu.addMenu("Load")
        network_load = load_menu.addAction("Load Network3D Object")
        network_load.triggered.connect(self.load_from_network_obj)
        for i in range(4):
            load_action = load_menu.addAction(f"Load {self.channel_names[i]}")
            load_action.triggered.connect(lambda checked, ch=i: self.load_channel(ch))
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
        load_action = misc_menu.addAction("Merge Nodes")
        load_action.triggered.connect(lambda: self.load_misc('Merge Nodes'))
        load_action = misc_menu.addAction("Merge Node IDs from Images")
        load_action.triggered.connect(self.show_merge_node_id_dialog)

        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analyze")
        network_menu = analysis_menu.addMenu("Network")
        netshow_action = network_menu.addAction("Show Network")
        netshow_action.triggered.connect(self.show_netshow_dialog)
        report_action = network_menu.addAction("Generic Network Report")
        report_action.triggered.connect(self.handle_report)
        partition_action = network_menu.addAction("Community Partition + Generic Community Stats")
        partition_action.triggered.connect(self.show_partition_dialog)
        com_identity_action = network_menu.addAction("Identity Makeup of Network Communities (and UMAP)")
        com_identity_action.triggered.connect(self.handle_com_id)
        com_neighbor_action = network_menu.addAction("Convert Network Communities into Neighborhoods?")
        com_neighbor_action.triggered.connect(self.handle_com_neighbor)
        com_cell_action = network_menu.addAction("Create Communities Based on Cuboidal Proximity Cells?")
        com_cell_action.triggered.connect(self.handle_com_cell)
        stats_menu = analysis_menu.addMenu("Stats")
        allstats_action = stats_menu.addAction("Calculate Generic Network Stats")
        allstats_action.triggered.connect(self.stats)
        histos_action = stats_menu.addAction("Calculate Generic Network Histograms")
        histos_action.triggered.connect(self.histos)
        radial_action = stats_menu.addAction("Radial Distribution Analysis")
        radial_action.triggered.connect(self.show_radial_dialog)
        degree_dist_action = stats_menu.addAction("Degree Distribution Analysis")
        degree_dist_action.triggered.connect(self.show_degree_dist_dialog)
        neighbor_id_action = stats_menu.addAction("Identity Distribution of Neighbors")
        neighbor_id_action.triggered.connect(self.show_neighbor_id_dialog)
        ripley_action = stats_menu.addAction("Ripley Clustering Analysis")
        ripley_action.triggered.connect(self.show_ripley_dialog)
        heatmap_action = stats_menu.addAction("Community Cluster Heatmap")
        heatmap_action.triggered.connect(self.show_heatmap_dialog)
        nearneigh_action = stats_menu.addAction("Average Nearest Neighbors")
        nearneigh_action.triggered.connect(self.show_nearneigh_dialog)
        vol_action = stats_menu.addAction("Calculate Volumes")
        vol_action.triggered.connect(self.volumes)
        rad_action = stats_menu.addAction("Calculate Radii")
        rad_action.triggered.connect(self.show_rad_dialog)
        inter_action = stats_menu.addAction("Calculate Node < > Edge Interaction")
        inter_action.triggered.connect(self.show_interaction_dialog)
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

        rand_menu = analysis_menu.addMenu("Randomize")
        random_action = rand_menu.addAction("Generate Equivalent Random Network")
        random_action.triggered.connect(self.show_random_dialog)
        random_nodes = rand_menu.addAction("Scramble Nodes (Centroids)")
        random_nodes.triggered.connect(self.show_randnode_dialog)



        # Process menu
        process_menu = menubar.addMenu("Process")
        calculate_menu = process_menu.addMenu("Calculate")
        calc_all_action = calculate_menu.addAction("Calculate Connectivity Network (Find Node-Edge-Node Network)")
        calc_all_action.triggered.connect(self.show_calc_all_dialog)
        calc_prox_action = calculate_menu.addAction("Calculate Proximity Network (connect nodes by distance)")
        calc_prox_action.triggered.connect(self.show_calc_prox_dialog)
        centroid_action = calculate_menu.addAction("Calculate Centroids (Active Image)")
        centroid_action.triggered.connect(self.show_centroid_dialog)

        image_menu = process_menu.addMenu("Image")
        resize_action = image_menu.addAction("Resize (Up/Downsample)")
        resize_action.triggered.connect(self.show_resize_dialog)
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
        slabel_action = image_menu.addAction("Neighborhood Labels")
        slabel_action.triggered.connect(self.show_slabel_dialog)
        thresh_action = image_menu.addAction("Threshold/Segment")
        thresh_action.triggered.connect(self.show_thresh_dialog)
        mask_action = image_menu.addAction("Mask Channel")
        mask_action.triggered.connect(self.show_mask_dialog)
        crop_action = image_menu.addAction("Crop Channels")
        crop_action.triggered.connect(self.show_crop_dialog)
        type_action = image_menu.addAction("Channel dtype")
        type_action.triggered.connect(self.show_type_dialog)
        skeletonize_action = image_menu.addAction("Skeletonize")
        skeletonize_action.triggered.connect(self.show_skeletonize_dialog)
        watershed_action = image_menu.addAction("Watershed")
        watershed_action.triggered.connect(self.show_watershed_dialog)
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
        branch_action.triggered.connect(self.show_branch_dialog)
        genvor_action = generate_menu.addAction("Generate Voronoi Diagram (From Node Centroids) - goes in Overlay2")
        genvor_action.triggered.connect(self.voronoi)

        modify_action = process_menu.addAction("Modify Network")
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
        idoverlay_action = overlay_menu.addAction("Create ID Overlay")
        idoverlay_action.triggered.connect(self.show_idoverlay_dialog)
        coloroverlay_action = overlay_menu.addAction("Color Nodes (or Edges)")
        coloroverlay_action.triggered.connect(self.show_coloroverlay_dialog)
        #searchoverlay_action = overlay_menu.addAction("Show Search Regions")
        #searchoverlay_action.triggered.connect(self.show_search_dialog)
        shuffle_action = overlay_menu.addAction("Shuffle")
        shuffle_action.triggered.connect(self.show_shuffle_dialog)
        arbitrary_action = image_menu.addAction("Select Objects")
        arbitrary_action.triggered.connect(self.show_arbitrary_dialog)
        show3d_action = image_menu.addAction("Show 3D (Napari)")
        show3d_action.triggered.connect(self.show3d_dialog)

        # Help

        help_button = menubar.addAction("Help")
        help_button.triggered.connect(self.help_me)

    def help_me(self):

        import webbrowser
        try:
            webbrowser.open('https://nettracer3d.readthedocs.io/en/latest/')
            return True
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False


    def stats(self):
        """Method to get and display the network stats"""
        # Get the stats dictionary
        try:
            stats = my_network.get_network_stats()

            self.format_for_upperright_table(stats, title = 'Network Stats')
        except Exception as e:
            print(f"Error finding stats: {e}")

    def histos(self):

        """from networkx documentation"""

        try:

            G = my_network.network

            shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
            diameter = max(nx.eccentricity(G, sp=shortest_path_lengths).values())
            # We know the maximum shortest path length (the diameter), so create an array
            # to store values from 0 up to (and including) diameter
            path_lengths = np.zeros(diameter + 1, dtype=int)



            # Extract the frequency of shortest path lengths between two nodes
            for pls in shortest_path_lengths.values():
                pl, cnts = np.unique(list(pls.values()), return_counts=True)
                path_lengths[pl] += cnts

            # Express frequency distribution as a percentage (ignoring path lengths of 0)
            freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()

            # Plot the frequency distribution (ignoring path lengths of 0) as a percentage
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.bar(np.arange(1, diameter + 1), height=freq_percent)
            ax.set_title(
                "Distribution of shortest path length in G", fontdict={"size": 35}, loc="center"
            )
            ax.set_xlabel("Shortest Path Length", fontdict={"size": 22})
            ax.set_ylabel("Frequency (%)", fontdict={"size": 22})

            plt.show()
            freq_dict = {freq: length for length, freq in enumerate(freq_percent, start=1)}
            self.format_for_upperright_table(freq_dict, metric='Frequency (%)', value='Shortest Path Length', title="Distribution of shortest path length in G")

            degree_centrality = nx.centrality.degree_centrality(G)
            plt.figure(figsize=(15, 8))
            plt.hist(degree_centrality.values(), bins=25)
            plt.xticks(ticks=[0, 0.025, 0.05, 0.1, 0.15, 0.2])  # set the x axis ticks
            plt.title("Degree Centrality Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Degree Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.format_for_upperright_table(degree_centrality, metric='Node', value='Degree Centrality', title="Degree Centrality Table")


            betweenness_centrality = nx.centrality.betweenness_centrality(
                G
            )
            plt.figure(figsize=(15, 8))
            plt.hist(betweenness_centrality.values(), bins=100)
            plt.xticks(ticks=[0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5])  # set the x axis ticks
            plt.title("Betweenness Centrality Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Betweenness Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.format_for_upperright_table(betweenness_centrality, metric='Node', value='Betweenness Centrality', title="Betweenness Centrality Table")


            closeness_centrality = nx.centrality.closeness_centrality(
                G
            )
            plt.figure(figsize=(15, 8))
            plt.hist(closeness_centrality.values(), bins=60)
            plt.title("Closeness Centrality Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Closeness Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.format_for_upperright_table(closeness_centrality, metric='Node', value='Closeness Centrality', title="Closeness Centrality Table")


            eigenvector_centrality = nx.centrality.eigenvector_centrality(
                G
            )
            plt.figure(figsize=(15, 8))
            plt.hist(eigenvector_centrality.values(), bins=60)
            plt.xticks(ticks=[0, 0.01, 0.02, 0.04, 0.06, 0.08])  # set the x axis ticks
            plt.title("Eigenvector Centrality Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Eigenvector Centrality", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.format_for_upperright_table(eigenvector_centrality, metric='Node', value='Eigenvector Centrality', title="Eigenvector Centrality Table")



            clusters = nx.clustering(G)
            plt.figure(figsize=(15, 8))
            plt.hist(clusters.values(), bins=50)
            plt.title("Clustering Coefficient Histogram ", fontdict={"size": 35}, loc="center")
            plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
            plt.ylabel("Counts", fontdict={"size": 20})
            plt.show()
            self.format_for_upperright_table(clusters, metric='Node', value='Clustering Coefficient', title="Clustering Coefficient Table")

            bridges = list(nx.bridges(G))
            self.format_for_upperright_table(bridges, metric = 'Node Pair', title="Bridges")

        except Exception as e:
            print(f"Error generating histograms: {e}")

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

        

    def format_for_upperright_table(self, data, metric='Metric', value='Value', title=None):
        """
        Format dictionary or list data for display in upper right table.
        
        Args:
            data: Dictionary with keys and single/multiple values, or a list of values
            metric: String for the key/index column header
            value: String or list of strings for value column headers (used for dictionaries only)
            title: Optional custom title for the tab
        """
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
        
        try:

            if isinstance(data, (list, tuple, np.ndarray)):
                # Handle list input - create single column DataFrame
                df = pd.DataFrame({
                    metric: [convert_to_numeric(val) for val in data]
                })
                
                # Format floating point numbers
                df[metric] = df[metric].apply(lambda x: f"{x:.3f}" if isinstance(x, (float, np.float64)) else str(x))
                
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
                        df[header] = df[header].apply(lambda x: f"{x:.3f}" if isinstance(x, (float, np.float64)) else str(x))
                        
                else:
                    # Single-value case
                    df = pd.DataFrame({
                        metric: data.keys(),
                        value: [convert_to_numeric(val) for val in data.values()]
                    })
                    
                    # Format floating point numbers
                    df[value] = df[value].apply(lambda x: f"{x:.3f}" if isinstance(x, (float, np.float64)) else str(x))
            
            # Create new table
            table = CustomTableView(self)
            table.setModel(PandasModel(df))
            
            # Add to tabbed widget
            if title is None:
                self.tabbed_data.add_table(f"{metric} Analysis", table)
            else:
                self.tabbed_data.add_table(f"{title}", table)
            
            # Adjust column widths to content
            for column in range(table.model().columnCount(None)):
                table.resizeColumnToContents(column)

        except:
            pass

    def show_merge_node_id_dialog(self):

        dialog = MergeNodeIdDialog(self)
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
        dialog.exec()

    def show_calc_prox_dialog(self):
        """Show the proximity calc dialog"""
        dialog = ProxDialog(self)
        dialog.exec()

    def show_centroid_dialog(self):
        """show the centroid dialog"""
        dialog = CentroidDialog(self)
        dialog.exec()

    def show_dilate_dialog(self):
        """show the dilate dialog"""
        dialog = DilateDialog(self)
        dialog.exec()

    def show_erode_dialog(self):
        """show the erode dialog"""
        dialog = ErodeDialog(self)
        dialog.exec()

    def show_hole_dialog(self):
        """show the hole dialog"""
        dialog = HoleDialog(self)
        dialog.exec()

    def show_label_dialog(self):
        """Show the label dialog"""
        dialog = LabelDialog(self)
        dialog.exec()

    def show_slabel_dialog(self):
        """Show the slabel dialog"""
        dialog = SLabelDialog(self)
        dialog.exec()

    def show_thresh_dialog(self):
        """Show threshold dialog"""
        if self.machine_window is not None:
            return

        dialog = ThresholdDialog(self)
        dialog.exec()


    def show_mask_dialog(self):
        """Show the mask dialog"""
        dialog = MaskDialog(self)
        dialog.exec()

    def show_crop_dialog(self):
        """Show the crop dialog"""
        dialog = CropDialog(self)
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

    def show_centroid_node_dialog(self):
        """show the centroid node dialog"""
        dialog = CentroidNodeDialog(self)
        dialog.exec()


    def show_gennodes_dialog(self, down_factor = None, called = False):
        """show the gennodes dialog"""
        gennodes = GenNodesDialog(self, down_factor = down_factor, called = called)
        gennodes.exec()

    def show_branch_dialog(self):
        """Show the branch label dialog"""
        dialog = BranchDialog(self)
        dialog.exec()

    def voronoi(self):

        try:

            if my_network.nodes is not None:
                shape = my_network.nodes.shape
            else:
                shape = None

            if my_network.node_centroids is None:
                self.show_centroid_dialog()
                if my_network.node_centroids is None:
                    print("Node centroids must be set")
                    return

            array = pxt.create_voronoi_3d_kdtree(my_network.node_centroids, shape)
            self.load_channel(3, array, True)

        except Exception as e:
            print(f"Error generating voronoi: {e}")


    def show_modify_dialog(self):
        """Show the network modify dialog"""
        dialog = ModifyDialog(self)
        dialog.exec()


    def show_binarize_dialog(self):
        """show the binarize dialog"""
        dialog = BinarizeDialog(self)
        dialog.exec()


    def show_resize_dialog(self):
        """show the resize dialog"""
        dialog = ResizeDialog(self)
        dialog.exec()


    def show_properties_dialog(self):
        """Show the properties dialog"""
        dialog = PropertiesDialog(self)
        dialog.exec()
    
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

    def show_search_dialog(self):
        """Show the search dialog"""
        dialog = SearchOverlayDialog(self)
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

                        first_value = list(my_network.node_identities.values())[0]  # Check that there are not multiple IDs
                        if isinstance(first_value, (list, tuple)):
                            trump_value, ok = QInputDialog.getText(
                                self,
                                'Multiple IDs Detected',
                                'The node identities appear to contain multiple ids per node in a list.\n'
                                'If you desire one node ID to trump all others, enter it here.\n'
                                '(Enter "-" to have the first IDs trump all others)\n'
                                '(Enter "/" to have multi-ID nodes be split into many nodes sharing a centroid)\n'
                                '(Close this window to continue with multi-ID nodes)'
                            )
                            if not ok or trump_value.strip() == '':
                                trump_value = None
                            elif trump_value.upper() == '-':
                                trump_value = '-'
                            elif trump_value.upper() == "/":
                                trump_value = '/'
                            my_network.node_identities = uncork(my_network.node_identities, trump_value)
                        else:
                            trump_value = None
                            my_network.node_identities = uncork(my_network.node_identities, trump_value)


                        if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                            try:
                                self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                            except Exception as e:
                                print(f"Error loading node identity table: {e}")

                    elif sort == 'Node Centroids':
                        my_network.load_node_centroids(file_path = filename)

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

                if len(np.unique(my_network.nodes)) < 3:
                    self.show_label_dialog()

                # First ask user what they want to select
                msg = QMessageBox()
                msg.setWindowTitle("Selection Type")
                msg.setText("Would you like to select a TIFF file or a directory?")
                tiff_button = msg.addButton("TIFF File", QMessageBox.ButtonRole.AcceptRole)
                dir_button = msg.addButton("Directory", QMessageBox.ButtonRole.AcceptRole)
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

                msg.exec()

                if msg.clickedButton() == tiff_button:
                    # Code for selecting TIFF files
                    filename, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select TIFF file",
                        "",
                        "TIFF files (*.tiff *.tif)"
                    )
                    if filename:
                        selected_path = filename

                elif msg.clickedButton() == dir_button:
                    # Code for selecting directories
                    dialog = QFileDialog(self)
                    dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
                    dialog.setOption(QFileDialog.Option.ReadOnly)
                    dialog.setFileMode(QFileDialog.FileMode.Directory)
                    dialog.setViewMode(QFileDialog.ViewMode.Detail)

                    if dialog.exec() == QFileDialog.DialogCode.Accepted:
                        selected_path = dialog.directory().absolutePath()

                my_network.merge_nodes(selected_path, root_id = self.node_name)
                self.load_channel(0, my_network.nodes, True)


                if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                    try:
                        self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                    except Exception as e:
                        print(f"Error loading node identity table: {e}")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Merging",
                    f"Failed to load {sort}: {str(e)}"
                )



    # Modify load_from_network_obj method
    def load_from_network_obj(self):
        try: 
            directory = QFileDialog.getExistingDirectory(
                self,
                f"Select Directory for Network3D Object",
                "",
                QFileDialog.Option.ShowDirsOnly
                )

            if directory != "":

                self.reset(network = True, xy_scale = 1, z_scale = 1, edges = True, network_overlay = True, id_overlay = True)

                my_network.assemble(directory)

                # Load image channels
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

                # Update slider range based on new data
                for channel in self.channel_data:
                    if channel is not None:
                        self.slice_slider.setEnabled(True)
                        self.slice_slider.setMinimum(0)
                        self.slice_slider.setMaximum(channel.shape[0] - 1)
                        self.slice_slider.setValue(0)
                        self.current_slice = 0
                        break

                # Display network_lists in the network table
                # Create empty DataFrame for network table if network_lists is None
                if not hasattr(my_network, 'network_lists') or my_network.network_lists is None:
                    empty_df = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
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
            QMessageBox.critical(
                self,
                "Error Loading Network 3D Object",
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
    
    def handle_excel_data(self, data_dict, property_name):
        """Handle data received from Excelotron"""
        print(f"Received data for property: {property_name}")
        print(f"Data keys: {list(data_dict.keys())}")

        if property_name == 'Node Centroids':

            try:

                ys = data_dict['Y']
                xs = data_dict['X']
                if 'Numerical IDs' in data_dict:
                    nodes = data_dict['Numerical IDs']
                else:
                    nodes = np.arange(1, len(ys) + 1)


                if 'Z' in data_dict:
                    zs = data_dict['Z']
                else:
                    zs = np.zeros(len(ys))

                centroids = {}

                for i in range(len(nodes)):

                    centroids[nodes[i]] = [int(zs[i]), int(ys[i]), int(xs[i])]

                my_network.node_centroids = centroids

                self.format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')

                print("Centroids succesfully set")

            except Exception as e:
                print(f"Error: {e}")

        elif property_name == 'Node Identities':

            try:

                idens = data_dict['Identity Column']

                if 'Numerical IDs' in data_dict:
                    nodes = data_dict['Numerical IDs']
                else:
                    nodes = np.arange(1, len(idens) + 1)

                identities = {}


                for i in range(len(nodes)):

                    identities[nodes[i]] = str(idens[i])

                my_network.node_identities = identities

                self.format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', title = 'Node Identities')

                print("Identities succesfully set")

            except Exception as e:
                print(f"Error: {e}")

        elif property_name == 'Node Communities':

            try:

                coms = data_dict['Community Identifier']

                if 'Numerical IDs' in data_dict:
                    nodes = data_dict['Numerical IDs']
                else:
                    nodes = np.arange(1, len(coms) + 1)

                communities = {}

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
                btn.setStyleSheet("background-color: lightblue;")
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
        
        if method not in ['first', 'mean', 'max', 'min']:
            raise ValueError(f"Unknown method: {method}")
        
        if method == 'first':
            return array[..., 0]
        elif method == 'mean':
            return np.mean(array, axis=-1)
        elif method == 'max':
            return np.max(array, axis=-1)
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

    def confirm_resize_dialog(self):
        """Shows a dialog asking user to resize image"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText("Image Format Alert")
        msg.setInformativeText(f"This image is a different shape than the ones loaded into the viewer window. Trying to run processes with images of different sizes has a high probability of crashing the program.\nPress yes to resize the new image to the other images. Press no to load it anyway.")
        msg.setWindowTitle("Resize")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def load_channel(self, channel_index, channel_data=None, data=False, assign_shape = True):
        """Load a channel and enable active channel selection if needed."""

        try:
            if not data:  # For solo loading
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
                        self.channel_data[channel_index] = tifffile.imread(filename)

                        
                    elif file_extension == 'nii':
                        import nibabel as nib
                        nii_img = nib.load(filename)
                        # Get data and transpose to match TIFF orientation
                        # If X needs to become Z, we move axis 2 (X) to position 0 (Z)
                        arraydata = nii_img.get_fdata()
                        self.channel_data[channel_index] = np.transpose(arraydata, (2, 1, 0))
                        
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

            if len(self.channel_data[channel_index].shape) == 2:  # handle 2d data
                self.channel_data[channel_index] = np.expand_dims(self.channel_data[channel_index], axis=0)

            try:
                if len(self.channel_data[channel_index].shape) == 3:  # potentially 2D RGB
                    if self.channel_data[channel_index].shape[-1] in (3, 4):  # last dim is 3 or 4
                        if self.confirm_rgb_dialog():
                            # User confirmed it's 2D RGB, expand to 4D
                            self.channel_data[channel_index] = np.expand_dims(self.channel_data[channel_index], axis=0)
            except:
                pass
            
            try:
                if len(self.channel_data[channel_index].shape) == 4 and (channel_index == 0 or channel_index == 1):
                    self.channel_data[channel_index] = self.reduce_rgb_dimension(self.channel_data[channel_index])
            except:
                pass

            reset_resize = False

            for i in range(4): #Try to ensure users don't load in different sized arrays
                if self.channel_data[i] is None or i == channel_index or data:
                    if self.highlight_overlay is not None: #Make sure highlight overlay is always the same shape as new images
                        try:
                            if self.channel_data[i].shape[:3] != self.highlight_overlay.shape:
                                self.resizing = True
                                reset_resize = True
                                self.highlight_overlay = None
                        except:
                            pass
                    if not data:
                        self.original_xlim = None
                        self.original_ylim = None
                    continue
                else:
                    old_shape = self.channel_data[i].shape[:3] #Ask user to resize images that are shaped differently
                    if old_shape != self.channel_data[channel_index].shape[:3]:
                        if self.confirm_resize_dialog():
                            self.channel_data[channel_index] = n3d.upsample_with_padding(self.channel_data[channel_index], original_shape = old_shape)
                            break


            if channel_index == 0:
                my_network.nodes = self.channel_data[channel_index]
            elif channel_index == 1:
                my_network.edges = self.channel_data[channel_index]
            elif channel_index == 2:
                my_network.network_overlay = self.channel_data[channel_index]
            elif channel_index == 3:
                my_network.id_overlay = self.channel_data[channel_index]
            
            # Enable the channel button
            self.channel_buttons[channel_index].setEnabled(True)
            self.delete_buttons[channel_index].setEnabled(True) 

            
            # Enable active channel selector if this is the first channel loaded
            if not self.active_channel_combo.isEnabled():
                self.active_channel_combo.setEnabled(True)
            
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
            self.min_max[channel_index][1] = np.max(self.channel_data[channel_index])
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

            self.shape = self.channel_data[channel_index].shape

            self.img_height, self.img_width = self.shape[1], self.shape[2]


            self.update_display(reset_resize = reset_resize)


                
        except Exception as e:

            if not data:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Error Loading File",
                    f"Failed to load tiff file: {str(e)}"
                )

    def delete_channel(self, channel_index, called = True):
        """Delete the specified channel and update the display."""
        if called:
            # Confirm deletion
            reply = QMessageBox.question(
                self,
                'Delete Channel',
                f'Are you sure you want to delete the {self.channel_names[channel_index]} channel?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
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
            self.delete_buttons[channel_index].setEnabled(False)
            self.channel_visible[channel_index] = False
            
            # If this was the active channel, switch to the first available channel
            if self.active_channel == channel_index:
                for i in range(4):
                    if self.channel_data[i] is not None:
                        self.set_active_channel(i)
                        break
                else:
                    # If no channels are available, disable active channel selector
                    self.active_channel_combo.setEnabled(False)
            
            # Update display
            self.update_display()

    def reset(self, nodes = False, network = False, xy_scale = 1, z_scale = 1, edges = False, search_region = False, network_overlay = False, id_overlay = False):
        """Method to flexibly reset certain fields to free up the RAM as desired"""
        
        # Set scales first before any clearing operations
        my_network.xy_scale = xy_scale
        my_network.z_scale = z_scale

        if network:
            my_network.network = None
            my_network.communities = None

            # Create empty DataFrame
            empty_df = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
            
            # Clear network table
            self.network_table.setModel(PandasModel(empty_df))
            
            # Clear selection table
            self.selection_table.setModel(PandasModel(empty_df))

        if nodes:
            self.delete_channel(0, False)

        if edges:
            self.delete_channel(1, False)
        try:
            if search_region:
                my_network.search_region = None
        except:
            pass

        if network_overlay:
            self.delete_channel(2, False)

        if id_overlay:
            self.delete_channel(3, False)



    def save_network_3d(self, asbool=True):
        try:
            if asbool:  # Save As
                # First let user select parent directory
                parent_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Location for Network3D Object Outputs",
                    "",
                    QFileDialog.Option.ShowDirsOnly
                )
                if not parent_dir:  # If user canceled the directory selection
                    return  # Exit the method early
                    
                # Prompt user for new folder name
                new_folder_name, ok = QInputDialog.getText(
                    self,
                    "New Folder",
                    "Enter name for new output folder:"
                )
                
                # Check if user canceled the folder name dialog
                if not ok or not new_folder_name:
                    return  # Exit the method early
            
            else:  # Save
                parent_dir = None  # Let the backend handle default save location
            
            # Call appropriate save method
            if asbool:
                my_network.dump(parent_dir=parent_dir, name=new_folder_name)
            else:
                my_network.dump(name='my_network')
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving File",
                f"Failed to save file: {str(e)}"
            )


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
                filename = None  # Let the backend handle default save location
            
            # Call appropriate save method
            if filename is not None or not asbool:  # Proceed if we have a filename OR if it's a regular save
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
        if not self.resizing:
            current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
            current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
        else:
            current_xlim = None
            current_ylim = None
        
        # Store the pending slice and view settings
        self.pending_slice = (self.slice_slider.value(), (current_xlim, current_ylim))
        
        # Reset and restart timer
        self._slice_update_timer.start(20)  # 20ms delay
        
    def _do_slice_update(self):
        """Actually perform the slice update after debounce delay."""
        if self.pending_slice is not None:
            slice_value, view_settings = self.pending_slice
            self.current_slice = slice_value
            if self.mini_overlay == True: #If we are rendering the highlight overlay for selected values one at a time.
                self.create_mini_overlay(node_indices = self.clicked_values['nodes'], edge_indices = self.clicked_values['edges'])
            self.update_display(preserve_zoom=view_settings)
            if self.machine_window is not None:
                self.machine_window.poke_segmenter()
            self.pending_slice = None

    def update_brightness(self, channel_index, values):
        """Update brightness/contrast settings for a channel."""

        # Store current zoom settings before toggling
        current_xlim = self.ax.get_xlim() if hasattr(self, 'ax') and self.ax.get_xlim() != (0, 1) else None
        current_ylim = self.ax.get_ylim() if hasattr(self, 'ax') and self.ax.get_ylim() != (0, 1) else None
        # Convert slider values (0-100) to data values (0-1)
        min_val, max_val = values
        self.channel_brightness[channel_index]['min'] = min_val / 65535 #Accomodate 32 bit data?
        self.channel_brightness[channel_index]['max'] = max_val / 65535
        self.update_display(preserve_zoom = (current_xlim, current_ylim))



    
    def update_display(self, preserve_zoom=None, dims = None, called = False, reset_resize = False, begin_paint = False):
        """Update the display with currently visible channels and highlight overlay."""

        try:

            if begin_paint:
                # Store/update the static background with current zoom level
                self.static_background = self.canvas.copy_from_bbox(self.ax.bbox)


            self.figure.clear()

            # Get active channels and their dimensions
            active_channels = [i for i in range(4) if self.channel_data[i] is not None]
            if dims is None:
                if active_channels:
                    dims = [(self.channel_data[i].shape[1:3] if len(self.channel_data[i].shape) >= 3 else 
                            self.channel_data[i].shape) for i in active_channels]
                    min_height = min(d[0] for d in dims)
                    min_width = min(d[1] for d in dims)
                else:
                    min_height = 1
                    min_width = 1
            else:
                min_height = dims[0]
                min_width = dims[1]

            # Set axes limits before displaying any images
            self.ax.set_xlim(-0.5, min_width - 0.5)
            self.ax.set_ylim(min_height - 0.5, -0.5)
            
            # Create subplot with tight layout and white figure background
            self.figure.patch.set_facecolor('white')
            self.ax = self.figure.add_subplot(111)
            
            # Store current zoom limits if they exist and weren't provided
     
            current_xlim, current_ylim = preserve_zoom if preserve_zoom else (None, None)
            
            # Define base colors for each channel with increased intensity
            base_colors = self.base_colors
            # Set only the axes (image area) background to black
            self.ax.set_facecolor('black')
            
            # Display each visible channel
            for channel in range(4):
                if (self.channel_visible[channel] and 
                    self.channel_data[channel] is not None):
                    
                    # Check if we're dealing with RGB data
                    is_rgb = len(self.channel_data[channel].shape) == 4 and (self.channel_data[channel].shape[-1] == 3 or self.channel_data[channel].shape[-1] == 4)
                    
                    if len(self.channel_data[channel].shape) == 3 and not is_rgb:
                        current_image = self.channel_data[channel][self.current_slice, :, :]
                    elif is_rgb:
                        current_image = self.channel_data[channel][self.current_slice]  # Already has RGB channels
                    else:
                        current_image = self.channel_data[channel]

                    if is_rgb and self.channel_data[channel].shape[-1] == 3:
                        # For RGB images, just display directly without colormap
                        self.ax.imshow(current_image,
                                     alpha=0.7)
                    elif is_rgb and self.channel_data[channel].shape[-1] == 4:
                        self.ax.imshow(current_image) #For images that already have an alpha value and RGB, don't update alpha

                    else:
                        # Regular channel processing with colormap
                        # Calculate brightness/contrast limits from entire volume
                        if self.min_max[channel][0] == None:
                            self.min_max[channel][0] = np.min(channel)
                        if self.min_max[channel][1] == None:
                            self.min_max[channel][1] = np.max(channel)

                        img_min = self.min_max[channel][0]
                        img_max = self.min_max[channel][1]
                        
                        # Calculate vmin and vmax, ensuring we don't get a zero range
                        if img_min == img_max:
                            vmin = img_min
                            vmax = img_min + 1
                        else:
                            vmin = img_min + (img_max - img_min) * self.channel_brightness[channel]['min']
                            vmax = img_min + (img_max - img_min) * self.channel_brightness[channel]['max']



                        
                        # Normalize the image safely
                        if vmin == vmax:
                            normalized_image = np.zeros_like(current_image)
                        else:
                            normalized_image = np.clip((current_image - vmin) / (vmax - vmin), 0, 1)
                        
                        if channel == 2 and self.machine_window is not None:
                            custom_cmap = LinearSegmentedColormap.from_list(
                                f'custom_{channel}',
                                [(0, 0, 0, 0),          # transparent for 0
                                 (0.5, 1, 0.5, 1),      # light green for 1
                                 (1, 0.5, 0.5, 1)]      # light red for 2
                            )
                            self.ax.imshow(current_image,
                                         cmap=custom_cmap,
                                         vmin=0,
                                         vmax=2,
                                         alpha=0.7,
                                         interpolation='nearest',
                                         extent=(-0.5, min_width-0.5, min_height-0.5, -0.5))
                        else:
                            # Create custom colormap with higher intensity
                            color = base_colors[channel]
                            custom_cmap = LinearSegmentedColormap.from_list(
                                f'custom_{channel}',
                                [(0,0,0,0), (*color,1)]
                            )
                            
                            # Display the image with slightly higher alpha
                            self.ax.imshow(normalized_image,
                                         alpha=0.7,
                                         cmap=custom_cmap,
                                         vmin=0,
                                         vmax=1,
                                         extent=(-0.5, min_width-0.5, min_height-0.5, -0.5))

            if self.preview and not called:
                self.create_highlight_overlay_slice(self.targs, bounds = self.bounds)

            # Add highlight overlay if it exists
            if self.mini_overlay and self.highlight and self.machine_window is None:
                highlight_cmap = LinearSegmentedColormap.from_list(
                    'highlight',
                    [(0, 0, 0, 0), (1, 1, 0, 1)]  # yellow
                )
                self.ax.imshow(self.mini_overlay_data,
                             cmap=highlight_cmap,
                             alpha=0.5)
            elif self.highlight_overlay is not None and self.highlight and self.machine_window is None:
                highlight_slice = self.highlight_overlay[self.current_slice]
                highlight_cmap = LinearSegmentedColormap.from_list(
                    'highlight',
                    [(0, 0, 0, 0), (1, 1, 0, 1)]  # yellow
                )
                self.ax.imshow(highlight_slice,
                             cmap=highlight_cmap,
                             alpha=0.5)
            elif self.highlight_overlay is not None and self.highlight:
                highlight_slice = self.highlight_overlay[self.current_slice]
                highlight_cmap = LinearSegmentedColormap.from_list(
                    'highlight',
                    [(0, 0, 0, 0),          # transparent for 0
                     (1, 1, 0, 1),          # bright yellow for 1
                     (0, 0.7, 1, 1)]        # cool blue for 2
                )
                self.ax.imshow(highlight_slice,
                             cmap=highlight_cmap,
                             vmin=0,
                             vmax=2,         # Important: set vmax to 2 to accommodate both values
                             alpha=0.5)



            
            # Style the axes
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_title(f'Slice {self.current_slice}')

            # Make axis labels and ticks black for visibility against white background
            self.ax.xaxis.label.set_color('black')
            self.ax.yaxis.label.set_color('black')
            self.ax.title.set_color('black')
            self.ax.tick_params(colors='black')
            for spine in self.ax.spines.values():
                spine.set_color('black')

            # Adjust the layout to ensure the plot fits well in the figure
            self.figure.tight_layout()

            # Redraw measurement points and their labels
            for point in self.measurement_points:
                x1, y1, z1 = point['point1']
                x2, y2, z2 = point['point2']
                pair_idx = point['pair_index']
                
                # Draw points and labels if they're on current slice
                if z1 == self.current_slice:
                    self.ax.plot(x1, y1, 'yo', markersize=8)
                    self.ax.text(x1, y1+5, str(pair_idx), 
                                color='white', ha='center', va='bottom')
                if z2 == self.current_slice:
                    self.ax.plot(x2, y2, 'yo', markersize=8)
                    self.ax.text(x2, y2+5, str(pair_idx), 
                                color='white', ha='center', va='bottom')
                    
                # Draw line if both points are on current slice
                if z1 == z2 == self.current_slice:
                    self.ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.5)
        
            if active_channels:
                self.ax.set_xlim(-0.5, min_width - 0.5)
                self.ax.set_ylim(min_height - 0.5, -0.5)

            if self.resizing:
                self.original_xlim = self.ax.get_xlim()
                self.original_ylim = self.ax.get_ylim()
            # Restore zoom limits if they existed
            if current_xlim is not None and current_ylim is not None:
                self.ax.set_xlim(current_xlim)
                self.ax.set_ylim(current_ylim)
            if reset_resize:
                self.resizing = False
            
            self.canvas.draw()

        except:
            import traceback
            print(traceback.format_exc())

    def update_display_slice_optimized(self, channel, preserve_zoom=None):
        """Ultra minimal update that only changes the paint channel's data - OPTIMIZED VERSION"""
        if not self.channel_visible[channel]:
            return
            
        if preserve_zoom:
            current_xlim, current_ylim = preserve_zoom
            if current_xlim is not None and current_ylim is not None:
                self.ax.set_xlim(current_xlim)
                self.ax.set_ylim(current_ylim)
        
        # Find the existing image for channel (paint channel)
        channel_image = None
        for img in self.ax.images:
            if img.cmap.name == f'custom_{channel}':
                channel_image = img
                break
                
        if channel_image is not None:
            # Update the data of the existing image with thread safety
            with self.paint_lock:
                channel_image.set_array(self.channel_data[channel][self.current_slice])
            
            # Restore the static background (all other channels) at current zoom level
            # This is the key - use static_background from update_display, not paint_background
            if hasattr(self, 'static_background') and self.static_background is not None:
                self.canvas.restore_region(self.static_background)
                # Draw just our paint channel
                self.ax.draw_artist(channel_image)
                # Blit everything
                self.canvas.blit(self.ax.bbox)
                self.canvas.flush_events()
            else:
                # Fallback to full draw if no static background
                self.canvas.draw()
        else:
            # Fallback if channel image not found
            self.canvas.draw()

    def get_channel_image(self, channel):
        """Find the matplotlib image object for a specific channel."""
        if not hasattr(self.ax, 'images'):
            return None
            
        for img in self.ax.images:
            if hasattr(img, 'cmap') and hasattr(img.cmap, 'name'):
                if img.cmap.name == f'custom_{channel}':
                    return img
        return None

    def show_netshow_dialog(self):
        dialog = NetShowDialog(self)
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
            stats['num_edges'] = my_network.network.number_of_edges()
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

    def show_radial_dialog(self):
        dialog = RadialDialog(self)
        dialog.exec()

    def show_degree_dist_dialog(self):
        dialog = DegreeDistDialog(self)
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

    def show_interaction_dialog(self):
        dialog = InteractionDialog(self)
        dialog.exec()

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
    def __init__(self, parent=None, is_top_table=False):
        super().__init__(parent)
        self.search_widget = SearchWidget(self)
        self.search_widget.hide()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.parent = parent  # Store reference to parent window
        self.is_top_table = is_top_table  # Flag to distinguish top tables
        
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
            if self in self.parent.data_table:  # Top table
                save_menu = context_menu.addMenu("Save As")
                save_csv = save_menu.addAction("CSV")
                save_excel = save_menu.addAction("Excel")
                
                # Connect the actions
                save_csv.triggered.connect(lambda: self.save_table_as('csv'))
                save_excel.triggered.connect(lambda: self.save_table_as('xlsx'))
            else:  # Bottom tables
                # Add Find action
                find_menu = context_menu.addMenu("Find")
                find_action = find_menu.addAction("Find Node/Edge")
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
                
                # Connect the actions - ensure we're saving the active table
                save_csv.triggered.connect(lambda: self.parent.active_table.save_table_as('csv'))
                save_excel.triggered.connect(lambda: self.parent.active_table.save_table_as('xlsx'))
                save_gephi.triggered.connect(lambda: self.parent.active_table.save_table_as('gexf'))
                save_graphml.triggered.connect(lambda: self.parent.active_table.save_table_as('graphml'))
                save_pajek.triggered.connect(lambda: self.parent.active_table.save_table_as('net'))


                if self == self.parent.selection_table:
                    set_action = context_menu.addAction("Swap with network table (also sets internal network properties - may affect related functions)")
                    set_action.triggered.connect(self.set_selection_to_active)
            
            # Show the menu at cursor position
            cursor_pos = QCursor.pos()
            context_menu.exec(cursor_pos)

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
            QMessageBox.critical(
                self,
                "Error",
                f"Error sorting table: {str(e)}"
            )

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

                model = PandasModel(my_network.network_lists)
                self.parent.network_table.setModel(model)
                # Adjust column widths to content
                for column in range(model.columnCount(None)):
                    self.parent.network_table.resizeColumnToContents(column)

                #move old model to selection
                new_lists = [list(old.iloc[:, 0]), list(old.iloc[:, 1]), list(old.iloc[:, 2])]
                model = PandasModel(new_lists)
                self.parent.selection_table.setModel(model)
                for column in range(model.columnCount(None)):
                    self.parent.selection_table.resizeColumnToContents(column)

        except Exception as e:
            print(f"Error setting new network: {e}")



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
            data = pd.DataFrame(columns=['Node 1A', 'Node 1B', 'Edge 1C'])
        elif type(data) == list:
            data = self.lists_to_dataframe(data[0], data[1], data[2], column_names=['Node 1A', 'Node 1B', 'Edge 1C'])
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
        self.setModal(True)
        
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

        # Add Run button
        run_button = QPushButton("Enter")
        run_button.clicked.connect(self.run_properties)
        layout.addWidget(run_button)

        report_button = QPushButton("Report Properties (Show in Top Right Tables)")
        report_button.clicked.connect(self.report)
        layout.addWidget(report_button)

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

            self.parent().reset(nodes = nodes, edges = edges, search_region = search_region, network_overlay = network_overlay, id_overlay = id_overlay, network = network, xy_scale = xy_scale, z_scale = z_scale)
            
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
            
            # Connect signals
            slider.valueChanged.connect(lambda values, ch=i: self.on_slider_change(ch, values))
            min_input.editingFinished.connect(lambda ch=i: self.on_min_input_change(ch))
            max_input.editingFinished.connect(lambda ch=i: self.on_max_input_change(ch))
            
    def on_slider_change(self, channel, values):
        """Update text inputs when slider changes"""
        min_val, max_val = values
        self.min_inputs[channel].setText(str(min_val))
        self.max_inputs[channel].setText(str(max_val))
        self.parent().update_brightness(channel, values)
        
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
        self.parent().update_display()
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
        self.setWindowTitle("Merging Node Identities From Folder Dialog.\nNote that you should prelabel or prewatershed your current node objects before doing this. (See Process -> Image) It does not label them for you.")
        self.setModal(True)
        
        layout = QFormLayout(self)

        self.search = QLineEdit("")
        layout.addRow("Step-out distance (from current nodes image - ignore if you dilated them previously or don't want):", self.search)

        self.xy_scale = QLineEdit(f"{my_network.xy_scale}")
        layout.addRow("xy_scale:", self.xy_scale)

        self.z_scale = QLineEdit(f"{my_network.z_scale}")
        layout.addRow("z_scale:", self.z_scale)

        # Add Run button
        run_button = QPushButton("Get Directory")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

    def run(self):

        try:

            search = float(self.search.text()) if self.search.text().strip() else 0
            xy_scale = float(self.xy_scale.text()) if self.xy_scale.text().strip() else 1
            z_scale = float(self.z_scale.text()) if self.z_scale.text().strip() else 1


            data = self.parent().channel_data[0]

            if data is None:
                return



            dialog = QFileDialog(self)
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
            dialog.setOption(QFileDialog.Option.ReadOnly)
            dialog.setFileMode(QFileDialog.FileMode.Directory)
            dialog.setViewMode(QFileDialog.ViewMode.Detail)

            if dialog.exec() == QFileDialog.DialogCode.Accepted:
                selected_path = dialog.directory().absolutePath()

            if search > 0:
                data = sdl.smart_dilate(data, 1, 1, GPU = False, fast_dil = False, use_dt_dil_amount = search, xy_scale = xy_scale, z_scale = z_scale)

            my_network.merge_node_ids(selected_path, data)

            self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity')

            QMessageBox.critical(
                self,
                "Success",
                "Node Identities Merged. New IDs represent presence of corresponding img foreground with +, absence with -. Please save your new identities as csv, then use File -> Load -> Load From Excel Helper to bulk search and rename desired combinations. (Press Help [above] for more info)"
            )

            self.accept()

        except Exception as e:
            print(f"Error: {e}")


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
        layout.addRow("Use cubic downsample (Slower but preserves shape better potentially)?", self.cubic)

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

            if self.parent().highlight_overlay is not None or self.parent().mini_overlay_data is not None:
                if self.parent().mini_overlay == True:
                    self.parent().create_highlight_overlay(node_indices = self.parent().clicked_values['nodes'], edge_indices = self.parent().clicked_values['edges'])
                arrays_3d.append(self.parent().highlight_overlay)
                colors.append(color_template[4])
        
            n3d.show_3d(arrays_3d, arrays_4d, down_factor = downsample, order = order, xy_scale = my_network.xy_scale, z_scale = my_network.z_scale, colors = colors, box = box)
            
            self.accept()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            print(traceback.format_exc())


class NetOverlayDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Generate Network Overlay?")
        self.setModal(True)

        layout = QFormLayout(self)

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

            my_network.network_overlay = my_network.draw_network()

            self.parent().load_channel(2, channel_data = my_network.network_overlay, data = True)

            self.accept()

        except Exception as e:

            print(f"Error with Overlay Generation: {e}")

class SearchOverlayDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Generate Search Region Overlay?")
        self.setModal(True)

        layout = QFormLayout(self)

        # Add Run button
        run_button = QPushButton("Generate (Will go to Overlay 2)")
        run_button.clicked.connect(self.searchoverlay)
        layout.addWidget(run_button)

    def searchoverlay(self):

        try:

            my_network.id_overlay = my_network.search_region

            self.parent().load_channel(3, channel_data = my_network.search_region, data = True)

            self.accept()

        except Exception as e:

            print(f"Error with Overlay Generation: {e}")

class IdOverlayDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Generate ID Overlay?")
        self.setModal(True)

        layout = QFormLayout(self)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # Add Run button
        run_button = QPushButton("Generate (Will go to Overlay 2)")
        run_button.clicked.connect(self.idoverlay)
        layout.addWidget(run_button)

    def idoverlay(self):

        accepted_mode = self.mode_selector.currentIndex()

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

            my_network.id_overlay = my_network.draw_node_indices()

        elif accepted_mode == 1:

            my_network.id_overlay = my_network.draw_edge_indices()


        self.parent().load_channel(3, channel_data = my_network.id_overlay, data = True)

        self.accept()

class ColorOverlayDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Generate Node (or Edge) -> Color Overlay?")
        self.setModal(True)

        layout = QFormLayout(self)

        self.down_factor = QLineEdit("")
        layout.addRow("down_factor (for speeding up overlay generation - optional):", self.down_factor)

        # Add Run button
        run_button = QPushButton("Generate (Will go to Overlay 2)")
        run_button.clicked.connect(self.coloroverlay)
        layout.addWidget(run_button)

    def coloroverlay(self):

        down_factor = float(self.down_factor.text()) if self.down_factor.text().strip() else None

        if self.parent().active_channel == 0:
            mode = 0
            self.sort = 'Node'
        else:
            mode = 1
            self.sort = 'Edge'


        result, legend = my_network.node_to_color(down_factor = down_factor, mode = mode)

        self.parent().format_for_upperright_table(legend, f'{self.sort} Id', f'Encoding Val: {self.sort}', 'Legend')


        self.parent().load_channel(3, channel_data = result, data = True)

        self.accept()


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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        # geo checkbox (default True)
        self.geo_layout = QPushButton("geo_layout")
        self.geo_layout.setCheckable(True)
        self.geo_layout.setChecked(False)
        layout.addRow("Use Geographic Layout:", self.geo_layout)
        
        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Default", "Community Coded", "Node ID Coded"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # weighted checkbox (default True)
        self.weighted = QPushButton("weighted")
        self.weighted.setCheckable(True)
        self.weighted.setChecked(True)
        layout.addRow("Use Weighted Network (Only for community graphs):", self.weighted)

        # Optional saving:
        self.directory = QLineEdit()
        self.directory.setPlaceholderText("Does not save when empty")
        layout.addRow("Output Directory:", self.directory)
        
        # Add Run button
        run_button = QPushButton("Show Network")
        run_button.clicked.connect(self.show_network)
        layout.addWidget(run_button)
    
    def show_network(self):
        # Get parameters and run analysis
        if my_network.communities is None:
            self.parent().show_partition_dialog()
            if my_network.communities is None:
                return
        geo = self.geo_layout.isChecked()
        if geo:
            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()
        accepted_mode = self.mode_selector.currentIndex()  # Convert to 1-based index
        # Get directory (None if empty)
        directory = self.directory.text() if self.directory.text() else None

        weighted = self.weighted.isChecked()

        try:
            if accepted_mode == 0:
                my_network.show_network(geometric=geo, directory = directory)
            elif accepted_mode == 1:
                my_network.show_communities_flex(geometric=geo, directory = directory, weighted = weighted, partition = my_network.communities)
                self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'CommunityID')
            elif accepted_mode == 2:
                my_network.show_identity_network(geometric=geo, directory = directory)
            
            self.accept()
        except Exception as e:
            print(f"Error showing network: {e}")
            import traceback
            print(traceback.format_exc())

class PartitionDialog(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Partition Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        # weighted checkbox (default True)
        self.weighted = QPushButton("weighted")
        self.weighted.setCheckable(True)
        self.weighted.setChecked(True)
        layout.addRow("Use Weighted Network:", self.weighted)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Label Propogation", "Louvain"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # stats checkbox (default True)
        self.stats = QPushButton("Stats")
        self.stats.setCheckable(True)
        self.stats.setChecked(True)
        layout.addRow("Community Stats:", self.stats)

        self.seed = QLineEdit("")
        layout.addRow("Seed (int):", self.seed)

        # Add Run button
        run_button = QPushButton("Partition")
        run_button.clicked.connect(self.partition)
        layout.addWidget(run_button)

    def partition(self):

        accepted_mode = self.mode_selector.currentIndex()
        weighted = self.weighted.isChecked()
        dostats = self.stats.isChecked()

        try:
            seed = int(self.seed.text()) if self.seed.text() else 42
        except:
            seed = None


        my_network.communities = None

        try:
            stats = my_network.community_partition(weighted = weighted, style = accepted_mode, dostats = dostats, seed = seed)
            print(f"Discovered communities: {my_network.communities}")

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

        # umap checkbox (default True)
        self.umap = QPushButton("UMAP")
        self.umap.setCheckable(True)
        self.umap.setChecked(True)
        layout.addRow("Generate UMAP?:", self.umap)

        # weighted checkbox (default True)
        self.label = QPushButton("Label")
        self.label.setCheckable(True)
        self.label.setChecked(False)
        layout.addRow("If using above - label UMAP points?:", self.label)


        # Add Run button
        run_button = QPushButton("Get Community ID Info")
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

            umap = self.umap.isChecked()
            label = self.label.isChecked()

            if mode == 1:

                info = my_network.community_id_info()

                self.parent().format_for_upperright_table(info, 'Node Identity Type', 'Weighted Proportion in Communities', 'Weighted Average of Community Makeup')

            else:

                info, names = my_network.community_id_info_per_com(umap = umap, label = label)

                self.parent().format_for_upperright_table(info, 'Community', names, 'Average of Community Makeup')

            self.accept()

        except Exception as e:

            print(f"Error: {e}")



class ComNeighborDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Reassign Communities Based on Identity Similarity?")
        self.setModal(True)

        layout = QFormLayout(self)

        self.neighborcount = QLineEdit("5")
        layout.addRow("Num Neighborhoods:", self.neighborcount)

        self.seed = QLineEdit("")
        layout.addRow("Clustering Seed:", self.seed)

        self.limit = QLineEdit("")
        layout.addRow("Min Community Size to be grouped (Smaller communities will be placed in neighborhood 0 - does not apply if empty)", self.limit)

        # Add Run button
        run_button = QPushButton("Get Communities")
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

            seed = float(self.seed.text()) if self.seed.text().strip() else 42

            limit = int(self.limit.text()) if self.limit.text().strip() else None


            neighborcount = int(self.neighborcount.text()) if self.neighborcount.text().strip() else 5

            if self.parent().prev_coms is None:

                self.parent().prev_coms = copy.deepcopy(my_network.communities)
                my_network.assign_neighborhoods(seed, neighborcount, limit = limit)
            else:
                my_network.assign_neighborhoods(seed, neighborcount, limit = limit, prev_coms = self.parent().prev_coms)

            self.parent().format_for_upperright_table(my_network.communities, 'NodeID', 'NeighborhoodID', title = 'Neighborhood Partition')

            print("Neighborhoods have been assigned to communities based on similarity")

            self.accept()

        except Exception as e:

            print(f"Error assigning neighborhoods: {e}")

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
        run_button = QPushButton("Get Neighborhoods (Note this overwrites current communities - save your coms first)")
        run_button.clicked.connect(self.run)
        layout.addWidget(run_button)

    def run(self):

        try:

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







class RadialDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Radial Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        self.distance = QLineEdit("50")
        layout.addRow("Bucket Distance for Searching For Node Neighbors (automatically scaled by xy and z scales):", self.distance)

        self.directory = QLineEdit("")
        layout.addRow("Output Directory:", self.directory)

        # Add Run button
        run_button = QPushButton("Get Radial Distribution")
        run_button.clicked.connect(self.radial)
        layout.addWidget(run_button)

    def radial(self):

        try:

            distance = float(self.distance.text()) if self.distance.text().strip() else 50

            directory = str(self.distance.text()) if self.directory.text().strip() else None

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()

            radial = my_network.radial_distribution(distance, directory = directory)

            self.parent().format_for_upperright_table(radial, 'Radial Distance From Any Node', 'Average Number of Neighboring Nodes', title = 'Radial Distribution Analysis')

            self.accept()

        except Exception as e:
            print(f"An error occurred: {e}")

class DegreeDistDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle("Degree Distribution Parameters")
        self.setModal(True)

        layout = QFormLayout(self)

        self.directory = QLineEdit("")
        layout.addRow("Output Directory:", self.directory)

        # Add Run button
        run_button = QPushButton("Get Degree Distribution")
        run_button.clicked.connect(self.degreedist)
        layout.addWidget(run_button)

    def degreedist(self):

        try:

            directory = str(self.distance.text()) if self.directory.text().strip() else None

            degrees = my_network.degree_distribution(directory = directory)


            self.parent().format_for_upperright_table(degrees, 'Degree (k)', 'Proportion of nodes with degree (p(k))', title = 'Degree Distribution Analysis')

            self.accept()

        except Exception as e:
            print(f"An error occurred: {e}")

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
            self.root.addItems(list(set(my_network.node_identities.values())))  
            self.root.setCurrentIndex(0)
            identities_layout.addRow("Root Identity to Search for Neighbor's IDs?", self.root)
            
            self.targ = QComboBox()
            neighs = list(set(my_network.node_identities.values()))
            neighs.append("All Others (Excluding Self)")
            self.targ.addItems(neighs)  
            self.targ.setCurrentIndex(0)
            identities_layout.addRow("Neighbor Identities to Search For?", self.targ)
        else:
            self.root = None
            self.targ = None

        self.num = QLineEdit("1")
        identities_layout.addRow("Number of Nearest Neighbors to Evaluate Per Node?:", self.num)

        
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
        
        main_layout.addWidget(heatmap_group)
        
        # Get Distribution group box
        distribution_group = QGroupBox("Get Distribution")
        distribution_layout = QVBoxLayout(distribution_group)
        
        run_button = QPushButton("Get Average Nearest Neighbor (Plus Distribution)")
        run_button.clicked.connect(self.run)
        distribution_layout.addWidget(run_button)
        
        main_layout.addWidget(distribution_group)
        
        # Get All Averages group box (only if node_identities exists)
        if my_network.node_identities is not None:
            averages_group = QGroupBox("Get All Averages")
            averages_layout = QVBoxLayout(averages_group)
            
            run_button2 = QPushButton("Get Average Nearest All ID Combinations (No Distribution, No Heatmap)")
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

            heatmap = self.map.isChecked()
            threed = self.threed.isChecked()
            numpy = self.numpy.isChecked()
            num = int(self.num.text()) if self.num.text().strip() else 1

            if root is not None and targ is not None:
                title = f"Nearest {num} Neighbor(s) Distance of {targ} from {root}"
                header = f"Shortest Distance to Closest {num} {targ}(s)"
                header2 = f"{root} Node ID"
            else:
                title = f"Nearest {num} Neighbor(s) Distance Between Nodes"
                header = f"Shortest Distance to Closest {num} Nodes"
                header2 = "Root Node ID"

            if my_network.node_centroids is None:
                self.parent().show_centroid_dialog()
                if my_network.node_centroids is None:
                    return

            if not numpy:
                avg, output = my_network.nearest_neighbors_avg(root, targ, my_network.xy_scale, my_network.z_scale, num = num, heatmap = heatmap, threed = threed)
            else:
                avg, output, overlay = my_network.nearest_neighbors_avg(root, targ, my_network.xy_scale, my_network.z_scale, num = num, heatmap = heatmap, threed = threed, numpy = True)
                self.parent().load_channel(3, overlay, data = True)

            self.parent().format_for_upperright_table([avg], metric = f'Avg {title}', title = f'Avg {title}')
            self.parent().format_for_upperright_table(output, header2, header, title = title)

            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())

            print(f"Error: {e}")

    def run2(self):

        try:

            available = list(set(my_network.node_identities.values()))

            num = int(self.num.text()) if self.num.text().strip() else 1

            output_dict = {}

            while len(available) > 1:

                root = available[0]

                for targ in available:

                    avg, _ = my_network.nearest_neighbors_avg(root, targ, my_network.xy_scale, my_network.z_scale, num = num)

                    output_dict[f"{root} vs {targ}"] = avg

                del available[0]

            self.parent().format_for_upperright_table(output_dict, "ID Combo", "Avg Distance to Nearest", title = "Average Distance to Nearest Neighbors for All ID Combos")

            self.accept()

        except Exception as e:

            print(f"Error: {e}")


class NeighborIdentityDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle(f"Neighborhood Identity Distribution Parameters \n(Note - the same node is not included more than once as a neighbor even if it borders multiple nodes of the root ID)")
        self.setModal(True)

        layout = QFormLayout(self)

        if my_network.node_identities is not None:
            self.root = QComboBox()
            self.root.addItems(list(set(my_network.node_identities.values())))  
            self.root.setCurrentIndex(0)
            layout.addRow("Root Identity to Search for Neighbor's IDs (search uses nodes of this ID, finds what IDs they connect to", self.root)
        else:
            self.root = None

        self.directory = QLineEdit("")
        layout.addRow("Output Directory:", self.directory)

        self.mode = QComboBox()
        self.mode.addItems(["From Network - Based on Absolute Connectivity", "Use Labeled Nodes - Based on Morphological Neighborhood Densities"])
        self.mode.setCurrentIndex(0)
        layout.addRow("Mode", self.mode)

        self.search = QLineEdit("")
        layout.addRow("Search Radius (Ignore if using network):", self.search)

        self.fastdil = QPushButton("Fast Dilate")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(False)
        layout.addRow("(If not using network) Use Fast Dilation (Higher speed, less accurate with search regions much larger than nodes):", self.fastdil)

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

            directory = self.directory.text() if self.directory.text().strip() else None

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
            print(f"Error: {e}")




class RipleyDialog(QDialog):

    def __init__(self, parent=None):

        super().__init__(parent)
        self.setWindowTitle(f"Find Ripley's H Function From Centroids")
        self.setModal(True)

        layout = QFormLayout(self)

        if my_network.node_identities is not None:
            self.root = QComboBox()
            self.root.addItems(list(set(my_network.node_identities.values())))  
            self.root.setCurrentIndex(0)
            layout.addRow("Root Identity to Search for Neighbors", self.root)
        else:
            self.root = None

        if my_network.node_identities is not None:
            self.targ = QComboBox()
            self.targ.addItems(list(set(my_network.node_identities.values())))  
            self.targ.setCurrentIndex(0)
            layout.addRow("Targ Identity to be Searched For", self.targ)
        else:
            self.targ = None

        self.distance = QLineEdit("5")
        layout.addRow("Bucket Distance for Searching For Clusters (automatically scaled by xy and z scales):", self.distance)


        self.proportion = QLineEdit("0.5")
        layout.addRow("Proportion of image to search? (0-1, high vals increase border artifacts): ", self.proportion)

        self.edgecorrect = QPushButton("Border Correction")
        self.edgecorrect.setCheckable(True)
        self.edgecorrect.setChecked(False)
        layout.addRow("Use Border Correction (Extrapolate for points beyond the border):", self.edgecorrect)

        self.ignore = QPushButton("Ignore Border Roots")
        self.ignore.setCheckable(True)
        self.ignore.setChecked(False)
        layout.addRow("Exclude Root Nodes Near Borders?:", self.ignore)

        # Add Run button
        run_button = QPushButton("Get Ripley's H")
        run_button.clicked.connect(self.ripley)
        layout.addWidget(run_button)

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

            if proportion > 1 or proportion <= 0:
                print("Utilizing proportion = 0.5")
                proportion = 0.5


            edgecorrect = self.edgecorrect.isChecked()

            ignore = self.ignore.isChecked()

            if my_network.nodes is not None:

                if my_network.nodes.shape[0] == 1:
                    bounds = (np.array([0, 0]), np.array([my_network.nodes.shape[2], my_network.nodes.shape[1]]))
                else:
                    bounds = (np.array([0, 0, 0]), np.array([my_network.nodes.shape[2], my_network.nodes.shape[1], my_network.nodes.shape[0]]))
            else:
                bounds = None

            r_vals, k_vals, h_vals = my_network.get_ripley(root, targ, distance, edgecorrect, bounds, ignore, proportion)
            
            k_dict = dict(zip(r_vals, k_vals))
            h_dict = dict(zip(r_vals, h_vals))


            self.parent().format_for_upperright_table(k_dict, metric='Radius (scaled)', value='L Value', title="Ripley's K")
            self.parent().format_for_upperright_table(h_dict, metric='Radius (scaled)', value='L Normed', title="Ripley's H")


            self.accept()

        except Exception as e:
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
                self.parent().load_channel(3, overlay, data = True)


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
        new_model = PandasModel(df)
        self.parent().selection_table.setModel(new_model)
        
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
                max_shape = tuple(max_coord + 1 for max_coord in max_coords)
                min_coords = centroid_points.min(axis=0)
                min_shape = tuple(min_coord + 1 for min_coord in min_coords)
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
        layout.addRow("Use GPU:", self.GPU)


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

        self.fastdil = QPushButton("Fast Dilate")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(False)
        layout.addRow("Use Fast Dilation (Higher speed, less accurate with search regions much larger than nodes):", self.fastdil)

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

            result = my_network.interactions(search = node_search, cores = accepted_mode, fastdil = fastdil)

            self.parent().format_for_upperright_table(result, 'Node ID', ['Volume of Nearby Edge (Scaled)', 'Volume of Search Region'], title = 'Node/Edge Interactions')

            self.accept()

        except Exception as e:

            import traceback
            print(traceback.format_exc())

            print(f"Error finding interactions: {e}")


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
                self.parent().load_channel(3, channel_data = nodes, data = True)


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

                self.parent().load_channel(3, channel_data = img, data = True)


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
                self.parent().load_channel(2, channel_data = result, data = True)

            degree_dict = {}

            for node in G.nodes():
                degree_dict[node] = my_network.network.degree(node)

            self.parent().format_for_upperright_table(degree_dict, 'Mother ID', 'Degree', title = 'Mother Nodes')


            self.accept()

        except Exception as e:

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


            self.parent().load_channel(3, image, True)
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
        self.cubic = QPushButton("Use Cubic Resize? (Will alter labels and require re-binarization -> labelling, but preserves shape better)")
        self.cubic.setCheckable(True)
        self.cubic.setChecked(False)
        layout.addRow("Use cubic algorithm:", self.cubic)
        
        if self.parent().original_shape is not None:
            undo_button = QPushButton(f"Resample to original shape: {self.parent().original_shape}")
            undo_button.clicked.connect(lambda: self.run_resize(undo = True))
            layout.addRow(undo_button)

        if my_network.xy_scale != my_network.z_scale:
            norm_button_upsize = QPushButton(f"Normalize Scaling with Upsample")
            norm_button_upsize.clicked.connect(lambda: self.run_resize(upsize = True, special = True))
            layout.addRow(norm_button_upsize)

            norm_button_downsize = QPushButton("Normalize Scaling with Downsample")
            norm_button_downsize.clicked.connect(lambda: self.run_resize(upsize = False, special = True))
            layout.addRow(norm_button_downsize)

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
                
            if any(dim < 1 for dim in new_shape):
                QMessageBox.critical(self, "Error", f"Resize would result in invalid dimensions: {new_shape}")
                self.reset_fields()
                return

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
                        self.parent().load_channel(channel, channel_data=resized_data, data=True, assign_shape = False)


                
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
                        self.parent().load_channel(channel, channel_data=resized_data, data=True, assign_shape = False)

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

            try:
                if my_network.node_centroids is not None:
                    centroids = copy.deepcopy(my_network.node_centroids)
                    if isinstance(resize, (int, float)):
                        for item in my_network.node_centroids:
                            centroids[item] = np.round((my_network.node_centroids[item]) * resize)
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
            self.reset_fields()
            self.parent().resizing = False
            self.accept()
            
        except Exception as e:
            print(f"Error during resize operation: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to resize: {str(e)}")


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

            try:
                # Call binarize method with parameters
                result = n3d.binarize(
                    active_data
                    )

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
                self.parent().channel_data[self.parent().active_channel] = result


                # Update the corresponding property in my_network
                setattr(my_network, network_properties[self.parent().active_channel], result)

                self.parent().update_display()
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
        self.setWindowTitle("Smart Label (Use label array to assign label neighborhoods to binary array)?")
        self.setModal(True)
        
        layout = QFormLayout(self)


        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Prelabeled Array:", self.mode_selector)

        layout.addRow(QLabel("Will Label Neighborhoods in: "))

        # Add mode selection dropdown
        self.target_selector = QComboBox()
        self.target_selector.addItems(["Nodes", "Edges", "Overlay 1", "Overlay 2"])
        self.target_selector.setCurrentIndex(1)  # Default to Mode 1
        layout.addRow("Binary Array:", self.target_selector)

        # GPU checkbox (default True)
        self.GPU = QPushButton("GPU")
        self.GPU.setCheckable(True)
        self.GPU.setChecked(False)
        layout.addRow("Use GPU:", self.GPU)

        self.down_factor = QLineEdit("")
        layout.addRow("Internal Downsample for GPU (if needed):", self.down_factor)

       # Add Run button
        run_button = QPushButton("Run Smart Label")
        run_button.clicked.connect(self.run_slabel)
        layout.addRow(run_button)

    def run_slabel(self):

        try:

            accepted_source = self.mode_selector.currentIndex()
            accepted_target = self.target_selector.currentIndex()
            GPU = self.GPU.isChecked()


            if accepted_source == accepted_target:
                return

            binary_array = self.parent().channel_data[accepted_target]

            label_array = self.parent().channel_data[accepted_source]

            down_factor = float(self.down_factor.text()) if self.down_factor.text().strip() else None


            try:

                # Update both the display data and the network object
                binary_array = sdl.smart_label(binary_array, label_array, directory = None, GPU = GPU, predownsample = down_factor, remove_template = True)

                label_array = sdl.invert_array(label_array)

                binary_array = binary_array * label_array

                self.parent().load_channel(accepted_target, binary_array, True)

                self.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error running smart label: {str(e)}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running smart label: {str(e)}"
            )


class ThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Threshold Mode")
        self.setModal(True)
        
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

        # Add ML button
        #ML2 = QPushButton("Machine Learning (GPU)")
        #ML2.clicked.connect(lambda: self.start_ml(GPU = True))
        #layout.addRow(ML2)


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

            if self.parent().mini_overlay_data is not None:
                self.parent().mini_overlay_data = None

            thresh_window = ThresholdWindow(self.parent(), accepted_mode)
            thresh_window.show()  # Non-modal window
            self.highlight_overlay = None
            self.accept()
        except:
            import traceback
            traceback.print_exc()   
            pass

    def start_ml(self, GPU = False):


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
    data_received = pyqtSignal(dict, str)  # dictionary, property_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.excelotron_window = None
        self.last_data = None
        self.last_property = None
    
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
    
    def _on_data_exported(self, data_dict, property_name):
        """Internal slot to handle data from Excelotron"""
        self.last_data = data_dict
        self.last_property = property_name
        # Re-emit the signal for parent to handle
        self.data_received.emit(data_dict, property_name)
    
    def _on_window_destroyed(self):
        """Handle when the Excelotron window is destroyed/closed"""
        self.excelotron_window = None
    
    def get_last_data(self):
        """Get the last exported data"""
        return self.last_data, self.last_property

class MachineWindow(QMainWindow):

    def __init__(self, parent=None, GPU = False):
        super().__init__(parent)

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
            array1 = np.zeros_like(active_data).astype(np.uint8)
        except:
            print("No data in nodes channel")
            return

        self.setWindowTitle("Threshold")
        
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

        array3 = np.zeros_like(active_data).astype(np.uint8)
        self.parent().highlight_overlay = array3 #Clear this out for the segmenter to use

        self.parent().load_channel(2, array1, True)
        # Enable the channel button
        # Not exactly sure why we need all this but the channel buttons weren't loading like they normally do when load_channel() is called:
        if not self.parent().channel_buttons[2].isEnabled():
            self.parent().channel_buttons[2].setEnabled(True)
            self.parent().channel_buttons[2].click()
        self.parent().delete_buttons[2].setEnabled(True)

        self.parent().base_colors[act_channel] = self.parent().color_dictionary['WHITE']
        self.parent().base_colors[2] = self.parent().color_dictionary['LIGHT_GREEN']

        self.parent().update_display()
        
        # Set a reasonable default size for the window
        self.setMinimumWidth(600)  # Increased to accommodate grouped buttons
        self.setMinimumHeight(500)

        # Create main layout container
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Group 1: Drawing tools (Brush + Foreground/Background)
        drawing_group = QGroupBox("Drawing Tools")
        drawing_layout = QHBoxLayout()

        # Brush button
        self.brush_button = QPushButton("🖌️")
        self.brush_button.setCheckable(True)
        self.brush_button.setFixedSize(40, 40)
        self.brush_button.clicked.connect(self.toggle_brush_mode)
        self.brush_button.click()

        # Foreground/Background buttons in their own horizontal layout
        fb_layout = QHBoxLayout()
        self.fore_button = QPushButton("Foreground")
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
        training_group = QGroupBox("Training")
        training_layout = QHBoxLayout()
        train_quick = QPushButton("Train Quick Model")
        train_quick.clicked.connect(lambda: self.train_model(speed=True))
        train_detailed = QPushButton("Train More Detailed Model")
        train_detailed.clicked.connect(lambda: self.train_model(speed=False))
        save = QPushButton("Save Model")
        save.clicked.connect(self.save_model)
        load = QPushButton("Load Model")
        load.clicked.connect(self.load_model)
        training_layout.addWidget(train_quick)
        training_layout.addWidget(train_detailed)
        training_layout.addWidget(save)
        training_layout.addWidget(load)
        training_group.setLayout(training_layout)

        # Group 4: Segmentation Options
        segmentation_group = QGroupBox("Segmentation")
        segmentation_layout = QHBoxLayout()
        seg_button = QPushButton("Preview Segment")
        self.seg_button = seg_button
        seg_button.clicked.connect(self.start_segmentation)
        self.pause_button = QPushButton("▶/⏸️")
        self.pause_button.clicked.connect(self.pause)
        self.lock_button = QPushButton("🔒 Memory lock - (Prioritize RAM)")
        self.lock_button.setCheckable(True)
        self.lock_button.setChecked(True)
        self.lock_button.clicked.connect(self.toggle_lock)
        self.mem_lock = True
        full_button = QPushButton("Segment All")
        full_button.clicked.connect(self.segment)
        segmentation_layout.addWidget(seg_button)
        #segmentation_layout.addWidget(self.pause_button)   # <--- for some reason the segmenter preview is still running even when killed, may be regenerating itself somewhere. May or may not actually try to resolve this because this feature isnt that necessary.
        #segmentation_layout.addWidget(self.lock_button)   # Also turned this off
        segmentation_layout.addWidget(full_button)
        segmentation_group.setLayout(segmentation_layout)

        # Add all groups to main layout
        main_layout.addWidget(drawing_group)
        if not GPU:
            main_layout.addWidget(processing_group)
        main_layout.addWidget(training_group)
        main_layout.addWidget(segmentation_group)

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

    def toggle_lock(self):

        self.mem_lock = self.lock_button.isChecked()

    def pause(self):

        if self.segmentation_worker is not None:
            try:
                print("Pausing segmenter")
                self.previewing = False
                self.segmentation_finished
                del self.segmentation_worker
                self.segmentation_worker = None
            except:
                pass

        else:
            try:
                print("Restarting segmenter")
                self.previewing = True
                self.start_segmentation
            except:
                pass

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
        if self.parent().brush_mode:
            self.parent().pan_button.setChecked(False)
            self.parent().zoom_button.setChecked(False)
            self.parent().pan_mode = False
            self.parent().zoom_mode = False
            self.parent().update_brush_cursor()
        else:
            self.threed = False
            self.can = False
            self.parent().zoom_button.click()

    def silence_button(self):
        self.brush_button.setChecked(False)

    def toggle_brush_button(self):

        self.brush_button.click()

    def train_model(self, speed = True):

        self.kill_segmentation()
        # Wait a bit for cleanup
        time.sleep(0.1)
        if not self.use_two:
            self.previewing = False
        try:
            try:
                self.segmenter.train_batch(self.parent().channel_data[2], speed = speed, use_gpu = self.use_gpu, use_two = self.use_two, mem_lock = self.mem_lock)
                self.trained = True
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

        self.kill_segmentation()
        time.sleep(0.1)

        if self.use_two:
            self.previewing = True
        else:
            print("Beginning new segmentation...")


        if self.parent().active_channel == 0:
            if self.parent().channel_data[0] is not None:
                active_data = self.parent().channel_data[0]
            else:
                active_data = self.parent().channel_data[1]

        array3 = np.zeros_like(active_data).astype(np.uint8)
        self.parent().highlight_overlay = array3 #Clear this out for the segmenter to use

        if not self.trained:
            return
        else:
            self.segmentation_worker = SegmentationWorker(self.parent().highlight_overlay, self.segmenter, self.use_gpu, self.use_two, self.previewing, self, self.mem_lock)
            self.segmentation_worker.chunk_processed.connect(self.update_display)  # Just update display
            self.segmentation_worker.finished.connect(self.segmentation_finished)
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



    def check_for_z_change(self):
        current_z = self.parent().current_slice
        if not hasattr(self, '_last_z'):
            self._last_z = current_z
            return False
        
        changed = (self._last_z != current_z)
        self._last_z = current_z
        
        if changed and self.previewing and self.segmentation_worker is not None:
            self.segmentation_worker.stop()
            time.sleep(0.1)
            
            # Force regeneration of chunks
            self.segmenter.realtimechunks = None
            
            # Restart the worker
            self.start_segmentation()
            
        return changed

    def update_display(self):
        if not hasattr(self, '_last_update'):
            self._last_update = 0
        
        current_z = self.parent().current_slice
        if not hasattr(self, '_last_z'):
            self._last_z = current_z

        self._last_z = current_z

        if self.previewing:
            changed = self.check_for_z_change()

        current_time = time.time()
        if current_time - self._last_update >= 1:  # Match worker's interval
            try:
                # Store current view state
                current_xlim = self.parent().ax.get_xlim()
                current_ylim = self.parent().ax.get_ylim()

                try:
                    x, y = self.parent().get_current_mouse_position()
                except:
                    x, y = 0, 0
                self.segmenter.update_position(self.parent().current_slice, x, y)

                if not self.parent().painting:
                    # Only update if view limits are valid
                    self.parent().update_display(preserve_zoom=(current_xlim, current_ylim))

                    
                    self._last_update = current_time
            except Exception as e:
                print(f"Display update error: {e}")

    def poke_segmenter(self):
        if self.use_two and self.previewing:
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

    def segmentation_finished(self):
        
        current_xlim = self.parent().ax.get_xlim()
        current_ylim = self.parent().ax.get_ylim()
        self.parent().update_display(preserve_zoom=(current_xlim, current_ylim))
        
        # Store the current z position before killing the worker
        current_z = self.parent().current_slice
        
        # Clean up the worker
        self.kill_segmentation()
        self.segmentation_worker = None
        time.sleep(0.1)
        
        # Auto-restart for 2D preview mode only if certain conditions are met
        if self.previewing and self.use_two:
            # Track when this slice was last processed
            if not hasattr(self, '_processed_slices'):
                self._processed_slices = {}
                
            current_time = time.time()
            
            # Check if we've recently tried to process this slice (to prevent loops)
            recently_processed = False
            if current_z in self._processed_slices:
                time_since_last_attempt = current_time - self._processed_slices[current_z]
                recently_processed = time_since_last_attempt < 5.0  # 5 second cooldown
                
            if not recently_processed:
                self._processed_slices[current_z] = current_time
                
                # Reset any processing flags in the segmenter
                if hasattr(self.segmenter, '_currently_processing'):
                    self.segmenter._currently_processing = None
                
                if 0 in self.parent().highlight_overlay[current_z, :, :]:
                    # Create a new worker after a brief delay
                    QTimer.singleShot(500, self.start_segmentation)



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

            if self.parent().active_channel == 0:
                if self.parent().channel_data[0] is not None:
                    active_data = self.parent().channel_data[0]
                else:
                    active_data = self.parent().channel_data[1]

            array3 = np.zeros_like(active_data).astype(np.uint8)
            self.parent().highlight_overlay = array3 #Clear this out for the segmenter to use

            print("Segmenting entire volume with model...")
            #foreground_coords, background_coords = self.segmenter.segment_volume(array = self.parent().highlight_overlay)
            try:
                self.parent().highlight_overlay = self.segmenter.segment_volume(array = self.parent().highlight_overlay)
            except Exception as e:
                print(f"Error segmenting (Perhaps retrain the model...): {e}")
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
            if self.parent().isVisible():
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
                    
                    self.parent().machine_window = None
                else:
                    event.ignore()
        except:
            pass




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
        self.update_interval = 1  # Increased to 500ms
        self.chunks_since_update = 0
        self.chunks_per_update = 5  # Only update every 5 chunks
        self.poked = False # If it should wake up or not
        self.last_update = time.time()
        
    def stop(self):
        self._stop = True

    def get_poked(self):
        self.machine_window.poke_segmenter()
        
    def run(self):
        try:
            self.overlay.fill(False)
            
            # Remember the starting z position
            self.starting_z = self.segmenter.current_z
            
            if self.previewing and self.use_two:
                # Process current z-slice in chunks
                current_z = self.segmenter.current_z
                
                # Process the slice with chunked generator
                for foreground, background in self.segmenter.segment_slice_chunked(current_z):
                    if foreground == None and background == None:
                        self.get_poked()

                    if self._stop:
                        break
                    
                    # Update the overlay
                    for z,y,x in foreground:
                        self.overlay[z,y,x] = 1
                    for z,y,x in background:
                        self.overlay[z,y,x] = 2
                    
                    # Signal update after each chunk
                    self.chunks_since_update += 1
                    current_time = time.time()
                    if (self.chunks_since_update >= self.chunks_per_update and 
                        current_time - self.last_update >= self.update_interval):
                        self.chunk_processed.emit()
                        self.chunks_since_update = 0
                        self.last_update = current_time
                
            else:
                # Original 3D approach
                for foreground_coords, background_coords in self.segmenter.segment_volume_realtime(gpu=self.use_gpu):
                    if self._stop:
                        break
                    
                    for z,y,x in foreground_coords:
                        self.overlay[z,y,x] = 1
                    for z,y,x in background_coords:
                        self.overlay[z,y,x] = 2
                    
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

    def run_batch(self):
        try:
            foreground_coords, _ = self.segmenter.segment_volume()
            
            # Modify the array directly
            self.overlay.fill(False)
            for z,y,x in foreground_coords:
                self.overlay[z,y,x] = True
                
            self.finished.emit()
            
        except Exception as e:
            print(f"Error in segmentation: {e}")
            raise
            



class ThresholdWindow(QMainWindow):
    def __init__(self, parent=None, accepted_mode=0):
        super().__init__(parent)
        self.setWindowTitle("Threshold")

        self.accepted_mode = accepted_mode
        
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
            if 0 in img_nodes:
                del img_nodes[0]
            for node in img_nodes:
                if node in nodes:
                    self.parent().degree_dict[int(node)] = my_network.network.degree(node)
                else:
                    self.parent().degree_dict[int(node)] = 0

            self.histo_list = list(self.parent().degree_dict.values())
            self.bounds = False
            self.parent().bounds = False

        elif accepted_mode == 0:
            targ_shape = self.parent().channel_data[self.parent().active_channel].shape
            if (targ_shape[0] + targ_shape[1] + targ_shape[2]) > 2500: #Take a simpler histogram on big arrays
                temp_max = np.max(self.parent().channel_data[self.parent().active_channel])
                temp_min = np.min(self.parent().channel_data[self.parent().active_channel])
                temp_array = n3d.downsample(self.parent().channel_data[self.parent().active_channel], 5)
                self.histo_list = temp_array.flatten().tolist()
                self.histo_list.append(temp_min)
                self.histo_list.append(temp_max)
            else: #Otherwise just use full array data
                self.histo_list = self.parent().channel_data[self.parent().active_channel].flatten().tolist()
            self.bounds = True
            self.parent().bounds = True

        self.chan = self.parent().active_channel

            
        # Create matplotlib figure
        fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        
        # Pre-compute histogram with numpy
        counts, bin_edges = np.histogram(self.histo_list, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot pre-computed histogram
        self.ax = fig.add_subplot(111)
        self.ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], alpha=0.5)
        
        # Add vertical lines for thresholds
        self.min_line = self.ax.axvline(min(self.histo_list), color='r')
        self.max_line = self.ax.axvline(max(self.histo_list), color='b')
        
        # Connect events for dragging
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.dragging = None
        
        # Store histogram bounds
        if self.bounds:
            self.data_min = 0
        else:
            self.data_min = min(self.histo_list)
        self.data_max = max(self.histo_list)

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

        run_button = QPushButton("Apply Threshold")
        run_button.clicked.connect(self.thresh)
        form_layout.addRow(run_button)
        
        layout.addLayout(form_layout)
        
        # Set a reasonable default size
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    def closeEvent(self, event):
        self.parent().preview = False
        self.parent().targs = None
        self.parent().bounds = False
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


    def get_values_in_range_all_vols(self, chan, min_val, max_val):
        output = []
        if self.accepted_mode == 1:
            for node, vol in self.parent().volume_dict[chan].items():
                if min_val <= vol <= max_val:
                    output.append(node)
        elif self.accepted_mode == 2:
            for node, vol in self.parent().radii_dict[chan].items():
                if min_val <= vol <= max_val:
                    output.append(node)
        elif self.accepted_mode == 3:
            for node, vol in self.parent().degree_dict.items():
                if min_val <= vol <= max_val:
                    output.append(node)
        return output

    def get_values_in_range(self, lst, min_val, max_val):
        values = [x for x in lst if min_val <= x <= max_val]
        output = []
        if self.accepted_mode == 1:
            for item in self.parent().volume_dict[self.parent().active_channel]:
                if self.parent().volume_dict[self.parent().active_channel][item] in values:
                    output.append(item)
        elif self.accepted_mode == 2:
            for item in self.parent().radii_dict[self.parent().active_channel]:
                if self.parent().radii_dict[self.parent().active_channel][item] in values:
                    output.append(item)
        elif self.accepted_mode == 3:
            for item in self.parent().degree_dict:
                if self.parent().degree_dict[item] in values:
                    output.append(item)
        return output


    def min_value_changed(self):
        try:
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
                    self.max.setText(str(round(self.data_max, 2)))
                    self.max_line.set_xdata([self.data_max, self.data_max])
                    # And set min to the previous max value
                    value = max_val
                    self.min.setText(str(round(value, 2)))

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
                
                # Update the line
                self.min_line.set_xdata([value, value])
                self.canvas.draw()


                
            except ValueError:
                # If invalid number, reset to current line position
                self.min.setText(str(round(self.min_line.get_xself.data_mindata()[0], 2)))
        except:
            pass

    def max_value_changed(self):
        try:
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
                    self.min.setText(str(round(self.data_min, 2)))
                    self.min_line.set_xdata([self.data_min, self.data_min])
                    # And set max to the previous min value
                    value = min_val
                    self.max.setText(str(round(value, 2)))

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
                
                # Update the line
                self.max_line.set_xdata([value, value])
                self.canvas.draw()





                
            except ValueError:
                # If invalid number, reset to current line position
                self.max.setText(str(round(self.max_line.get_xdata()[0], 2)))
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
                
            if self.dragging == 'min':
                if event.xdata < self.max_line.get_xdata()[0]:
                    self.min_line.set_xdata([event.xdata, event.xdata])
                    self.min.setText(str(round(event.xdata, 2)))
            else:
                if event.xdata > self.min_line.get_xdata()[0]:
                    self.max_line.set_xdata([event.xdata, event.xdata])
                    self.max.setText(str(round(event.xdata, 2)))
                    
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

        # GPU checkbox (default True)
        self.GPU = QPushButton("GPU")
        self.GPU.setCheckable(True)
        self.GPU.setChecked(False)
        layout.addRow("Use GPU:", self.GPU)

        # dt checkbox (default False)
        self.predt = QPushButton("Fast Dilation")
        self.predt.setCheckable(True)
        self.predt.setChecked(False)
        layout.addRow("Use Fast Dilation (Higher speed, less accurate with search regions much larger than nodes):", self.predt)

        self.down_factor = QLineEdit("")
        layout.addRow("Internal Downsample for GPU (if needed):", self.down_factor)

        self.params = params

        # Add Run button
        run_button = QPushButton("Dilate")
        run_button.clicked.connect(self.smart_dilate)
        layout.addRow(run_button)

    def smart_dilate(self):

        GPU = self.GPU.isChecked()
        down_factor = float(self.down_factor.text()) if self.down_factor.text().strip() else None
        predt = self.predt.isChecked()
        active_data, amount, xy_scale, z_scale = self.params

        dilate_xy, dilate_z = n3d.dilation_length_to_pixels(xy_scale, z_scale, amount, amount)

        result = sdl.smart_dilate(active_data, dilate_xy, dilate_z, GPU = GPU, predownsample = down_factor, fast_dil = predt, use_dt_dil_amount = amount, xy_scale = xy_scale, z_scale = z_scale)

        self.parent().load_channel(self.parent().active_channel, result, True)
        self.accept()



class DilateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dilate Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)

        self.amount = QLineEdit("1")
        layout.addRow("Dilation Radius:", self.amount)

        if my_network.xy_scale is not None:
            xy_scale = f"{my_network.xy_scale}"
        else:
            xy_scale = "1"

        self.xy_scale = QLineEdit(xy_scale)
        layout.addRow("xy_scale:", self.xy_scale)

        if my_network.z_scale is not None:
            z_scale = f"{my_network.z_scale}"
        else:
            z_scale = "1"

        self.z_scale = QLineEdit(z_scale)
        layout.addRow("z_scale:", self.z_scale)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Pseudo3D Binary Kernels (For Fast, small dilations)", "Preserve Labels (slower)", "Distance Transform-Based (Slower but more accurate at larger dilations)"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
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

            if accepted_mode == 1:
                dialog = SmartDilateDialog(self.parent(), [active_data, amount, xy_scale, z_scale])
                dialog.exec()
                self.accept()
                return

            if accepted_mode == 2:
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Erosion Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)

        self.amount = QLineEdit("1")
        layout.addRow("Erosion Radius:", self.amount)

        if my_network.xy_scale is not None:
            xy_scale = f"{my_network.xy_scale}"
        else:
            xy_scale = "1"

        self.xy_scale = QLineEdit(xy_scale)
        layout.addRow("xy_scale:", self.xy_scale)

        if my_network.z_scale is not None:
            z_scale = f"{my_network.z_scale}"
        else:
            z_scale = "1"

        self.z_scale = QLineEdit(z_scale)
        layout.addRow("z_scale:", self.z_scale)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Pseudo3D Binary Kernels (For Fast, small erosions)", "Distance Transform-Based (Slower but more accurate at larger dilations)"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
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
                mode = mode
            )


            self.parent().load_channel(self.parent().active_channel, result, True)


            self.parent().update_display(preserve_zoom=(current_xlim, current_ylim))
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
            
            # Call dilate method with parameters
            result = n3d.fill_holes_3d(
                active_data,
                head_on = headon,
                fill_borders = borders
            )

            if not sep_holes:
                self.parent().load_channel(self.parent().active_channel, result, True)
            else:
                self.parent().load_channel(3, active_data - result, True)


            self.parent().update_display()
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error running fill holes: {str(e)}"
            )

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
                    self.parent().load_channel(output_target, channel_data = result, data = True)

                self.parent().update_display()

                self.accept()

            except Exception as e:
                print(f"Error masking: {e}")

        except Exception as e:
            print(f"Error masking: {e}")

class CropDialog(QDialog):

    def __init__(self, parent=None):

        try:

            super().__init__(parent)
            self.setWindowTitle("Crop Image (Will transpose any centroids)?")
            self.setModal(True)

            layout = QFormLayout(self)

            self.xmin = QLineEdit("0")
            layout.addRow("X Min", self.xmin)

            self.xmax = QLineEdit(f"{self.parent().shape[2]}")
            layout.addRow("X Max", self.xmax)

            self.ymin = QLineEdit("0")
            layout.addRow("Y Min", self.ymin)

            self.ymax = QLineEdit(f"{self.parent().shape[1]}")
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
                    
                    # Boolean mask for valid coordinates
                    valid_mask = ((transformed >= 0) & 
                                  (transformed <= np.array([zmax, ymax, xmax]))).all(axis=1)
                    
                    # Rebuild dictionary with only valid entries
                    my_network.node_centroids = {
                        nodes[int(i)]: [int(transformed[i, 0]), int(transformed[i, 1]), int(transformed[i, 2])]
                        for i in range(len(nodes)) if valid_mask[i]
                    }
                    
                    self.parent().format_for_upperright_table(my_network.node_centroids, 'NodeID', ['Z', 'Y', 'X'], 'Node Centroids')

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

            self.parent().load_channel(self.active_chan, active_data, True)


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
        self.auto.setChecked(False)
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
                result = n3d.remove_branches(result, remove)


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
                f"Error running skeletonize: {str(e)}"
            )   


class WatershedDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Watershed Parameters")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        # Directory (empty by default)
        self.directory = QLineEdit()
        self.directory.setPlaceholderText("Leave empty for None")
        layout.addRow("Output Directory:", self.directory)

        try:

            active_shape = self.parent().channel_data[self.parent().active_channel].shape[0]

            if active_shape == 1:
                self.default = 0.2
            else:
                self.default = 0.05

        except:
            self.default = 0.05

        
        # Proportion (default 0.1)
        self.proportion = QLineEdit(f"{self.default}")
        layout.addRow("Proportion:", self.proportion)
        
        # GPU checkbox (default True)
        self.gpu = QPushButton("GPU")
        self.gpu.setCheckable(True)
        self.gpu.setChecked(False)
        layout.addRow("Use GPU:", self.gpu)
        
        # Smallest radius (empty by default)
        self.smallest_rad = QLineEdit()
        self.smallest_rad.setPlaceholderText("Leave empty for None")
        layout.addRow("Smallest Radius:", self.smallest_rad)
        
        # Predownsample (empty by default)
        self.predownsample = QLineEdit()
        self.predownsample.setPlaceholderText("Leave empty for None")
        layout.addRow("Kernel Obtainment GPU Downsample:", self.predownsample)
        
        # Predownsample2 (empty by default)
        self.predownsample2 = QLineEdit()
        self.predownsample2.setPlaceholderText("Leave empty for None")
        layout.addRow("Smart Label GPU Downsample:", self.predownsample2)
        
        layout.addRow("Note:", QLabel(f"If the optimal proportion watershed output is still labeling spatially seperated objects with the same label, try right placing the result in nodes or edges\nthen right click the image and choose 'select all', followed by right clicking and 'selection' -> 'split non-touching labels'."))


        # Add Run button
        run_button = QPushButton("Run Watershed")
        run_button.clicked.connect(self.run_watershed)
        layout.addRow(run_button)

    def run_watershed(self):
        try:
            # Get directory (None if empty)
            directory = self.directory.text() if self.directory.text() else None
            
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
            except ValueError:
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
                predownsample=predownsample,
                predownsample2=predownsample2
            )

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
                self.parent().load_channel(i, self.parent().channel_data[i], True)
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

                my_network.nodes = my_network.centroid_array()

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
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        self.called = called
        
        # Set down_factor and cubic
        if not down_factor:
            down_factor = None
        
        if down_factor is None:
            # --- Processing Options Group ---
            process_group = QGroupBox("Processing Options")
            process_layout = QGridLayout()
            
            # Downsample factor
            self.down_factor = QLineEdit("0")
            process_layout.addWidget(QLabel("Downsample Factor (Speeds up calculation at the cost of fidelity):"), 0, 0)
            process_layout.addWidget(self.down_factor, 0, 1)
            
            # Cubic checkbox
            self.cubic = QPushButton("Cubic Downsample")
            self.cubic.setCheckable(True)
            self.cubic.setChecked(False)
            process_layout.addWidget(QLabel("Use cubic downsample? (Slower but preserves structure better):"), 1, 0)
            process_layout.addWidget(self.cubic, 1, 1)
            
            # Fast dilation checkbox
            self.fast_dil = QPushButton("Fast-Dil")
            self.fast_dil.setCheckable(True)
            self.fast_dil.setChecked(True)
            process_layout.addWidget(QLabel("Use Fast Dilation (Higher speed, less accurate with large search regions):"), 2, 0)
            process_layout.addWidget(self.fast_dil, 2, 1)
            
            process_group.setLayout(process_layout)
            main_layout.addWidget(process_group)
        else:
            self.down_factor = down_factor[0]
            self.cubic = down_factor[1]
            
            # Fast dilation checkbox (still needed even if down_factor is provided)
            process_group = QGroupBox("Processing Options")
            process_layout = QGridLayout()
            
            self.fast_dil = QPushButton("Fast-Dil")
            self.fast_dil.setCheckable(True)
            self.fast_dil.setChecked(True)
            process_layout.addWidget(QLabel("Use Fast Dilation (Higher speed, less accurate with large search regions):"), 0, 0)
            process_layout.addWidget(self.fast_dil, 0, 1)
            
            process_group.setLayout(process_layout)
            main_layout.addWidget(process_group)
        
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
        rec_layout.addWidget(QLabel("Attempt to Auto Correct Skeleton Looping:"), 1, 0)
        rec_layout.addWidget(self.auto, 1, 1)
        
        rec_group.setLayout(rec_layout)
        main_layout.addWidget(rec_group)
        
        # --- Optional Corrections Group ---
        opt_group = QGroupBox("Optional Corrections")
        opt_layout = QGridLayout()
        
        # Max volume
        self.max_vol = QLineEdit("0")
        opt_layout.addWidget(QLabel("Maximum Voxel Volume to Retain (Compensates for skeleton looping):"), 0, 0)
        opt_layout.addWidget(self.max_vol, 0, 1)
        
        # Component dilation
        self.comp_dil = QLineEdit("0")
        opt_layout.addWidget(QLabel("Voxel distance to merge nearby nodes (Compensates for multi-branch regions):"), 1, 0)
        opt_layout.addWidget(self.comp_dil, 1, 1)
        
        opt_group.setLayout(opt_layout)
        main_layout.addWidget(opt_group)
        
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
            if type(self.down_factor) is int:
                down_factor = self.down_factor
                cubic = self.cubic
            else:
                try:
                    down_factor = int(self.down_factor.text()) if self.down_factor.text() else 0
                except ValueError:
                    down_factor = 0
                cubic = self.cubic.isChecked()
                
            try:
                retain = self.retain.isChecked()
            except:
                retain = True

            if cubic:
                order = 3
            else:
                order = 0

            auto = self.auto.isChecked()

            fastdil = self.fast_dil.isChecked()


            if auto:
                my_network.edges = n3d.skeletonize(my_network.edges)
                my_network.edges = n3d.fill_holes_3d(my_network.edges)

            
            result, skele = n3d.label_vertices(
                my_network.edges,
                max_vol=max_vol,
                branch_removal=branch_removal,
                comp_dil=comp_dil,
                down_factor=down_factor,
                order = order,
                return_skele = True,
                fastdil = fastdil

            )

            if down_factor > 0 and not self.called:
                self.parent().resizing = True

                my_network.edges = n3d.downsample(my_network.edges, down_factor, order = order)
                my_network.xy_scale = my_network.xy_scale * down_factor
                my_network.z_scale = my_network.z_scale * down_factor
                print("xy_scales and z_scales have been adjusted per downsample. Check image -> properties to manually reset them to 1 if desired.")

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Branches (of edges)")
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # --- Correction Options Group ---
        correction_group = QGroupBox("Correction Options")
        correction_layout = QGridLayout()
        
        # Branch Fix checkbox
        self.fix = QPushButton("Auto-Correct 1")
        self.fix.setCheckable(True)
        self.fix.setChecked(False)
        correction_layout.addWidget(QLabel("Auto-Correct Branches by Collapsing Busy Neighbors: "), 0, 0)
        correction_layout.addWidget(self.fix, 0, 1)
        
        # Fix value
        self.fix_val = QLineEdit('4')
        correction_layout.addWidget(QLabel("Avg Degree of Nearby Branch Communities to Merge (4-6 recommended):"), 1, 0)
        correction_layout.addWidget(self.fix_val, 1, 1)
        
        # Seed
        self.seed = QLineEdit('')
        correction_layout.addWidget(QLabel("Random seed for auto correction (int - optional):"), 2, 0)
        correction_layout.addWidget(self.seed, 2, 1)

        self.fix2 = QPushButton("Auto-Correct 2")
        self.fix2.setCheckable(True)
        self.fix2.setChecked(True)
        correction_layout.addWidget(QLabel("Auto-Correct Branches by Collapsing Internal Labels: "), 3, 0)
        correction_layout.addWidget(self.fix2, 3, 1)

        self.fix3 = QPushButton("Split Nontouching Branches?")
        self.fix3.setCheckable(True)
        self.fix3.setChecked(True)
        correction_layout.addWidget(QLabel("Split Nontouching Branches? (Useful if branch pruning - may want to threshold out small, split branches after): "), 4, 0)
        correction_layout.addWidget(self.fix3, 4, 1)
        
        correction_group.setLayout(correction_layout)
        main_layout.addWidget(correction_group)
        
        # --- Processing Options Group ---
        processing_group = QGroupBox("Processing Options")
        processing_layout = QGridLayout()
        
        # Downsample factor
        self.down_factor = QLineEdit("0")
        processing_layout.addWidget(QLabel("Internal downsample factor (will recompute nodes):"), 0, 0)
        processing_layout.addWidget(self.down_factor, 0, 1)
        
        # Cubic checkbox
        self.cubic = QPushButton("Cubic Downsample")
        self.cubic.setCheckable(True)
        self.cubic.setChecked(False)
        processing_layout.addWidget(QLabel("Use cubic downsample? (Slower but preserves structure better):"), 1, 0)
        processing_layout.addWidget(self.cubic, 1, 1)
        
        processing_group.setLayout(processing_layout)
        main_layout.addWidget(processing_group)
        
        # --- Misc Options Group ---
        misc_group = QGroupBox("Misc Options")
        misc_layout = QGridLayout()
        
        # Nodes checkbox
        self.nodes = QPushButton("Generate Nodes")
        self.nodes.setCheckable(True)
        self.nodes.setChecked(True)
        misc_layout.addWidget(QLabel("Generate nodes from edges? (Skip if already completed):"), 0, 0)
        misc_layout.addWidget(self.nodes, 0, 1)
        
        # GPU checkbox
        self.GPU = QPushButton("GPU")
        self.GPU.setCheckable(True)
        self.GPU.setChecked(False)
        misc_layout.addWidget(QLabel("Use GPU (May downsample large images):"), 1, 0)
        misc_layout.addWidget(self.GPU, 1, 1)
        
        misc_group.setLayout(misc_layout)
        main_layout.addWidget(misc_group)
        
        # Add Run button
        run_button = QPushButton("Run Branch Label")
        run_button.clicked.connect(self.branch_label)
        main_layout.addWidget(run_button)

        if self.parent().channel_data[0] is not None or self.parent().channel_data[3] is not None:
            QMessageBox.critical(
                self,
                "Alert",
                "The nodes and overlay 2 channels will be intermittently overwritten when running this method"
            )

    def branch_label(self):

        try:

            try:
                down_factor = int(self.down_factor.text()) if self.down_factor.text() else 0
            except ValueError:
                down_factor = 0

            nodes = self.nodes.isChecked()
            GPU = self.GPU.isChecked()
            cubic = self.cubic.isChecked()
            fix = self.fix.isChecked()
            fix2 = self.fix2.isChecked()
            fix3 = self.fix3.isChecked()
            fix_val = float(self.fix_val.text()) if self.fix_val.text() else None
            seed = int(self.seed.text()) if self.seed.text() else None

            original_shape = my_network.edges.shape
            original_array = copy.deepcopy(my_network.edges)

            if down_factor > 0:
                self.parent().show_gennodes_dialog(down_factor = [down_factor, cubic], called = True)
            elif nodes or my_network.nodes is None:
                self.parent().show_gennodes_dialog(called = True)
                down_factor = None

            if my_network.edges is not None and my_network.nodes is not None and my_network.id_overlay is not None:

                output = n3d.label_branches(my_network.edges, nodes = my_network.nodes, bonus_array = original_array, GPU = GPU, down_factor = down_factor, arrayshape = original_shape)

                if fix2:

                    temp_network = n3d.Network_3D(nodes = output)

                    max_val = np.max(temp_network.nodes)

                    background = temp_network.nodes == 0

                    background = background * max_val

                    temp_network.nodes = temp_network.nodes + background

                    del background

                    temp_network.morph_proximity(search = [3,3], fastdil = True) #Detect network of nearby branches

                    output = n3d.fix_branches(output, temp_network.network, max_val)


                if fix:

                    temp_network = n3d.Network_3D(nodes = output)

                    temp_network.morph_proximity(search = [3,3], fastdil = True) #Detect network of nearby branches

                    temp_network.community_partition(weighted = False, style = 1, dostats = False, seed = seed) #Find communities with louvain, unweighted params

                    targs = n3d.fix_branches_network(temp_network.nodes, temp_network.network, temp_network.communities, fix_val)

                    temp_network.com_to_node(targs)

                    output = temp_network.nodes

                if fix3:

                    output = self.parent().separate_nontouching_objects(output, max_val=np.max(output))


                if down_factor is not None:

                    self.parent().reset(nodes = True, id_overlay = True, edges = True)

                else:
                    self.parent().reset(id_overlay = True)
                self.parent().update_display(dims = (output.shape[1], output.shape[2]))

                self.parent().load_channel(1, channel_data = output, data = True)

            self.parent().update_display()
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
        self.setModal(True)
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
                
                # Add reverse pair with same edge value
                my_network.network_lists[0].append(node2)
                my_network.network_lists[1].append(node1)
                my_network.network_lists[2].append(edge if edge is not None else 0)
            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.parent().network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.parent().network_table.resizeColumnToContents(column)
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

            try:
                if hasattr(my_network, 'network_lists'):
                    model = PandasModel(my_network.network_lists)
                    self.parent().network_table.setModel(model)
                    # Adjust column widths to content
                    for column in range(model.columnCount(None)):
                        self.parent().network_table.resizeColumnToContents(column)
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
        self.setModal(True)
        layout = QFormLayout(self)

        self.revid = QPushButton("Remove Unassigned")
        self.revid.setCheckable(True)
        self.revid.setChecked(False)
        layout.addRow("Remove Unassigned IDs from Centroid List?:", self.revid)

        self.remove = QPushButton("Remove Missing")
        self.remove.setCheckable(True)
        self.remove.setChecked(False)
        layout.addRow("Remove Any Nodes Not in Nodes Channel From Properties?:", self.remove)
        
        # trunk checkbox (default false)
        self.trunk = QPushButton("Remove Trunk")
        self.trunk.setCheckable(True)
        self.trunk.setChecked(False)
        layout.addRow("Remove Trunk? (Most connected edge - overrides below):", self.trunk)
        
        # trunk checkbox (default false)
        self.trunknode = QPushButton("Trunk -> Node")
        self.trunknode.setCheckable(True)
        self.trunknode.setChecked(False)
        layout.addRow("Convert Trunk to Node? (Most connected edge):", self.trunknode)
        
        # edgenode checkbox (default false)
        self.edgenode = QPushButton("Edges -> Nodes")
        self.edgenode.setCheckable(True)
        self.edgenode.setChecked(False)
        layout.addRow("Convert 'Edges (Labeled objects)' to node objects?:", self.edgenode)
        
        # edgeweight checkbox (default false)
        self.edgeweight = QPushButton("Remove weights")
        self.edgeweight.setCheckable(True)
        self.edgeweight.setChecked(False)
        layout.addRow("Remove network weights?:", self.edgeweight)
        
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
        dialog.exec()

    def run_changes(self):

        try:

            revid = self.revid.isChecked()
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
            except Exception as e:
                print(f"Error showing network table: {e}")

            if hasattr(my_network, 'node_identities') and my_network.node_identities is not None:
                try:
                    self.parent().format_for_upperright_table(my_network.node_identities, 'NodeID', 'Identity', 'Node Identities')
                except Exception as e:
                    print(f"Error loading node identity table: {e}")

            self.parent().update_display()
            self.accept()

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"An error occurred: {e}")







class CentroidDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Centroids")
        self.setModal(True)

        layout = QFormLayout(self)

        self.directory = QLineEdit()
        self.directory.setPlaceholderText("Leave empty for active directory")
        layout.addRow("Output Directory:", self.directory)

        self.downsample = QLineEdit("1")
        layout.addRow("Downsample Factor:", self.downsample)

        # Add mode selection dropdown
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Nodes and Edges", "Nodes", "Edges"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        layout.addRow("Execution Mode:", self.mode_selector)

        # Add Run button
        run_button = QPushButton("Run Calculate Centroids")
        run_button.clicked.connect(self.run_centroids)
        layout.addRow(run_button)

    def run_centroids(self):

        try:

            chan = self.mode_selector.currentIndex()

            # Get directory (None if empty)
            directory = self.directory.text() if self.directory.text() else None
            
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
                if directory:
                    my_network.save_node_centroids(directory = directory)

            elif chan == 2:
                my_network.calculate_edge_centroids(
                    down_factor = downsample
                )
                if directory:
                    my_network.save_edge_centroids(directory = directory)

            elif chan == 0:
                try:
                    my_network.calculate_node_centroids(
                        down_factor = downsample
                    )
                    if directory:
                        my_network.save_node_centroids(directory = directory)
                except:
                    pass

                try:

                    my_network.calculate_edge_centroids(
                        down_factor = downsample
                    )
                    if directory:
                        my_network.save_edge_centroids(directory = directory)

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
    prev_directory = ""
    prev_search = ""
    prev_diledge = ""
    prev_down_factor = ""
    prev_GPU_downsample = ""
    prev_other_nodes = ""
    prev_remove_trunk = ""
    prev_gpu = False
    prev_label_nodes = True
    prev_inners = True
    prev_fastdil = False
    prev_overlays = False
    prev_updates = True
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Connectivity Network Parameters")
        self.setModal(True)
        
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
        optional_layout.addRow("Filepath or directory containing additional node images:", self.other_nodes)
        
        self.remove_trunk = QLineEdit(self.prev_remove_trunk)
        self.remove_trunk.setPlaceholderText("Leave empty for 0")
        optional_layout.addRow("Times to remove edge trunks (int):", self.remove_trunk)
        
        self.inners = QPushButton("Inner Edges")
        self.inners.setCheckable(True)
        self.inners.setChecked(self.prev_inners)
        optional_layout.addRow("Use Inner Edges:", self.inners)
        
        main_layout.addWidget(optional_group)
        
        # Speed Up Options Group
        speedup_group = QGroupBox("Speed Up Options")
        speedup_layout = QFormLayout(speedup_group)
        
        self.down_factor = QLineEdit(self.prev_down_factor)
        self.down_factor.setPlaceholderText("Leave empty for None")
        speedup_layout.addRow("Downsample for Centroids (int):", self.down_factor)
        
        self.GPU_downsample = QLineEdit(self.prev_GPU_downsample)
        self.GPU_downsample.setPlaceholderText("Leave empty for None")
        speedup_layout.addRow("Downsample for Distance Transform (GPU) (int):", self.GPU_downsample)
        
        self.gpu = QPushButton("GPU")
        self.gpu.setCheckable(True)
        self.gpu.setChecked(self.prev_gpu)
        speedup_layout.addRow("Use GPU:", self.gpu)
        
        self.fastdil = QPushButton("Fast Dilate")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(self.prev_fastdil)
        speedup_layout.addRow("Use Fast Dilation (Higher speed, less accurate with search regions much larger than nodes):", self.fastdil)
        
        main_layout.addWidget(speedup_group)
        
        # Output Options Group
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)
        
        self.directory = QLineEdit(self.prev_directory)
        self.directory.setPlaceholderText("Will Have to Save Manually If Empty")
        output_layout.addRow("Output Directory:", self.directory)
        
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
            directory = self.directory.text() if self.directory.text() else None
            
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
                
            # Get button states
            gpu = self.gpu.isChecked()
            label_nodes = self.label_nodes.isChecked()
            inners = self.inners.isChecked()
            fastdil = self.fastdil.isChecked()
            overlays = self.overlays.isChecked()
            update = self.update.isChecked()

            if not update:
                temp_nodes = my_network.nodes.copy()
                temp_edges = my_network.edges.copy()
            
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
            CalcAllDialog.prev_directory = self.directory.text()
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


            # Then handle overlays
            if overlays:
                if directory is None:
                    directory = 'my_network'
                
                # Generate and update overlays
                my_network.network_overlay = my_network.draw_network(directory=directory)
                my_network.id_overlay = my_network.draw_node_indices(directory=directory)
                
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
            QMessageBox.critical(
                self,
                "Error",
                f"Error running calculate all: {str(e)}"
            )


class ProxDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Proximity Network")
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Important Parameters Group
        important_group = QGroupBox("Important Parameters")
        important_layout = QFormLayout(important_group)
        
        self.search = QLineEdit()
        self.search.setPlaceholderText("search")
        important_layout.addRow("Search Region Distance? (enter true value corresponding to scaling, ie in microns):", self.search)
        
        self.xy_scale = QLineEdit(f"{my_network.xy_scale}")
        important_layout.addRow("xy_scale:", self.xy_scale)
        
        self.z_scale = QLineEdit(f"{my_network.z_scale}")
        important_layout.addRow("z_scale:", self.z_scale)
        
        main_layout.addWidget(important_group)
        
        # Mode Group
        mode_group = QGroupBox("Mode")
        mode_layout = QFormLayout(mode_group)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["From Centroids (fast but ignores shape - use for small or spherical objects - search STARTS at centroid)", "From Morphological Shape (slower but preserves shape - use for oddly shaped objects - search STARTS at object border)"])
        self.mode_selector.setCurrentIndex(0)  # Default to Mode 1
        mode_layout.addRow("Execution Mode:", self.mode_selector)
        
        if my_network.node_identities is not None:
            self.id_selector = QComboBox()
            # Add all options from id dictionary
            self.id_selector.addItems(['None'] + list(set(my_network.node_identities.values())))
            self.id_selector.setCurrentIndex(0)  # Default to Mode 1
            mode_layout.addRow("Create Networks only from a specific node identity?:", self.id_selector)
        else:
            self.id_selector = None
        
        main_layout.addWidget(mode_group)
        
        # Output Options Group
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)
        
        self.directory = QLineEdit('')
        self.directory.setPlaceholderText("Leave empty for 'my_network'")
        output_layout.addRow("Output Directory:", self.directory)
        
        self.overlays = QPushButton("Overlays")
        self.overlays.setCheckable(True)
        self.overlays.setChecked(True)
        output_layout.addRow("Generate Overlays:", self.overlays)
        
        self.populate = QPushButton("Populate Nodes from Centroids?")
        self.populate.setCheckable(True)
        self.populate.setChecked(False)
        output_layout.addRow("If using centroid search:", self.populate)
        
        main_layout.addWidget(output_group)
        
        # Speed Up Options Group
        speedup_group = QGroupBox("Speed Up Options")
        speedup_layout = QFormLayout(speedup_group)
        
        self.fastdil = QPushButton("Fast Dilate")
        self.fastdil.setCheckable(True)
        self.fastdil.setChecked(False)
        speedup_layout.addRow("(If using morphological) Use Fast Dilation (Higher speed, less accurate with search regions much larger than nodes):", self.fastdil)
        
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
                target = self.id_selector.currentText()
                targets = []
                for node in my_network.node_identities:
                    if target == my_network.node_identities[node]:
                        targets.append(int(node))
            else:
                targets = None

            try:
                directory = self.directory.text() if self.directory.text() else None
            except:
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

            overlays = self.overlays.isChecked()  
            fastdil = self.fastdil.isChecked()

            my_network.xy_scale = xy_scale
            my_network.z_scale = z_scale


            if mode == 1:
                if len(np.unique(my_network.nodes)) < 3:
                    my_network.nodes, _ = n3d.label_objects(my_network.nodes)
                if my_network.node_centroids is None:
                    self.parent().show_centroid_dialog()
                my_network.morph_proximity(search = search, targets = targets, fastdil = fastdil)

                self.parent().load_channel(0, channel_data = my_network.nodes, data = True)
            elif mode == 0:

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
                    my_network.nodes = my_network.kd_network(distance = search, targets = targets, make_array = True)
                    self.parent().load_channel(0, channel_data = my_network.nodes, data = True)
                else:
                    my_network.kd_network(distance = search, targets = targets)

            if directory is not None:
                my_network.dump(directory = directory)


            # Then handle overlays
            if overlays:

                if my_network.node_centroids is not None:
                    if directory is None:
                        directory = 'my_network'
                    
                    # Generate and update overlays
                    my_network.network_overlay = my_network.draw_network(directory=directory)
                    my_network.id_overlay = my_network.draw_node_indices(directory=directory)
                    
                    # Update channel data
                    self.parent().load_channel(2, channel_data = my_network.network_overlay, data = True)
                    self.parent().load_channel(3, channel_data = my_network.id_overlay, data = True)
                    
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
            except:
                pass

            if my_network.network is None:
                my_network.network = my_network.network_lists

        except Exception as e:
            print(f"Error running proximity network: {str(e)}")
            import traceback
            print(traceback.format_exc())








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