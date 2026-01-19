from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import copy
import numpy as np


class PaintManager(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resume = False
        
        # Initialize stroke tracking storage once
        if parent is not None:
            if not hasattr(parent, 'completed_paint_strokes'):
                parent.completed_paint_strokes = []
            if not hasattr(parent, 'current_stroke_points'):
                parent.current_stroke_points = []
            if not hasattr(parent, 'current_stroke_type'):
                parent.current_stroke_type = None
            
            # PyQtGraph visual items
            if not hasattr(parent, 'virtual_paint_items'):
                parent.virtual_paint_items = []  # Store all visual items
            if not hasattr(parent, 'current_paint_items'):
                parent.current_paint_items = []  # Current stroke visuals

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

    def initiate_paint_session(self, channel, current_xlim, current_ylim):
        if self.parent().machine_window is not None:
            if self.parent().machine_window.segmentation_worker is not None:
                if not self.parent().machine_window.segmentation_worker._paused:
                    self.resume = True
                self.parent().machine_window.segmentation_worker.pause()

        if not self.parent().channel_visible[channel]:
            self.parent().channel_visible[channel] = True

        if self.resume:
            self.parent().machine_window.segmentation_worker.resume()
            self.resume = False

    def start_virtual_paint_session(self, channel, current_xlim, current_ylim):
        """Start a virtual paint session that doesn't modify arrays until the end."""
        self.parent().painting = True
        self.parent().paint_channel = channel
        
        # Store original state
        if not self.parent().channel_visible[channel]:
            self.parent().channel_visible[channel] = True
            
        # Initialize stroke tracking storage
        if not hasattr(self.parent(), 'completed_paint_strokes'):
            self.parent().completed_paint_strokes = []
        if not hasattr(self.parent(), 'current_stroke_points'):
            self.parent().current_stroke_points = []
        if not hasattr(self.parent(), 'current_stroke_type'):
            self.parent().current_stroke_type = None
            
        # Initialize PyQtGraph visual storage
        if not hasattr(self.parent(), 'virtual_paint_items'):
            self.parent().virtual_paint_items = []
        if not hasattr(self.parent(), 'current_paint_items'):
            self.parent().current_paint_items = []

    def reset_all_paint_storage(self):
        """Reset all paint storage."""
        # Clear visual items from view
        if hasattr(self.parent(), 'virtual_paint_items'):
            for item in self.parent().virtual_paint_items:
                try:
                    self.parent().view.removeItem(item)
                except:
                    pass
        
        if hasattr(self.parent(), 'current_paint_items'):
            for item in self.parent().current_paint_items:
                try:
                    self.parent().view.removeItem(item)
                except:
                    pass
        
        self.parent().completed_paint_strokes = []
        self.parent().current_stroke_points = []
        self.parent().current_stroke_type = None
        self.parent().virtual_paint_items = []
        self.parent().current_paint_items = []

    def add_virtual_paint_point(self, x, y, brush_size, erase=False, foreground=True):
        """Add a single paint point to the virtual layer using PyQtGraph."""
        
        # Determine operation type and visual properties
        if erase:
            paint_color = (0, 0, 0)  # Black for erase
            operation_type = 'erase'
        else:
            if self.parent().machine_window is not None:
                if foreground:
                    paint_color = (0, 255, 0)  # Green for foreground (value 1)
                else:
                    paint_color = (255, 0, 0)  # Red for background (value 2)
            else:
                paint_color = (255, 255, 255)  # White for normal paint
            operation_type = 'draw'
        
        # Store the operation data
        operation_data = {
            'x': x,
            'y': y,
            'brush_size': brush_size,
            'erase': erase,
            'foreground': foreground,
            'channel': self.parent().paint_channel,
            'threed': getattr(self.parent(), 'threed', False),
            'threedthresh': getattr(self.parent(), 'threedthresh', 1)
        }
        
        # Add to stroke tracking
        if self.parent().current_stroke_type != operation_type:
            self.finish_current_stroke()
            self.parent().current_stroke_type = operation_type
        
        self.parent().current_stroke_points.append(operation_data)
        
        # Create visual circle using ScatterPlotItem
        scatter = pg.ScatterPlotItem(
            [x], [y], 
            size=brush_size,
            pen=pg.mkPen(paint_color, width=1),
            brush=pg.mkBrush(*paint_color, 127)  # 50% alpha
        )
        
        # Add to view
        self.parent().view.addItem(scatter)
        self.parent().current_paint_items.append(scatter)

    def finish_current_stroke(self):
        """Finish the current stroke and add it to completed strokes."""
        if not self.parent().current_stroke_points:
            return
        
        # Store the completed stroke with its type AND visual items
        stroke_data = {
            'points': self.parent().current_stroke_points.copy(),
            'type': self.parent().current_stroke_type,
            'visual_items': self.parent().current_paint_items.copy()  # Store visual items with stroke
        }
        
        self.parent().completed_paint_strokes.append(stroke_data)
        
        # Move current visual items to completed
        self.parent().virtual_paint_items.extend(self.parent().current_paint_items)
        self.parent().current_paint_items = []
        
        # Clear current stroke data
        self.parent().current_stroke_points = []
        self.parent().current_stroke_type = None

    def undo_last_virtual_stroke(self):
        """Undo the most recent virtual paint stroke (not yet converted to data)."""
        
        # First try to undo the current stroke in progress
        if hasattr(self.parent(), 'current_stroke_points') and self.parent().current_stroke_points:
            # Remove visual items for current stroke
            if hasattr(self.parent(), 'current_paint_items'):
                for item in self.parent().current_paint_items:
                    try:
                        self.parent().view.removeItem(item)
                    except:
                        pass
                self.parent().current_paint_items = []
            
            # Clear current stroke data
            self.parent().current_stroke_points = []
            self.parent().current_stroke_type = None
            return True  # Successfully undid current stroke
        
        # If no current stroke, undo the most recent completed stroke
        if hasattr(self.parent(), 'completed_paint_strokes') and self.parent().completed_paint_strokes:
            # Get the last completed stroke
            last_stroke = self.parent().completed_paint_strokes.pop()
            
            # Remove its visual items from the view
            visual_items = last_stroke.get('visual_items', [])
            for item in visual_items:
                try:
                    self.parent().view.removeItem(item)
                    # Also remove from virtual_paint_items list
                    if item in self.parent().virtual_paint_items:
                        self.parent().virtual_paint_items.remove(item)
                except:
                    pass
            
            return True  # Successfully undid completed stroke
        
        # Nothing to undo
        return False

    def add_virtual_paint_stroke(self, x, y, brush_size, erase=False, foreground=True):
        """Add a paint stroke."""
        self.add_virtual_paint_point(x, y, brush_size, erase, foreground)
        self.parent().last_virtual_pos = (x, y)

    def connect_virtual_paint_points(self):
        """Connect points with lines matching the brush size."""
        if not hasattr(self.parent(), 'current_stroke_points') or len(self.parent().current_stroke_points) < 2:
            return
        
        point_data = self.parent().current_stroke_points
        
        if len(point_data) < 2:
            return
        
        # Get visual properties from first point
        first_data = point_data[0]
        brush_size = first_data['brush_size']
        
        if first_data['erase']:
            line_color = (0, 0, 0)  # Black
        else:
            if self.parent().machine_window is not None:
                if first_data['foreground']:
                    line_color = (0, 255, 0)  # Green
                else:
                    line_color = (255, 0, 0)  # Red
            else:
                line_color = (255, 255, 255)  # White
        
        # Create line segments
        x_coords = [p['x'] for p in point_data]
        y_coords = [p['y'] for p in point_data]
        
        # Create connected line
        line = pg.PlotDataItem(
            x_coords, y_coords,
            pen=pg.mkPen(color=line_color, width=brush_size)
        )
        line.setOpacity(0.5)
        
        # Add to view
        self.parent().view.addItem(line)
        self.parent().current_paint_items.append(line)

    def finish_current_virtual_operation(self):
        """Finish the current operation."""
        self.finish_current_stroke()

    def convert_virtual_strokes_to_data(self):
        """Convert each stroke separately to actual array data."""
        
        # Finish the current stroke first
        self.finish_current_stroke()
        
        # Process completed strokes
        for stroke in self.parent().completed_paint_strokes:
            stroke_points = stroke['points']
            
            if len(stroke_points) == 0:
                continue
            
            # Apply interpolation within this stroke
            last_pos = None
            for point_data in stroke_points:
                current_pos = (point_data['x'], point_data['y'])
                
                if last_pos is not None:
                    # Interpolate between consecutive points
                    points = self.get_line_points(last_pos[0], last_pos[1], current_pos[0], current_pos[1])
                    for px, py in points:
                        self.paint_at_position_vectorized(
                            px, py,
                            erase=point_data['erase'],
                            channel=point_data['channel'],
                            brush_size=point_data['brush_size'],
                            threed=point_data['threed'],
                            threedthresh=point_data['threedthresh'],
                            foreground=point_data['foreground'],
                            machine_window=self.parent().machine_window
                        )
                else:
                    # First point in stroke
                    self.paint_at_position_vectorized(
                        point_data['x'], point_data['y'],
                        erase=point_data['erase'],
                        channel=point_data['channel'],
                        brush_size=point_data['brush_size'],
                        threed=point_data['threed'],
                        threedthresh=point_data['threedthresh'],
                        foreground=point_data['foreground'],
                        machine_window=self.parent().machine_window
                    )
                
                last_pos = current_pos
        
        # Clean up visual elements
        for item in self.parent().virtual_paint_items:
            try:
                self.parent().view.removeItem(item)
            except:
                pass
        
        for item in self.parent().current_paint_items:
            try:
                self.parent().view.removeItem(item)
            except:
                pass
        
        # Reset storage
        self.parent().completed_paint_strokes = []
        self.parent().current_stroke_points = []
        self.parent().current_stroke_type = None
        self.parent().virtual_paint_items = []
        self.parent().current_paint_items = []

    def paint_at_position_vectorized(self, center_x, center_y, erase=False, channel=2, 
                                   slice_idx=None, brush_size=None, threed=None, 
                                   threedthresh=None, foreground=True, machine_window=None):
        """Vectorized paint operation for better performance."""
        if self.parent().channel_data[channel] is None:
            return
        slice_idx = slice_idx if slice_idx is not None else self.parent().current_slice
        brush_size = brush_size if brush_size is not None else getattr(self.parent(), 'brush_size', 5)
        threed = threed if threed is not None else getattr(self.parent(), 'threed', False)
        threedthresh = threedthresh if threedthresh is not None else getattr(self.parent(), 'threedthresh', 1)
        
        # Handle 3D painting
        if threed and threedthresh > 1:
            half_range = (threedthresh - 1) // 2
            low = max(0, slice_idx - half_range)
            high = min(self.parent().channel_data[channel].shape[0] - 1, slice_idx + half_range)
            
            for i in range(low, high + 1):
                self.paint_at_position_vectorized(
                    center_x, center_y, 
                    erase=erase, 
                    channel=channel,
                    slice_idx=i,
                    brush_size=brush_size,
                    threed=False,
                    threedthresh=1,
                    foreground=foreground,
                    machine_window=machine_window
                )
            return
        
        # Determine paint value
        if erase:
            val = 0
        elif machine_window is None:
            try:
                val = max(1, self.parent().min_max[channel][1])
            except:
                val = 255
        elif foreground:
            val = 1
        else:
            val = 2
        
        height, width = self.parent().channel_data[channel][slice_idx].shape
        radius = brush_size // 2
        
        # Calculate affected region bounds
        y_min = max(0, center_y - radius)
        y_max = min(height, center_y + radius + 1)
        x_min = max(0, center_x - radius)
        x_max = min(width, center_x + radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            return
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Calculate distances and create mask
        distances_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        mask = distances_sq <= radius ** 2
        
        # Paint on this slice
        self.parent().channel_data[channel][slice_idx][y_min:y_max, x_min:x_max][mask] = val