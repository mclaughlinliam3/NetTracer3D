from PyQt6.QtWidgets import QApplication, QMainWindow
import matplotlib.pyplot as plt
import copy
import numpy as np


class PaintManager(QMainWindow):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.resume = False
        
        # Initialize stroke tracking storage once
        if parent is not None:
            if not hasattr(parent, 'completed_paint_strokes'):
                parent.completed_paint_strokes = []  # List of individual completed strokes
            if not hasattr(parent, 'current_stroke_points'):
                parent.current_stroke_points = []    # Current stroke being drawn
            if not hasattr(parent, 'current_stroke_type'):
                parent.current_stroke_type = None    # 'draw' or 'erase'
            
            # Keep the old properties for display purposes
            if not hasattr(parent, 'virtual_draw_operations'):
                parent.virtual_draw_operations = []
            if not hasattr(parent, 'virtual_erase_operations'):
                parent.virtual_erase_operations = []
            if not hasattr(parent, 'current_operation'):
                parent.current_operation = []
            if not hasattr(parent, 'current_operation_type'):
                parent.current_operation_type = None

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
        # Create static background (same as selection rectangle)

        if self.parent().machine_window is not None:
            if self.parent().machine_window.segmentation_worker is not None:
                if not self.parent().machine_window.segmentation_worker._paused:
                    self.resume = True
                self.parent().machine_window.segmentation_worker.pause()


        if not self.parent().channel_visible[channel]:
            self.parent().channel_visible[channel] = True
            
        # Capture the background once
        self.parent().static_background = self.parent().canvas.copy_from_bbox(self.parent().ax.bbox)

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
            
        # Initialize stroke tracking storage ONLY if they don't exist
        if not hasattr(self.parent(), 'completed_paint_strokes'):
            self.parent().completed_paint_strokes = []
        if not hasattr(self.parent(), 'current_stroke_points'):
            self.parent().current_stroke_points = []
        if not hasattr(self.parent(), 'current_stroke_type'):
            self.parent().current_stroke_type = None
            
        # Initialize display storage ONLY if they don't exist
        if not hasattr(self.parent(), 'virtual_draw_operations'):
            self.parent().virtual_draw_operations = []
        if not hasattr(self.parent(), 'virtual_erase_operations'):
            self.parent().virtual_erase_operations = []
        if not hasattr(self.parent(), 'current_operation'):
            self.parent().current_operation = []
        if not hasattr(self.parent(), 'current_operation_type'):
            self.parent().current_operation_type = None

    def reset_all_paint_storage(self):
        """Reset all paint storage - call this when you want to start completely fresh."""
        self.parent().completed_paint_strokes = []
        self.parent().current_stroke_points = []
        self.parent().current_stroke_type = None
        self.parent().virtual_draw_operations = []
        self.parent().virtual_erase_operations = []
        self.parent().current_operation = []
        self.parent().current_operation_type = None

    

    def add_virtual_paint_point(self, x, y, brush_size, erase=False, foreground=True):
        """Add a single paint point to the virtual layer."""
        
        # Determine operation type and visual properties
        if erase:
            paint_color = 'black'  # Visual indicator for erase
            alpha = 0.5
            operation_type = 'erase'
        else:
            if self.parent().machine_window is not None:
                if foreground:
                    paint_color = 'green'  # Visual for foreground (value 1)
                    alpha = 0.7
                else:
                    paint_color = 'red'  # Visual for background (value 2)
                    alpha = 0.7
            else:
                paint_color = 'white'  # Normal paint
                alpha = 0.7
            operation_type = 'draw'
                
        # Store the operation data (for later conversion to real paint)
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
        
        # Add to stroke tracking (NEW - separate stroke tracking)
        if self.parent().current_stroke_type != operation_type:
            # Finish previous stroke if switching between draw/erase
            self.finish_current_stroke()
            self.parent().current_stroke_type = operation_type
        
        self.parent().current_stroke_points.append(operation_data)
        
        # Create visual circle for display
        circle = plt.Circle((x, y), brush_size/2, 
                           color=paint_color, alpha=alpha, animated=True)
        
        # Add to display operations (OLD - for visual display)
        if self.parent().current_operation_type != operation_type:
            # Finish previous operation if switching between draw/erase
            self.finish_current_virtual_operation()
            self.parent().current_operation_type = operation_type
        
        self.parent().current_operation.append({
            'circle': circle,
            'data': operation_data
        })
        
        self.parent().ax.add_patch(circle)

    def finish_current_stroke(self):
        """Finish the current stroke and add it to completed strokes."""
        if not self.parent().current_stroke_points:
            return
            
        # Store the completed stroke with its type
        stroke_data = {
            'points': self.parent().current_stroke_points.copy(),
            'type': self.parent().current_stroke_type
        }
        
        self.parent().completed_paint_strokes.append(stroke_data)
        
        # Clear current stroke
        self.parent().current_stroke_points = []
        self.parent().current_stroke_type = None

    def add_virtual_paint_stroke(self, x, y, brush_size, erase=False, foreground=True):
        """Add a paint stroke - simple visual, interpolation happens during data conversion."""
        # Just add the current point for visual display (no interpolation yet)
        self.add_virtual_paint_point(x, y, brush_size, erase, foreground)
        
        # Store the last position for data conversion later
        self.parent().last_virtual_pos = (x, y)

    def connect_virtual_paint_points(self):
        """Connect points with lines matching the circle size by converting to screen coordinates."""
        
        if not hasattr(self.parent(), 'current_operation') or len(self.parent().current_operation) < 2:
            return
        
        # Get existing points but DON'T remove them
        existing_points = self.parent().current_operation.copy()
        point_data = [item['data'] for item in existing_points if 'data' in item]
        
        if len(point_data) < 2:
            return
        
        # Get visual properties and brush size from first point
        first_data = point_data[0]
        brush_size_data = first_data['brush_size']  # This is in data coordinates
        
        # Convert brush size from data coordinates to points for linewidth
        # Get the transformation from data to display coordinates
        ax = self.parent().ax
        
        # Transform two points to see the scaling
        p1_data = [0, 0]
        p2_data = [brush_size_data, 0]  # One brush_size unit away
        
        p1_display = ax.transData.transform(p1_data)
        p2_display = ax.transData.transform(p2_data)
        
        # Calculate pixels per data unit
        pixels_per_data_unit = abs(p2_display[0] - p1_display[0])
        
        # Convert to points (matplotlib uses 72 points per inch, figure.dpi pixels per inch)
        fig = ax.figure
        points_per_pixel = 72.0 / fig.dpi
        brush_size_points = pixels_per_data_unit * points_per_pixel
        
        if first_data['erase']:
            line_color = 'black'
            alpha = 0.5
        else:
            if self.parent().machine_window is not None:
                if first_data['foreground']:
                    line_color = 'green'
                    alpha = 0.7
                else:
                    line_color = 'red'
                    alpha = 0.7
            else:
                line_color = 'white'
                alpha = 0.7
        
        # Create line segments for connections using LineCollection
        from matplotlib.collections import LineCollection
        
        segments = []
        for i in range(len(point_data) - 1):
            x1, y1 = point_data[i]['x'], point_data[i]['y']
            x2, y2 = point_data[i+1]['x'], point_data[i+1]['y']
            segments.append([(x1, y1), (x2, y2)])
        
        # Create line collection with converted linewidth
        if segments:
            lc = LineCollection(segments, 
                               colors=line_color, 
                               alpha=alpha,
                               linewidths=brush_size_points,  # Now in points, matching circles
                               animated=True)
            self.parent().ax.add_collection(lc)
            
            # Add the line collection as a visual-only element
            self.parent().current_operation.append({
                'line_collection': lc,
                'is_connection_visual': True
            })

    def finish_current_virtual_operation(self):
       """Finish the current operation (draw or erase) and add it to the appropriate list."""
       
       if not self.parent().current_operation:
           return
       
       # Filter out connection visuals from the operation before storing
       data_items = []
       visual_items = []
       
       for item in self.parent().current_operation:
           if item.get('is_connection_visual', False):
               visual_items.append(item)
           else:
               data_items.append(item)
       
       # Only store the data items for this specific stroke
       if data_items:
           if self.parent().current_operation_type == 'draw':
               self.parent().virtual_draw_operations.append(data_items)
           elif self.parent().current_operation_type == 'erase':
               self.parent().virtual_erase_operations.append(data_items)
       
       # Clean up visual items that are connection-only
       for item in visual_items:
           try:
               if 'line_collection' in item:
                   item['line_collection'].remove()
               elif 'line' in item:
                   item['line'].remove()
           except:
               pass
       
       self.parent().current_operation = []
       self.parent().current_operation_type = None

    def update_virtual_paint_display(self):
        """Update display with virtual paint strokes - handles different object types."""
        if not hasattr(self.parent(), 'static_background') or self.parent().static_background is None:
            return
        
        # Restore the clean background
        self.parent().canvas.restore_region(self.parent().static_background)
        
        # Draw all completed operations
        for operation_list in [self.parent().virtual_draw_operations, self.parent().virtual_erase_operations]:
            for operation in operation_list:
                for item in operation:
                    self._draw_virtual_item(item)
        
        # Draw current operation being painted
        if hasattr(self.parent(), 'current_operation'):
            for item in self.parent().current_operation:
                self._draw_virtual_item(item)
        
        # Blit everything at once
        self.parent().canvas.blit(self.parent().ax.bbox)

    def _draw_virtual_item(self, item):
        """Helper method to draw different types of virtual paint items."""
        try:
            # Skip items that are marked as visual-only connections
            if item.get('is_connection_visual', False):
                if 'line' in item:
                    self.parent().ax.draw_artist(item['line'])
                elif 'line_collection' in item:
                    self.parent().ax.draw_artist(item['line_collection'])
            elif 'circle' in item:
                self.parent().ax.draw_artist(item['circle'])
            elif 'line' in item:
                self.parent().ax.draw_artist(item['line'])
            elif 'line_collection' in item:
                self.parent().ax.draw_artist(item['line_collection'])
        except Exception as e:
            # Skip items that can't be drawn (might have been removed)
            pass

    def convert_virtual_strokes_to_data(self):
        """Convert each stroke separately to actual array data using ONLY the new stroke tracking system."""
        
        # Finish the current stroke first
        self.finish_current_stroke()
        
        # Process ONLY the completed_paint_strokes (ignore old display operations)
        for stroke in self.parent().completed_paint_strokes:
            stroke_points = stroke['points']
            stroke_type = stroke['type']
            
            if len(stroke_points) == 0:
                continue
                
            # Apply interpolation within this stroke only
            last_pos = None
            for point_data in stroke_points:
                current_pos = (point_data['x'], point_data['y'])
                
                if last_pos is not None:
                    # Interpolate between consecutive points in this stroke
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
        
        # Clean up ALL visual elements (both old and new systems)
        for operation_list in [self.parent().virtual_draw_operations, self.parent().virtual_erase_operations]:
            for operation in operation_list:
                for item in operation:
                    try:
                        if 'circle' in item:
                            item['circle'].remove()
                        elif 'line_collection' in item:
                            item['line_collection'].remove()
                        elif 'line' in item:
                            item['line'].remove()
                    except:
                        pass
        
        if hasattr(self.parent(), 'current_operation'):
            for item in self.parent().current_operation:
                try:
                    if 'circle' in item:
                        item['circle'].remove()
                    elif 'line_collection' in item:
                        item['line_collection'].remove()
                    elif 'line' in item:
                        item['line'].remove()
                except:
                    pass
        
        # Reset all storage for next paint session
        self.parent().completed_paint_strokes = []
        self.parent().current_stroke_points = []
        self.parent().current_stroke_type = None
        self.parent().virtual_draw_operations = []
        self.parent().virtual_erase_operations = []
        self.parent().current_operation = []
        self.parent().current_operation_type = None

    def end_virtual_paint_session(self):
        """Convert virtual paint to actual array modifications when exiting paint mode."""
        if not hasattr(self.parent(), 'virtual_paint_strokes'):
            return
        
        # Now apply all the virtual strokes to the actual arrays
        for stroke in self.parent().virtual_paint_strokes:
            for circle in stroke:
                center = circle.center
                radius = circle.radius
                is_erase = circle.get_facecolor()[0] == 0  # Black = erase
                
                # Apply to actual array
                self.paint_at_position_vectorized(
                    int(center[0]), int(center[1]), 
                    erase=is_erase, 
                    channel=self.paint_channel,
                    brush_size=int(radius * 2)
                )
                
                # Remove the virtual circle
                circle.remove()
        
        # Clean up virtual paint data
        self.virtual_paint_strokes = []
        self.current_stroke = []
        
        # Reset background
        self.static_background = None
        self.painting = False
        
        # Full refresh to show final result
        self.update_display()

    def paint_at_position_vectorized(self, center_x, center_y, erase=False, channel=2, 
                                   slice_idx=None, brush_size=None, threed=None, 
                                   threedthresh=None, foreground=True, machine_window=None):
        """Vectorized paint operation for better performance."""
        if self.parent().channel_data[channel] is None:
            return
        
        # Use provided parameters or fall back to instance variables
        slice_idx = slice_idx if slice_idx is not None else self.parent().current_slice
        brush_size = brush_size if brush_size is not None else getattr(self.parent(), 'brush_size', 5)
        threed = threed if threed is not None else getattr(self.parent(), 'threed', False)
        threedthresh = threedthresh if threedthresh is not None else getattr(self.parent(), 'threedthresh', 1)
        
        # Handle 3D painting by recursively calling for each slice
        if threed and threedthresh > 1:
            half_range = (threedthresh - 1) // 2
            low = max(0, slice_idx - half_range)
            high = min(self.parent().channel_data[channel].shape[0] - 1, slice_idx + half_range)
            
            
            for i in range(low, high + 1):
                
                # Recursive call for each slice, but with threed=False to avoid infinite recursion
                self.paint_at_position_vectorized(
                    center_x, center_y, 
                    erase=erase, 
                    channel=channel,
                    slice_idx=i,  # Paint on slice i
                    brush_size=brush_size,
                    threed=False,  # Important: turn off 3D for recursive calls
                    threedthresh=1,
                    foreground=foreground,
                    machine_window=machine_window
                )
                
                
            return  # Exit early, recursive calls handle everything
        
        # Regular 2D painting (single slice)
        
        # Determine paint value
        if erase:
            val = 0
        elif machine_window is None:
            try:
                val = self.parent().min_max[channel][1]
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
            return  # No valid region to paint
        
        # Create coordinate grids for the affected region
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
        
        # Calculate distances squared (avoid sqrt for performance)
        distances_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        mask = distances_sq <= radius ** 2
        
        # Paint on this single slice
        
        self.parent().channel_data[channel][slice_idx][y_min:y_max, x_min:x_max][mask] = val