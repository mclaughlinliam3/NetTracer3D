from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QMenuBar, QMenu,
                             QTextEdit, QToolBar)
from PyQt6.QtCore import Qt, QRect, QRectF, QPoint, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QAction
import sys


class TutorialOverlay(QWidget):
    """Overlay widget that covers the entire window and highlights specific elements"""
    next_clicked = pyqtSignal()
    back_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        self.highlight_rect = None
        self.highlight_type = "circle"
        self.message = ""
        self.message_position = "bottom"
        self.show_back_button = False
        
        # Scroll support
        self.scroll_offset = 0
        self.max_scroll = 0
        self.needs_scroll = False
        self.scrollbar_rect = None
        
    def set_highlight(
            self,
            widget_or_rect,
            highlight_type="circle",
            message="",
            message_position="bottom",
            show_back_button=False
        ):
        """Set which widget/rect to highlight and what message to display.

        If highlight_type is None / "" / "none", no highlight will be drawn,
        even if widget_or_rect is provided.
        """
        # Normalize highlight type
        if not highlight_type or highlight_type == "none":
            self.highlight_type = "none"
            self.highlight_rect = None
        else:
            self.highlight_type = highlight_type

            if widget_or_rect is None:
                self.highlight_rect = None
            elif isinstance(widget_or_rect, QRect):
                # It's already a rect in global coordinates, convert to local
                local_pos = self.mapFromGlobal(widget_or_rect.topLeft())
                self.highlight_rect = QRect(local_pos, widget_or_rect.size())
            else:
                # It's a widget, get its global position and convert to overlay coordinates
                global_pos = widget_or_rect.mapToGlobal(QPoint(0, 0))
                local_pos = self.mapFromGlobal(global_pos)
                self.highlight_rect = QRect(local_pos, widget_or_rect.size())
            
        if self.message != message:
            self.scroll_offset = 0
        
        self.message = message
        self.message_position = message_position
        self.show_back_button = show_back_button
        self.update()

        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create a path for the entire overlay
        full_path = QPainterPath()
        full_path.addRect(QRectF(self.rect()))
        
        # If there's a highlight, subtract it from the overlay
        if self.highlight_rect:
            highlight_path = QPainterPath()
            
            if self.highlight_type == "circle":
                # Create circular highlight
                center = self.highlight_rect.center()
                radius = max(self.highlight_rect.width(), self.highlight_rect.height()) // 2 + 20
                highlight_path.addEllipse(QPointF(center), radius, radius)
            else:  # rectangle
                padding = 10
                highlight_rect = self.highlight_rect.adjusted(-padding, -padding, padding, padding)
                highlight_path.addRoundedRect(QRectF(highlight_rect), 10, 10)
            
            # Subtract the highlight area from the full overlay
            full_path = full_path.subtracted(highlight_path)
        
        # Draw semi-transparent overlay (excluding the highlight area)
        painter.fillPath(full_path, QColor(0, 0, 0, 180))
        
        # Draw highlight border
        if self.highlight_rect:
            painter.setPen(QPen(QColor(0, 191, 255), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            if self.highlight_type == "circle":
                center = self.highlight_rect.center()
                radius = max(self.highlight_rect.width(), self.highlight_rect.height()) // 2 + 20
                painter.drawEllipse(QPointF(center), radius, radius)
            else:
                padding = 10
                highlight_rect = self.highlight_rect.adjusted(-padding, -padding, padding, padding)
                painter.drawRoundedRect(QRectF(highlight_rect), 10, 10)
            
            # Draw arrow pointing to the highlighted area
            if self.highlight_type == "circle":
                self._draw_arrow_to_circle(painter)
            
        # Draw message box
        if self.message:
            self._draw_message_box(painter)
            
    def _draw_arrow_to_circle(self, painter):
        """Draw an arrow pointing to the highlighted circle"""
        center = self.highlight_rect.center()
        radius = max(self.highlight_rect.width(), self.highlight_rect.height()) // 2 + 20
        
        # Determine arrow position based on message position
        if self.message_position == "bottom":
            arrow_start = QPoint(center.x(), center.y() + radius + 100)
            arrow_end = QPoint(center.x(), center.y() + radius)
        elif self.message_position == "top":
            arrow_start = QPoint(center.x(), center.y() - radius - 100)
            arrow_end = QPoint(center.x(), center.y() - radius)
        elif self.message_position == "left":
            arrow_start = QPoint(center.x() - radius - 100, center.y())
            arrow_end = QPoint(center.x() - radius, center.y())
        else:  # right
            arrow_start = QPoint(center.x() + radius + 100, center.y())
            arrow_end = QPoint(center.x() + radius, center.y())
        
        # Draw arrow line
        painter.setPen(QPen(QColor(255, 215, 0), 4))
        painter.drawLine(arrow_start, arrow_end)
        
        # Draw arrowhead
        self._draw_arrowhead(painter, arrow_start, arrow_end)
        
    def _draw_arrowhead(self, painter, start, end):
        """Draw an arrowhead at the end of a line"""
        # Calculate angle
        import math
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        
        arrow_size = 15
        angle1 = angle + math.pi * 0.8
        angle2 = angle - math.pi * 0.8
        
        p1 = QPointF(end.x() + arrow_size * math.cos(angle1),
                     end.y() + arrow_size * math.sin(angle1))
        p2 = QPointF(end.x() + arrow_size * math.cos(angle2),
                     end.y() + arrow_size * math.sin(angle2))
        
        path = QPainterPath()
        path.moveTo(QPointF(end))
        path.lineTo(p1)
        path.lineTo(p2)
        path.closeSubpath()
        
        painter.setBrush(QBrush(QColor(255, 215, 0)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)
        
    def _draw_message_box(self, painter):
        """Draw the tutorial message box"""
        # Calculate message box position
        padding = 20
        box_width = 350
        
        # Calculate required height based on text
        font = QFont("Arial", 11)
        painter.setFont(font)
        fm = painter.fontMetrics()
        
        # Calculate text area
        text_width = box_width - 30
        button_area_height = 50
        text_padding = 30
        
        # Calculate required height for text
        text_rect_temp = QRect(0, 0, text_width, 10000)
        bounding_rect = fm.boundingRect(text_rect_temp, 
                                         Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop, 
                                         self.message)
        
        required_text_height = bounding_rect.height()
        ideal_box_height = required_text_height + text_padding + button_area_height
        
        # Calculate maximum available height
        max_available_height = self.height() - 2 * padding
        
        # Determine if we need scrolling
        self.needs_scroll = ideal_box_height > max_available_height
        
        if self.needs_scroll:
            box_height = max_available_height
            scrollbar_width = 20  # Made wider for arrow buttons
            actual_text_width = text_width - scrollbar_width - 10
            self.max_scroll = required_text_height - (box_height - text_padding - button_area_height)
            self.max_scroll = max(0, self.max_scroll + 100)  # Add 30px padding
        else:
            box_height = max(150, ideal_box_height)
            scrollbar_width = 0
            actual_text_width = text_width
            self.scroll_offset = 0
            self.max_scroll = 0

        # Position calculation (same as before)
        if self.message_position == "beside":
            box_x = padding
            box_y = self.height() - box_height - padding
        elif self.message_position == "top_right":
            box_x = self.width() - box_width - padding
            box_y = padding
        elif self.message_position == 'top_left':
            box_x = padding
            box_y = padding
        elif self.highlight_rect:
            center = self.highlight_rect.center()
            radius = max(self.highlight_rect.width(), self.highlight_rect.height()) // 2 + 20
            
            if self.message_position == "bottom":
                box_x = center.x() - box_width // 2
                box_y = center.y() + radius + 120
            elif self.message_position == "top":
                box_x = center.x() - box_width // 2
                box_y = center.y() - radius - 120 - box_height
            elif self.message_position == "left":
                box_x = center.x() - radius - 120 - box_width
                box_y = center.y() - box_height // 2
            else:  # right
                box_x = center.x() + radius + 120
                box_y = center.y() - box_height // 2
        else:
            if self.message_position == "bottom":
                box_x = (self.width() - box_width) // 2
                box_y = self.height() - box_height - padding - 100
            elif self.message_position == "top":
                box_x = (self.width() - box_width) // 2
                box_y = padding + 100
            elif self.message_position == "left":
                box_x = padding + 50
                box_y = (self.height() - box_height) // 2
            elif self.message_position == "right":
                box_x = self.width() - box_width - padding - 50
                box_y = (self.height() - box_height) // 2
            else:
                box_x = (self.width() - box_width) // 2
                box_y = (self.height() - box_height) // 2
        
        # Ensure box stays within bounds
        box_x = max(padding, min(box_x, self.width() - box_width - padding))
        box_y = max(padding, min(box_y, self.height() - box_height - padding))
        
        message_rect = QRect(box_x, box_y, box_width, box_height)
        self.message_rect_for_scroll = message_rect
        
        # Draw message box background
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 191, 255), 2))
        painter.drawRoundedRect(message_rect, 10, 10)
        
        # Set up clipping for text area
        text_rect = message_rect.adjusted(15, 15, -15 - scrollbar_width, -button_area_height)
        
        # Save painter state and set clipping
        painter.save()
        painter.setClipRect(text_rect)
        
        # Draw message text with scroll offset
        painter.setPen(QColor(0, 0, 0))
        scrolled_text_rect = text_rect.adjusted(0, -self.scroll_offset, 0, required_text_height)
        painter.drawText(scrolled_text_rect, Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop, self.message)
        
        # Restore painter state
        painter.restore()
        
        # Draw scroll arrows if needed
        if self.needs_scroll:
            arrow_x = message_rect.right() - scrollbar_width - 8
            arrow_width = scrollbar_width
            arrow_height = 25
            
            # Up arrow
            up_arrow_y = text_rect.top()
            self.scroll_up_rect = QRect(arrow_x, up_arrow_y, arrow_width, arrow_height)
            
            # Draw up arrow button
            up_color = QColor(150, 150, 150) if self.scroll_offset == 0 else QColor(0, 191, 255)
            painter.setBrush(QBrush(up_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.scroll_up_rect, 3, 3)
            
            # Draw up arrow triangle
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            arrow_path = QPainterPath()
            center_x = self.scroll_up_rect.center().x()
            arrow_path.moveTo(center_x, up_arrow_y + 8)
            arrow_path.lineTo(center_x - 5, up_arrow_y + 17)
            arrow_path.lineTo(center_x + 5, up_arrow_y + 17)
            arrow_path.closeSubpath()
            painter.drawPath(arrow_path)
            
            # Down arrow
            down_arrow_y = text_rect.bottom() - arrow_height
            self.scroll_down_rect = QRect(arrow_x, down_arrow_y, arrow_width, arrow_height)
            
            # Draw down arrow button
            down_color = QColor(150, 150, 150) if self.scroll_offset >= self.max_scroll else QColor(0, 191, 255)
            painter.setBrush(QBrush(down_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.scroll_down_rect, 3, 3)
            
            # Draw down arrow triangle
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            arrow_path = QPainterPath()
            center_x = self.scroll_down_rect.center().x()
            arrow_path.moveTo(center_x, down_arrow_y + 17)
            arrow_path.lineTo(center_x - 5, down_arrow_y + 8)
            arrow_path.lineTo(center_x + 5, down_arrow_y + 8)
            arrow_path.closeSubpath()
            painter.drawPath(arrow_path)
        
        # Draw buttons (same as before)
        button_width = 80
        button_height = 30
        button_spacing = 10
        
        if self.show_back_button:
            next_rect = QRect(message_rect.right() - button_width - 15,
                              message_rect.bottom() - button_height - 10,
                              button_width, button_height)
            
            skip_rect = QRect(next_rect.left() - button_width - button_spacing,
                              next_rect.top(),
                              button_width, button_height)
            
            back_rect = QRect(skip_rect.left() - button_width - button_spacing,
                              skip_rect.top(),
                              button_width, button_height)
            
            self.back_button_rect = back_rect
            
            painter.setBrush(QBrush(QColor(150, 150, 150)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(back_rect, 5, 5)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(back_rect, Qt.AlignmentFlag.AlignCenter, "Back")
        else:
            next_rect = QRect(message_rect.right() - button_width - 15,
                              message_rect.bottom() - button_height - 10,
                              button_width, button_height)
            
            skip_rect = QRect(next_rect.left() - button_width - button_spacing,
                              next_rect.top(),
                              button_width, button_height)
            
            self.back_button_rect = None
        
        self.next_button_rect = next_rect
        self.skip_button_rect = skip_rect
        
        painter.setBrush(QBrush(QColor(0, 191, 255)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(next_rect, 5, 5)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(next_rect, Qt.AlignmentFlag.AlignCenter, "Next")
        
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(skip_rect, 5, 5)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(skip_rect, Qt.AlignmentFlag.AlignCenter, "Skip")


    def keyPressEvent(self, event):

        """Key press shortcuts for main class"""
        print('hello')
        if event.key() == Qt.Key_Space:
            self.next_clicked.emit()


    def mousePressEvent(self, event):
        """Handle clicks - treat any click as Next unless it's Skip, Back, or scroll arrows"""
        event.accept()
        
        # Check scroll up arrow
        if self.needs_scroll and hasattr(self, 'scroll_up_rect') and self.scroll_up_rect.contains(event.pos()):
            self.scroll_offset = max(0, self.scroll_offset - 30)
            self.update()
            return
        
        # Check scroll down arrow
        if self.needs_scroll and hasattr(self, 'scroll_down_rect') and self.scroll_down_rect.contains(event.pos()):
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + 30)
            self.update()
            return
        
        # Check if clicking Skip button
        if hasattr(self, 'skip_button_rect') and self.skip_button_rect.contains(event.pos()):
            self.skip_clicked.emit()
        # Check if clicking Back button
        elif hasattr(self, 'back_button_rect') and self.back_button_rect and self.back_button_rect.contains(event.pos()):
            self.back_clicked.emit()
        elif hasattr(self, 'next_button_rect') and self.next_button_rect and self.next_button_rect.contains(event.pos()):
            self.next_clicked.emit()
        elif event.button() == Qt.MouseButton.RightButton:
            self.next_clicked.emit()
        # Any other click (including Next button or anywhere else) advances
        #else:
        #    self.next_clicked.emit()

    def wheelEvent(self, event):
        if self.needs_scroll and hasattr(self, 'message_rect_for_scroll'):
            # Check if mouse is over the message box
            if self.message_rect_for_scroll.contains(event.position().toPoint()):
                delta = event.angleDelta().y()
                scroll_amount = delta // 120 * 20  # Scroll by 20 pixels per notch
                
                self.scroll_offset -= scroll_amount
                self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))
                
                self.update()
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

class TutorialManager:
    """Manages the tutorial steps and progression"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.overlay = None
        self.current_step = 0
        self.steps = []
        
    def add_step(self, widget, message, highlight_type="circle", message_position="bottom", 
                 action=None, pre_action=None):
        """
        Add a tutorial step
        
        Args:
            widget: The widget to highlight
            message: The message to display
            highlight_type: "circle" or "rect"
            message_position: "top", "bottom", "left", or "right"
            action: Callable to execute when moving to next step
            pre_action: Callable to execute before showing this step (e.g., to open a menu)
        """
        self.steps.append({
            'widget': widget,
            'message': message,
            'highlight_type': highlight_type,
            'message_position': message_position,
            'action': action,
            'pre_action': pre_action
        })
        
    def start(self):
        """Start the tutorial"""
        if not self.overlay:
            self.overlay = TutorialOverlay(self.main_window)
            self.overlay.next_clicked.connect(self.next_step)
            self.overlay.back_clicked.connect(self.previous_step)
            self.overlay.skip_clicked.connect(self.end_tutorial)
            
        self.current_step = 0
        self.overlay.setGeometry(self.main_window.rect())
        self.overlay.show()
        self.overlay.raise_()
        self.show_current_step()
        
    def show_current_step(self):
        """Display the current tutorial step"""
        if self.current_step >= len(self.steps):
            self.end_tutorial()
            return
            
        step = self.steps[self.current_step]
        
        # Execute pre-action if any (e.g., open menu)
        if step['pre_action']:
            step['pre_action']()
            # Small delay to let the action complete
            QTimer.singleShot(50, lambda: self._show_step_highlight(step))
        else:
            self._show_step_highlight(step)
            
    def _show_step_highlight(self, step):
        """Show the highlight for a step"""
        widget_or_rect = step['widget']() if callable(step['widget']) else step['widget']
        
        # Determine if back button should be shown (not on first step)
        show_back = self.current_step > 0
        
        if widget_or_rect is not None:
            self.overlay.set_highlight(
                widget_or_rect,
                step['highlight_type'],
                step['message'],
                step['message_position'],
                show_back_button=show_back
            )
        else:
            # If widget not found, show message only
            self.overlay.set_highlight(None, "rect", step['message'], step['message_position'], show_back_button=show_back)
            
    def next_step(self):
        """Move to the next tutorial step"""
        step = self.steps[self.current_step]
        
        # Execute step action if any
        if step['action']:
            step['action']()
            
        self.current_step += 1
        
        if self.current_step < len(self.steps):
            self.show_current_step()
        else:
            self.end_tutorial()
    
    def previous_step(self):
        """Move to the previous tutorial step"""
        if self.current_step > 0:
            # Execute cleanup action from current step if going back
            step = self.steps[self.current_step]
            if step['action']:
                step['action']()
            
            self.current_step -= 1
            self.show_current_step()
            
    def end_tutorial(self):
        """End the tutorial"""
        if self.overlay:
            self.overlay.hide()
            self.overlay.deleteLater()
            self.overlay = None
        self.current_step = 0


class MenuHelper:
    """Helper class for interacting with menus programmatically"""
    
    @staticmethod
    def open_menu(window, menu_name):
        """Open a menu by name"""
        menubar = window.menuBar()
        for action in menubar.actions():
            if action.text() == menu_name:
                menu = action.menu()
                if menu:
                    # Get the geometry of this specific action to position correctly
                    action_rect = menubar.actionGeometry(action)
                    popup_pos = window.mapToGlobal(menubar.mapToParent(action_rect.bottomLeft()))
                    menu.popup(popup_pos)
                return menu
        return None
    
    @staticmethod
    def close_menu(window, menu_name):
        """Close a menu by name"""
        menubar = window.menuBar()
        for action in menubar.actions():
            if action.text() == menu_name:
                menu = action.menu()
                if menu:
                    menu.hide()
                return
    
    @staticmethod
    def get_menu(window, menu_name):
        """Get a menu object by name"""
        menubar = window.menuBar()
        for action in menubar.actions():
            if action.text() == menu_name:
                return action.menu()
        return None
    
    @staticmethod
    def open_submenu(menu, submenu_name):
        """Open a submenu within a menu"""
        if not menu:
            return None
        for action in menu.actions():
            if action.text() == submenu_name:
                submenu = action.menu()
                if submenu:
                    # Get the geometry of this specific action in the parent menu
                    action_rect = menu.actionGeometry(action)
                    # Position submenu to the right of the parent menu item
                    popup_pos = menu.mapToGlobal(action_rect.topRight())
                    submenu.popup(popup_pos)
                return submenu
        return None
    
    @staticmethod
    def trigger_action(menu, action_name):
        """Trigger a menu action by name"""
        if not menu:
            return False
        for action in menu.actions():
            if action.text() == action_name:
                action.trigger()
                return True
        return False
    
    @staticmethod
    def click_menu_item(window, menu_name, item_name):
        """Click a menu item by menu and item name"""
        menu = MenuHelper.open_menu(window, menu_name)
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.trigger_action(menu, item_name))
            return True
        return False
    
    @staticmethod
    def click_submenu_item(window, menu_name, submenu_name, item_name):
        """Click a submenu item"""
        menu = MenuHelper.open_menu(window, menu_name)
        if menu:
            def trigger_submenu():
                submenu = MenuHelper.open_submenu(menu, submenu_name)
                if submenu:
                    QTimer.singleShot(100, lambda: MenuHelper.trigger_action(submenu, item_name))
            QTimer.singleShot(100, trigger_submenu)
            return True
        return False
    
    @staticmethod
    def get_action_rect(menu, action_name, window):
        """Get the global rect for a menu action to highlight it"""
        if not menu:
            return None
        for action in menu.actions():
            if action.text() == action_name:
                # Get the action's geometry within the menu
                action_rect = menu.actionGeometry(action)
                # Convert to global coordinates
                global_pos = menu.mapToGlobal(action_rect.topLeft())
                # Create a QRect in global coordinates
                return QRect(global_pos, action_rect.size())
        return None
    
    @staticmethod
    def get_submenu_action_rect(parent_menu, submenu_name, action_name, window):
        """Get the global rect for an action within a submenu"""
        submenu = None
        for action in parent_menu.actions():
            if action.text() == submenu_name:
                submenu = action.menu()
                break
        
        if submenu:
            return MenuHelper.get_action_rect(submenu, action_name, window)
        return None
    
    @staticmethod
    def create_menu_step_rect_getter(window, menu_name):
        """Create a function that gets a menu item rect for tutorial highlighting"""
        def get_menu_rect():
            menubar = window.menuBar()
            for action in menubar.actions():
                if action.text() == menu_name:
                    action_rect = menubar.actionGeometry(action)
                    global_pos = window.mapToGlobal(menubar.mapToParent(action_rect.topLeft()))
                    return QRect(global_pos, action_rect.size())
            # Fallback
            menubar_rect = menubar.rect()
            global_pos = window.mapToGlobal(menubar.mapToParent(menubar_rect.topLeft()))
            return QRect(global_pos.x() + 50, global_pos.y(), 100, menubar_rect.height())
        return get_menu_rect
    
    @staticmethod
    def create_submenu_item_rect_getter(window, menu_name, submenu_name):
        """Create a function that gets a submenu item rect for tutorial highlighting"""
        def get_submenu_rect():
            menu = MenuHelper.get_menu(window, menu_name)
            if menu and menu.isVisible():
                rect = MenuHelper.get_action_rect(menu, submenu_name, window)
                if rect:
                    return rect
            # Fallback
            menubar = window.menuBar()
            menubar_rect = menubar.rect()
            global_pos = window.mapToGlobal(menubar.mapToParent(menubar_rect.topLeft()))
            return QRect(global_pos.x() + 150, global_pos.y(), 100, menubar_rect.height())
        return get_submenu_rect
    
    @staticmethod
    def create_submenu_action_rect_getter(window, menu_name, submenu_name, action_name):
        """Create a function that gets a submenu action rect for tutorial highlighting"""
        def get_action_rect():
            menu = MenuHelper.get_menu(window, menu_name)
            if menu and menu.isVisible():
                submenu = None
                for action in menu.actions():
                    if action.text() == submenu_name:
                        submenu = action.menu()
                        break
                
                if submenu and submenu.isVisible():
                    rect = MenuHelper.get_action_rect(submenu, action_name, window)
                    if rect:
                        return rect
            # Fallback
            menubar = window.menuBar()
            menubar_rect = menubar.rect()
            global_pos = window.mapToGlobal(menubar.mapToParent(menubar_rect.topLeft()))
            return QRect(global_pos.x() + 250, global_pos.y() + 100, 150, 30)
        return get_action_rect
    
    @staticmethod
    def create_dialog_opener(window, tutorial, dialog_method_name, dialog_class_name, store_attr_name, 
                             widget_type=None, *method_args, **method_kwargs):
        """Create functions to open a dialog/window and store its reference
        
        Args:
            widget_type: The base class to check for (QDialog, QMainWindow, etc.). 
                         If None, will accept any QWidget.
        """
        def open_dialog():
            # Clear any existing reference to the old dialog/window
            if hasattr(tutorial, store_attr_name):
                old_widget = getattr(tutorial, store_attr_name)
                if old_widget:
                    try:
                        old_widget.close()
                        old_widget.deleteLater()
                    except:
                        pass
                setattr(tutorial, store_attr_name, None)
            
            # Close any open menus first
            for action in window.menuBar().actions():
                menu = action.menu()
                if menu:
                    menu.hide()
            
            # Open the dialog/window with provided args/kwargs
            QTimer.singleShot(200, lambda: getattr(window, dialog_method_name)(*method_args, **method_kwargs))
            # Store reference
            QTimer.singleShot(300, lambda: store_dialog())
        
        def store_dialog():
            from PyQt6.QtWidgets import QDialog, QMainWindow, QWidget
            
            # Determine what type to check for
            if widget_type is None:
                check_type = QWidget
            else:
                check_type = widget_type
            
            # Find the most recently created widget that matches and is visible
            matching_widgets = []
            for child in window.children():
                if child.__class__.__name__ == dialog_class_name and isinstance(child, check_type):
                    if child.isVisible():
                        matching_widgets.append(child)
            
            if matching_widgets:
                setattr(tutorial, store_attr_name, matching_widgets[-1])
            else:
                QTimer.singleShot(200, store_dialog)
        
        return open_dialog, store_dialog
    
    @staticmethod
    def create_widget_getter(tutorial, dialog_attr, widget_attr):
        """
        Create a function that gets a widget from a dialog for highlighting
        
        Args:
            tutorial: Tutorial manager instance
            dialog_attr: Dialog attribute name (e.g., 'dilate_dialog')
            widget_attr: Widget attribute name (e.g., 'mode_selector')
        
        Returns:
            Function that returns the widget or None
        """
        def getter():
            if hasattr(tutorial, dialog_attr):
                dialog = getattr(tutorial, dialog_attr)
                if dialog and hasattr(dialog, widget_attr):
                    return getattr(dialog, widget_attr)
            return None
        return getter
    
    @staticmethod
    def create_widget_interaction(tutorial, dialog_attr, widget_attr, interaction, delay=0):
        """
        Create an action that performs an interaction on a widget
        
        Args:
            tutorial: Tutorial manager instance
            dialog_attr: Dialog attribute name (e.g., 'dilate_dialog')
            widget_attr: Widget attribute name (e.g., 'mode_selector')
            interaction: Interaction string (e.g., 'showPopup()', 'setText("5")', 'click()')
            delay: Optional delay in ms before executing interaction
        
        Returns:
            Function that performs the interaction
            
        Example:
            action=MenuHelper.create_widget_interaction(
                tutorial, 'dilate_dialog', 'mode_selector', 'showPopup()', delay=100
            )
        """
        def action():
            if hasattr(tutorial, dialog_attr):
                dialog = getattr(tutorial, dialog_attr)
                if dialog and hasattr(dialog, widget_attr):
                    if interaction == 'close()':
                        dialog.close()
                        return
                    widget = getattr(dialog, widget_attr)
                    
                    def do_interaction():
                        # Special handling for click() on buttons
                        if interaction == 'click()':
                            def blink_sequence():
                                widget.click()  # Turn off
                                QTimer.singleShot(200, lambda: widget.click())  # Turn on
                                QTimer.singleShot(400, lambda: widget.click())  # Turn off
                                QTimer.singleShot(600, lambda: widget.click())  # Turn on (final state)
                            from PyQt6.QtWidgets import QPushButton
                            # Check if it's a checkable button that's already checked
                            if isinstance(widget, QPushButton) and widget.isCheckable() and widget.isChecked():
                                # Blink effect: toggle off and on multiple times to draw attention
                                blink_sequence()
                            else:
                                # Just click normally (button is off or not checkable)
                                widget.click()
                                blink_sequence()
                        elif interaction.startswith('setText'):
                            exec(f'widget.{interaction}', {'widget': widget})
                            if interaction.endswith('("")'):
                                widget.deselect()
                            else:
                                widget.selectAll()
                        else:
                            # Execute the interaction on the widget for all other cases
                            # Pass locals to exec so it can access 'widget'
                            exec(f'widget.{interaction}', {'widget': widget})
                    
                    if delay > 0:
                        QTimer.singleShot(delay, do_interaction)
                    else:
                        do_interaction()
        
        return action

def setup_start_tutorial(window):
    """
    Set up the basic interface tutorial for NetTracer3D
    
    Args:
        window: ImageViewerWindow instance from nettracer_gui
    
    Returns:
        TutorialManager instance
    """
    tutorial = TutorialManager(window)

    # Step 1: Welcome
    tutorial.add_step(
        window.canvas,
        "Welcome to NetTracer3D! This tutorial will give you a basic overview of this application. Click 'Next' or use Right-Click to continue.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "This program is designed to analysis of two or three dimensional images, such as those aquired via microscopy or medical imaging.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "The major form of analysis is done by creating undirected networks between objects of interest, called nodes. These can be biological structures such as cells or functional tissue units.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "Analysis can also be done on more direct measures of morphology or spatial arrangement, such as analyzing object measures like volumes or making clustering heatmaps.",
        highlight_type=None,
        message_position="bottom"
    )

    # Threshold Tool
    tutorial.add_step(
        window.thresh_button,
        "Any quantifications need to be done on segmented data. A segmented image is one where all your objects of interest have either been assigned a binary value (ie 1 or 255), or assigned a discrete integer label (ie each cell contains the val 1, 2, 3, etc). Raw images should be segmented first, either here or with another software.",
        highlight_type="circle",
        message_position="top"
    )

    tutorial.add_step(
        window.canvas,
        "When it comes to making networks, there are three major modalities that NetTracer3D offers.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "The first is the 'connectivity network', where your node objects are connected via a secondary structure, deemed 'edges'. For example, we can evaluate how groups of segmented cell aggregates are connected via vasculature.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        window.channel_buttons[0],
        "This would require providing two segmentations, one for your nodes.",
        highlight_type="rect",
        message_position="top"
    )

    tutorial.add_step(
        window.channel_buttons[1],
        "And a second for your edges.",
        highlight_type="rect",
        message_position="top")

    tutorial.add_step(
        window.canvas,
        "The second modality is making networks directly from branched structures. First, you would provide a binary segmentation of a branching structure like a nerve or a blood vessel. Next, you can algorithmically label the branches in NetTracer3D.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        window.channel_buttons[1],
        "Typically you would queue your image to be branch-labeled in the edges channel.",
        highlight_type="rect",
        message_position="top")

    tutorial.add_step(
        window.channel_buttons[0],
        "But you can also load them into the nodes channel, although note whatever is in the 'edges' channel takes priority. This is because the program has to actually make nodes at the branchpoints of your edges so it temporarily treats branches like edges.",
        highlight_type="rect",
        message_position="top"
    )

    tutorial.add_step(
        window.canvas,
        "Labeled branches can be turned into two types of networks. The first way is to connect the branchpoints. The second is to connect the branches themselves, just based on what other branches they come off of.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "The final modality is making networks based on proximity. This is an option to evaluate spatial clusters in your image, for example, deciphering what sort of groups a set of cells are arranged in. This would be an ideal way to analyze a multiplexed image with a lot of different channels bearing cellular fluorescent labels, for example.",
        highlight_type=None,
        message_position="bottom"
    )

    tutorial.add_step(
        None,
        "Networks can be directly quantified, but there are still many more options for direct morphological/spatial analysis, and for making interesting visualizations!",
        message_position="bottom"
    )

    def open_to_properties():
        menu = MenuHelper.open_menu(window, "Image")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Properties"))
    
    tutorial.add_step(
        MenuHelper.create_submenu_item_rect_getter(window, "Image", "Properties"),
        "Your current session has several stored properties, some of which are available here.",
        highlight_type=None,
        message_position="beside",
        pre_action=open_to_properties,
        action=lambda: MenuHelper.close_menu(window, "Image")
    )
    
    # Step 5: Open the dialog
    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_properties_dialog", "PropertiesDialog", "properties_dialog"
    )
    
    tutorial.add_step(
        None,
        "Let's open the Properties menu to see all the options available. Click 'Next' to open it.",
        message_position="beside",
        action=open_dialog
    )

    tutorial.add_step(
        None,
        "A blue button means the property has data, an unselected button means it's empty. Deselecting any blue property and pressing enter below will erase that property from the current session.",
        message_position="beside"
        )
    
    
    # Step 6: Explain xy_scale field (and demonstrate interaction)
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'xy_scale'),
        "xy_scale affects how NetTracer3D interprets distances in the X and Y dimensions. If your image has anisotropic voxels (different spacing in X/Y vs Z), you may need to adjust this to compensate. Note that your data is always presumed to have an equal resolution in the xy plane itself.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'xy_scale', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'xy_scale', 'deselect()')
    )
    
    # Step 7: Explain z_scale field
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'z_scale'),
        "z_scale adjusts the evaluation of distances in the Z dimension. Many microscopy images have a different Z step size than XY resolution, so you might set this differently than xy_scale.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'z_scale', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'z_scale', 'deselect()')
    )

    # Step 8: Explain nodes button
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'nodes'),
        "This signifies your nodes channel has data.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'nodes', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'nodes', 'toggle()')
    )

    # Step 8: Explain nodes button
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'edges'),
        "This signifies your edges channel has data.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'edges', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'edges', 'toggle()')
    )

    # Step 8: Explain nodes button
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'network_overlay'),
        "This signifies your overlay channel 1 has data. (and same with the second overlay below)",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'network_overlay', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'network_overlay', 'toggle()')
    )

    # Step 8: Explain nodes button
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'network'),
        "This signifies your network has been calculated, or instead loaded into the current session",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'network', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'network', 'toggle()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'node_identities'),
        "This signifies your nodes have 'identities' associated with them, such as 'T-Cell'. Nodes with identities can be used to analyze how different types of nodes aggregate.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'node_identities', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'node_identities', 'toggle()')
    )

    tutorial.add_step(
        None,
        "Other properties include centroids for your nodes and edges, as well as community/neighborhood groupings for your nodes.",
        message_position="beside"
        )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'properties_dialog', 'report_button'),
        "Click on the report button to view a summary of these other properties in the upper right table",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'report_button', 'click()')
        )



    
    
    # Step 9: Close dialog and finish
    def close_dialog():
        if hasattr(tutorial, 'properties_dialog') and tutorial.properties_dialog:
            tutorial.properties_dialog.close()
            tutorial.properties_dialog = None

    tutorial.add_step(
        None,
        "That's it for the Intro tutorial! Select the Basic Interface Tour next to see how to use the main GUI elements.",
        message_position="bottom",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'properties_dialog', 'xy_scale', 'close()'),
        action=close_dialog
    )

    return tutorial


def setup_basics_tutorial(window):
    """
    Set up the basic interface tutorial for NetTracer3D
    
    Args:
        window: ImageViewerWindow instance from nettracer_gui
    
    Returns:
        TutorialManager instance
    """
    tutorial = TutorialManager(window)
    
    # Step 1: Welcome
    tutorial.add_step(
        window.canvas,
        "This tutorial will guide you through the main features of the GUI window. Click 'Next' or use 'Right-Click' to continue.",
        highlight_type="rect",
        message_position="bottom"
    )

    # Step 2: Canvas explanation
    tutorial.add_step(
        window.canvas,
        "This canvas is where your loaded images will render.",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "Clicking a node or edge in this canvas will select it (if the nodes or edges channels are set as the 'active channel', respectively). Click and drag to select multiple objects. This is intended mainly for segmented, labeled data rather than interacting directly with raw images.",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "Selected objects will be highlighted yellow and can be used for certain functions. Clicking a background val in an image (ie voxel with value 0) will deselect your objects.",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        window.canvas,
        "Use right click to interact with highlighted objects (ie, delete them or merge them into one object); or rather to select objects algorithmically (for example, the neighbors of a node in your network)",
        highlight_type="rect",
        message_position="bottom"
    )

    # Active Channel Selector
    tutorial.add_step(
        window.active_channel_combo,
        "This dropdown lets you select which image channel is active for operations. Many of the 'process' functions will execute on the 'active image', instead of you having to select one each time. You can choose between Nodes, Edges, and either of the Overlay channels. To select nodes or edges when clicking on the canvas, you will also need to have them as the active image here.",
        highlight_type="rect",
        message_position="top"
    )

    # Scale Button
    tutorial.add_step(
        window.toggle_scale,
        "Click this to toggle a scale bar that shows distances in the canvas.",
        highlight_type="circle",
        message_position="top"
    )
    
    # Home/Reset View Button
    tutorial.add_step(
        window.reset_view,
        "Click this Home button to reset your view to the original zoom and position.",
        highlight_type="circle",
        message_position="top"
    )
    
    # Zoom Tool
    tutorial.add_step(
        window.zoom_button,
        "(Shortcut Z) Use the Zoom tool to zoom into specific areas of your image. Left click to zoom in, right click to zoom out. Click and drag to draw a rectangle and zoom into a specific area of your image.",
        highlight_type="circle",
        message_position="top"
    )
    
    # Pan Tool
    tutorial.add_step(
        window.pan_button,
        "(Shortcut middle mouse) The Pan tool lets you click and drag to move around your image when zoomed in.",
        highlight_type="circle",
        message_position="top"
    )
    
    # Highlight Toggle
    tutorial.add_step(
        window.high_button,
        "(Shortcut X) Toggle this to show/hide highlighting of selected nodes and edges in your network.",
        highlight_type="circle",
        message_position="top"
    )
    
    # Pen/Brush Tool
    tutorial.add_step(
        window.pen_button,
        "The Pen tool allows you to manually paint write foreground regions into your data. Note this modifies data directly and is for making minor corrections of segmented data. Ctrl + Mousewheel will change the brush size.",
        highlight_type="circle",
        message_position="top"
    )

    # Pen/Brush Tool
    tutorial.add_step(
        window.pen_button,
        "Pressing 'D' in the pen tool will enable a pen that writes into multiple image planes at once. In this mode, the mousewheel will change how many planes are being written into. Pressing 'F' will enable a fill can.",
        highlight_type="circle",
        message_position="top"
    )
    
    # Threshold Tool
    tutorial.add_step(
        window.thresh_button,
        "Use this to open the threshold dialog to segment your images by intensity or volume; or instead to open the Machine Learning segmenter.",
        highlight_type="circle",
        message_position="top"
    )
    
    # Channel Buttons
    tutorial.add_step(
        window.channel_buttons[0],
        "These channel buttons let you toggle the visibility of different image layers. The '×' button next to each channel allows you to delete that channel's data.",
        highlight_type="rect",
        message_position="top"
    )

    tutorial.add_step(
        window.channel_buttons[0],
        "The Nodes Channel is where you will typically load data that you want to convert into a network.",
        highlight_type="rect",
        message_position="top"
    )

    tutorial.add_step(
        window.channel_buttons[1],
        "The Edges Channel is where you can load data for a secondary structure that you want to connect your nodes. For example, this might be an image showing blood vessels. However, this is not the only way to connect your nodes. Furthermore, either the nodes or the edges channels can be used as generic overlays for visualization or direct analysis if you are not particularly interested in using them for networks.",
        highlight_type="rect",
        message_position="top"
    )

    tutorial.add_step(
        window.channel_buttons[2],
        "These Overlay Channels will show informative overlay outputs that NetTracer3D generates. They can also be loaded to directly to visualize multiple channels stacked together.",
        highlight_type="rect",
        message_position="top"
    )

    tutorial.add_step(
        window.channel_buttons[3],
        "These Overlay Channels will show informative overlay outputs that NetTracer3D generates. They can also be loaded to directly to visualize multiple channels stacked together.",
        highlight_type="rect",
        message_position="top"
    )
    
    # Slice Slider
    tutorial.add_step(
        window.slice_slider,
        "(Shortcut - 'Shift + Mouse Wheel' or 'Ctrl + Shift + Mouse Wheel'). Use this slider to navigate through different Z-slices (depth) of your 3D image stack. The arrow buttons allow continuous scrolling.",
        highlight_type="rect",
        message_position="top"
    )
    
    # Data Tables
    tutorial.add_step(
        window.network_button,
        "Switch between Network and Selection views to see different data tables. The Network table shows all nodes/edges, while Selection shows only selected items.",
        highlight_type="rect",
        message_position="top"
    )
    
    # The actual table
    tutorial.add_step(
        window.network_table,
        "This table displays your network data. You can click rows to highlight corresponding elements in the image, and sort columns by clicking headers. Right click to export any tables in spreadsheet format, or in a format for a few other types of network analysis software.",
        highlight_type="rect",
        message_position="left"
    )

    tutorial.add_step(
        window.tabbed_data,
        "This table displays outputs of most of the quantifications NetTracer3D runs. Right click to export these as a spreadsheets, to close all tables, or to use a table to threshold your nodes (possible with any two column table quantifying the nodes with some parameter).",
        highlight_type="rect",
        message_position="left"
    )

    tutorial.add_step(
        window.load_button,
        "Use this to open a spreadsheet back into the upper right tables. You would mainly do this if you wanted to use it to threshold your nodes.",
        highlight_type="circle",
        message_position="bottom"
    )

    tutorial.add_step(
        window.cam_button,
        "This button can be used to take a screenshot of what you currently see in your main canvas.",
        highlight_type="circle",
        message_position="bottom"
    )

    tutorial.add_step(
        window.popup_button,
        "This button can be used to eject the canvas from the main window, to make it larger. Just click back in the main window to return it.",
        highlight_type="circle",
        message_position="bottom"
    )

    # File Menu - show where to load data
    tutorial.add_step(
        lambda: window.menuBar(),
        "The File menu contains options to load images, save your work, and export data. Start by loading a TIF image stack to begin analyzing your network.",
        highlight_type="rect",
        message_position="bottom",
        pre_action=lambda: MenuHelper.open_menu(window, "File"),
        action=lambda: MenuHelper.close_menu(window, "File")
    )

    tutorial.add_step(
        lambda: window.menuBar(),
        "The Analyze menu contains options to quantify your segmented data and networks.",
        highlight_type="rect",
        message_position="bottom",
        pre_action=lambda: MenuHelper.open_menu(window, "Analyze"),
        action=lambda: MenuHelper.close_menu(window, "Analyze")
    )

    tutorial.add_step(
        lambda: window.menuBar(),
        "The Process menu contains functions to actually create your networks, and to alter your segmented data, such as via watershedding (seperate fused objects), improve your segmentations, labeling branches, and more.",
        highlight_type="rect",
        message_position="bottom",
        pre_action=lambda: MenuHelper.open_menu(window, "Process"),
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    tutorial.add_step(
        lambda: window.menuBar(),
        "The Image menu contains ways to alter the visualization, such as changing channel color, brightness, creating informative overlays, and showing a 3D renders.",
        highlight_type="rect",
        message_position="bottom",
        pre_action=lambda: MenuHelper.open_menu(window, "Image"),
        action=lambda: MenuHelper.close_menu(window, "Image")
    )

    tutorial.add_step(
        lambda: window.menuBar(),
        "The Help menu can be used to access the documentation, which contains an in-depth description of every available function. It can also be used to access the tutorial.",
        highlight_type="rect",
        message_position="bottom",
        pre_action=lambda: MenuHelper.open_menu(window, "Help"),
        action=lambda: MenuHelper.close_menu(window, "Help")
    )
    
    # Completion
    tutorial.add_step(
        None,
        "That's it! You're ready to use NetTracer3D. Load an image from the File menu to get started, then use the tools to analyze your 3D network structures.",
        message_position="bottom"
    )
    
    return tutorial

def setup_file_tutorial(window):
    """
    Set up the basic interface tutorial for NetTracer3D
    
    Args:
        window: ImageViewerWindow instance from nettracer_gui
    
    Returns:
        TutorialManager instance
    """
    tutorial = TutorialManager(window)

    # Step 1: Welcome
    tutorial.add_step(
        None,
        "This tutorial will guide you through saving and loading data.",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "File"),
        "The File menu contains the options for saving/loading your data. This includes all images and properties.",
        highlight_type="rect",
        message_position="beside",
        pre_action=lambda: MenuHelper.open_menu(window, "File"),
        action=lambda: MenuHelper.close_menu(window, "File")
    )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "File")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Save As"))

    # Step 3: Point to Image submenu
    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "File", "Save As", "Save Nodes As"),
        f"""--Saving occurs from the SaveAs menu.
        \n\n--Use 'Save Network3D Object As' as the primary save function. This will dump all the relevant properties. First, you will be prompted to select a folder on your computer. Next, you will enter the name of a new folder to create in the aforementioned parent folder. All the outputs will be saved to this new folder.
        \n\n--The other SaveAs options can be used to save any of the image channels as a .tif.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    def open_to_load():
        menu = MenuHelper.open_menu(window, "File")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Load"))
    
    # Step 3: Point to Image submenu
    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "File", "Load", "Load Network3D Object"),
        f"""--Loading occurs from the load menu. Acceptable image types are .tif, .tiff, .nii, .png, .jpeg, and .jpg.
        \n\n--'Load Network3D Object' can be used to load in an entire previously saved session, assuming it had been saved with the correspoinding 'Save Network3D Object' method. Navigate your way to the directory the 'Network3D Object' dumped to. Select it to reload all properties within.
        \n\n--Use 'load nodes' to load an image into the nodes channel. Similarly, use load edges to load edges, and either of the load overlays to load the overlays.
        \n\n--Use 'load network' to load your saved network data from .csv or .xlsx format. Note this will expect to see the corresponding spreadsheet in the layout that NetTracer3D saves it.
        \n\n--'Loading from the excel helper' opens a secondary gui where (mainly node identities) can be reassigned with a set of string keywords. For example, a node with identity 'x' and 'y' can be configured to be loaded as 'identity xy'
        \n\n--'Load misc properties' can be used to load node identities, node centroids, edge centroids, or node communities directly from a .csv or .xlsx spreadsheet, expecting the format that NetTracer3D saves these properties in.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_load,
        )

    def open_to_merge():
        menu = MenuHelper.open_menu(window, "File")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Images -> Node Identities"))

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "File", "Images -> Node Identities", "Merge Labeled Images Into Nodes"),
        f"""--This 'Images -> Node Identities' menu will be a primary way to assign identities from nodes you are trying to load in.
        \n\n--The option 'Merge Labeled Images Into Nodes' will prompt you to find another .tif or a folder of .tif images containing additional segmented nodes. These nodes will be merged with your current nodes image, assigned an IDs based on the file names themselves. Use this to evaluate multiple pre-segmented channels as nodes.
        \n\n--The option 'Assign Node Identities From Overlap With Other Images' is specifically designed for assigning cell identities in multiplex images (ie has many channels) based on a single cell segmentation (usually nuclei).
        """,
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_merge,
        )

    tutorial.add_step(
        None,
        "In short, load your segmented cells/nuclei into the nodes channel. Arrange the rest of your RAW channels of interest into a seperate folder. You will be prompted to find this folder.",
        highlight_type=None,
        message_position="top_right"
        )

    tutorial.add_step(
        None,
        "Next, for each channel in the folder, you will be asked to assign intensity threshold boundaries where the segmented cells are assigned an identity based on whether their average intensity of expression falls within the assigned bounds.",
        highlight_type=None,
        message_position="top_right"
        )

    tutorial.add_step(
        None,
        "This can be used to rapidly assign differential identities to your cells. Note that all the channels should have the same shape.",
        highlight_type=None,
        message_position="top_right"
        )

    tutorial.add_step(
        None,
        "When you're done with the identity assignments, you will be prompted to save a resultant intensity table containing average intensity of each channel for each cell.",
        highlight_type=None,
        message_position="top_right"
        )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_violin_dialog", "ViolinDialog", "violin_dialog",
        called=True
    )

    tutorial.add_step(
        None,
        "This data table can be used to access this menu (Also available from 'Analyze -> Stats -> Show Identity Violins...'), where you can generate informative violin plots about shared marker expression amongst cells of different communities or shared expression between cells defined as some identity.",
        message_position="top_right",
        pre_action=open_dialog
        )

    tutorial.add_step(
        None,
        "You can also algorithmically cluster your cells into higher order neighborhoods based on shared marker expression, for deciphering phenotypes",
        highlight_type=None,
        message_position="top_right",
        )    
    
    # Step 9: Close dialog and finish
    def close_dialog():
        if hasattr(tutorial, 'violin_dialog') and tutorial.violin_dialog:
            tutorial.violin_dialog.close()
            tutorial.violin_dialog = None

    tutorial.add_step(
        None,
        "That's it for the data loading tutorial!",
        message_position="top_right",
        action=close_dialog
    )



    return tutorial

def setup_connectivity_tutorial(window):

    tutorial = TutorialManager(window)

    tutorial.add_step(
        None,
        "This tutorial will guide you through generating the first sort of network, the 'connectivity network'. These networks should be used to connect one type of object (your nodes), which can be for example cells or functional tissue units, via a secondary structure (your edges), which should be some kind of connector medium, such as nerves or blood vessels.",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        window.channel_buttons[0],
        "Start by loading your data into the nodes channel (binary or numerically labeled). Make sure to segment it first if you haven't.",
        highlight_type="circle",
        message_position="top"
    )

    tutorial.add_step(
        window.channel_buttons[1],
        "Also load your segmented edges into the edges channel (should be binary).",
        highlight_type="circle",
        message_position="top"
    )

    def open_to_connect():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Calculate Network"))
    
    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Calculate Network", "Calculate Connectivity Network (Find Node-Edge-Node Network)"),
        "Next, select 'Process -> Calculate Network -> Calculate Connectivity Network'.",
        highlight_type=None,
        message_position="beside",
        pre_action=open_to_connect,
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_calc_all_dialog", "CalcAllDialog", "con_dialog"
    )
    
    tutorial.add_step(
        None,
        "Let's open the connectivity network calculator to see all the options available. Click 'Next' to open it.",
        message_position="bottom",
        action=open_dialog
    )
    
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'xy_scale'),
        "xy_scale affects how NetTracer3D interprets distances in the X and Y dimensions. If your image has anisotropic voxels (different spacing in X/Y vs Z), you may need to adjust this to compensate. Note that your data is always presumed to have an equal resolution in the xy plane itself.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'xy_scale', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'xy_scale', 'deselect()')
    )
    
    # Step 7: Explain z_scale field
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'z_scale'),
        "z_scale adjusts the evaluation of distances in the Z dimension. Many microscopy images have a different Z step size than XY resolution, so you might set this differently than xy_scale.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'z_scale', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'z_scale', 'deselect()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'search'),
        "Node search can expand your nodes by a set distance to look for connections nearby. Otherwise, they will just consider what edges pass directly through them. Note this distance is affected by the xy and z scales.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'search', 'setText("FLOAT!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'search', 'setText("")')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'diledge'),
        "Edge Search similarly expands your edges by the entered distance. While nodes that expand keep their identity when they push up into each other, expanding your edges will actually fuse them. This is intended to account for segmentation artifacts, such as small holes. Edges must be contiguous in space to connect a node pair. Note you can preprocess your edges via dilating or closing to skip this, or ignore it entirely.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'diledge', 'setText("FLOAT!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'diledge', 'setText("")')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'label_nodes'),
        "Having this option on will have the system use a basic labeling algorithm to label your nodes, where each discrete object in space takes on a seperate integer label. This is for binary data mainly. If your nodes are already labeled, please toggle this option off!",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'label_nodes', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'label_nodes', 'toggle()'))

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'remove_trunk'),
        "'Time to Remove Edge Trunks' can be given an integer which will tell the system to remove that many trunks from the edges before connecting your nodes. A trunk is a large confluence of edges, such as a large nerve from which all other nerves arise. Sometimes this will result in most of your nodes being connected by the trunk alone and result in a hard-to-analyze network. If 1 is given here, for example, the single largest volume edge will be removed before making the network. If 2 is given, the single largest two edges are removed, etc. I would recommend skipping this at first, then coming back and trying 1 if there is a large trunk causing problems.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'remove_trunk', 'setText("INTEGER!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'remove_trunk', 'setText("")')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'inners'),
        "Deselecting this button will have the system not consider 'inner edges'. Inner edges are portions of your edge image that exist solely within nodes (as well as their expanded search regions). You can deselect this to ignore inner connections between within node clusters, for example if you only wanted to consider more distal connections to get a simpler network. However, I would recommend keeping this enabled unless you had a good reason to not.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'inners', 'click()'))

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'down_factor'),
        "Enter an int here to downsample your nodes prior to finding their centroids. The resultant centroids will be scaled back up to their proper values. This can speed up the centroid calculation and is recommended for large images. Note that small nodes may be completely erased if the downsample is too large. A larger int equates to a greater downsample. Downsampling here will also enlarge any overlays generated in this window.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'down_factor', 'setText("INTEGER!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'down_factor', 'setText("")')
    )

    """ # <-- so I am trying out removing these because their use cases are confusing

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'GPU_downsample'),
        "If you want to try and use the GPU, you can likewise enter an arbitrary integer downsample factor here to speed it up. Note the GPU calculation can be greedy with VRAM and will automatically try to downsample itself in a lot of cases."
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'GPU_downsample', 'setText("INTEGER!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'GPU_downsample', 'setText("")')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'GPU'),
        "Enable this to have your system attempt to use the GPU. You will need a CUDA toolkit and a corresponding cupy package installed. Note that I consider this function somewhat experimental. In short, the cupy implementation uses a distance transform calculation that can be very greedy with VRAM. If it overflows, it will attempt to iteratively downsample itself until the calculation works (specifically containing to calculating the 'node search' volume). Note this risks kicking out small nodes from your image. Furthermore, it is only really applicable of 'fast dilation' is enabled. Therefore, generally skip using this option. However, it can be a way to rapidly assess the general network structure of a large image.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'GPU', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'GPU', 'toggle()'))

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'fastdil'),
        "Enable this to have the algorithm use fast dilation. Fast dilation "
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'fastdil', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'fastdil', 'toggle()'))
    """

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'overlays'),
        "Selecting 'Overlays' will have the system also generate a 'network overlay' which literally draws white lines into an image between your nodes. This will be loaded into Overlay 1. It will also generate an 'ID overlay', which draws the integer labels of nodes directly at their centroids. This will be loaded into Overlay 2. These overlays can also be generated after the fact",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'overlays', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'overlays', 'toggle()'))


    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'con_dialog', 'update'),
        "This is enabled by default and will just make it so your 'nodes' are replaced by the 'labeled nodes' if you are labelling them. It will also replace your 'edges' with the labeled edges that this function always generates and utilizes to make connections. Note that you should generally enable this and just save both sets of images. Selecting objects with the highlight overlay and having them correspond to the actual calculated network will only work if you also make sure these images were updated to correspond.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'update', 'click()'))

    def close_dialog():
        if hasattr(tutorial, 'con_dialog') and tutorial.con_dialog:
            tutorial.con_dialog.close()
            tutorial.con_dialog = None

    tutorial.add_step(
        None,
        "That's it for the connectivity network creation!",
        message_position="bottom",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'con_dialog', 'xy_scale', 'close()'),
        action=close_dialog
    )


    return tutorial

def setup_branch_tutorial(window):
    tutorial = TutorialManager(window)

    tutorial.add_step(
        None,
        "This tutorial will guide you through generating the second sort of network, the 'branch network'. These networks should be used to create branch graphs of segmentations of branchy images, such as nerves or vessels.",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        window.channel_buttons[1],
        "First, load your segmented branch image into the edges channel. Make sure it is segmented, not raw data.",
        highlight_type="circle",
        message_position="top"
    )

    tutorial.add_step(
        window.channel_buttons[0],
        "You can also load to the nodes channel first, but the program will prioritize whatever is in edges for any branch analysis.",
        highlight_type="circle",
        message_position="top"
    )

    def open_to_connect():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Calculate Network"))
    
    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Calculate Network", "Calculate Connectivity Network (Find Node-Edge-Node Network)"),
        "--There are two options for making branch networks. \n\n --1. ('Process -> Calculate Network -> Calculate Branchpoint Network') is 'branchpoint networks', where the vertices of your branches are joined in a network\n\n--2. (Process -> Calculate Network -> Calculate Branch Adjacency Network) is 'branch adjacency networks', where the branches themselves are connected based on which other branches they happen to touch.",
        highlight_type=None,
        message_position="beside",
        pre_action=open_to_connect,
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Calculate Network", "Calculate Connectivity Network (Find Node-Edge-Node Network)"),
        "We will start with the 'branch adjacency network' for this demo.",
        highlight_type=None,
        message_position="beside",
        pre_action=open_to_connect,
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_branch_dialog", "BranchDialog", "branch_dialog",
        tutorial_example = True
    )
    
    tutorial.add_step(
        None,
        "This menu will appear when calculating the 'branch adjacency network'. It is the same menu that you'll get when just trying to label branches with 'Process -> Generate -> Label Branches'. For the most part the parameters are set to the recommended defaults, however I will go over what options are available.",
        message_position="beside",
        pre_action=open_dialog
    )

    """
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'fix'),
        "This first auto-correction option is designed if you feel like the branch labels are generally too busy. Selecting this will have the program attempt to collapse overly-dense regions of branches into a single label. Note that this behavior is somewhat tricky to predict so I generally don't use it but feel free to give it a shot and see how it looks.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix', 'click()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix', 'toggle()')
    )
    
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'fix_val'),
        "This integer value tells the above parameter (if enabled) what degree of branch-busyness should get merged. In short, a lower value is more aggressive with merging while a higher value only merges very busy regions. By default it is set to 4.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix_val', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix_val', 'deselect()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'seed'),
        "The random seed for grouping branches above can be changed here with an integer value, if the behavior of the above option is desired to be tweaked somewhat. It will use 42 by default.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'seed', 'setText("INTEGER!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'seed', 'setText("")')
    )

    """

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'fix2'),
        "The second auto-correction option will automatically merge any internal labels that arise with their outer-neighbors. This is something that can occasionally happen with fat, trunk-like branches that are tricky to algorithmically decipher. I have found that this merge handles these issues quite well, so this option is enabled by default.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix2', 'click()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'fix3'),
        "This auto-correction step will automatically correct any branches that aren't contiguous in space. Rarely (Depending on the segmentation, really) a branch can initially be labeled non-contiguously, which is usually not correct. This is because the 'meat' of any branch is at first labeled based on which internal filament it's closest to. So if you have a very wide branch it may rarely aquire labels of nearby smaller branches across gaps. Enabling this will split those labels into seperate regions as to not confound the connectivity graph. The largest component is considered the 'correct one' and keeps its label, while smaller components inherit the label of the largest shared border of a 'real' branch they are bordering. It is enabled here by default to mitigate any potential errors, although note this does not apply to the branchpoint networks since they don't actually utilize the branches themselves.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix3', 'click()')
        )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'fix4'),
        "This final auto-correction step will try to automatically merge any similarly sized branches moving in the same direction, instead of just letting a larger branch with many sub-branches get chopped up. It is off by default because of its less predictable behavior, although its good if you want your branches to be more continuous. Just note each of these fixes does add extra processing time.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix4', 'click()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix4', 'toggle()')
        )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'fix4_val'),
        "This threshold values controls how likely a junction is to merge any pair of its nearby branches. Regardless of what you enter here, only two branches at a time can merge at a junction. Values between 20-40 are more meaningful, while those lower tend to merge everything and those higher usually emrge nothing.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix4_val', 'setText("FLOAT!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'fix4_val', 'setText("")')
        )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'down_factor'),
        "This integer value can be used to temporarily downsample the image while creating branches. Aside from speeding up the process, this may actually improve branch-labeling behavior with thick branches but will lose labeling for smaller branches (instead merging them with nearby thicker branches they arise from). It is disabled by default. Larger values will downsample more aggressively.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'down_factor', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'down_factor', 'deselect()')
    )


    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'compute'),
        "Leaving this option enabled will have the program also compute branch lengths and branch tortuosities for the labeled branches. These computations are pretty cheap so I usually leave this enabled. This will get skipped if you use a downsample but you can calculate these again from the Analyze menu.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'compute', 'click()')
    )


    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'branch_dialog', 'nodes'),
        "If you have already created node vertices for your branch labeling scheme (ie via 'Process -> Generate -> Generate Nodes from Edge Vertices) and have loaded those nodes into the nodes channel (with your branches in the edges channel) you can forgo regenerating these vertices by disabling this. However, it is usually presumed you will be generating them from scratch, so this is enabled by default'.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'nodes', 'click()')
    )

    def close_dialog():
        if hasattr(tutorial, 'branch_dialog') and tutorial.branch_dialog:
            tutorial.branch_dialog.close()
            tutorial.branch_dialog = None

    tutorial.add_step(
        None,
        "Press 'Run Branch Label' to move on. This will move you to the step to generate nodes for your branch vertices.",
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'branch_dialog', 'down_factor', 'close()'),
        action=close_dialog
    )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_gennodes_dialog", "GenNodesDialog", "gen_dialog",
        tutorial_example = True
    )

    tutorial.add_step(
        None,
        "This also happens to be the only menu that will appear if you were to have chosen to create a 'Branchpoint Network', and so the description of this menu applies to both the initial step for the 'branchpoint network' and the second step for the 'branch adjacency network'.",
        message_position="beside",
        pre_action=open_dialog
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'gen_dialog', 'down_factor'),
        "This integer value can be used to temporarily downsample the image while creating branchpoints. Aside from speeding up the process, this may alter branch detection, possibly performing a cleaner branch appraisal of very thick branches but losing network identification of smaller branches (Much like in the prior menu - note that any value entered in the prior menu will be applied by default here for consistency, and you won't see this option). It is disabled by default. Larger values will downsample more aggressively.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'down_factor', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'down_factor', 'deselect()')
    )


    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'gen_dialog', 'branch_removal'),
        "IMPORTANT - This branch removal parameter (Skeleton voxel branch to remove...) is something I would consider entering a value for. This is the length of terminal branches that will be removed prior to any vertex/branch labeling. Any branch shorter than the value here will be removed, but only if it is a terminal branch. For more jagged segmentations, this may be a necessity to prevent branchpoints from arising from spine-like artifacts. More internal branches will not be removed, so as a test it is generally safe to enter a large value here, which will preserve the majority of the branch schema and just risk losing occasional terminal branches.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'branch_removal', 'setText("INTEGER!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'branch_removal', 'setText("")')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'gen_dialog', 'auto'),
        "This 'attempt to auto correct skeleton looping' option should generally be enabled for 3D data. In short it applies an extra algorithmic step to improve the branch detection algorithm. However, this does not really apply to 2D data. It will be enabled by default for 3D data and disabled by default for 2D data.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'auto', 'click()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'auto', 'toggle()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'gen_dialog', 'comp_dil'),
        "This final 'attempt to expand nodes' will cause your nodes (branchpoint labels) to grow in size by the specified amount. They will fuse with any neighbors they encounter. Doing this will decrease the label splitting along a single branch that has many branches emerge from it in a tightly packed stretch, just as an example, because the system would instead see a single branchpoint there. This can generally be skipped, but if you notice a plethora of tightly packed vertices that you'd want to be treated as a single vertice, you could consider using it.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'comp_dil', 'setText("INTEGER!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'comp_dil', 'setText("")')
    )

    def close_dialog():
        if hasattr(tutorial, 'gen_dialog') and tutorial.gen_dialog:
            tutorial.gen_dialog.close()
            tutorial.gen_dialog = None

    tutorial.add_step(
        None,
        "That's it for creating branch networks!.",
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'gen_dialog', 'down_factor', 'close()'),
        action=close_dialog
    )


    return tutorial

def setup_prox_tutorial(window):

    tutorial = TutorialManager(window)

    tutorial.add_step(
        None,
        "This tutorial will guide you through generating the third sort of network, the 'proximity network'. These networks should be used to evaluate spatial clustering and general arrangement of objects in an image, for example, just looking at groups of cells and what subtypes group together.",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        window.channel_buttons[0],
        "First, load your segmented objects into the nodes channel. Alternatively, you can load in the data for just the node centroids, either through 'File -> Load' or 'File -> Load From Excel Helper'",
        highlight_type="circle",
        message_position="top"
    )

    def open_to_connect():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Calculate Network"))
    
    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Calculate Network", "Calculate Proximity Network (connect nodes by distance)"),
        "From the calculate menu, select 'Calculate Proximity Network...'",
        highlight_type=None,
        message_position="beside",
        pre_action=open_to_connect,
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_calc_prox_dialog", "ProxDialog", "prox_dialog",
        tutorial_example = True
    )
    
    tutorial.add_step(
        None,
        "You will then see this menu. Let's walk through what options are available.",
        message_position="beside",
        pre_action=open_dialog
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'search'),
        "The search region value will tell the program how close you want a pair of nodes to be before they are connected. You must provide a value here to use this function.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'search', 'setText("FLOAT!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'search', 'setText("")')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'xy_scale'),
        "xy_scale affects how NetTracer3D interprets distances in the X and Y dimensions. If your image has anisotropic voxels (different spacing in X/Y vs Z), you may need to adjust this to compensate. Note that your data is always presumed to have an equal resolution in the xy plane itself.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'xy_scale', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'xy_scale', 'deselect()')
    )
    
    # Step 7: Explain z_scale field
    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'z_scale'),
        "z_scale adjusts the evaluation of distances in the Z dimension. Many microscopy images have a different Z step size than XY resolution, so you might set this differently than xy_scale.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'z_scale', 'selectAll()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'z_scale', 'deselect()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'mode_selector'),
        "--Execution Mode tells the program if you want to link nodes by comparing the distances between their centroids or their borders.\n\n--The first option will utilize centroids, which is usually faster and good for objects that are rougly circular or spheroid, such as cells. Note the search distance will start at the centroid and only create a pair if the search encounter's another centroid, so you may need to increase that value to compensate if the borders are larger than the centroids.\n\n--The second option will search from the actual object's boundary and may be slower to process, but is ideal for more oddly shaped nodes whose location cannot be described well by a centroid. Nodes will be linked based on their boundary-to-boundary distance.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'mode_selector', 'showPopup()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'mode_selector', 'hidePopup()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'id_selector'),
        "If your nodes have been assigned identities, this menu will allow you to only use one as a basis for finding connections. So if I were to choose 'identity A' here, then any nodes bearing 'identity A' could connect to any other node type (including A, B, C, etc), but nodes of 'identity B' or 'C' could not connect to each other. This can be used to evaluate relationships around specific subtypes of objects. Note that a network exclusively between two identity types can also be created in post with 'Process -> Modify Network...'",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'id_selector', 'showPopup()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'id_selector', 'hidePopup()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'overlays'),
        "Selecting 'Overlays' will have the system also generate a 'network overlay' which literally draws white lines into an image between your nodes. This will be loaded into Overlay 1. It will also generate an 'ID overlay', which draws the integer labels of nodes directly at their centroids. This will be loaded into Overlay 2. These overlays can also be generated after the fact",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'overlays', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'overlays', 'toggle()'))

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'downsample'),
        "The downsample factor integer value will downsample when generating the overlays, which is essentially just a trick to make them render larger. This only matters if you want to generate the overlays here.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'downsample', 'setText("INT!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'downsample', 'setText("")')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'populate'),
        "Since you can skip using an actual image and just use the data in the centroids property (if nothing is in nodes) to generate this network, enabling this 'Populate Nodes From Centroids' option will cause the program to generate a new image and place the centroids in it as single, labeled points. This will be loaded into the nodes channel. The labeled points can be dilated after the fact if you'd like to make them more visible.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'populate', 'click()'),
        action = MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'populate', 'toggle()'))

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'prox_dialog', 'max_neighbors'),
        "The integer entered here will cause any node to only be able to have a maximum of that many connections. It will preferentially take connections to its closest neighbors. You can enter a cap here to simplify network structure in dense images. Alternatively, if you are using the centroid search you can enter a very large distance for your search region (note this sort of distance might slow down the border search substantially) and then pass a value here as a way to appraise the 'n' closest neighbors for each node.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'max_neighbors', 'setText("INT!")'),
        action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'max_neighbors', 'setText("")')
    )

    def close_dialog():
        if hasattr(tutorial, 'prox_dialog') and tutorial.prox_dialog:
            tutorial.prox_dialog.close()
            tutorial.prox_dialog = None

    tutorial.add_step(
        None,
        "That's it for creating proximity networks!.",
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'prox_dialog', 'downsample', 'close()'),
        action=close_dialog
    )



    return tutorial

def setup_seg_tutorial(window):

    tutorial = TutorialManager(window)

    tutorial.add_step(
        None,
        "This tutorial will guide you through options for segmenting your data within NetTracer3D",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        None,
        "Alternatively, you can segment your data with a seperate software and bring it to NetTracer3D. Prepare your segmented data into a .tif format, where the image has been reduced to binary (ie 1 or 255 for the foreground, 0 for the background), or where each discrete object has its own numerical label (ie 1, 2, 3, etc).",
        highlight_type="rect",
        message_position="bottom"
    )

    tutorial.add_step(
        window.thresh_button,
        "Some of the main segmentation options are accessible by clicking this widget.",
        highlight_type="circle",
        message_position="top"
    )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_thresh_dialog", "ThresholdDialog", "thresh_dialog",
        tutorial_example = True
    )
    
    tutorial.add_step(
        None,
        "You will then see this menu. Let's walk through what options are available.",
        message_position="beside",
        pre_action=open_dialog
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'thresh_dialog', 'mode_selector'),
        "This menu shows some default thresholding options. Thresholding by label/brightness is an easy way to produce a segmentation, although it will only work for regions that are brighter than the background. Thresholding by volume can be used to remove noise after a segmentation is produced.",
        highlight_type=None,
        message_position="beside",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'thresh_dialog', 'mode_selector', 'showPopup()'),
        action=MenuHelper.create_widget_interaction(tutorial, 'thresh_dialog', 'mode_selector', 'hidePopup()')
    )

    tutorial.add_step(
        window.active_channel_combo,
        "Choose 'select' to open the thresholding histogram. This thresholding will execute on whatever the active channel is",
        highlight_type="rect",
        message_position="top"
    )

    def close_dialog():
        if hasattr(tutorial, 'thresh_dialog') and tutorial.thresh_dialog:
            tutorial.thresh_dialog.close()
            tutorial.thresh_dialog = None

    tutorial.add_step(
        None,
        "For data with less signal-to-noise, you can try to use the 'machine learning' segmenter.",
        message_position="beside",
        action = close_dialog
    )

    tutorial.add_step(
        window.channel_buttons[0],
        "First, load the data you'd like to segment into the nodes channel. Then, click 'machine learning' from the threshold menu.",
        highlight_type="circle",
        message_position="top"
    )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_machine_window_tutorial", "MachineWindow", "machine_window"
    )

    tutorial.add_step(
        None,
        "This window will then appear and is used to control the machine learning segmenter.",
        message_position="top_left",
        pre_action=open_dialog
    )

    tutorial.add_step(
        window.channel_buttons[2],
        "Note that at the moment the segmenter will use this overlay channel to store training data. You can directly save and reload this overlay if you'd like to save your training data and use it again later.",
        highlight_type="circle",
        message_position="top_left"
    )

    tutorial.add_step(
        window.high_button,
        "It will also use the highlight overlay to render your segmentation preview. As a result, make sure you have enough RAM to accomodate these additional arrays. In general, I would advise against segmenting something over 5 GB with this, so please downsample any data larger than that with 'Process -> Image -> Resize'.",
        highlight_type="circle",
        message_position="top_left"
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'machine_window', 'brush_button'),
        "Use this button to access the brush. Use left click to draw markings for the segmenter, and right click to erase them. Ctrl + Mousewheel can increase the size of the brush.",
        highlight_type=None,
        message_position="top_left",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'machine_window', 'brush_button', 'click()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'machine_window', 'brush_button'),
        "You will use the brush to mark the 'foreground' of the image (what you'd like to keep) and the 'background' (what you'd like to remove).",
        highlight_type=None,
        message_position="top_left",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'machine_window', 'brush_button', 'click()')
    )

    tutorial.add_step(
        window.canvas,
        "You will mark these regions directly on the canvas.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        window.canvas,
        "The program will use your markings to train itself. When you train a model, it will learn to segment out regions that look like those you marked as foreground, while ignoring regions that you marked as background.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'machine_window', 'fore_button'),
        "You can toggle whether you're labeling foreground or background with the foreground or background buttons here.",
        highlight_type=None,
        message_position="top_left",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'machine_window', 'fore_button', 'click()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'machine_window', 'GPU'),
        "Click the GPU button to use the GPU, available if you set up a CUDA toolkit and installed the corresponding cupy package.",
        highlight_type=None,
        message_position="top_left",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'machine_window', 'GPU', 'click()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'machine_window', 'two'),
        "Selecting 'Train by 2D Slice Patterns' will have the program only consider two dimensional patterns around your marked regions when learning to segment. This is faster but does not consider 3D structures",
        highlight_type=None,
        message_position="top_left",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'machine_window', 'two', 'click()')
    )

    tutorial.add_step(
        MenuHelper.create_widget_getter(tutorial, 'machine_window', 'three'),
        "Selecting 'Train by 3D Slice Patterns' will likewise consider 3D patterns, but is slower.",
        highlight_type=None,
        message_position="top_left",
        pre_action=MenuHelper.create_widget_interaction(tutorial, 'machine_window', 'three', 'click()')
    )

    tutorial.add_step(
        None,
        "When you've marked your data a bit, you can select 'Train Quick Model' or Train Detailed Model' to train the segmenter. The quick model is both faster and lighter and a good option for when the signal-to-noise is decent. It also doesn't require as much training.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "The 'Detailed Model' considers structure more than signal and is an option for less distinct data. It will be slower and likely require more training data than the quick model.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "Select 'Preview Segment' to make the segmenter start segmenting the image without interrupting the training session. Its preview segmentation will begin to render in the highlight overlay. During this period, you should observe how it's segmenting and correct it if it's making mistakes.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "At some point you'll reach some kind of ceiling where additional training data won't really help more. This might take 20-40 minutes of training. So try to end the training session by then, or earlier if it looks satisfactory.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "The button with ▶/⏸️ can be pressed to pause the segmentation preview, or to start it again.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "Select 'segment all' to have the program calculate the segmentation for the entire image. This will freeze the NetTracer3D window until it's done.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        window.channel_buttons[3],
        "The finished binary segmentation will be placed here. Make sure to save it with 'File -> Save As...'.",
        highlight_type="circle",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "If you'd like to reuse a model you trained, select 'Save Model' to save it to your computer somewhere.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "Likewise, 'Load Model' can be used to reopen a saved model. You can train over an old model to have it combine all the training data, although note the model might become more ponderous the more you train over it.",
        highlight_type="rect",
        message_position="top_left"
    )

    tutorial.add_step(
        None,
        "Select 'Load Image' from the segmenter window to load a new image into the nodes to segment. IMPORTANT - This option allows you to segment color images, such as an H&E slide. The 'nodes' channel will not let you load color images otherwise.",
        highlight_type="rect",
        message_position="top_left"
    )

    def close_dialog():
        if hasattr(tutorial, 'machine_window') and tutorial.machine_window:
            tutorial.machine_window.close()
            tutorial.machine_window = None

    tutorial.add_step(
        None,
        "This is a way to produce binary segmentations, however for cellular data, we may want to instead have labeled data.",
        message_position="top_right",
        pre_action=close_dialog
    )

    def open_to_connect():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Image"))
    
    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Image", "Binary Watershed"),
        "--Watershedding can be used to split a binary image into labeled components, or to directly label an image\n\n--From the displayed menu, 'Binary Watershed' can be used to split apart fused components of your binary segmentation, assuming they are distinct enough.\n\n--'Gray Watershed' can be used to direcly label objects like cell nuclei from a raw image, provided they have distinct peaks of intensity in the image. Note both of these methods can be prickly about their default parameters so may require some testing on your specific dataset. Please reference the documentation for more info.'",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_connect,
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    def open_to_connect():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Generate"))

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Generate", "Trace Filaments"),
        "--Trace Filaments can be used to try to automatically trace a cleaner segmentation of a rough segmentation of filamental objects, like nerves or vessels. This can be used to improve a segmentation either generated by rote intensity thresholding, or by the ML segmenter. For alternative ways to remove noise, use morphological calculations (Analyze -> Stats -> Morphological) to characterize objects in the image, then place your segmentation in the 'Nodes' channel and threshold them from the upper-right table. Alternatively, you can select and manually delete noise from the canvas by clicking it, then right clicking and choosing 'Selection -> Delete Object'.'",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_connect,
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    tutorial.add_step(
        lambda: window.menuBar(),
        "Another option is to use other software. One that's useful for segmenting cells is Cellpose. If you package cellpose with NetTracer3D, you can actually open it from here, although it generally requires you to have a decent GPU.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Image"),
        action=lambda: MenuHelper.close_menu(window, "Image")
    )


    return tutorial

def setup_analysis_tutorial(window):

    tutorial = TutorialManager(window)
    # Step 1: Welcome
    tutorial.add_step(
        None,
        "This tutorial will guide you through the options for analysis. It will briefly describe what is available. Please reference the documentation for full information.",
        highlight_type="rect",
        message_position="top_right"
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Analyze"),
        "The Analyze menu contains the options for analyzing your data. This applies to both analysis of the networks and of the images themselves.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Analyze"),
        action=lambda: MenuHelper.close_menu(window, "Analyze")
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Analyze"),
        "The Network submenu contains options for network visualization, getting network statistics, and grouping networks into communities or neighborhoods.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Analyze"),
        action=lambda: MenuHelper.close_menu(window, "Analyze")
    )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Analyze")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Network"))

    # Step 3: Point to Image submenu
    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Network", "Show Network"),
        f"""--Use 'Show Network' to visualize your network in an interactive matplotlib graph. You can enable geo_layout to position nodes based on their 3D coordinates, and color-code nodes by community or node-ID. Note this visualization is slow with networks with lots of nodes (100k+) or edges. In those cases, render the network overlay instead, followed by the 3D visualization or z-projection. 

        \n\n--Use 'Generic Network Report' to get basic statistics about your Network 3D Object. This includes node count, edge count, nodes per identity category, and nodes per community (if assigned).

        \n\n--Use 'Community Partition + Generic Network Stats' to group nodes into communities using either Label Propagation or Louvain algorithms. This function also calculates comprehensive community statistics including modularity, clustering coefficients, and per-community metrics like density and conductance.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Network", "Show Network"),

        f"""--Use 'Calculate Composition of Network Communities (And Show UMAP)' to analyze the compositional makeup of your communities based on node identities. This function can provide per-community identity proportions or a weighted average across all communities, and can generate a UMAP to visualize compositional similarity between communities.
        \n\n--Use 'Convert Network Communities Into Neighborhoods' to group similar communities into a smaller set of neighborhoods using K-means or DBSCAN clustering. This function returns compositional heatmap graphs showing identity distributions across neighborhoods, including optional robust heatmaps that highlight overrepresented node types. Note this will reassign the 'communities' property to neighborhoods.

        \n\n--Use 'Create Communities Based on Cuboidal Proximity Cells' as an alternative spatial method for grouping nodes into communities. This splits the image into user-defined cuboidal cells and assigns nodes to communities based on whether they share a cell, independent of the network structure. You would mostly use it for images where the nodes were chaotically arranged (and so not in meaningful network communities), and you were just interested in creating neighborhoods to describe what is clustered with what.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Analyze"),
        "The Stats submenu contains options for quantifying your networks, analyzing object morphology, and comparing spatial distribution of objects.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Analyze"),
        action=lambda: MenuHelper.close_menu(window, "Analyze")
    )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Analyze")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Stats"))

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Stats", "Network Related"),

        f"""--The first submenu (Network Related) is for calculating more comprehenive stats about your network.

       \n\n --Use 'Calculate Generic Network Stats' to generate basic network statistics including node/edge counts, density, connectivity, degree metrics, and centrality averages. 

        \n\n--Use 'Network Statistics Histograms' to generate and visualize distributions of various network properties using matplotlib histograms. Options include degree distribution, centrality metrics (betweenness, closeness, eigenvector), clustering coefficients, and many others based on NetworkX functions. These stats are displayed in the tabulated data widget and can be used to directly threshold your nodes.

        \n\n--Use 'Radial Distribution Analysis' to create a graph showing the average number of neighboring nodes versus distance from any given node. This helps evaluate how far apart connected nodes are in 3D space and assess network efficiency.,
        
        \n\n--Use 'Community Cluster Heatmap' to visualize community density in 2D or 3D, with nodes colored by whether they're in higher (red) or lower (blue) density communities than expected. Can be output as a matplotlib graph or RGB overlay.""",

        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Stats", "Network Related"),

        f"""--The second submenu (Spatial) is for calculating spatial relationships between objects that are not network-dependent.

        \n\n--Use 'Identity Distribution of Neighbors' to explore what types of nodes tend to be located near or connected to nodes of a specific identity. Choose between network-based connectivity analysis or morphological density-based analysis within a search radius.

        \n\n--Use 'Ripley Clustering Analysis' to generate a Ripley's K curve that compares relative object clustering versus distance. This identifies whether nodes are clustered or dispersed and how clustering behavior varies across the image, with optional border exclusion and edge correction.

        \n\n--Use 'Average Nearest Neighbors' to analyze nearest neighbor distances, either for specific identity pairs or all nodes. Can generate heatmaps showing nodes colored by their proximity to neighbors and output quantifiable overlays with distance values.
        
        \n\n--Use 'Calculate Node < > Edge Interactions' to quantify how much edge channel image surrounds each labeled node. Can measure either volumes or lengths of adjacent edges within a specified search distance, with options to include or exclude regions inside nodes.""",

        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Stats", "Network Related"),

        f"""--The second submenu (Morphological) is for calculating morphological characteristics of objects.

        \n\n--Use 'Calculate Volumes' to find the volumes of all labeled objects in the Active Image, scaled by axis scalings and returned as a table in the tabulated data widget.

        \n\n--Use 'Calculate Radii' to find the largest radius of each labeled object in the Active Image, useful for evaluating thickness of structures like branches.

        \n\n--Use 'Calculate Surface Area' to find the approximate surface area of each labeled object in the Active Image. Note that this will slightly overestimate smooth surfaces as it just rotely counts the voxel faces, but it is fine for comparing objects in the same image.

        \n\n--Use 'Calculate Sphericities' to find the sphericities of each labeled object in the Active Image. Values closer to 1 are more spherical while those closer to 0 are less so. This also computes volumes and surface areas as described above.

        \n\n--Use 'Calculate Branch Stats' to get stats for branches that you've labeled first, including lengths and tortuosities.""",

        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Stats", "Network Related"),

        f"""--The last stats functions are as follows:

        \n\n--Use 'Significance Testing' to open a dedicated GUI for statistical testing on your data. Arrange data in Excel format, drag columns to compare, and select from various tests including t-tests, ANOVA, Mann-Whitney U, Pearson, Shapiro-Wilk, and Chi-squared tests.

        \n\n--Use 'Show Identities Violin/UMAP' to visualize normalized violin plots and UMAPs for nodes assigned identities via multiple channel markers. Displays intensity expression patterns for specific identities or communities/neighborhoods based on channel marker data. Nodes can also be grouped into neighborhoods based on shared intensity expressions across channels. This requires use of the table obtained from 'File -> Images -> Node Identities -> Assign Node Identities from Overlap with Other Images.""",

        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )


    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Analyze"),
        "The Data/Overlays submenu contains options to generate informative overlays based on different analytical outputs.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Analyze"),
        action=lambda: MenuHelper.close_menu(window, "Analyze")
    )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Analyze")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Data/Overlays"))


    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Data/Overlays", "Get Degree Information"),

        f"""--Use 'Get Degree Information' to extract and visualize node connectivity. Options include creating a data table, drawing degree values as text overlays, relabeling nodes by their degree for thresholding, or generating an RGB heatmap where high-degree nodes are red and low-degree nodes are blue. Can optionally filter to show only top proportion of high-degree nodes.

        \n\n--Use 'Get Hub Information' to identify hub nodes with the fewest degrees of separation from other nodes. Can optionally create an overlay isolating hubs and specify the proportion of most connected hubs to return (e.g., top 10%). Hubs are evaluated independently per network component.

        \n\n--Use 'Get Mother Nodes' to identify nodes that bridge connections between different communities. These nodes enable inter-community interactions. Can optionally create an overlay isolating mother nodes, which goes into the Overlay 1 channel.""",

        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Data/Overlays", "Get Degree Information"),

        f"""--Use 'Code Communities' to generate overlays showing community membership. Choose between a color-coded RGB overlay for easy visualization or a grayscale overlay labeled by community number for downstream thresholding and analysis. A legend table is also generated.

        \n\n--Use 'Code Identities' to generate overlays showing node identity membership. Choose between a color-coded RGB overlay for easy visualization or a grayscale overlay labeled by numerical identity for downstream thresholding and analysis. A legend table is also generated.

        \n\n--Use 'Centroid UMAP' to create a UMAP clustering nodes based on spatial similarity of their centroids. Nodes are colored by identity if available. This is useful for 3D data to quickly identify spatial groupings.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Analyze"),
        "The Randomize submenu contains options to randomize either the position of your nodes (just the centroids) or the connections within your network. This can serve as a way to demonstrate non-randomness, by comparing these distributions to your observed.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Analyze"),
        action=lambda: MenuHelper.close_menu(window, "Analyze")
    )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Analyze")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Randomize"))

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Analyze", "Randomize", "Scramble Nodes (Centroids)"),
        f"""--Use 'Generate Equivalent Random Network' to create a random network with the same number of edges and nodes as your current network. This is useful for comparing your network to a random control to demonstrate non-randomness. The random network is placed in the 'Selection' table where it can be saved or swapped to active. Optional weighted parameter allows edges to stack into weighted edges.

        \n\n--Use 'Scramble Nodes (Centroids)' to randomize node locations for comparison against random distributions. Choose where nodes can be repositioned: anywhere in image bounds, within dimensional bounds of current nodes, or within masked bounds of edge/overlay channels. Only centroids are randomized.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )


    return tutorial

def setup_process_tutorial(window):
    """
    Set up the image processing tutorial for NetTracer3D
    
    Args:
        window: ImageViewerWindow instance from nettracer_gui
    
    Returns:
        TutorialManager instance
    """
    tutorial = TutorialManager(window)

    # Step 1: Welcome
    tutorial.add_step(
        None,
        "This tutorial will guide you through the options for processing your data. It will briefly describe what is available. Please reference the documentation for full information.",
        highlight_type="rect",
        message_position="top_right"
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        "The Process menu contains the options for processing your data. This is where you can generate your networks, label your branches, improve segmentations, and modify the network.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Process"),
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        "The 'Calculate Network' submenu contains options for creating your networks, and also calculating the centroids for your objects. All of the network calculation options have their own detailed tutorials from the tutorial window.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Process"),
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Calculate Network"))

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Calculate Network", "Calculate Connectivity Network"),
        f"""--Use 'Calculate Connectivity Network' to connect nodes via edges in your images. Key parameters include node search distance (how far nodes look for edges), edge reconnection distance (to fill segmentation holes).

        \n\n--Use 'Calculate Proximity Network' to connect nodes based on spatial proximity within a user-defined distance. Choose between centroid-based search (faster, ideal for spherical nodes) or morphological search (slower but better for irregular shapes). Can optionally restrict connections to nearest neighbors only and create networks from specific node identities.

        \n\n--Use 'Calculate Branchpoint Network' to convert branchpoints in branchy structures (like blood vessels) into network nodes, which are then joined in a network based on which branches they border.

        \n\n--Use 'Calculate Branch Adjacency Network' to connect adjacent branches in branchy structures by converting the branches themselves (not branchpoints) into network nodes and joining them into a network based on which branches border each other.

        \n\n--Use 'Calculate Centroids' to find the center of mass for nodes and/or edges. Centroids provide a low-memory way to track object locations and are required for many other functions. Can downsample temporarily for speed.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Image"))

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        "The 'Image' submenu contains options altering your segmented data.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Process"),
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Image", "Dilate"),
        f"""--Use 'Resize' to resize images by upsampling or downsampling in any dimension. Enter values between 0-1 for downsampling or above 1 for upsampling. Can resize all dimensions uniformly or individual axes. Includes options to normalize scaling between xy and z dimensions and restore to original shape after prior resampling.

        \n\n--Use 'Clean Segmentation' to access quick cleaning operations including Close (dilation then erosion to fill gaps), Open (erosion then dilation to remove noise, smooth edges), Fill Holes, trace cleaner filamental segmentations of vessels/nerves, or Threshold Noise by Volume for removing small objects.

        \n\n--Use 'Dilate' to expand objects in an image. Dilation radius is scaled by xy_scale and z_scale properties.

        \n\n--Use 'Erode' to shrink objects in an image.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Image", "Dilate"),
        f"""--Use 'Fill Holes' to eliminate artifacts in binary segmentations by filling enclosed gaps. Can operate in 2D slicing mode only (XY plane) or across all 3D planes, with optional border hole filling and ability to output hole mask to Overlay 2 for selective filling.

        \n\n--Use 'Binarize' to convert images to binary format. Choose between total binarize (sets all nonzero to 255) or predict foreground (uses Otsu's method to automatically segment signal from background).

        \n\n--Use 'Label Objects' to assign distinct numerical identities to all touching, nonzero regions in an image. Useful for separating binary segmentations into unique objects.

        \n\n--Use 'Neighbor Labels' to label objects in one image based on proximity to labeled objects in another image. The first option is all nonzero objects take on the label of the closest labeled object. The second option is all nonzero objects take on the label of the closest labeled object that they are continuous with in space. Useful for defining spatial relationships between images.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Image", "Dilate"),
        f"""--Use 'Threshold/Segment' to access intensity-based, volume-based, radius-based, or degree-based thresholding windows. Alternatively, launch the Machine Learning segmenter which uses Random Forest Classifier trained on user-designated regions to segment based on morphological patterns.

        \n\n--Use 'Mask Channel' to use the binarized version of one channel to mask another, preserving only regions that exist in the mask.

        \n\n--Use 'Crop Channels' to crop all available channels to specified Z, Y, X boundaries. Can be auto-called by Shift+left-click-dragging in the Image Viewer Window.

        \n\n--Use 'Channel dtype' to change the data type of a channel (8-bit, 16-bit, 32-bit int, or 32/64-bit float) to preserve memory when larger data types aren't needed.,
        \n\n--Use 'Skeletonize' to reduce images to their medial axis. Can optionally remove terminal branches of specified pixel length and auto-correct loop artifacts that appear in thick regions.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Image", "Dilate"),
        """--Use 'Binary Watershed' to split fused objects in binary segmentations that appear as separate objects. Control aggressiveness with smallest radius or proportion parameters. Ideal for separating overlapping binary objects.

        \n\n--Use 'Gray Watershed' to watershed grayscale images, separating and labeling objects based on size and blobbiness. Best for quickly segmenting cells without ML training. Set minimum peak distance between labeled components for optimal results.

        \n\n--Use 'Invert' to invert an image, swapping high and low values.

        \n\n--Use 'Z-Project' to superimpose all XY slices into a single 2D slice using max, mean, min, sum, or standard deviation projection modes.""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Generate"))

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        "The 'Generate' submenu contains options for creating data.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Process"),
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Process", "Generate", "Label Branches"),
        f"""--Use 'Generate Nodes (From Node Centroids)' to convert your node_centroids property into a labeled image in the nodes channel, where each centroid becomes a labeled point. This is useful when centroids were loaded from a previous session or extracted from another analysis tool and you want to access image functions.

        \n\n--Use 'Generate Nodes (From Edge Vertices)' to generate nodes at the vertices of a branch-like segmented structure loaded into the edges channel.

        \n\n--Use 'Label Branches' to label the branches of a binary mask from branchy structures. Ideal for analyzing branchy structures like blood vessels.

        \n\n--Use 'Trace Filaments' to open a window for automatically cleaning segmentations of filament-like structures (such as vessels, nerves) by tracing a new, cleaner mask over pathways the program can detect.

        \n\n--Use 'Generate Voronoi Diagram' to create a Voronoi diagram from node_centroids, where labeled cells represent the region closest to each centroid. This provides an alternative way to define node neighborhoods for connectivity networks, particularly useful for small or homogeneous spheroid nodes. The diagram is loaded into Overlay2.""",                
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Process")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Modify Network/Properties"))

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        "The 'Modify Network/Properties' menu will help you tweak elements of a network in post.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Process"),
        action=lambda: MenuHelper.close_menu(window, "Process")
    )

    open_dialog, _ = MenuHelper.create_dialog_opener(
        window, tutorial, "show_modify_dialog", "ModifyDialog", "modify_dialog"
    )
    

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        f"""--Use 'Remove Unassigned IDs from Centroid List' to remove centroids of nodes without associated identities. This prepares the data for ID-oriented functions that expect all nodes to have an identity.

        \n\n--Use 'Force Any Multiple IDs to Pick a Random Single ID' to randomly assign a single identity to nodes with multiple identities. This simplifies identity visualization when there are many identity permutations.

        \n\n--Use 'Remove Any Nodes Not in Nodes Channel From Properties' to clean up node_centroids and node_identities properties by removing any nodes whose labels aren't present in the nodes channel image. Useful after cropping datasets.

        \n\n--Use 'Remove Trunk' to eliminate the most interconnected edge from the network, which can dominate analysis when evaluating downstream connections in trunk-heavy networks.""",
        highlight_type="rect",
        message_position="top_right",
        pre_action=open_dialog
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        f"""--Use 'Convert Trunk to Node' to transform the trunk into a new node instead of removing it, preserving network structure by treating the trunk as a central hub rather than shattering the network into subgraphs. Also moves the trunk from edges image to nodes image.

        \n\n--Use 'Convert Edges to Node Objects' to merge all edges into the nodes image and update the network to pair nodes based on their previously shared edges. Edges receive new labels and gain the identity 'edge'. Useful for visualizing exact connectivity between objects.

        \n\n--Use 'Remove Network Weights' to eliminate edge weights from connectivity networks, reducing each edge to a parameter of absolute connectivity rather than weighted by the number of joining edges.

        \n\n--Use 'Prune Connections Between Nodes of the Same Type' to remove all connections between nodes with matching identities, isolating only connections between different node types.""",
        highlight_type="rect",
        message_position="top_right"
    )

    def close_dialog():
        if hasattr(tutorial, 'modify_dialog') and tutorial.modify_dialog:
            tutorial.modify_dialog.close()
            tutorial.modify_dialog = None

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Process"),
        f"""--Use 'Isolate Connections Between Two Specific Node Types' to filter the network to show only connections involving two user-selected node identities, removing all other connections.

        \n\n--Use 'Rearrange Community IDs by Size' to renumber communities by node count, with 1 being the largest community. Makes community IDs more meaningful for visualization and analysis.

        \n\n--Use 'Convert Communities to Nodes' to replace the node-to-node network with a community-to-community network, and relabel nodes in the image by their community ID rather than original ID.

        \n\n--Use 'Add/Remove Network Pairs' to manually add or remove specific node pairs from the network with optional edge IDs. Allows arbitrary network modification beyond what table widgets support.""",
        highlight_type="rect",
        message_position="top_right",
        action = close_dialog
    )



    return tutorial

def setup_image_tutorial(window):
    tutorial = TutorialManager(window)

    # Step 1: Welcome
    tutorial.add_step(
        None,
        "This tutorial will guide you through the options for visualizing your data.",
        highlight_type="rect",
        message_position="top_right"
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Image"),
        "The Image menu contains the options visualizing your data. Here you can adjust the brightness, generate some overlays, move channels around, and show the 3D display.",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Image"),
        action=lambda: MenuHelper.close_menu(window, "Image")
    )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Image"),
        f"""--Use 'Properties' to view and modify current session properties including xy_scale and z_scale settings. You can also purge specific channels (nodes, edges, overlays) and properties (network, identities, centroids) by unchecking them and pressing Enter. The Report Properties button populates spreadsheet-based properties to the upper-right table.

        \n\n--Use 'Adjust Brightness/Contrast' to modify the visibility of each channel using dual-knobbed slider bars or by entering min/max values (0-65535). Essential for making loaded images visible when they appear too dark or bright.

        \n\n--Use 'Channel Colors' to change the display colors for each channel. Default colors are light_red for nodes, light_green for edges, and white for Overlay1 and Overlay2.""",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Image"),
        action=lambda: MenuHelper.close_menu(window, "Image")
    )

    def open_to_save():
        menu = MenuHelper.open_menu(window, "Image")
        if menu:
            QTimer.singleShot(100, lambda: MenuHelper.open_submenu(menu, "Overlays"))

    tutorial.add_step(
        MenuHelper.create_submenu_action_rect_getter(window, "Image", "Overlays", "Shuffle"),
        f"""-Use 'Overlays -> Create Network Overlay' to draw 1-voxel thick white lines between all node centroids in the network, placed in Overlay1. Provides convenient network structure visualization, especially in 3D. Optional downsampling enlarges the rendered output.

        \n\n--Use 'Overlays -> Create ID Overlay' to write the numerical ID of each node over its centroid, placed in Overlay2. Provides convenient node label visualization with optional downsampling to enlarge rendered output.

        \n\n--Use 'Overlays -> Color Nodes (or edges)' to create an RGB overlay where each grayscale label receives a unique color, placed in Overlay2 with a color legend in the data tables. Excellent for visualizing labeled objects.

        \n\n--Use 'Overlays -> Shuffle' to swap data between channels. Useful for moving outputs to correct channels since many functions expect content in specific channels (e.g., Nodes for network generation).""",
        highlight_type=None,
        message_position="top_right",
        pre_action=open_to_save,
        )

    tutorial.add_step(
        MenuHelper.create_menu_step_rect_getter(window, "Image"),

        f"""--Use 'Select Objects' to arbitrarily select and highlight groups of nodes or edges by entering comma-separated IDs or importing from a spreadsheet. Automatically navigates to the Z-plane of the first selected object for easy searching.

        \n\n--Use 'Show 3D (Napari)' to launch Napari for interactive 3D visualization of all visible channels. Uses GPU for smooth rendering if available. Can downsample for speed and optionally include a bounding box. Requires Napari to be installed.

        \n\n--Use 'Cellpose' to open the Cellpose3 GUI for cell segmentation. Opens 3D-stack version for 3D images and 2D-stack version for 2D images. Requires Cellpose3 to be installed in NetTracer3D's package environment.""",
        highlight_type="rect",
        message_position="top_right",
        pre_action=lambda: MenuHelper.open_menu(window, "Image"),
        action=lambda: MenuHelper.close_menu(window, "Image")
    )

    return tutorial

def setup_nettracer_tutorial(window):
    """
    DEPRECATED: Use setup_basics_tutorial instead
    This function is kept for backwards compatibility
    """
    return setup_basics_tutorial(window)
