import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, 
                           QWidget, QTableWidget, QTableWidgetItem, QPushButton, 
                           QLabel, QLineEdit, QScrollArea, QFrame, QMessageBox,
                           QHeaderView, QAbstractItemView, QSplitter, QTabWidget)
from PyQt6.QtCore import Qt, QMimeData, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QDrag, QPainter, QPixmap
import os
from PyQt6.QtWidgets import QComboBox
from ast import literal_eval
from PyQt6.QtCore import QObject, pyqtSignal

class DraggableTableWidget(QTableWidget):
    """Custom table widget that supports drag and drop operations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        
    def startDrag(self, supportedActions):
        if self.currentColumn() >= 0:
            # Create drag data with column index and header
            drag = QDrag(self)
            mimeData = QMimeData()
            
            # Store column index and header text
            col_idx = self.currentColumn()
            header_text = self.horizontalHeaderItem(col_idx).text() if self.horizontalHeaderItem(col_idx) else f"Column_{col_idx}"
            
            mimeData.setText(f"excel_column:{col_idx}:{header_text}")
            drag.setMimeData(mimeData)
            
            # Create drag pixmap
            pixmap = QPixmap(100, 30)
            pixmap.fill(Qt.GlobalColor.lightGray)
            painter = QPainter(pixmap)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, header_text)
            painter.end()
            drag.setPixmap(pixmap)
            
            drag.exec(Qt.DropAction.CopyAction)


class DropZoneWidget(QFrame):
    """Widget that accepts file drops for Excel/CSV files"""
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)
        
        layout = QVBoxLayout()
        label = QLabel("Drag Excel (.xlsx) or CSV files here")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(label)
        self.setLayout(layout)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith(('.xlsx', '.csv')):
                    event.acceptProposedAction()
                    return
        event.ignore()
        
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.file_dropped.emit(file_path)
            event.acceptProposedAction()

class DictColumnWidget(QFrame):
    """Widget representing a dictionary column that can accept drops"""
    column_dropped = pyqtSignal(str, int, str)  # widget_id, col_idx, col_name
    delete_requested = pyqtSignal(str)  # widget_id
    
    def __init__(self, widget_id, parent=None):
        super().__init__(parent)
        self.widget_id = widget_id
        self.column_data = None
        self.column_name = None
        self.setAcceptDrops(True)
        self.setFixedHeight(80)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: white;
                margin: 2px;
            }
            QFrame:hover {
                border-color: #007acc;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Header input
        self.header_input = QLineEdit()
        self.header_input.setPlaceholderText("Dictionary key name...")
        self.header_input.textChanged.connect(self.on_header_changed)
        layout.addWidget(self.header_input)
        
        # Drop zone / content area
        self.content_label = QLabel("Drop column here")
        self.content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.content_label)
        
        # Delete button
        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(20, 20)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
        """)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.widget_id))
        
        # Position delete button in top-right
        self.delete_btn.setParent(self)
        self.delete_btn.move(self.width() - 25, 5)
        
        self.setLayout(layout)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.delete_btn.move(self.width() - 25, 5)
        
    def on_header_changed(self):
        # Update display when header changes
        if self.column_data is not None:
            self.content_label.setText(f"Column: {self.column_name}\nKey: {self.header_input.text()}")
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasText() and event.mimeData().text().startswith("excel_column:"):
            event.acceptProposedAction()
        else:
            event.ignore()
            
    def dropEvent(self, event: QDropEvent):
        text = event.mimeData().text()
        if text.startswith("excel_column:"):
            parts = text.split(":", 2)
            if len(parts) >= 3:
                col_idx = int(parts[1])
                col_name = parts[2]
                self.column_name = col_name
                self.content_label.setText(f"Column: {col_name}\nKey: {self.header_input.text()}")
                self.column_dropped.emit(self.widget_id, col_idx, col_name)
                event.acceptProposedAction()


class ClassifierWidget(QFrame):
    """Widget representing a single classifier with substrings and new ID"""
    
    def __init__(self, classifier_id, classifier_group_widget, parent=None):
        super().__init__(parent)
        self.classifier_id = classifier_id
        self.classifier_group_widget = classifier_group_widget  # Store reference to parent group
        self.substrings = []
        
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f8f9fa;
                margin: 2px;
                padding: 5px;
            }
        """)
        
        # Header with classifier number and buttons
        # Header with classifier number and buttons
        layout = QVBoxLayout()
        header_layout = QHBoxLayout()
        self.header_label = QLabel(f"Classifier {classifier_id}")  # Store reference to label
        self.header_label.setStyleSheet("font-weight: bold; color: #495057;")
        header_layout.addWidget(self.header_label)

        header_layout.addStretch()

        # Move up button
        self.up_btn = QPushButton("↑")
        self.up_btn.setFixedSize(20, 20)
        self.up_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.up_btn.clicked.connect(self.move_up)  # Connect to instance method
        header_layout.addWidget(self.up_btn)

        # Move down button
        self.down_btn = QPushButton("↓")
        self.down_btn.setFixedSize(20, 20)
        self.down_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.down_btn.clicked.connect(self.move_down)  # Connect to instance method
        header_layout.addWidget(self.down_btn)

        # Copy button
        self.copy_btn = QPushButton("⎘")
        self.copy_btn.setFixedSize(20, 20)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.copy_btn.clicked.connect(self.copy_classifier)  # Connect to instance method
        header_layout.addWidget(self.copy_btn)

        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(20, 20)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_requested)
        header_layout.addWidget(self.delete_btn)
        
        layout.addLayout(header_layout)
        
        # Substring input area
        substring_layout = QHBoxLayout()
        substring_label = QLabel("Substrings:")
        substring_label.setStyleSheet("font-weight: bold; color: #6c757d;")
        substring_layout.addWidget(substring_label)
        
        self.substring_input = QLineEdit()
        self.substring_input.setPlaceholderText("Enter substring to match...")
        substring_layout.addWidget(self.substring_input)
        
        add_substring_btn = QPushButton("Add")
        add_substring_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        add_substring_btn.clicked.connect(self.add_substring)
        substring_layout.addWidget(add_substring_btn)
        
        layout.addLayout(substring_layout)
        
        # Connect Enter key to add substring
        self.substring_input.returnPressed.connect(self.add_substring)
        
        # Substrings display
        self.substrings_display = QLabel("Substrings: (none)")
        self.substrings_display.setStyleSheet("color: #6c757d; font-style: italic; margin: 5px 0;")
        self.substrings_display.setWordWrap(True)
        layout.addWidget(self.substrings_display)
        
        # New ID input
        new_id_layout = QHBoxLayout()
        new_id_label = QLabel("New ID:")
        new_id_label.setStyleSheet("font-weight: bold; color: #6c757d;")
        new_id_layout.addWidget(new_id_label)
        
        self.new_id_input = QLineEdit()
        self.new_id_input.setPlaceholderText("Enter new ID for matches...")
        new_id_layout.addWidget(self.new_id_input)
        
        layout.addLayout(new_id_layout)
        
        self.setLayout(layout)
    
    def add_substring(self):
        substring = self.substring_input.text().strip()
        if substring and substring not in self.substrings:
            self.substrings.append(substring)
            self.update_substrings_display()
            self.substring_input.clear()
    
    def update_substrings_display(self):
        if self.substrings:
            # Create clickable labels for each substring
            if hasattr(self, 'substrings_container') and self.substrings_container is not None:
                try:
                    self.substrings_container.deleteLater()
                except RuntimeError:
                    pass  # Object already deleted
                self.substrings_container = None
            
            self.substrings_container = QWidget()
            container_layout = QHBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            for i, substring in enumerate(self.substrings):
                substring_widget = QFrame()
                substring_widget.setStyleSheet("""
                    QFrame {
                        background-color: #e9ecef;
                        border: 1px solid #adb5bd;
                        border-radius: 3px;
                        padding: 2px 5px;
                        margin: 1px;
                    }
                """)
                
                substring_layout = QHBoxLayout()
                substring_layout.setContentsMargins(2, 2, 2, 2)
                
                label = QLabel(f'"{substring}"')
                label.setStyleSheet("background: transparent; border: none;")
                substring_layout.addWidget(label)
                
                remove_btn = QPushButton("×")
                remove_btn.setFixedSize(16, 16)
                remove_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #dc3545;
                        color: white;
                        border: none;
                        border-radius: 8px;
                        font-size: 8px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #c82333;
                    }
                """)
                remove_btn.clicked.connect(lambda checked, idx=i: self.remove_substring(idx))
                substring_layout.addWidget(remove_btn)
                
                substring_widget.setLayout(substring_layout)
                container_layout.addWidget(substring_widget)
            
            container_layout.addStretch()
            self.substrings_container.setLayout(container_layout)
            
            # Replace the old display
            layout = self.layout()
            old_display_index = -1
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget() == getattr(self, 'substrings_display', None):
                    old_display_index = i
                    break
            
            if old_display_index >= 0:
                layout.removeWidget(self.substrings_display)
                try:
                    self.substrings_display.deleteLater()
                except RuntimeError:
                    pass
                layout.insertWidget(old_display_index, self.substrings_container)
            else:
                # Insert after substring input layout (index 2)
                layout.insertWidget(2, self.substrings_container)
        else:
            if hasattr(self, 'substrings_container') and self.substrings_container is not None:
                try:
                    self.substrings_container.deleteLater()
                except RuntimeError:
                    pass
                self.substrings_container = None
            
            self.substrings_display = QLabel("Substrings: (none)")
            self.substrings_display.setStyleSheet("color: #6c757d; font-style: italic; margin: 5px 0;")
            self.substrings_display.setWordWrap(True)
            
            layout = self.layout()
            # Find where to insert (after substring input layout)
            insert_index = 2  # After header and substring input
            layout.insertWidget(insert_index, self.substrings_display)

    def remove_substring(self, index):
        if 0 <= index < len(self.substrings):
            self.substrings.pop(index)
            self.update_substrings_display()
    
    def delete_requested(self):
        # Use the stored reference instead of parent()
        self.classifier_group_widget.remove_classifier(self.classifier_id)
    
    def matches_identity(self, identity_str):
        """Check if this classifier matches the given identity string"""
        if not self.substrings:  # Empty classifier matches everything
            return True
        
        identity_str = str(identity_str)
        return all(substring in identity_str for substring in self.substrings)
    
    def get_new_id(self):
        """Get the new ID for this classifier"""
        return self.new_id_input.text().strip()

    def move_up(self):
        """Move this classifier up using current classifier_id"""
        self.classifier_group_widget.move_classifier_up(self.classifier_id)

    def move_down(self):
        """Move this classifier down using current classifier_id"""
        self.classifier_group_widget.move_classifier_down(self.classifier_id)

    def copy_classifier(self):
        """Copy this classifier using current classifier_id"""
        self.classifier_group_widget.copy_classifier(self.classifier_id)

    def update_header_label(self):
        """Update the header label text"""
        if hasattr(self, 'header_label'):
            self.header_label.setText(f"Classifier {self.classifier_id}")


class ClassifierGroupWidget(QFrame):
    """Widget containing multiple classifiers for enhanced search functionality"""
    
    def __init__(self, identity_remap_widget, parent=None):
        super().__init__(parent)
        self.identity_remap_widget = identity_remap_widget
        self.classifier_counter = 0
        self.classifiers = {}  # classifier_id -> ClassifierWidget
        
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #007acc;
                border-radius: 5px;
                background-color: #f8f9fa;
                margin: 5px;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        header = QLabel("Enhanced Search & Classification")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #007acc; margin-bottom: 5px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Add classifier button
        add_btn = QPushButton("+ Add Classifier")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        add_btn.clicked.connect(self.add_classifier)
        header_layout.addWidget(add_btn)
        
        layout.addLayout(header_layout)
        
        # Scroll area for classifiers
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container for classifiers
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container_layout.setSpacing(5)
        self.container_layout.addStretch()
        self.container.setLayout(self.container_layout)
        scroll.setWidget(self.container)
        
        layout.addWidget(scroll)
        
        # Preview button
        preview_btn = QPushButton("🔍 Preview Classification")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #212529;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        preview_btn.clicked.connect(self.preview_classification)
        layout.addWidget(preview_btn)
        
        self.setLayout(layout)
    
    def add_classifier(self):
        self.classifier_counter += 1
        classifier_id = self.classifier_counter
        
        # Pass reference to self (ClassifierGroupWidget) as second parameter
        classifier_widget = ClassifierWidget(classifier_id, self, self)
        self.classifiers[classifier_id] = classifier_widget
        
        # Insert before the stretch
        self.container_layout.insertWidget(self.container_layout.count() - 1, classifier_widget)
    
    def remove_classifier(self, classifier_id):
        if classifier_id in self.classifiers:
            widget = self.classifiers[classifier_id]
            self.container_layout.removeWidget(widget)
            widget.deleteLater()
            del self.classifiers[classifier_id]
            self.renumber_classifiers()
    
    def preview_classification(self):
        """Apply classification rules to the identity remapping widget"""
        if not hasattr(self.identity_remap_widget, 'identity_mappings'):
            QMessageBox.warning(self, "Warning", "No identity data loaded yet.")
            return
        
        if not self.classifiers:
            QMessageBox.warning(self, "Warning", "No classifiers defined.")
            return
        
        # Apply classifiers in order (hierarchical)
        classifier_ids = sorted(self.classifiers.keys())  # MOVE THIS LINE UP HERE
        
        # Get all original identities
        original_identities = list(self.identity_remap_widget.identity_mappings.keys())
        matched_identities = set()
        classifier_usage = {classifier_id: 0 for classifier_id in classifier_ids}  # Now classifier_ids is defined
        
        for identity in original_identities:
            identity_str = str(identity)
            
            # Check classifiers in order
            for classifier_id in classifier_ids:
                classifier = self.classifiers[classifier_id]
                
                if classifier.matches_identity(identity_str):
                    # This classifier matches
                    matched_identities.add(identity)
                    classifier_usage[classifier_id] += 1  # Track usage
                    
                    # Set the new ID if provided
                    new_id = classifier.get_new_id()
                    if new_id and identity in self.identity_remap_widget.identity_mappings:
                        self.identity_remap_widget.identity_mappings[identity]['new_edit'].setText(new_id)
                    
                    # Move to next identity (hierarchical - first match wins)
                    break
        
        # Remove identities that didn't match any classifier
        unmatched_identities = set(original_identities) - matched_identities
        for identity in unmatched_identities:
            self.identity_remap_widget.remove_identity(identity)
        
        # Show results
        matched_count = len(matched_identities)
        removed_count = len(unmatched_identities)
        
        # Create usage report
        usage_report = ""
        for classifier_id in classifier_ids:
            classifier = self.classifiers[classifier_id]
            count = classifier_usage[classifier_id]
            new_id = classifier.get_new_id() or "(no new ID set)"
            substrings = classifier.substrings or ["(empty - matches all)"]
            usage_report += f"  Classifier {classifier_id}: {count} matches → '{new_id}'\n"
            usage_report += f"    Substrings: {substrings}\n"

        QMessageBox.information(
            self, 
            "Classification Preview Applied", 
            f"Classification complete!\n\n"
            f"• Matched identities: {matched_count}\n"
            f"• Removed identities: {removed_count}\n"
            f"• Total classifiers used: {len(self.classifiers)}\n\n"
            f"Classifier Usage:\n{usage_report}\n"
            f"Check the Identity Remapping widget to see the results."
        )

    def copy_classifier(self, classifier_id):
        if classifier_id in self.classifiers:
            original = self.classifiers[classifier_id]
            
            # Create new classifier
            self.classifier_counter += 1
            new_classifier_id = self.classifier_counter
            
            new_classifier = ClassifierWidget(new_classifier_id, self, self)
            
            # Copy data
            new_classifier.substrings = original.substrings.copy()
            new_classifier.new_id_input.setText(original.new_id_input.text())
            new_classifier.update_substrings_display()
            
            self.classifiers[new_classifier_id] = new_classifier
            
            # Insert after the original
            original_index = self.get_classifier_index(classifier_id)
            self.container_layout.insertWidget(original_index + 1, new_classifier)
            
            self.renumber_classifiers()

    def move_classifier_up(self, classifier_id):
        current_index = self.get_classifier_index(classifier_id)
        if current_index > 0:
            self.swap_classifiers(current_index, current_index - 1)

    def move_classifier_down(self, classifier_id):
        current_index = self.get_classifier_index(classifier_id)
        classifier_count = len(self.classifiers)
        if current_index < classifier_count - 1:
            self.swap_classifiers(current_index, current_index + 1)

    def get_classifier_index(self, classifier_id):
        """Get the current layout index of a classifier by its ID"""
        for i in range(self.container_layout.count() - 1):  # -1 for stretch
            widget = self.container_layout.itemAt(i).widget()
            if (widget is not None and 
                hasattr(widget, 'classifier_id') and 
                widget.classifier_id == classifier_id):
                return i
        return -1

    def swap_classifiers(self, index1, index2):
        # Get widgets at the positions
        widget1 = self.container_layout.itemAt(index1).widget()
        widget2 = self.container_layout.itemAt(index2).widget()
        
        if not (hasattr(widget1, 'classifier_id') and hasattr(widget2, 'classifier_id')):
            return
        
        # Remove widgets in reverse order to maintain indices
        if index1 > index2:
            self.container_layout.removeWidget(widget1)  # Remove higher index first
            self.container_layout.removeWidget(widget2)
            # Now reinsert: widget1 goes to index2, widget2 goes to index1
            self.container_layout.insertWidget(index2, widget1)
            self.container_layout.insertWidget(index1, widget2)
        else:
            self.container_layout.removeWidget(widget2)  # Remove higher index first
            self.container_layout.removeWidget(widget1)
            # Now reinsert: widget1 goes to index2, widget2 goes to index1
            self.container_layout.insertWidget(index1, widget2)
            self.container_layout.insertWidget(index2, widget1)
        
        # Renumber all classifiers to maintain correct order and references
        self.renumber_classifiers()
        
    def renumber_classifiers(self):
        # Create new dictionary to avoid issues during iteration
        new_classifiers = {}
        
        # Renumber all classifiers to maintain order
        for i in range(self.container_layout.count() - 1):  # -1 for stretch
            widget = self.container_layout.itemAt(i).widget()
            if hasattr(widget, 'classifier_id'):
                old_id = widget.classifier_id
                new_id = i + 1
                
                # Update the widget's ID
                widget.classifier_id = new_id
                
                # Update the header label using the new method
                widget.update_header_label()
                
                # Add to new dictionary with new ID
                new_classifiers[new_id] = widget
        
        # Replace the old dictionary
        self.classifiers = new_classifiers
        
        # Update counter to the highest number
        self.classifier_counter = len(self.classifiers)

class IdentityRemapWidget(QFrame):
    """Widget for remapping node identities"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #007acc;
                border-radius: 5px;
                background-color: #f8f9fa;
                margin: 5px;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Identity Remapping & Filtering")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #007acc; margin-bottom: 5px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Create scroll area for the mapping table
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(250)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container for mapping rows
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container_layout.setSpacing(2)
        self.container.setLayout(self.container_layout)
        scroll.setWidget(self.container)
        
        layout.addWidget(scroll)
        self.setLayout(layout)
        
        # Store mapping data
        self.identity_mappings = {}  # original_id -> {'new_edit': QLineEdit, 'row_widget': QWidget}
        self.removed_identities = set()  # Track removed identities
        
    def populate_identities(self, identities):
        """Populate the left column with unique identities from the data"""
        # Clear existing mappings
        for i in reversed(range(self.container_layout.count())):
            item = self.container_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        
        self.identity_mappings.clear()
        self.removed_identities.clear()
        
        # Get unique identities
        unique_identities = sorted(list(set(identities)))
        
        # Create header row
        header_layout = QHBoxLayout()
        orig_label = QLabel("Original ID")
        orig_label.setStyleSheet("font-weight: bold; padding: 5px;")
        new_label = QLabel("New ID (leave blank to keep original)")
        new_label.setStyleSheet("font-weight: bold; padding: 5px;")
        delete_label = QLabel("Remove")
        delete_label.setStyleSheet("font-weight: bold; padding: 5px; text-align: center;")
        delete_label.setFixedWidth(60)
        
        header_layout.addWidget(orig_label, 2)
        header_layout.addWidget(new_label, 3)
        header_layout.addWidget(delete_label, 1)
        
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        self.container_layout.addWidget(header_widget)
        
        # Create mapping rows
        for identity in unique_identities:
            row_layout = QHBoxLayout()
            
            # Original identity (read-only)
            orig_edit = QLineEdit(str(identity))
            orig_edit.setReadOnly(True)
            orig_edit.setStyleSheet("background-color: #e9ecef; border: 1px solid #ced4da;")
            orig_edit.setMinimumWidth(120)  # Set initial minimum width

            # New identity (editable)
            new_edit = QLineEdit()
            new_edit.setPlaceholderText(f"Enter new ID for {identity}")
            new_edit.setStyleSheet("border: 1px solid #ced4da;")
            new_edit.setMinimumWidth(180)  # Set initial minimum width
            
            # Delete button
            delete_btn = QPushButton("×")
            delete_btn.setFixedSize(25, 25)
            delete_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
            """)
            
            row_layout.addWidget(orig_edit, 2)
            row_layout.addWidget(new_edit, 3)
            row_layout.addWidget(delete_btn, 1)
            
            row_widget = QWidget()
            row_widget.setLayout(row_layout)
            self.container_layout.addWidget(row_widget)
            
            # Store the mapping
            self.identity_mappings[identity] = {
                'new_edit': new_edit,
                'row_widget': row_widget
            }
            
            # Connect delete button
            delete_btn.clicked.connect(lambda checked, id=identity: self.remove_identity(id))
    
    def remove_identity(self, identity):
        """Remove an identity from the mapping widget"""
        if identity in self.identity_mappings:
            # Add to removed set
            self.removed_identities.add(identity)
            
            # Remove the widget
            row_widget = self.identity_mappings[identity]['row_widget']
            self.container_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            
            # Remove from mappings
            del self.identity_mappings[identity]
    
    def get_remapped_identities(self, original_identities):
        """Return the remapped identities based on user input, filtering out removed ones"""
        remapped = []
        for orig_id in original_identities:
            # Skip if identity was removed
            if orig_id in self.removed_identities:
                continue
                
            if orig_id in self.identity_mappings:
                new_id = self.identity_mappings[orig_id]['new_edit'].text().strip()
                if new_id:  # If user entered a new ID
                    remapped.append(new_id)
                else:  # If blank, keep original
                    remapped.append(orig_id)
            else:
                # If not in mappings but not removed, keep original
                if orig_id not in self.removed_identities:
                    remapped.append(orig_id)
        return remapped
    
    def get_filtered_indices(self, original_identities):
        """Return indices of identities that should be kept (not removed)"""
        kept_indices = []
        for i, orig_id in enumerate(original_identities):
            if orig_id not in self.removed_identities:
                kept_indices.append(i)
        return kept_indices
    
    def update_font_sizes(self, scale_factor):
        """Update widget sizes based on scale but keep font sizes constant"""
        # Don't change font sizes - just let the widgets resize naturally
        # The horizontal layout will automatically make the text fields wider
        # when the parent widget gets larger
        
        # Set minimum widths based on scale factor to ensure readability
        min_orig_width = max(80, int(120 * scale_factor))
        min_new_width = max(120, int(180 * scale_factor))
        
        # Update all line edits in the mapping
        for identity_data in self.identity_mappings.values():
            new_edit = identity_data['new_edit']
            
            # Set minimum widths but don't change font
            new_edit.setMinimumWidth(min_new_width)
            
            # Also update the original ID field
            row_widget = identity_data['row_widget']
            layout = row_widget.layout()
            if layout and layout.itemAt(0):
                orig_edit = layout.itemAt(0).widget()
                if isinstance(orig_edit, QLineEdit):
                    orig_edit.setMinimumWidth(min_orig_width)


class TabbedIdentityWidget(QFrame):
    """Widget that contains both identity remapping and classifier widgets with tabs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        # Tab buttons
        tab_layout = QHBoxLayout()
        
        self.remap_tab_btn = QPushButton("Identity Remapping")
        self.remap_tab_btn.setCheckable(True)
        self.remap_tab_btn.setChecked(True)
        self.remap_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px 5px 0 0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:checked {
                background-color: #004d7a;
            }
        """)
        self.remap_tab_btn.clicked.connect(self.show_remap_tab)
        
        self.classifier_tab_btn = QPushButton("Enhanced Search")
        self.classifier_tab_btn.setCheckable(True)
        self.classifier_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px 5px 0 0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked {
                background-color: #007acc;
            }
        """)
        self.classifier_tab_btn.clicked.connect(self.show_classifier_tab)
        
        tab_layout.addWidget(self.remap_tab_btn)
        tab_layout.addWidget(self.classifier_tab_btn)
        tab_layout.addStretch()
        
        layout.addLayout(tab_layout)
        
        # Save/Load buttons
        save_load_layout = QHBoxLayout()

        save_btn = QPushButton("💾 Save Config")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        save_btn.clicked.connect(self.save_configuration)

        load_btn = QPushButton("📁 Load Config")
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #6f42c1;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a2d91;
            }
        """)
        load_btn.clicked.connect(self.load_configuration)

        save_load_layout.addWidget(save_btn)
        save_load_layout.addWidget(load_btn)
        save_load_layout.addStretch()

        layout.addLayout(save_load_layout)

        # Create both widgets
        self.identity_remap_widget = IdentityRemapWidget()
        self.classifier_group_widget = ClassifierGroupWidget(self.identity_remap_widget)
        
        # Initially hide classifier widget
        self.classifier_group_widget.hide()
        
        layout.addWidget(self.identity_remap_widget)
        layout.addWidget(self.classifier_group_widget)
        
        self.setLayout(layout)
    
    def show_remap_tab(self):
        self.remap_tab_btn.setChecked(True)
        self.classifier_tab_btn.setChecked(False)
        
        self.identity_remap_widget.show()
        self.classifier_group_widget.hide()
        
        # Update button styles
        self.remap_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px 5px 0 0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:checked {
                background-color: #004d7a;
            }
        """)
        
        self.classifier_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px 5px 0 0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked {
                background-color: #007acc;
            }
        """)
    
    def show_classifier_tab(self):
        self.remap_tab_btn.setChecked(False)
        self.classifier_tab_btn.setChecked(True)
        
        self.identity_remap_widget.hide()
        self.classifier_group_widget.show()
        
        # Update button styles
        self.classifier_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px 5px 0 0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:checked {
                background-color: #004d7a;
            }
        """)
        
        self.remap_tab_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px 5px 0 0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked {
                background-color: #007acc;
            }
        """)
    
    def populate_identities(self, identities):
        """Delegate to the identity remap widget"""
        self.identity_remap_widget.populate_identities(identities)
    
    def get_remapped_identities(self, original_identities):
        """Delegate to the identity remap widget"""
        return self.identity_remap_widget.get_remapped_identities(original_identities)
    
    def get_filtered_indices(self, original_identities):
        """Delegate to the identity remap widget"""
        return self.identity_remap_widget.get_filtered_indices(original_identities)
    
    def update_font_sizes(self, scale_factor):
        """Delegate to the identity remap widget"""
        self.identity_remap_widget.update_font_sizes(scale_factor)

    def save_configuration(self):
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Identity Configuration", 
            "", 
            "JSON Files (*.json)"
        )
        
        if file_path:
            config = {
                'identity_mappings': {},
                'removed_identities': list(self.identity_remap_widget.removed_identities),
                'classifiers': []
            }
            
            # Save identity mappings
            for orig_id, data in self.identity_remap_widget.identity_mappings.items():
                config['identity_mappings'][str(orig_id)] = {
                    'new_id': data['new_edit'].text()
                }
            
            # Save classifiers in order
            for i in range(self.classifier_group_widget.container_layout.count() - 1):
                widget = self.classifier_group_widget.container_layout.itemAt(i).widget()
                if hasattr(widget, 'classifier_id'):
                    classifier_config = {
                        'id': widget.classifier_id,
                        'substrings': widget.substrings,
                        'new_id': widget.get_new_id()
                    }
                    config['classifiers'].append(classifier_config)
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                QMessageBox.information(self, "Success", "Configuration saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")

    def load_configuration(self):
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Identity Configuration", 
            "", 
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Clear existing classifiers
                for classifier_id in list(self.classifier_group_widget.classifiers.keys()):
                    self.classifier_group_widget.remove_classifier(classifier_id)
                
                # Load identity mappings
                for orig_id_str, data in config.get('identity_mappings', {}).items():
                    # Convert string back to original type if needed
                    orig_id = orig_id_str
                    if orig_id in self.identity_remap_widget.identity_mappings:
                        self.identity_remap_widget.identity_mappings[orig_id]['new_edit'].setText(data['new_id'])
                
                # Load removed identities
                for removed_id in config.get('removed_identities', []):
                    if removed_id in self.identity_remap_widget.identity_mappings:
                        self.identity_remap_widget.remove_identity(removed_id)
                
                # Load classifiers
                for classifier_config in config.get('classifiers', []):
                    self.classifier_group_widget.add_classifier()
                    # Get the last added classifier
                    last_classifier = list(self.classifier_group_widget.classifiers.values())[-1]
                    last_classifier.substrings = classifier_config['substrings']
                    last_classifier.new_id_input.setText(classifier_config['new_id'])
                    last_classifier.update_substrings_display()
                
                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")


class ExcelToDictGUI(QMainWindow):
    # Add this signal
    data_exported = pyqtSignal(dict, str, bool)  # dictionary, property_name, add_status
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.dict_columns = {}  # widget_id -> column_data
        self.column_counter = 0
        self.identity_remap_widget = None

        self.templates = {
            'Node Identities': ['Numerical IDs', 'Identity Column'],
            'Node Centroids': ['Numerical IDs', 'Z', 'Y', 'X'],
            'Node Communities': ['Numerical IDs', 'Community Identifier']
        }
        
        self.setWindowTitle("Excel to Python Dictionary Converter")
        self.setGeometry(100, 100, 1200, 800)
        self.add = False
        
        self.setup_ui()

    def on_splitter_moved(self, pos, index):
        """Handle splitter movement to update font sizes"""
        splitter = self.sender()
        sizes = splitter.sizes()
        total_width = sum(sizes)
        
        if total_width > 0:
            right_width = sizes[1]
            # Calculate scale factor based on right panel width (300 is base width)
            scale_factor = max(0.7, min(2.0, right_width / 300))
            
            # Update identity remapping widget font sizes
            if self.identity_remap_widget.isVisible():
                self.identity_remap_widget.update_font_sizes(scale_factor)
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()

        # Template selector at top
        template_layout = QHBoxLayout()
        template_label = QLabel("Templates:")
        template_label.setStyleSheet("font-weight: bold;")
        template_layout.addWidget(template_label)

        self.template_combo = QComboBox()
        self.template_combo.addItem("Select Template...")
        self.template_combo.addItems(['Node Identities', 'Node Centroids', 'Node Communities'])
        self.template_combo.currentTextChanged.connect(self.load_template)
        template_layout.addWidget(self.template_combo)
        template_layout.addStretch()

        template_widget = QWidget()
        template_widget.setLayout(template_layout)
        template_widget.setMaximumHeight(40)

        # Add to main layout
        main_layout_with_template = QVBoxLayout()
        main_layout_with_template.addWidget(template_widget)
        main_layout_with_template.addLayout(main_layout)
        central_widget.setLayout(main_layout_with_template)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setStyleSheet("""
                    QSplitter::handle {
                        background-color: #cccccc;
                        border: 1px solid #999999;
                    }
                    QSplitter::handle:hover {
                        background-color: #007acc;
                    }
                """)
                
        # Left side - Excel data viewer
        left_widget = QWidget()
        left_widget.setMinimumWidth(400)  # Set minimum width
        left_layout = QVBoxLayout()

        left_label = QLabel("Excel Data Viewer")
        left_label.setStyleSheet("font-weight: bold; font-size: 16px; margin-bottom: 10px;")
        left_layout.addWidget(left_label)

        # File drop zone
        self.drop_zone = DropZoneWidget()
        self.drop_zone.file_dropped.connect(self.load_file)
        self.drop_zone.setFixedHeight(60)
        left_layout.addWidget(self.drop_zone)

        # Excel table
        self.excel_table = DraggableTableWidget()
        self.excel_table.setAlternatingRowColors(True)
        self.excel_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.excel_table)

        left_widget.setLayout(left_layout)

        # Right side - Dictionary builder
        right_widget = QWidget()
        right_widget.setMinimumWidth(300)  # Set minimum width
        right_layout = QVBoxLayout()

        # Header with controls
        header_layout = QHBoxLayout()
        right_label = QLabel("Python Dictionary Builder")
        right_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        header_layout.addWidget(right_label)

        # Add column button
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedSize(30, 30)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        self.add_btn.clicked.connect(self.add_dict_column)
        header_layout.addWidget(self.add_btn)

        right_layout.addLayout(header_layout)

        # Dictionary columns scroll area
        self.dict_scroll = QScrollArea()
        self.dict_scroll.setWidgetResizable(True)
        self.dict_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.dict_container = QWidget()
        self.dict_layout = QVBoxLayout()
        self.dict_layout.addStretch()
        self.dict_container.setLayout(self.dict_layout)
        self.dict_scroll.setWidget(self.dict_container)

        right_layout.addWidget(self.dict_scroll)

        # Tabbed identity remapping widget (initially hidden)
        self.identity_remap_widget = TabbedIdentityWidget()
        self.identity_remap_widget.hide()
        right_layout.addWidget(self.identity_remap_widget)

        # Export controls
        export_layout = QHBoxLayout()

        # Property selector
        self.property_combo = QComboBox()
        self.property_combo.addItem("Select Property...")
        self.property_combo.addItems(['Node Identities', 'Node Centroids', 'Node Communities'])
        export_layout.addWidget(self.property_combo)

        # Export button
        self.export_btn = QPushButton("→ Export to NetTracer3D")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.export_btn.clicked.connect(self.export_dictionary)
        export_layout.addWidget(self.export_btn)

        self.add_button = QPushButton("+")
        self.add_button.setFixedSize(20, 20)
        self.add_button.setCheckable(True)
        self.add_button.setChecked(False)
        self.add_button.clicked.connect(self.toggle_add)
        export_layout.addWidget(self.add_button)


        right_layout.addLayout(export_layout)

        right_widget.setLayout(right_layout)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # Set initial sizes (60% left, 40% right)
        splitter.setSizes([600, 400])

        # Connect splitter moved signal to update font sizes
        splitter.splitterMoved.connect(self.on_splitter_moved)

        # Add splitter to main layout
        main_layout.addWidget(splitter)

    def toggle_add(self):

        if self.add_button.isChecked():
            print("Exported Properties will be added onto existing ones")
            self.add = True
        else:
            print("Exported Properties will be override existing ones")
            self.add = False

    def load_template(self, template_name):
        if template_name in self.templates:
            # Clear existing columns
            for widget_id in list(self.dict_columns.keys()):
                self.remove_dict_column(widget_id)
            
            # Clear widgets
            for i in reversed(range(self.dict_layout.count())):
                item = self.dict_layout.itemAt(i)
                if item and item.widget() and hasattr(item.widget(), 'widget_id'):
                    widget = item.widget()
                    self.dict_layout.removeWidget(widget)
                    widget.deleteLater()
            
            # Add stretch back
            self.dict_layout.addStretch()
            
            # Add new columns for template
            for key_name in self.templates[template_name]:
                self.add_dict_column()
                # Get the last added widget and set its header
                for i in range(self.dict_layout.count()):
                    item = self.dict_layout.itemAt(i)
                    if item and item.widget() and hasattr(item.widget(), 'widget_id'):
                        widget = item.widget()
                        if widget.widget_id not in [w.widget_id for w in self.get_existing_widgets()]:
                            widget.header_input.setText(key_name)
                            break
            
            # Set property combo to match
            self.property_combo.setCurrentText(template_name)
            
            # Show/hide identity remapping widget
            if template_name == 'Node Identities':
                self.identity_remap_widget.show()
            else:
                self.identity_remap_widget.hide()

    def get_existing_widgets(self):
        widgets = []
        for i in range(self.dict_layout.count()):
            item = self.dict_layout.itemAt(i)
            if item and item.widget() and hasattr(item.widget(), 'widget_id'):
                widgets.append(item.widget())
        return widgets[:-1]  # Exclude the stretch
        
    def load_file(self, file_path):
        try:
            if file_path.lower().endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            elif file_path.lower().endswith('.csv'):
                self.df = pd.read_csv(file_path)
            else:
                QMessageBox.warning(self, "Error", "Unsupported file format")
                return
                
            self.populate_excel_table()
            QMessageBox.information(self, "Success", f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
            
    def populate_excel_table(self):
        if self.df is None:
            return
            
        # Limit display to 200 rows but keep full dataframe
        display_rows = min(200, len(self.df))
        
        self.excel_table.setRowCount(display_rows)
        self.excel_table.setColumnCount(len(self.df.columns))
        
        # Set headers
        self.excel_table.setHorizontalHeaderLabels([str(col) for col in self.df.columns])
        
        # Populate data
        for i in range(display_rows):
            for j, col in enumerate(self.df.columns):
                item = QTableWidgetItem(str(self.df.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
                self.excel_table.setItem(i, j, item)
                
        # Resize columns to content
        self.excel_table.resizeColumnsToContents()
        
    def add_dict_column(self):
        self.column_counter += 1
        widget_id = f"col_{self.column_counter}"
        
        dict_widget = DictColumnWidget(widget_id)
        dict_widget.column_dropped.connect(self.on_column_dropped)
        dict_widget.delete_requested.connect(self.remove_dict_column)
        
        # Insert before the stretch
        self.dict_layout.insertWidget(self.dict_layout.count() - 1, dict_widget)
        
    def remove_dict_column(self, widget_id):
        # Find and remove the widget
        for i in range(self.dict_layout.count()):
            item = self.dict_layout.itemAt(i)
            if item and item.widget() and hasattr(item.widget(), 'widget_id'):
                if item.widget().widget_id == widget_id:
                    widget = item.widget()
                    self.dict_layout.removeWidget(widget)
                    widget.deleteLater()
                    break
                    
        # Remove from data storage
        if widget_id in self.dict_columns:
            del self.dict_columns[widget_id]
            
    def on_column_dropped(self, widget_id, col_idx, col_name):
        if self.df is not None and col_idx < len(self.df.columns):
            # Store the column data
            column_data = self.df.iloc[:, col_idx].values
            self.dict_columns[widget_id] = {
                'column_name': col_name,
                'column_index': col_idx,
                'data': column_data
            }
            
            # If this is the identity column in Node Identities template, populate remapping widget
            current_template = self.template_combo.currentText()
            if current_template == 'Node Identities':
                # Find the widget that received the drop
                for i in range(self.dict_layout.count()):
                    item = self.dict_layout.itemAt(i)
                    if item and item.widget() and hasattr(item.widget(), 'widget_id'):
                        if item.widget().widget_id == widget_id:
                            key_name = item.widget().header_input.text().strip()
                            if key_name == 'Identity Column':
                                # Populate the identity remapping widget
                                self.identity_remap_widget.populate_identities(column_data)
                            break

    def export_dictionary(self):
        if not self.dict_columns:
            QMessageBox.warning(self, "Warning", "No dictionary columns defined")
            return
            
        property_name = self.property_combo.currentText()
        if property_name == "Select Property...":
            QMessageBox.warning(self, "Warning", "Please select a property")
            return
            
        try:
            result_dict = {}
            
            # Build dictionary from all defined columns
            for widget_id in self.dict_columns:
                # Find the corresponding widget to get the key name
                for i in range(self.dict_layout.count()):
                    item = self.dict_layout.itemAt(i)
                    if item and item.widget() and hasattr(item.widget(), 'widget_id'):
                        if item.widget().widget_id == widget_id:
                            key_name = item.widget().header_input.text().strip()
                            if key_name:
                                column_data = self.dict_columns[widget_id]['data']
                                
                                # Apply identity remapping and filtering if this is Node Identities
                                if property_name == 'Node Identities':
                                    if key_name == 'Identity Column':
                                        # Get filtered indices and remapped identities
                                        filtered_indices = self.identity_remap_widget.get_filtered_indices(column_data.tolist())
                                        filtered_data = [column_data[i] for i in filtered_indices]
                                        remapped_data = self.identity_remap_widget.get_remapped_identities(filtered_data)
                                        result_dict[key_name] = remapped_data
                                    elif key_name == 'Numerical IDs':
                                        
                                        # Check if user actually dropped a numerical IDs column
                                        if widget_id not in self.dict_columns or 'data' not in self.dict_columns[widget_id]:
                                            # Auto-generate sequential IDs and assign to column_data
                                            column_data = np.array(list(range(1, len(self.df) + 1)))
                                        
                                        # Now use the exact same logic as if user provided the data
                                        identity_column_data = None
                                        # Find the identity column data
                                        for other_widget_id in self.dict_columns:
                                            for j in range(self.dict_layout.count()):
                                                item_j = self.dict_layout.itemAt(j)
                                                if item_j and item_j.widget() and hasattr(item_j.widget(), 'widget_id'):
                                                    if item_j.widget().widget_id == other_widget_id:
                                                        other_key_name = item_j.widget().header_input.text().strip()
                                                        if other_key_name == 'Identity Column':
                                                            identity_column_data = self.dict_columns[other_widget_id]['data']
                                                            break
                                                if identity_column_data is not None:
                                                    break
                                        
                                        if identity_column_data is not None:
                                            filtered_indices = self.identity_remap_widget.get_filtered_indices(identity_column_data.tolist())
                                            filtered_numerical_ids = [column_data[i] for i in filtered_indices]
                                            result_dict[key_name] = filtered_numerical_ids
                                        else:
                                            result_dict[key_name] = column_data.tolist()
                                        

                                    else:
                                        result_dict[key_name] = column_data.tolist()
                                else:
                                    result_dict[key_name] = column_data.tolist()
                            break

            for i in range(self.dict_layout.count()):
                item = self.dict_layout.itemAt(i)
                if item and item.widget() and hasattr(item.widget(), 'widget_id'):
                    widget = item.widget()
                    widget_id = widget.widget_id
                    key_name = widget.header_input.text().strip()
                    
                    # Skip if already processed (has dropped data) or no key name
                    if widget_id in self.dict_columns or not key_name:
                        continue
                        
                    # Handle auto-generation for Node Identities template
                    if property_name == 'Node Identities' and key_name == 'Numerical IDs':
                        
                        # Find the identity column data
                        identity_column_data = None
                        for other_widget_id in self.dict_columns:
                            for j in range(self.dict_layout.count()):
                                item_j = self.dict_layout.itemAt(j)
                                if item_j and item_j.widget() and hasattr(item_j.widget(), 'widget_id'):
                                    if item_j.widget().widget_id == other_widget_id:
                                        other_key_name = item_j.widget().header_input.text().strip()
                                        if other_key_name == 'Identity Column':
                                            identity_column_data = self.dict_columns[other_widget_id]['data']
                                            break
                                if identity_column_data is not None:
                                    break
                        
                        if identity_column_data is not None:
                            # Auto-generate sequential IDs
                            auto_generated_ids = np.array(list(range(1, len(self.df) + 1)))
                            
                            filtered_indices = self.identity_remap_widget.get_filtered_indices(identity_column_data.tolist())
                            
                            filtered_numerical_ids = [auto_generated_ids[i] for i in filtered_indices]
                            
                            result_dict[key_name] = filtered_numerical_ids
                        else:
                            # Fallback: generate sequential IDs for all rows
                            result_dict[key_name] = list(range(1, len(self.df) + 1))

            
            if not result_dict:
                QMessageBox.warning(self, "Warning", "No valid dictionary keys defined")
                return
                
            # Emit signal to parent application
            self.data_exported.emit(result_dict, property_name, self.add)
            
            # Still store in global variables for backward compatibility
            import builtins
            builtins.excel_dict = result_dict
            builtins.target_property = property_name
            builtins.add = self.add
            
            # Show success message with preview
            preview = str(result_dict)
            if len(preview) > 150:
                preview = preview[:150] + "..."
                
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Dictionary exported for property '{property_name}'.\n\nData sent to parent application.\n\nPreview:\n{preview}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export dictionary: {str(e)}")

def main(standalone=True):
    if standalone:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        window = ExcelToDictGUI()
        window.show()
        
        sys.exit(app.exec())
    else:
        # Return a fresh instance of the class
        return ExcelToDictGUI

if __name__ == "__main__":
    main(True)