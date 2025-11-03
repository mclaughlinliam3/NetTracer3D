import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
                            QPushButton, QComboBox, QTextEdit, QSplitter, 
                            QListWidget, QListWidgetItem, QFrame, QMessageBox,
                            QHeaderView, QAbstractItemView, QCheckBox)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QDrag
from scipy import stats
import os

class DragDropListWidget(QListWidget):
    """Custom QListWidget that accepts drag and drop of column names with remove buttons"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
            
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasText():
            column_name = event.mimeData().text()
            # Check if column already exists
            for i in range(self.count()):
                widget = self.itemWidget(self.item(i))
                if widget and widget.column_name == column_name:
                    return  # Column already exists
            
            self.add_column_item(column_name)
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def add_column_item(self, column_name):
        """Add a new column item with remove button"""
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
        
        # Create list item
        item = QListWidgetItem()
        self.addItem(item)
        
        # Create custom widget for the item
        widget = QWidget()
        widget.column_name = column_name  # Store column name for reference
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Column name label
        label = QLabel(column_name)
        label.setStyleSheet("QLabel { color: #333; }")
        layout.addWidget(label)
        
        # Remove button
        remove_btn = QPushButton("×")
        remove_btn.setMaximumSize(20, 20)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #cc3333;
            }
            QPushButton:pressed {
                background-color: #aa2222;
            }
        """)
        
        # Connect remove button to removal function
        remove_btn.clicked.connect(lambda: self.remove_column_item(item))
        layout.addWidget(remove_btn)
        
        # Set the custom widget for this item
        item.setSizeHint(widget.sizeHint())
        self.setItemWidget(item, widget)
    
    def remove_column_item(self, item):
        """Remove a column item from the list"""
        row = self.row(item)
        if row >= 0:
            self.takeItem(row)
    
    def get_selected_columns(self):
        """Get list of all selected column names"""
        columns = []
        for i in range(self.count()):
            widget = self.itemWidget(self.item(i))
            if widget and hasattr(widget, 'column_name'):
                columns.append(widget.column_name)
        return columns

class DragDropTableWidget(QTableWidget):
    """Custom QTableWidget that allows dragging column headers"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        
    def startDrag(self, supportedActions):
        if self.currentColumn() >= 0:
            # Create drag data with column index and header
            drag = QDrag(self)
            mimeData = QMimeData()
            
            # Store column index and header text
            col_idx = self.currentColumn()
            header_text = self.horizontalHeaderItem(col_idx).text() if self.horizontalHeaderItem(col_idx) else f"Column_{col_idx}"
            
            # Just store the header text (simpler than the reference)
            mimeData.setText(header_text)
            drag.setMimeData(mimeData)
            
            # Create drag pixmap for visual feedback
            from PyQt6.QtGui import QPixmap, QPainter
            pixmap = QPixmap(150, 30)
            pixmap.fill(Qt.GlobalColor.lightGray)
            painter = QPainter(pixmap)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, header_text)
            painter.end()
            drag.setPixmap(pixmap)
            
            drag.exec(Qt.DropAction.CopyAction)

class FileDropWidget(QWidget):
    """Widget that accepts file drops"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_window = parent
        
        # Setup UI
        layout = QVBoxLayout()
        self.label = QLabel("Drop .xlsx or .csv files here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
            }
        """)
        layout.addWidget(self.label)
        self.setLayout(layout)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith(('.xlsx', '.csv')):
                        event.acceptProposedAction()
                        return
        event.ignore()
        
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith(('.xlsx', '.csv')):
                        self.parent_window.load_file(file_path)
                        event.acceptProposedAction()
                        return
        event.ignore()

class StatisticalTestGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_dataframe = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Statistical Testing GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - File staging and data display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File drop area
        file_drop_label = QLabel("Data Staging Area")
        file_drop_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(file_drop_label)
        
        self.file_drop_widget = FileDropWidget(self)
        self.file_drop_widget.setMaximumHeight(100)
        left_layout.addWidget(self.file_drop_widget)
        
        # Data display table
        data_display_label = QLabel("Data Display")
        data_display_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(data_display_label)
        
        self.data_table = DragDropTableWidget()
        self.data_table.setAlternatingRowColors(True)
        left_layout.addWidget(self.data_table)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Column selection and testing
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Column selection area
        cocking_label = QLabel("Dataset Selection Area")
        cocking_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(cocking_label)
        
        cocking_instruction = QLabel("Drag column headers here to select datasets for comparison")
        cocking_instruction.setStyleSheet("color: #666; font-style: italic;")
        right_layout.addWidget(cocking_instruction)
        
        self.column_list = DragDropListWidget()
        self.column_list.setMaximumHeight(150)
        right_layout.addWidget(self.column_list)
        
        # Clear columns button
        clear_button = QPushButton("Clear Selected Columns")
        clear_button.clicked.connect(self.clear_columns)
        right_layout.addWidget(clear_button)
        
        # Test selection
        test_label = QLabel("Statistical Test Selection")
        test_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(test_label)
        
        self.test_combo = QComboBox()
        self.test_combo.addItems([
            "Student's t-test (independent)",
            "Student's t-test (paired)",
            "Welch's t-test (independent)",
            "Welch's t-test (paired)",
            "One-way ANOVA",
            "Mann-Whitney U test",
            "Correlation analysis (Pearson)",
            "Normality test (Shapiro-Wilk)",
            "Chi-square test of independence"
        ])
        self.test_combo.currentTextChanged.connect(self.update_test_info)
        right_layout.addWidget(self.test_combo)
        
        # Test information label
        self.test_info_label = QLabel()
        self.test_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        self.test_info_label.setWordWrap(True)
        right_layout.addWidget(self.test_info_label)
        
        # Update test info initially
        self.update_test_info()
        
        # Execute button
        self.execute_button = QPushButton("Execute Statistical Test")
        self.execute_button.clicked.connect(self.execute_test)
        self.execute_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        right_layout.addWidget(self.execute_button)
        
        # Output area
        output_label = QLabel("Test Results")
        output_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        right_layout.addWidget(output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(300)
        right_layout.addWidget(self.output_text)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([700, 500])
    
    def update_test_info(self):
        """Update the test information label based on selected test"""
        test_type = self.test_combo.currentText()
        info_text = ""
        
        if "t-test (independent)" in test_type:
            info_text = "Requires: 2 columns (independent groups)\nCompares means of two separate groups"
        elif "t-test (paired)" in test_type:
            info_text = "Requires: 2 columns (same subjects measured twice)\nCompares paired observations"
        elif "One-way ANOVA" in test_type:
            info_text = "Requires: 2+ columns\nCompares means across multiple groups"
        elif "Mann-Whitney U" in test_type:
            info_text = "Requires: 2 columns (independent groups)\nNon-parametric alternative to independent t-test"
        elif "Correlation" in test_type:
            info_text = "Requires: 2 columns\nMeasures linear relationship between variables"
        elif "Normality test" in test_type:
            info_text = "Requires: 1+ columns\nTests if data follows normal distribution"
        elif "Chi-square" in test_type:
            info_text = "Requires: 2 columns (categorical)\nTests independence between categorical variables"
        
        self.test_info_label.setText(info_text)
        
    def load_file(self, file_path):
        """Load CSV or Excel file into pandas DataFrame and display in table"""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                self.show_error("Unsupported file format. Please use .csv or .xlsx files.")
                return
                
            self.current_dataframe = df
            self.populate_table(df)
            
            # Update file drop label
            filename = os.path.basename(file_path)
            self.file_drop_widget.label.setText(f"Loaded: {filename}")
            self.file_drop_widget.label.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #e8f5e8;
                    font-size: 14px;
                    color: #2e7d2e;
                }
            """)
            
            self.output_text.append(f"Successfully loaded {filename} with {df.shape[0]} rows and {df.shape[1]} columns.")
            
        except Exception as e:
            self.show_error(f"Error loading file: {str(e)}")
            
    def populate_table(self, df):
        """Populate the table widget with DataFrame data"""
        self.data_table.setRowCount(min(df.shape[0], 1000))  # Limit display to 1000 rows
        self.data_table.setColumnCount(df.shape[1])
        
        # Set headers
        self.data_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        # Populate data (limit to first 1000 rows for performance)
        for i in range(min(df.shape[0], 1000)):
            for j in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.data_table.setItem(i, j, item)
        
        # Resize columns to content
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.resizeColumnsToContents()
        
        if df.shape[0] > 1000:
            self.output_text.append(f"Note: Displaying first 1000 rows of {df.shape[0]} total rows.")
            
    def clear_columns(self):
        """Clear all selected columns"""
        self.column_list.clear()
        
    def get_numeric_column_data(self, column_name):
        """Extract numeric data from a column, excluding NaN and non-numeric values"""
        if self.current_dataframe is None:
            return None
            
        if column_name not in self.current_dataframe.columns:
            return None
            
        column_data = self.current_dataframe[column_name]
        
        # Convert to numeric, coercing errors to NaN
        numeric_data = pd.to_numeric(column_data, errors='coerce')
        
        # Remove NaN values
        clean_data = numeric_data.dropna()
        
        return clean_data.values
    
    def get_categorical_column_data(self, column_name):
        """Extract categorical data from a column, excluding NaN values"""
        if self.current_dataframe is None:
            return None
            
        if column_name not in self.current_dataframe.columns:
            return None
            
        column_data = self.current_dataframe[column_name]
        
        # Remove NaN values
        clean_data = column_data.dropna()
        
        return clean_data.values
        
    def execute_test(self):
        """Execute the selected statistical test"""
        if self.current_dataframe is None:
            self.show_error("Please load a dataset first.")
            return
            
        # Get selected columns
        selected_columns = self.column_list.get_selected_columns()
            
        if len(selected_columns) == 0:
            self.show_error("Please select at least one column for testing.")
            return
            
        # Get test type
        test_type = self.test_combo.currentText()
        
        try:
            if "Student's t-test (independent)" in test_type:
                self.execute_ttest(selected_columns, paired=False, equal_var=True)
            elif "Student's t-test (paired)" in test_type:
                self.execute_ttest(selected_columns, paired=True, equal_var=True)
            elif "Welch's t-test (independent)" in test_type:
                self.execute_ttest(selected_columns, paired=False, equal_var=False)
            elif "Welch's t-test (paired)" in test_type:
                self.execute_ttest(selected_columns, paired=True, equal_var=False)
            elif "One-way ANOVA" in test_type:
                self.execute_anova(selected_columns)
            elif "Mann-Whitney U test" in test_type:
                self.execute_mannwhitney(selected_columns)
            elif "Correlation analysis" in test_type:
                self.execute_correlation(selected_columns)
            elif "Normality test" in test_type:
                self.execute_normality_test(selected_columns)
            elif "Chi-square test" in test_type:
                self.execute_chisquare(selected_columns)
        except Exception as e:
            self.show_error(f"Error executing test: {str(e)}")
            
    def execute_ttest(self, selected_columns, paired=False, equal_var=True):
        """Execute t-test (Student's or Welch's, paired or independent)"""
        if len(selected_columns) != 2:
            self.show_error(f"t-test requires exactly 2 columns. You have selected {len(selected_columns)} columns.")
            return
            
        # Extract numeric data from columns
        data1 = self.get_numeric_column_data(selected_columns[0])
        data2 = self.get_numeric_column_data(selected_columns[1])
        
        if data1 is None or len(data1) == 0:
            self.show_error(f"Column '{selected_columns[0]}' contains no numeric data.")
            return
        if data2 is None or len(data2) == 0:
            self.show_error(f"Column '{selected_columns[1]}' contains no numeric data.")
            return
            
        if paired and len(data1) != len(data2):
            self.show_error("Paired t-test requires equal sample sizes.")
            return
            
        # Perform appropriate t-test
        if paired:
            statistic, p_value = stats.ttest_rel(data1, data2)
        else:
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        
        # Display results
        self.output_text.clear()
        test_name = "WELCH'S" if not equal_var else "STUDENT'S"
        test_type_desc = "PAIRED" if paired else "INDEPENDENT"
        
        self.output_text.append("=" * 50)
        self.output_text.append(f"{test_name} T-TEST ({test_type_desc}) RESULTS")
        self.output_text.append("=" * 50)
        self.output_text.append(f"Group 1: {selected_columns[0]}")
        self.output_text.append(f"  Sample size (n₁): {len(data1)}")
        self.output_text.append(f"  Mean: {np.mean(data1):.4f}")
        self.output_text.append(f"  Std Dev: {np.std(data1, ddof=1):.4f}")
        self.output_text.append("")
        self.output_text.append(f"Group 2: {selected_columns[1]}")
        self.output_text.append(f"  Sample size (n₂): {len(data2)}")
        self.output_text.append(f"  Mean: {np.mean(data2):.4f}")
        self.output_text.append(f"  Std Dev: {np.std(data2, ddof=1):.4f}")
        self.output_text.append("")
        self.output_text.append("TEST STATISTICS:")
        self.output_text.append(f"  t-statistic: {statistic:.6f}")
        self.output_text.append(f"  p-value: {p_value:.6f}")
        
        if paired:
            df = len(data1) - 1
        else:
            df = len(data1) + len(data2) - 2
        self.output_text.append(f"  Degrees of freedom: {df}")
        self.output_text.append("")
        
        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            self.output_text.append(f"RESULT: Significant difference (p < {alpha})")
        else:
            self.output_text.append(f"RESULT: No significant difference (p ≥ {alpha})")
            
    def execute_anova(self, selected_columns):
        """Execute one-way ANOVA"""
        if len(selected_columns) < 2:
            self.show_error(f"One-way ANOVA requires at least 2 columns. You have selected {len(selected_columns)} columns.")
            return
            
        # Extract numeric data from all columns
        datasets = []
        dataset_info = []
        
        for col_name in selected_columns:
            data = self.get_numeric_column_data(col_name)
            if data is None or len(data) == 0:
                self.show_error(f"Column '{col_name}' contains no numeric data.")
                return
            datasets.append(data)
            dataset_info.append({
                'name': col_name,
                'n': len(data),
                'data': data
            })
            
        # Perform one-way ANOVA
        statistic, p_value = stats.f_oneway(*datasets)
        
        # Display results
        self.output_text.clear()
        self.output_text.append("=" * 50)
        self.output_text.append("ONE-WAY ANOVA RESULTS")
        self.output_text.append("=" * 50)
        
        for i, group in enumerate(dataset_info):
            self.output_text.append(f"Group {i+1}: {group['name']}")
            self.output_text.append(f"  Sample size (n): {group['n']}")
            self.output_text.append(f"  Mean: {np.mean(group['data']):.4f}")
            self.output_text.append(f"  Std Dev: {np.std(group['data'], ddof=1):.4f}")
            self.output_text.append("")
            
        self.output_text.append("TEST STATISTICS:")
        self.output_text.append(f"  F-statistic: {statistic:.6f}")
        self.output_text.append(f"  p-value: {p_value:.6f}")
        
        # Degrees of freedom
        k = len(datasets)  # number of groups
        N = sum(len(data) for data in datasets)  # total sample size
        df_between = k - 1
        df_within = N - k
        
        self.output_text.append(f"  Degrees of freedom (between): {df_between}")
        self.output_text.append(f"  Degrees of freedom (within): {df_within}")
        self.output_text.append("")
        
        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            self.output_text.append(f"RESULT: Significant difference between groups (p < {alpha})")
        else:
            self.output_text.append(f"RESULT: No significant difference between groups (p ≥ {alpha})")
            
    def execute_mannwhitney(self, selected_columns):
        """Execute Mann-Whitney U test"""
        if len(selected_columns) != 2:
            self.show_error(f"Mann-Whitney U test requires exactly 2 columns. You have selected {len(selected_columns)} columns.")
            return
            
        # Extract numeric data from columns
        data1 = self.get_numeric_column_data(selected_columns[0])
        data2 = self.get_numeric_column_data(selected_columns[1])
        
        if data1 is None or len(data1) == 0:
            self.show_error(f"Column '{selected_columns[0]}' contains no numeric data.")
            return
        if data2 is None or len(data2) == 0:
            self.show_error(f"Column '{selected_columns[1]}' contains no numeric data.")
            return
            
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Display results
        self.output_text.clear()
        self.output_text.append("=" * 50)
        self.output_text.append("MANN-WHITNEY U TEST RESULTS")
        self.output_text.append("=" * 50)
        self.output_text.append(f"Group 1: {selected_columns[0]}")
        self.output_text.append(f"  Sample size (n₁): {len(data1)}")
        self.output_text.append(f"  Median: {np.median(data1):.4f}")
        self.output_text.append(f"  Mean rank: {stats.rankdata(np.concatenate([data1, data2]))[:len(data1)].mean():.2f}")
        self.output_text.append("")
        self.output_text.append(f"Group 2: {selected_columns[1]}")
        self.output_text.append(f"  Sample size (n₂): {len(data2)}")
        self.output_text.append(f"  Median: {np.median(data2):.4f}")
        self.output_text.append(f"  Mean rank: {stats.rankdata(np.concatenate([data1, data2]))[-len(data2):].mean():.2f}")
        self.output_text.append("")
        self.output_text.append("TEST STATISTICS:")
        self.output_text.append(f"  U-statistic: {statistic:.6f}")
        self.output_text.append(f"  p-value: {p_value:.6f}")
        self.output_text.append("")
        
        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            self.output_text.append(f"RESULT: Significant difference (p < {alpha})")
        else:
            self.output_text.append(f"RESULT: No significant difference (p ≥ {alpha})")
            
    def execute_correlation(self, selected_columns):
        """Execute Pearson correlation analysis"""
        if len(selected_columns) != 2:
            self.show_error(f"Correlation analysis requires exactly 2 columns. You have selected {len(selected_columns)} columns.")
            return
            
        # Extract numeric data from columns
        data1 = self.get_numeric_column_data(selected_columns[0])
        data2 = self.get_numeric_column_data(selected_columns[1])
        
        if data1 is None or len(data1) == 0:
            self.show_error(f"Column '{selected_columns[0]}' contains no numeric data.")
            return
        if data2 is None or len(data2) == 0:
            self.show_error(f"Column '{selected_columns[1]}' contains no numeric data.")
            return
            
        # For correlation, we need paired data
        if len(data1) != len(data2):
            self.show_error("Correlation analysis requires equal sample sizes (paired observations).")
            return
            
        # Perform Pearson correlation
        correlation, p_value = stats.pearsonr(data1, data2)
        
        # Display results
        self.output_text.clear()
        self.output_text.append("=" * 50)
        self.output_text.append("PEARSON CORRELATION ANALYSIS RESULTS")
        self.output_text.append("=" * 50)
        self.output_text.append(f"Variable 1: {selected_columns[0]}")
        self.output_text.append(f"Variable 2: {selected_columns[1]}")
        self.output_text.append(f"Sample size (n): {len(data1)}")
        self.output_text.append("")
        self.output_text.append("CORRELATION STATISTICS:")
        self.output_text.append(f"  Pearson correlation coefficient (r): {correlation:.6f}")
        self.output_text.append(f"  p-value: {p_value:.6f}")
        self.output_text.append(f"  R-squared (r²): {correlation**2:.6f}")
        self.output_text.append("")
        
        # Interpretation of correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
            
        direction = "positive" if correlation > 0 else "negative"
        
        self.output_text.append(f"INTERPRETATION:")
        self.output_text.append(f"  Correlation strength: {strength}")
        self.output_text.append(f"  Correlation direction: {direction}")
        self.output_text.append("")
        
        # Statistical significance
        alpha = 0.05
        if p_value < alpha:
            self.output_text.append(f"RESULT: Significant correlation (p < {alpha})")
        else:
            self.output_text.append(f"RESULT: No significant correlation (p ≥ {alpha})")
            
    def execute_normality_test(self, selected_columns):
        """Execute Shapiro-Wilk normality test"""
        if len(selected_columns) == 0:
            self.show_error("Normality test requires at least 1 column.")
            return
            
        self.output_text.clear()
        self.output_text.append("=" * 50)
        self.output_text.append("SHAPIRO-WILK NORMALITY TEST RESULTS")
        self.output_text.append("=" * 50)
        
        for col_name in selected_columns:
            data = self.get_numeric_column_data(col_name)
            
            if data is None or len(data) == 0:
                self.output_text.append(f"Column '{col_name}': No numeric data available")
                self.output_text.append("")
                continue
                
            if len(data) < 3:
                self.output_text.append(f"Column '{col_name}': Insufficient data (n < 3)")
                self.output_text.append("")
                continue
                
            if len(data) > 5000:
                self.output_text.append(f"Column '{col_name}': Sample too large for Shapiro-Wilk (n > 5000)")
                self.output_text.append("Consider using Kolmogorov-Smirnov test for large samples.")
                self.output_text.append("")
                continue
                
            # Perform Shapiro-Wilk test
            statistic, p_value = stats.shapiro(data)
            
            self.output_text.append(f"Column: {col_name}")
            self.output_text.append(f"  Sample size (n): {len(data)}")
            self.output_text.append(f"  Mean: {np.mean(data):.4f}")
            self.output_text.append(f"  Std Dev: {np.std(data, ddof=1):.4f}")
            self.output_text.append(f"  Shapiro-Wilk statistic: {statistic:.6f}")
            self.output_text.append(f"  p-value: {p_value:.6f}")
            
            # Interpretation
            alpha = 0.05
            if p_value < alpha:
                self.output_text.append(f"  RESULT: Data significantly deviates from normal distribution (p < {alpha})")
            else:
                self.output_text.append(f"  RESULT: Data appears normally distributed (p ≥ {alpha})")
            self.output_text.append("")
            
    def execute_chisquare(self, selected_columns):
        """Execute Chi-square test of independence"""
        if len(selected_columns) != 2:
            self.show_error(f"Chi-square test requires exactly 2 columns. You have selected {len(selected_columns)} columns.")
            return
            
        # Extract categorical data from columns
        data1 = self.get_categorical_column_data(selected_columns[0])
        data2 = self.get_categorical_column_data(selected_columns[1])
        
        if data1 is None or len(data1) == 0:
            self.show_error(f"Column '{selected_columns[0]}' contains no data.")
            return
        if data2 is None or len(data2) == 0:
            self.show_error(f"Column '{selected_columns[1]}' contains no data.")
            return
            
        if len(data1) != len(data2):
            self.show_error("Chi-square test requires equal sample sizes (paired observations).")
            return
            
        # Create contingency table
        try:
            contingency_table = pd.crosstab(data1, data2)
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Display results
            self.output_text.clear()
            self.output_text.append("=" * 50)
            self.output_text.append("CHI-SQUARE TEST OF INDEPENDENCE RESULTS")
            self.output_text.append("=" * 50)
            self.output_text.append(f"Variable 1: {selected_columns[0]}")
            self.output_text.append(f"Variable 2: {selected_columns[1]}")
            self.output_text.append(f"Sample size (n): {len(data1)}")
            self.output_text.append("")
            
            self.output_text.append("CONTINGENCY TABLE:")
            self.output_text.append(str(contingency_table))
            self.output_text.append("")
            
            self.output_text.append("TEST STATISTICS:")
            self.output_text.append(f"  Chi-square statistic: {chi2:.6f}")
            self.output_text.append(f"  p-value: {p_value:.6f}")
            self.output_text.append(f"  Degrees of freedom: {dof}")
            self.output_text.append("")
            
            # Check assumptions
            min_expected = np.min(expected)
            cells_below_5 = np.sum(expected < 5)
            total_cells = expected.size
            
            self.output_text.append("ASSUMPTION CHECK:")
            self.output_text.append(f"  Minimum expected frequency: {min_expected:.2f}")
            self.output_text.append(f"  Cells with expected frequency < 5: {cells_below_5}/{total_cells}")
            
            if min_expected < 1 or (cells_below_5 / total_cells) > 0.2:
                self.output_text.append("  WARNING: Chi-square assumptions may be violated!")
                self.output_text.append("  Consider Fisher's exact test for small samples.")
            else:
                self.output_text.append("  Chi-square assumptions satisfied.")
            self.output_text.append("")
            
            # Calculate effect size (Cramér's V)
            n = len(data1)
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            self.output_text.append(f"EFFECT SIZE:")
            self.output_text.append(f"  Cramér's V: {cramers_v:.4f}")
            self.output_text.append("")
            
            # Interpretation
            alpha = 0.05
            if p_value < alpha:
                self.output_text.append(f"RESULT: Significant association between variables (p < {alpha})")
            else:
                self.output_text.append(f"RESULT: No significant association between variables (p ≥ {alpha})")
                
        except Exception as e:
            self.show_error(f"Error creating contingency table: {str(e)}")
            
    def show_error(self, message):
        """Show error message"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()
        self.output_text.append(f"ERROR: {message}")

def main(app=None):
    if app is None:
        app = QApplication(sys.argv)
        should_exec = True
    else:
        should_exec = False
    
    window = StatisticalTestGUI()
    window.show()
    
    if should_exec:
        sys.exit(app.exec())
    else:
        return window

if __name__ == "__main__":
    main()