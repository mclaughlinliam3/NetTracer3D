import subprocess
import sys
import threading
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QWidget

class CellposeGUILauncher:
    """Simple launcher for cellpose GUI in PyQt6 applications."""
    
    def __init__(self, parent_widget=None):
        """
        Initialize the launcher.
        
        Args:
            parent_widget: PyQt6 widget for showing message boxes (optional)
        """
        self.parent_widget = parent_widget
        self.cellpose_process = None
    
    def launch_cellpose_gui(self, image_path=None, working_directory=None):
        """
        Launch cellpose GUI in a separate thread.
        
        Args:
            image_path (str, optional): Path to image file to load automatically
            working_directory (str, optional): Directory to start cellpose in
        
        Returns:
            bool: True if launch was initiated successfully
        """
        def run_cellpose():
            """Function to run in separate thread."""
            try:
                # Build command
                cmd = [sys.executable, "-m", "cellpose"]
                
                # Add image path if provided
                if image_path and Path(image_path).exists():
                    cmd.extend(["--image_path", str(image_path)])
                
                # Set working directory
                cwd = working_directory if working_directory else None
                
                # Launch cellpose GUI
                self.cellpose_process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Optional: wait for process to complete
                # self.cellpose_process.wait()
                
            except Exception as e:
                if self.parent_widget:
                    # Show error in main thread
                    self.show_error(f"Failed to launch cellpose GUI: {str(e)}")
                else:
                    print(f"Failed to launch cellpose GUI: {str(e)}")
        
        try:
            # Start cellpose in separate thread
            thread = threading.Thread(target=run_cellpose, daemon=True)
            thread.start()
            
            if self.parent_widget:
                self.show_info("Cellpose GUI launched!")
            else:
                print("Cellpose GUI launched!")
            
            return True
            
        except Exception as e:
            if self.parent_widget:
                self.show_error(f"Failed to start cellpose thread: {str(e)}")
            else:
                print(f"Failed to start cellpose thread: {str(e)}")
            return False
    
    def launch_with_directory(self, directory_path):
        """
        Launch cellpose GUI with a specific directory.
        
        Args:
            directory_path (str): Directory containing images
        """
        cmd_args = ["--dir", str(directory_path)]
        return self.launch_cellpose_gui_with_args(cmd_args, working_directory=directory_path)
    
    def launch_cellpose_gui_with_args(self, additional_args=None, working_directory=None):
        """
        Launch cellpose GUI with custom arguments.
        
        Args:
            additional_args (list): List of additional command line arguments
            working_directory (str): Working directory for cellpose
        """
        def run_cellpose_custom():
            try:
                cmd = [sys.executable, "-m", "cellpose"]
                
                if additional_args:
                    cmd.extend(additional_args)
                
                cwd = working_directory if working_directory else None
                
                self.cellpose_process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
            except Exception as e:
                if self.parent_widget:
                    self.show_error(f"Failed to launch cellpose GUI: {str(e)}")
                else:
                    print(f"Failed to launch cellpose GUI: {str(e)}")
        
        try:
            thread = threading.Thread(target=run_cellpose_custom, daemon=True)
            thread.start()
            return True
        except Exception as e:
            if self.parent_widget:
                self.show_error(f"Failed to start cellpose: {str(e)}")
            return False
    
    def is_cellpose_running(self):
        """
        Check if cellpose process is still running.
        
        Returns:
            bool: True if cellpose is still running
        """
        if self.cellpose_process is None:
            return False
        
        return self.cellpose_process.poll() is None
    
    def close_cellpose(self):
        """Terminate the cellpose process if running."""
        if self.cellpose_process and self.is_cellpose_running():
            try:
                self.cellpose_process.terminate()
                self.cellpose_process.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                self.cellpose_process.kill()  # Force kill if it doesn't terminate
            except Exception as e:
                print(f"Error closing cellpose: {e}")
    
    def show_info(self, message):
        """Show info message if parent widget available."""
        if self.parent_widget:
            QMessageBox.information(self.parent_widget, "Cellpose Launcher", message)
    
    def show_error(self, message):
        """Show error message if parent widget available."""
        if self.parent_widget:
            QMessageBox.critical(self.parent_widget, "Cellpose Error", message)