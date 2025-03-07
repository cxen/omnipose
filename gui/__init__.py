"""
Omnipose GUI module - redirects to the real GUI implementation in cellpose_omni
"""
import os
import sys
import logging
import importlib.util
import traceback

# Configure logger
logger = logging.getLogger(__name__)

def launch_gui(*args, **kwargs):
    """Proxy function that launches the real GUI from the cellpose_omni package"""
    try:
        # Method 1: Try direct import
        try:
            from cellpose_omni.gui.gui import GUI
            gui = GUI(*args, **kwargs)
            return gui
        except ImportError:
            pass
        
        # Method 2: Try importing from gui module if it exists
        try:
            import cellpose_omni.gui
            if hasattr(cellpose_omni.gui, 'GUI'):
                gui = cellpose_omni.gui.GUI(*args, **kwargs)
                return gui
        except (ImportError, AttributeError):
            pass
            
        # Method 3: Try to locate gui.py and load it directly
        try:
            # Find cellpose_omni package
            import cellpose_omni
            gui_file = os.path.join(os.path.dirname(cellpose_omni.__file__), 'gui', 'gui.py')
            
            if os.path.exists(gui_file):
                spec = importlib.util.spec_from_file_location("gui_module", gui_file)
                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)
                gui = gui_module.GUI(*args, **kwargs)
                return gui
        except Exception:
            pass
            
        # If we get here, we couldn't find the GUI module
        logger.error("All GUI import methods failed")
        print("GUI could not be initialized due to missing dependencies")
        print("\nTo fix GUI issues, try:")
        print("1. Install GUI dependencies: pip install omnipose[gui]")
        print("2. Ensure PySide6 is installed: pip install PySide6")
        print("3. Check that pyqtgraph is installed: pip install pyqtgraph")
        return None
    except Exception as e:
        logger.error(f"Failed to import Cellpose GUI: {e}")
        print(f"GUI initialization error: {e}")
        traceback.print_exc()
        return None

# Define a GUI class that delegates to the real implementation
class OmniposeGUI:
    """Proxy class that delegates to the real GUI implementation"""
    def __init__(self, *args, **kwargs):
        self._gui = None
        try:
            gui = launch_gui(*args, **kwargs)
            if gui is not None:
                self._gui = gui
        except Exception as e:
            logger.error(f"Failed to create GUI: {e}")
            print(f"GUI could not be initialized: {e}")
    
    def __getattr__(self, name):
        if self._gui is None:
            raise AttributeError(f"GUI not initialized, cannot access {name}")
        return getattr(self._gui, name)

    def run(self):
        if self._gui is None:
            print("GUI not initialized, cannot run")
            return None
        return self._gui.run()
