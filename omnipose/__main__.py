"""
Main entry point for Omnipose
"""
import sys
import os
import logging
import importlib.util
import traceback
from .gpu import gpu_help, get_device

def main():
    # Get GPU info first
    device = get_device()
    
    # Try to launch GUI with robust error handling
    try:
        # Create QApplication first, before importing GUI modules
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        
        # Now that QApplication exists, import GUI
        try:
            from cellpose_omni.gui.gui import GUI
            gui = GUI()
            gui.show()  # Just show the GUI, don't call run() which creates another QApplication
            return app.exec()  # Use the existing QApplication's event loop
        except ImportError:
            pass
        
        # Alternative methods with QApplication already created
        try:
            import cellpose_omni
            gui_path = os.path.join(os.path.dirname(cellpose_omni.__file__), 'gui', 'gui.py')
            
            if os.path.exists(gui_path):
                print(f"Found GUI at {gui_path}")
                spec = importlib.util.spec_from_file_location("cellpose_gui", gui_path)
                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)
                
                if hasattr(gui_module, 'GUI'):
                    gui = gui_module.GUI()
                    gui.show()
                    return app.exec()
                else:
                    print(f"GUI class not found in {gui_path}")
            else:
                print(f"GUI file not found at {gui_path}")
        except Exception as e:
            print(f"Error loading GUI module: {e}")
            traceback.print_exc()
        
        # Last resort - through omnipose launcher
        try:
            from omnipose import launch_gui
            gui = launch_gui()
            if gui is not None:
                gui.show()  # Show instead of run
                return app.exec()  # Use existing QApplication
            else:
                print("GUI object was None")
        except Exception as e:
            print(f"GUI launch failed: {e}")
            
    except Exception as e:
        print(f"GUI could not be initialized due to import errors: {e}")
        traceback.print_exc()
        print("\nYou can still use Omnipose through its API or command line interface.")
        print("\nTo fix GUI issues, try:")
        print("1. Install GUI dependencies: pip install omnipose[gui]")
        print("2. Check that PySide6 and pyqtgraph are installed")
        print("3. Make sure your Python environment has all required packages")
        return 1

if __name__ == "__main__":
    sys.exit(main())

