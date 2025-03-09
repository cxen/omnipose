"""
Omnipose package initialization
"""
import logging
from .logger import setup_logger
from .gpu import get_device, gpu_help

__version__ = '0.10.0'  # Update with your actual version

# Set up main logger
logger = setup_logger('omnipose')

# Import compatibility modules early to ensure patches are applied
from . import compatibility

# Simple function to launch the GUI
def launch_gui():
    """
    Launch the Omnipose GUI
    
    This function properly imports and creates the GUI object
    """
    try:
        # Create QApplication first
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
            
        # Import shutdown handler
        from .shutdown import handler as shutdown_handler
        shutdown_handler.connect_to_app(app)
            
        # Now import GUI and create instance
        from cellpose_omni.gui.gui import GUI
        gui = GUI()
        
        # Connect cleanup to our shutdown handler
        shutdown_handler.shutdown_requested.connect(gui.cleanup_resources)
        
        return gui
    except ImportError as e:
        print(f"Failed to import GUI: {e}")
        print("Make sure all GUI dependencies are installed:")
        print("pip install omnipose[gui]")
        return None

# Command-line entry point function
def run_omnipose():
    """Command-line entry point for Omnipose"""
    import sys
    from .__main__ import main
    sys.exit(main())

# Set NUMEXPR_MAX_THREADS to the number of available CPU cores
import os
import multiprocessing
os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())

# controlled import to prevent MIP print statement 
# import mip
# from aicsimageio import AICSImage 

# Use of sets...
import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import pkg_resources

# Apply compatibility patches

__all__ = ['core', 'utils', 'loss', 'plot', 'misc', 'cli', 'data', 'gpu', 'stacks', 'measure']

def __getattr__(name):
    if name in __all__:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'omnipose' has no attribute '{name}'")
