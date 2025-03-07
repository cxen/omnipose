"""
Cellpose GUI module
"""
import sys
import logging
from logging import info
import os
import pathlib
import warnings
import datetime
import time
import signal
import importlib

# Set up Qt binding
def _setup_qt():
    try:
        import PySide6
        return
    except ImportError:
        pass
    raise ImportError("No Qt binding found. Please install PySide6")

_setup_qt()

# Configure logging
from omnipose.logger import setup_logger
logger = setup_logger('gui')

# Import PySide6 before PyQtGraph to ensure proper Qt binding
try:
    from PySide6 import QtCore
except ImportError:
    logger.warning("PySide6 not found, falling back to available Qt binding")

# Import utility functions with absolute imports
from cellpose_omni.utils import download_url_to_file, masks_to_outlines, diameters 

# Configure PyQtGraph
import pyqtgraph as pg

# Logo and test files setup
ICON_PATH = pathlib.Path.home().joinpath('.omnipose','logo.png')
ICON_URL = 'https://github.com/kevinjohncutler/omnipose/blob/main/gui/logo.png?raw=true'

# Test files
from cellpose_omni.io import check_dir
op_dir = pathlib.Path.home().joinpath('.omnipose','test_files')
check_dir(op_dir)
files = ['Sample000033.png','Sample000193.png','Sample000252.png','Sample000306.tiff','e1t1_crop.tif']
test_images = [pathlib.Path.home().joinpath(op_dir, f) for f in files]
for path,file in zip(test_images,files):
    if not path.is_file():
        download_url_to_file('https://github.com/kevinjohncutler/omnipose/blob/main/docs/test_files/'+file+'?raw=true',
                                path, progress=True)
PRELOAD_IMAGE = str(test_images[-1])
DEFAULT_MODEL = 'bact_phase_omni'

if not ICON_PATH.is_file():
    print('downloading logo from', ICON_URL,'to', ICON_PATH)
    download_url_to_file(ICON_URL, ICON_PATH, progress=True)

# Set up gamma icon
GAMMA_PATH = pathlib.Path.home().joinpath('.omnipose','gamma.svg')
BRUSH_PATH = pathlib.Path.home().joinpath('.omnipose','brush.svg')
GAMMA_URL = 'https://github.com/kevinjohncutler/omnipose/blob/main/gui/gamma.svg?raw=true'   
if not GAMMA_PATH.is_file():
    print('downloading gamma icon from', GAMMA_URL,'to', GAMMA_PATH)
    download_url_to_file(GAMMA_URL, GAMMA_PATH, progress=True)

# Import GUI class - Use only ABSOLUTE imports
GUI = None
try:
    # Method 1: Direct absolute import
    try:
        # Use absolute import ONLY - no relative imports
        from cellpose_omni.gui.gui import GUI as _GUI
        GUI = _GUI
        logger.info("Successfully imported GUI from cellpose_omni.gui.gui")
    except ImportError as e1:
        # Method 2: Check if gui.py exists in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        gui_file = os.path.join(current_dir, 'gui.py')
        
        if os.path.exists(gui_file):
            # Use importlib to load the module directly 
            spec = importlib.util.spec_from_file_location("gui_module", gui_file)
            gui_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gui_module)
            GUI = gui_module.GUI
            logger.info(f"Successfully imported GUI from file: {gui_file}")
        else:
            raise ImportError(f"gui.py not found in directory: {current_dir}")
except Exception as e:
    logger.error(f"Failed to import GUI: {e}")
    # Create placeholder GUI class to avoid errors
    class GUI:
        def __init__(self, *args, **kwargs):
            print("GUI could not be initialized due to import errors")
            print(f"Error: {e}")
        def run(self):
            print("GUI could not be initialized due to import errors")
            return None

# Make the GUI class available
__all__ = ['GUI']
