import signal, sys, os, pathlib, warnings, datetime, time
import inspect, importlib, pkgutil

# Add this import near the top of the file
from omnipose.shutdown import handler as shutdown_handler

import numpy as np
# np.seterr(all='raise')  # Raise exceptions instead of warnings


from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt, Slot, QCoreApplication
from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QComboBox, QGridLayout, QPushButton, QCheckBox, QLabel, QProgressBar, QLineEdit, QScrollArea
from PySide6.QtGui import QPalette

from PySide6.QtCore import QPoint
from PySide6.QtGui import QCursor, QGuiApplication



import importlib
import importlib
import inspect

import pyqtgraph as pg
try:
    pg.setConfigOptions(useOpenGL=True)
except Exception as e:
    print(f"Could not set OpenGL option: {e}")

os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

from scipy.stats import mode
# from scipy.ndimage import gaussian_filter

# Import GUI components directly
from cellpose_omni.gui import guiparts, menus, io
from cellpose_omni import models, dynamics
from cellpose_omni.utils import download_url_to_file, masks_to_outlines, diameters 
from cellpose_omni.io import get_image_files, imsave, imread, check_dir #OMNI_INSTALLED
from cellpose_omni.transforms import resize_image #fixed import
from cellpose_omni.plot import disk
from omnipose.utils import normalize99, to_8_bit


OMNI_INSTALLED = 1
from cellpose_omni.gui.guiutils import checkstyle, get_unique_points, avg3d, interpZ

from cellpose_omni.gui import logger


ALLOWED_THEMES = ['light','dark']



import darkdetect
import qdarktheme
import qtawesome as qta


# no more matplotlib just for colormaps
from cmap import Colormap

from cellpose_omni.gui import MainWindowModules as submodules

from cellpose_omni.gui import PRELOAD_IMAGE, ICON_PATH

import omnipose, cellpose_omni

import importlib
import types

from omnipose.gpu import is_cuda_available, is_mps_available


def run(image=PRELOAD_IMAGE):
    start_time = time.time()  # Record start time
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    QCoreApplication.setApplicationName('Omnipose')
    app = QApplication(sys.argv)
    
    # Connect our shutdown handler to the app
    shutdown_handler.connect_to_app(app)

    # screen = app.primaryScreen()
    # New: detect monitor from mouse cursor
    cursor_pos = QCursor.pos()
    screen = QGuiApplication.screenAt(cursor_pos)
    dpi = screen.logicalDotsPerInch()
    pxr = screen.devicePixelRatio()
    size = screen.availableGeometry()
    clipboard = app.clipboard()

    # Try to clear qdarktheme cache if the method exists
    try:
        if hasattr(qdarktheme, 'clear_cache'):
            qdarktheme.clear_cache()
        # Otherwise it's likely a newer version that doesn't need cache clearing
    except Exception as e:
        print(f"Note: Couldn't clear qdarktheme cache: {e}")

    app_icon = QtGui.QIcon()
    icon_path = str(ICON_PATH.resolve())
    for i in [16,24,32,48,64,256]:
        app_icon.addFile(icon_path, QtCore.QSize(i,i)) 
    app.setWindowIcon(app_icon) 
    
    # models.download_model_weights() # does not exist
    win = MainW(size, dpi, pxr, clipboard, image=image)

    # the below code block will automatically toggle the theme with the system,
    # but the manual color definitions (everywhere I set a style sheet) can mess that up
    @Slot()
    def sync_theme_with_system() -> None:
        theme = str(darkdetect.theme()).lower()
        theme = theme if theme in ALLOWED_THEMES else 'dark' #default to dark theme 
        stylesheet = qdarktheme.load_stylesheet(theme)
        QApplication.instance().setStyleSheet(stylesheet)
        win.darkmode = theme=='dark'
        win.accent = win.palette().brush(QPalette.ColorRole.Highlight).color()
        if hasattr(win,'win'):
            win.win.setBackground("k" if win.darkmode else '#f0f0f0') #pull out real colors from theme here from example
       
       # explicitly set colors for items that don't change automatically with theme
        win.set_hist_colors()
        win.set_button_color()
        win.set_crosshair_colors()
        win.SCheckBox.update_icons() 
        # win.update_plot()
    app.paletteChanged.connect(sync_theme_with_system)             
    sync_theme_with_system()

    end_time = time.time()  # Record end time
    print(f"Total Time: {end_time - start_time:.4f} seconds")


    ret = app.exec()
    sys.exit(ret)
    

class MainW(QMainWindow):
    def __init__(self, size=None, dpi=None, pxr=None, clipboard=None, image=None):
        start_time = time.time()  # Record start time

        super(MainW, self).__init__()
        
        # Initialize QApplication if it doesn't exist
        # This ensures we can get a clipboard and screen info
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        # Auto-detect values if not provided
        if clipboard is None:
            clipboard = self.app.clipboard()
        
        # Get screen info if not provided
        if size is None or dpi is None or pxr is None:
            cursor_pos = QCursor.pos()
            screen = QGuiApplication.screenAt(cursor_pos)
            if screen is None:
                # Fallback to primary screen if cursor isn't on any screen
                screen = QGuiApplication.primaryScreen()
            
            if size is None:
                size = screen.availableGeometry()
            if dpi is None:
                dpi = screen.logicalDotsPerInch()
            if pxr is None:
                pxr = screen.devicePixelRatio()
        
        # Initialize image dimensions with defaults
        self.Lx = 512
        self.Ly = 512
        self.NZ = 1
        self.dim = 2  # Default to 2D
        self.shape = (self.Ly, self.Lx)
        
        # Dict mapping { module_name: last_mtime }
        self.module_mtimes = {}
        self.module_sources = {}  # Track source code per module for change detection
        self.class_sources = {}   # maps (module_name, class_name) to the class's source code
        # Discover & load all submodules
        self.modules = self.load_all_submodules()
        self.patch_all_submodules()
        
        
        # --- New: Register external modules ---
        self.register_external_modules()
        
        # Now load extra modules from the base directory:
        self.additional_modules = self.load_additional_modules()
        self.patch_additional_modules(self.additional_modules)

        # Start a timer to check for changes every second
        self.timer_id = self.startTimer(1000)

        
        # palette = app.palette()
        # palette.setColor(QPalette.ColorRole.ColorRole.Link, dark_palette.link().color())
        # app.setPalette(palette)

        # print(qdarktheme.load_palette().link().color())
        self.darkmode = str(darkdetect.theme()).lower() in ['none','dark'] # have to initialize; str catches None on some systems

        try:
            pg.setConfigOptions(imageAxisOrder="row-major")
        except Exception as e:
            print(f"Could not set imageAxisOrder option: {e}")
        self.clipboard = clipboard

        # self.showMaximized()
        self.setWindowTitle("Omnipose GUI")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)
        menus.omnimenu(self)
        # menus.helpmenu(self) # all of these are outdated 

        self.model_strings = models.MODEL_NAMES.copy()
        self.loaded = False
        self.imask = 0

        self.make_main_widget()
    
        
        self.make_buttons() # no longer need to return b
        # self.win.adjustSize()
        
        
        # Instantiate the Colormap object
        cmap = Colormap("gist_ncar")

        # Generate evenly spaced color samples for 2**16-1 colors
        ncell = 2**16-1
        colormap = cmap(np.linspace(0, 1, ncell))  # Directly call the colormap
        colormap = (np.array(colormap) * 255).astype(np.uint8)  # Convert to uint8

        # Stable random shuffling of colors
        np.random.seed(42)
        self.colormap = colormap[np.random.permutation(ncell)]
    
        self.undo_stack = []  # Stack to store cellpix history
        self.redo_stack = []  # Stack to store redo states
        self.max_undo_steps = 50  # Limit the number of undo steps
    

        self.is_stack = True # always loading images of same FOV? Not sure about this assumption...
        # if called with image, load it
        if image is not None:
            self.filename = image
            print('loading', self.filename)
            io._load_image(self, self.filename)

        # training settings
        d = datetime.datetime.now()
        self.training_params = {'model_index': 0,
                                'learning_rate': 0.1, 
                                'weight_decay': 0.0001, 
                                'n_epochs': 100,
                                'model_name':'CP' + d.strftime("_%Y%m%d_%H%M%S")
                               }
        
        
        # Nx, Ny from image dimension
        Nx = self.Lx
        Ny = self.Ly

        # Create the overlay item
        self.pixelGridOverlay = guiparts.GLPixelGridOverlay(Nx, Ny, parent=self)
        self.pixelGridOverlay.setVisible(False) 
        self.p0.addItem(self.pixelGridOverlay)
        
        # Move and resize the window
        cursor_pos = QCursor.pos()
        screen = QGuiApplication.screenAt(cursor_pos)

        hint = self.cwidget.sizeHint()
        title_bar_height = self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_TitleBarHeight, None, self)
        self.resize(hint.width(), hint.height() + title_bar_height)
  
        if screen is not None:
            available_rect = screen.availableGeometry()
            self.move(available_rect.topLeft() + QtCore.QPoint(50, 50))
    

        self.setAcceptDrops(True)
        self.win.show()
        self.show()
        
        end_time = time.time()  # Record end time
        print(f"Init Time: {end_time - start_time:.4f} seconds")
        
        # Connect cleanup method to shutdown signal
        shutdown_handler.shutdown_requested.connect(self.cleanup_resources)

    def load_all_submodules(self):
        """
        Dynamically imports every .py module in submodules/ 
        and returns a dict of {mod_name: module_object}.
        """
        loaded_modules = {}
        for mod_info in pkgutil.iter_modules(submodules.__path__):
            mod_name = mod_info.name
            full_name = submodules.__name__ + "." + mod_name
            
            try:
                # Import the module
                mod = importlib.import_module(full_name)
                loaded_modules[mod_name] = mod

                # Track last modification time
                mod_path = os.path.join(os.path.dirname(submodules.__file__), mod_name + ".py")
                if os.path.exists(mod_path):
                    self.module_mtimes[mod_name] = self.get_mtime(mod_path)
                    print(f"🔄 Tracking {mod_name}: {self.module_mtimes[mod_name]}")
                else:
                    print(f"⚠️ Module file not found: {mod_path}")

            except Exception as e:
                print(f"❌ Error loading submodule {mod_name}: {e}")
    
        return loaded_modules
        
        
    def timerEvent(self, event):
        """
        Called every second; checks if any submodule .py changed on disk.
        If so, reload it and patch again.
        """
        for mod_name, mod in self.modules.items():
            mod_path = os.path.join(os.path.dirname(submodules.__file__), mod_name + ".py")
            new_mtime = self.get_mtime(mod_path)
            if new_mtime != self.module_mtimes[mod_name]:
                print(f"🔄 Reloading submodule '{mod_name}'...")
                new_mod = self.recursive_reload_and_update(mod)
                self.modules[mod_name] = new_mod
                self.module_mtimes[mod_name] = new_mtime
                self.patch_submodule(new_mod)
        # Check additional modules - fixed reference to non-existent method
        self.check_additional_modules()
        # Update instance fields
        self.update_mainw_fields()
        
    def mouse_moved(self, pos):
        """
        Handle mouse movement events to update cursor highlighting.
        This is connected to the scene's sigMouseMoved signal.
        
        Parameters
        ----------
        pos : QtCore.QPointF
            The position of the mouse cursor in scene coordinates
        """
        # Call update_highlight from the cursor module if it exists
        if hasattr(self, 'update_highlight'):
            self.update_highlight(pos)
            
    def update_mainw_fields(self):
        """
        Update methods on instance fields whose classes (from allowed modules)
        have changed in source code. Only methods defined directly on the class
        are updated.
        """
        allowed_mods = ("cellpose_omni.gui.guiparts",)
        fields_by_module = {}
        for field_name, instance in self.__dict__.items():
            if instance is None or not hasattr(instance, '__class__'):
                continue
            cls = instance.__class__
            module_name = cls.__module__
            if module_name not in allowed_mods:
                continue
            fields_by_module.setdefault(module_name, []).append((field_name, instance))
        
        for module_name, field_list in fields_by_module.items():
            try:
                module = importlib.import_module(module_name)
                mod_file = getattr(module, '__file__', None)
                if mod_file is None or not os.path.exists(mod_file):
                    continue
                new_mtime = self.get_mtime(mod_file)
            except Exception as e:
                print(f"Error checking module time for '{module_name}': {e}")
                continue
            
            stored_mtime = self.module_mtimes.get(module_name)
            if stored_mtime is not None and new_mtime == stored_mtime:
                continue  # Skip if the module file is unchanged

            for field_name, instance in field_list:
                old_cls = instance.__class__
                try:
                    new_cls = getattr(module, old_cls.__name__)
                except Exception as e:
                    print(f"Error retrieving new class for field '{field_name}': {e}")
                    continue

                # Even if new_cls is a new object, we only want to update if its source has changed.
                try:
                    import inspect
                    new_source = inspect.getsource(new_cls)
                except Exception as e:
                    print(f"Error getting source for {old_cls.__name__}: {e}")
                    new_source = None

                key = (module_name, old_cls.__name__)
                old_source = self.class_sources.get(key)
                # Normalize source code by stripping extra whitespace
                new_source_norm = new_source.strip() if new_source else None
                old_source_norm = old_source.strip() if old_source else None

                if old_source_norm is not None and new_source_norm == old_source_norm:
                    print(f"Source unchanged for {old_cls.__name__}; skipping update for instance '{field_name}'.")
                    continue

                print(f"Updating instance '{field_name}' of class {old_cls.__name__}: {id(old_cls)} -> {id(new_cls)}")
                   # Update the instance’s class pointer:
                for method_name, new_method in new_cls.__dict__.items():
                    if callable(new_method):
                        try:
                            bound_method = new_method.__get__(instance, new_cls)
                            setattr(instance, method_name, bound_method)
                            print(f"  Updated {field_name}.{method_name}")
                        except Exception as e:
                            print(f"  Failed to update {field_name}.{method_name}: {e}")
                
                if new_source:
                    self.class_sources[key] = new_source

            self.module_mtimes[module_name] = new_mtime
            
    def register_external_modules(self):
        """
        Register all submodules for each external package you care about (e.g. omnipose and cellpose_omni).
        This populates self.external_submodules as a dict mapping the full module name to a dict with keys:
        "module": the module object,
        "mtime": the module file's last modification time.
        """
        import pkgutil
        self.external_submodules = {}
        try:
            import omnipose
            import cellpose_omni
        except ImportError as e:
            print("Error importing external packages:", e)
            return

        packages = [omnipose, cellpose_omni]
        for pkg in packages:
            pkg_name = pkg.__name__
            try:
                for mod_info in pkgutil.iter_modules(pkg.__path__, prefix=pkg_name + "."):
                    mod_name = mod_info.name
                    try:
                        submod = importlib.import_module(mod_name)
                        mod_file = getattr(submod, '__file__', None)
                        if mod_file and os.path.exists(mod_file):
                            self.external_submodules[mod_name] = {
                                "module": submod,
                                "mtime": self.get_mtime(mod_file)
                            }
                            # Optionally, print for debugging:
                            # print(f"Registered external submodule: {mod_name}")
                    except Exception as e:
                        print(f"Failed to register submodule {mod_name}: {e}")
            except Exception as e:
                print(f"Error iterating modules in package {pkg_name}: {e}")

                    
                    
    def patch_all_submodules(self):
        """Call patch_submodule() for each loaded module."""
        for mod in self.modules.values():
            self.patch_submodule(mod)

    def patch_submodule(self, mod):
        """
        For each top-level function in a submodule, 
        bind it as a method on MainWindow.
        """
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            bound_method = obj.__get__(self, type(self))
            setattr(self, name, bound_method)
            # print(f"Patched {name}() from '{mod.__name__}' -> self.{name}")

    def call_patched_method(self, func_name):
        """Call a hot-reloaded method by name."""
        if hasattr(self, func_name):
            getattr(self, func_name)()
        else:
            print(f"No method '{func_name}' found on MainWindow")

    # def get_mtime(self, filepath):
    #     """Return last-modified time of a file, or 0 if missing."""
    #     return os.path.getmtime(filepath) if os.path.exists(filepath) else 0
        
    def get_mtime(self, filepath):
        return int(os.path.getmtime(filepath)) if os.path.exists(filepath) else 0
        
    def load_additional_modules(self):
        """
        Dynamically import any extra Python modules in the same directory as this file
        (e.g. the directory that contains __init__.py and gui.py) that you want to autoreload.
        Returns a dict mapping module names to module objects.
        """
        loaded = {}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # List all .py files in the base directory, excluding __init__.py and this file (e.g. gui.py)
        for fname in os.listdir(base_dir):
            if fname.endswith('.py') and fname not in ['__init__.py', os.path.basename(__file__)]:
                mod_name = fname[:-3]
                try:
                    # Construct the full module name based on the package name
                    full_name = __package__ + '.' + mod_name  # __package__ is defined because you have an __init__.py
                    mod = importlib.import_module(full_name)
                    loaded[mod_name] = mod
                    # Register its modification time using its file path.
                    mod_path = os.path.join(base_dir, fname)
                    self.module_mtimes[mod_name] = self.get_mtime(mod_path)
                    print(f"Loaded additional module: {full_name}")
                except Exception as e:
                    print(f"Error loading module {fname}: {e}")
        return loaded

    def patch_additional_modules(self, additional_mods):
        """
        For each module in the additional modules dict,
        patch every top-level function onto the MainW instance.
        """
        for mod in additional_mods.values():
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                bound_method = obj.__get__(self, type(self))
                setattr(self, name, bound_method)
                # Optionally, uncomment to print debug info:
                # print(f"Patched additional function {name} from {mod.__name__} onto self.")

    def check_additional_modules(self):
        """
        Check the extra modules (loaded from the base directory) for changes.
        If a file’s mtime has increased, reload it and patch its functions.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for mod_name, mod in self.additional_modules.items():
            mod_path = os.path.join(base_dir, mod_name + ".py")
            new_mtime = self.get_mtime(mod_path)
            if new_mtime != self.module_mtimes.get(mod_name, 0):
                print(f"🔄 Reloading additional module '{mod_name}'...")
                try:
                    # new_mod = importlib.reload(mod)
                    new_mod = self.recursive_reload_and_update(mod)
                    
                    self.additional_modules[mod_name] = new_mod
                    self.module_mtimes[mod_name] = new_mtime
                    self.patch_additional_modules({mod_name: new_mod})
                except Exception as e:
                    print(f"Error reloading additional module {mod_name}: {e}")
                    
    def recursive_reload_and_update(self, module, visited=None):
        """
        Recursively reloads a module (and any submodules whose names start with module.__name__)
        and then updates the __class__ pointer for all existing instances of classes defined in that module.
        This mimics the behavior of IPython's %autoreload 2.
        """
        if visited is None:
            visited = set()
        if module in visited:
            return module
        visited.add(module)
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, types.ModuleType) and attr.__name__.startswith(module.__name__):
                    self.recursive_reload_and_update(attr, visited)
            except Exception as e:
                print(f"Failed to recursively reload attribute {attr_name}: {e}")
                continue
        old_module = module
        new_module = importlib.reload(module)
        self.update_class_methods(old_module, new_module)
        return new_module

    def update_class_methods(self, old_module, new_module):
        """
        For every class defined in new_module that also existed in old_module,
        update the existing class’s dictionary with any callable attributes from new_module.
        This means that any subsequent lookups on existing instances (via self.method)
        will pick up the new code.
        """
        for name in dir(new_module):
            new_obj = getattr(new_module, name)
            if isinstance(new_obj, type):
                try:
                    old_obj = getattr(old_module, name)
                except AttributeError:
                    continue
                if old_obj is not new_obj:
                    for key, new_method in new_obj.__dict__.items():
                        if callable(new_method):
                            setattr(old_obj, key, new_method)
                            print(f"Updated {name}.{key}")
                            
    # Add a run method to use existing QApplication
    def run(self):
        """
        Run the GUI without creating a new QApplication.
        This is meant to be called after creating a MainW instance
        when a QApplication already exists.
        """
        self.show()
        if QApplication.instance() is not None:
            return QApplication.instance().exec()
        else:
            print("Error: No QApplication instance found")
            return 1

    def cleanup_resources(self):
        """
        Clean up resources before application exit.
        This prevents errors during shutdown by properly releasing graphics resources.
        """
        logger.info("Cleaning up GUI resources before shutdown")
        
        # Hide the window first to prevent additional rendering
        self.hide()
        
        # Clear pyqtgraph items that might cause issues
        if hasattr(self, 'win') and self.win is not None:
            # Remove items from the ViewBox that might cause issues
            if hasattr(self.win, 'scene'):
                self.win.clear()
        
        # Clear image data
        if hasattr(self, 'img'):
            self.img.clear()
        
        # Reset other resources that might be accessed during shutdown
        self.stack = None
        self.flows = None
        
        logger.info("GUI resources cleaned up successfully")
                            
# prevents gui from running under import 
if __name__ == "__main__":
    run()

# Export MainW as GUI for proper importing by other modules
GUI = MainW




