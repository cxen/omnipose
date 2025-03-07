"""
Cellpose Omnipose package
"""
# Define modules that should be available when imported
__all__ = ['models', 'utils', 'io', 'gui', 'core']

# Import GUI if available
try:
    from . import gui
except ImportError:
    pass

import pkg_resources
__version__ = pkg_resources.get_distribution("omnipose").version
def __getattr__(name):
    if name in __all__:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'omnipose' has no attribute '{name}'")
