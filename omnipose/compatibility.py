"""
Compatibility module - only providing functions when explicitly imported.
Does NOT patch any modules.
"""
import warnings
from cellpose_omni import io

# Create a wrapper for skimage.io functions that are often incorrectly referenced
class SkimageIoCompat:
    def __init__(self):
        pass
    
    def imsave(self, path, arr, **kwargs):
        warnings.warn("skimage.io.imsave is deprecated in this codebase, use io.imwrite instead.", DeprecationWarning)
        return io.imwrite(path, arr, **kwargs)
    
    def imwrite(self, path, arr, **kwargs):
        warnings.warn("skimage.io.imwrite does not exist, using io.imwrite instead.", DeprecationWarning)
        return io.imwrite(path, arr, **kwargs)

    # Add attribute checking to prevent direct attribute access errors
    def __getattr__(self, name):
        # Fallback to io module if attribute isn't found here
        if hasattr(io, name):
            return getattr(io, name)
        raise AttributeError(f"'SkimageIoCompat' object has no attribute '{name}' and it's not in io module")

# Create a singleton instance to be imported elsewhere
io_compat = SkimageIoCompat()

# Remove the automatic patching functionality
