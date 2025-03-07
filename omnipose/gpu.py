import os
import sys
import platform
import logging
import torch
import numpy as np
from pathlib import Path
from torch import autocast
import threading
import time
from functools import wraps

from .logger import setup_logger
gpu_logger = setup_logger('gpu')

# Configure logging
gpu_logger = logging.getLogger(__name__)

# Set PyTorch MPS environment variables for better Apple Silicon support
if sys.platform == 'darwin':
    # Enable MPS fallback for operations not yet supported in Metal
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Memory allocation settings for MPS
    os.environ["PYTORCH_MPS_MEMORY_ALLOCATOR"] = "1"
    
    # Disable sync to improve performance
    os.environ["PYTORCH_MPS_SYNC"] = "0"
    
    # Allow reserved memory for better performance
    os.environ["PYTORCH_MPS_ALLOW_RESERVED_MEMORY"] = "1"

# Platform detection
ARM = platform.machine() == 'arm64' or platform.processor() == 'arm'  # For Apple Silicon detection

# Device allocation helpers
torch_CPU = torch.device("cpu")
torch_CUDA = torch.device("cuda:0")
torch_MPS = torch.device("mps:0")  # Apple Silicon GPU

# Set default OMP threads to 1 on Apple Silicon for better performance
if platform.processor() == 'arm' and platform.system() == 'Darwin':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PARLAY_NUM_THREADS"] = "1"
    gpu_logger.info('On ARM, OMP_NUM_THREADS set to 1')

def is_cuda_available():
    """Check if CUDA is available on this system"""
    try:
        return torch.cuda.is_available()
    except (AssertionError, ImportError, RuntimeError):
        return False

def is_mps_available():
    """Check if MPS (Metal Performance Shaders) is available for Apple Silicon"""
    try:
        if ARM and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True
    except (AssertionError, ImportError, RuntimeError, AttributeError):
        pass
    return False

def timeout_detector(func):
    """Decorator that adds timeout protection to GPU operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)  # 10 second timeout
        
        if thread.is_alive():
            gpu_logger.warning("GPU operation timed out. Falling back to CPU.")
            # Force remaining operations to CPU
            os.environ["PYTORCH_MPS_ENABLE_FALLBACK"] = "1"
            # The thread will continue running but we'll use CPU from here on
            return torch_CPU
        
        if exception[0] is not None:
            gpu_logger.error(f"GPU error: {exception[0]}. Falling back to CPU.")
            os.environ["PYTORCH_MPS_ENABLE_FALLBACK"] = "1"
            return torch_CPU
            
        return result[0]
    return wrapper

@timeout_detector
def get_device_preference():
    """Get device preference from environment variables"""
    # Environment variable to force CPU usage
    if os.environ.get('OMNIPOSE_FORCE_CPU', '0') == '1':
        gpu_logger.info("OMNIPOSE_FORCE_CPU=1, using CPU")
        return torch_CPU
    
    # Check for MPS (Apple Silicon GPU) availability
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Check if user has disabled MPS
        if os.environ.get('OMNIPOSE_DISABLE_MPS', '0') == '1':
            gpu_logger.info("MPS available but disabled by OMNIPOSE_DISABLE_MPS=1")
            if torch.cuda.is_available():
                return torch_CUDA
            else:
                return torch_CPU
                
        # Use MPS for Apple Silicon
        gpu_logger.info("mps detected and enabled!")
        # Enable fallbacks for operations not supported by MPS
        os.environ["PYTORCH_MPS_ENABLE_FALLBACK"] = "1"
        return torch_MPS
        
    # Check for CUDA
    if torch.cuda.is_available():
        return torch_CUDA
    
    return torch_CPU

@timeout_detector
def get_device():
    """Get the device to use for torch operations based on preferences and GPU availability"""
    device_pref = get_device_preference()
    
    # If MPS (Apple Silicon GPU)
    if str(device_pref) == 'mps':
        try:
            # Test a small tensor operation to verify MPS is working
            test_tensor = torch.rand(10, 10, device=device_pref)
            test_result = test_tensor + test_tensor
            del test_tensor, test_result
            return device_pref
        except Exception as e:
            gpu_logger.warning(f"MPS test failed: {e}. Falling back to CPU.")
            return torch_CPU
    
    # If CUDA GPU
    elif device_pref == torch_CUDA:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_logger.info(f"** TORCH GPU version installed and working. **")
            gpu_logger.info(f"GPU: {gpu_name} with {gpu_memory_total:.2f} GB memory")
            return device_pref
        except Exception as e:
            gpu_logger.warning(f"CUDA GPU error: {e}. Falling back to CPU.")
            return torch_CPU
    
    # CPU fallback
    return torch_CPU

# Initialize device at module import time
device = get_device()

# Move tensor to appropriate device (GPU if available, otherwise CPU)
def torch_GPU(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return x.to(device)

# Move tensor to CPU
def torch_CPU(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    return x.cpu()

# Empty the GPU cache to free memory
def empty_cache():
    """Clear GPU memory cache if available"""
    try:
        if str(device) == 'cuda':
            torch.cuda.empty_cache()
        # MPS doesn't have explicit cache clearing, but we can force garbage collection
        elif str(device) == 'mps':
            import gc
            gc.collect()
    except Exception as e:
        gpu_logger.warning(f"Error clearing GPU cache: {e}")

# Get GPU info
def get_gpu_info():
    if is_cuda_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"NVIDIA GPU: {gpu_name}, Memory: {gpu_memory_total:.2f} GB"
    elif is_mps_available():
        return "Apple Silicon GPU (MPS)"
    else:
        return "No GPU detected, using CPU"

# Check if tensor operations use MKL
def mkl_enabled():
    try:
        import numpy as np
        from numpy.distutils.system_info import get_info
        mkl_info = get_info('mkl')
        return len(mkl_info) > 0
    except:
        return False

# Show help message for GPU support
def gpu_help():
    if not is_cuda_available() and not is_mps_available():
        gpu_logger.warning('No GPU detected. To use a GPU:')
        if sys.platform == 'darwin' and ARM:
            gpu_logger.warning('- You have Apple Silicon. Make sure PyTorch is 2.0+ with MPS support')
        elif sys.platform == 'darwin':
            gpu_logger.warning('- You have macOS on Intel. NVIDIA GPUs are not supported')
        else:
            gpu_logger.warning('- For NVIDIA GPUs, install CUDA and a compatible PyTorch version')
            gpu_logger.warning('- See https://pytorch.org/get-started/locally/')
    else:
        info = get_gpu_info()
        gpu_logger.info(f'Using: {info}')

# Print initial GPU info
gpu_help()

# Exception for non-torch frameworks
def non_torch_warning():
    raise ValueError('Omnipose only runs with PyTorch now')

# Safer function to move tensors to device with fallback
def to_device(tensor, dev=None):
    """Move tensor to specified device with automatic fallback to CPU if it fails"""
    if dev is None:
        dev = device
    
    try:
        return tensor.to(dev)
    except Exception as e:
        gpu_logger.warning(f"Failed to move tensor to {dev}: {e}. Using CPU instead.")
        return tensor.to(torch_CPU)

# Function to provide GPU information for troubleshooting
def gpu_info():
    """Return detailed information about available GPU devices"""
    info = {
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device_used": str(device)
    }
    
    # CUDA info
    if torch.cuda.is_available():
        info.update({
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB",
            "cuda_memory_cached": f"{torch.cuda.memory_reserved(0)/1024**3:.2f} GB",
        })
    else:
        info["cuda_available"] = False
        
    # MPS (Apple Silicon) info
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_env_vars": {
                "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "Not set"),
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "Not set"),
                "PYTORCH_MPS_ALLOCATOR_DEBUG": os.environ.get("PYTORCH_MPS_ALLOCATOR_DEBUG", "Not set")
            }
        })
    
    return info

# Add this function to maintain compatibility with cellpose_omni imports
def use_gpu(gpu_number=None, istorch=True, use_torch=True):
    """
    Check if GPU can be used based on availability and specified parameters.
    
    Parameters:
    -----------
    gpu_number: int or None
        Specific GPU device to use (None for default)
    istorch: bool
        Whether torch mode is available/requested
    use_torch: bool
        Whether to use torch (alternative would be MXNet which is deprecated)
        
    Returns:
    --------
    bool: True if GPU can be used, False otherwise
    tuple: (bool, device) if torch is available, returns both the availability and the device
    """
    if not (istorch or use_torch):
        # Legacy MXNet path - no longer supported
        gpu_logger.warning("MXNet GPU support is no longer maintained")
        return False
    
    # Check for MPS availability (Apple Silicon)
    if is_mps_available():
        if gpu_number is not None:
            gpu_logger.warning("GPU number specification not applicable for MPS (Apple Silicon)")
        return True
        
    # Check for CUDA availability
    if is_cuda_available():
        if gpu_number is not None:
            try:
                if gpu_number < torch.cuda.device_count():
                    # Set the specified GPU as active
                    torch.cuda.set_device(gpu_number)
                    return True
                else:
                    gpu_logger.warning(f"GPU number {gpu_number} not available, defaulting to device 0")
            except Exception as e:
                gpu_logger.warning(f"Error setting GPU device: {e}")
        
        # Default CUDA device
        return True
    
    # No GPU available
    return False
