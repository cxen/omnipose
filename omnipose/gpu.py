import os
import sys
import platform
import logging
import torch
import numpy as np
from pathlib import Path
from torch import autocast

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
torch_GPU = torch.device('cuda')  # Default CUDA device if available
torch_CPU = torch.device('cpu')   # CPU device

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

def get_device_preference():
    """Get device preference from environment variables"""
    device_pref = os.environ.get('OMNIPOSE_DEVICE', '').lower()
    if device_pref == 'cpu':
        return 'cpu'
    elif device_pref == 'cuda' and is_cuda_available():
        return 'cuda'
    elif device_pref == 'mps' and is_mps_available():
        return 'mps'
    return None  # Auto-select based on availability

def use_gpu(gpu_number=0):
    """
    Assign device to use for model - GPU if available, otherwise CPU
    
    Parameters
    ----------
    gpu_number: int (optional, default 0)
        which GPU to use (if more than one)
        
    Returns
    -------
    tuple (device, gpu_available)
        device: torch.device
            device object to use for computations
        gpu_available: bool
            whether a GPU was successfully assigned
    """
    # Check for preference from environment
    device_pref = get_device_preference()
    
    # If CUDA is available and preferred or auto-selecting
    if (device_pref in [None, 'cuda']) and is_cuda_available():
        device = torch.device(f'cuda:{gpu_number}')
        return device, True
    
    # If MPS is available and preferred or auto-selecting
    if (device_pref in [None, 'mps']) and is_mps_available():
        device = torch.device('mps')
        return device, True
    
    # Fallback to CPU
    device = torch.device('cpu')
    return device, False

# Make GPU device available if possible, otherwise CPU
def get_device():
    device, gpu_available = use_gpu()
    if gpu_available:
        gpu_logger.info(f'{device} detected and enabled!')
    else:
        gpu_logger.info('No GPU detected. Using CPU.')
    return device

# Initialize device
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
        if is_cuda_available():
            torch.cuda.empty_cache()
        # MPS doesn't have an empty_cache method yet
    except Exception:
        pass

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
