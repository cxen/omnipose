import platform  
import os
import multiprocessing
import sys
import logging

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

# import torch after setting env variables 
import torch

# Check for Apple Silicon
ARM = False
if sys.platform == 'darwin':
    try:
        # Check if running on Apple Silicon
        ARM = os.uname().machine == 'arm64'
    except:
        ARM = False

# Check if MPS is available for Apple Silicon GPU
def is_mps_available():
    if hasattr(torch, 'has_mps') and torch.has_mps and ARM:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    return False

# Check if CUDA is available for NVIDIA GPU
def is_cuda_available():
    return torch.cuda.is_available()

# Make GPU device available if possible, otherwise CPU
def get_device():
    if is_cuda_available():
        gpu_logger.info('CUDA GPU detected and enabled!')
        return torch.device('cuda')
    elif is_mps_available():
        gpu_logger.info('Apple Silicon GPU detected and enabled!')
        return torch.device('mps')
    else:
        gpu_logger.info('No GPU detected. Using CPU.')
        return torch.device('cpu')

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
    if is_cuda_available():
        torch.cuda.empty_cache()
    elif is_mps_available():
        # MPS doesn't have an equivalent yet, but we can try to manage memory
        import gc
        gc.collect()

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