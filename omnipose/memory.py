import gc
import os
import psutil
import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnipose.memory")

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024**3  # Convert to GB

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB if available."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
        return memory_allocated, memory_reserved
    return None, None

def force_cleanup(verbose=True):
    """Force garbage collection and clear CUDA cache."""
    if verbose:
        before_mem = get_memory_usage()
        before_gpu_allocated, before_gpu_reserved = get_gpu_memory_usage()
    
    # Clear any cached tensors and arrays
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose:
        after_mem = get_memory_usage()
        after_gpu_allocated, after_gpu_reserved = get_gpu_memory_usage()
        
        logger.info(f"Memory cleanup: {before_mem:.2f}GB → {after_mem:.2f}GB")
        if before_gpu_allocated is not None:
            logger.info(f"GPU memory (allocated): {before_gpu_allocated:.2f}GB → {after_gpu_allocated:.2f}GB")
            logger.info(f"GPU memory (reserved): {before_gpu_reserved:.2f}GB → {after_gpu_reserved:.2f}GB")

def batch_limiter(data, max_batch_size=None, max_memory_fraction=0.7):
    """
    Split data into smaller batches to prevent memory issues.
    
    Args:
        data: List of data items to process
        max_batch_size: Maximum number of items in a batch (if None, use max_memory_fraction)
        max_memory_fraction: Maximum fraction of system memory to use
    
    Returns:
        List of batched data
    """
    if max_batch_size is None:
        # Calculate based on available memory
        total_memory = psutil.virtual_memory().total / 1024**3  # GB
        current_memory = get_memory_usage()
        available_memory = total_memory * max_memory_fraction - current_memory
        
        # Estimate memory per item based on first item
        if len(data) > 0:
            if isinstance(data[0], np.ndarray):
                mem_per_item = data[0].nbytes / 1024**3  # GB
            elif isinstance(data[0], torch.Tensor):
                mem_per_item = data[0].element_size() * data[0].nelement() / 1024**3  # GB
            else:
                # Default to a conservative estimate
                mem_per_item = 0.1  # GB
            
            max_batch_size = max(1, int(available_memory / mem_per_item))
            logger.info(f"Auto batch size: {max_batch_size} (estimated {mem_per_item:.2f}GB per item)")
        else:
            max_batch_size = 1
    
    # Create batches
    batches = []
    for i in range(0, len(data), max_batch_size):
        batches.append(data[i:i + max_batch_size])
    
    return batches

def safe_process_batch(process_fn, data, *args, max_batch_size=None, **kwargs):
    """
    Safely process data in batches with memory cleanup.
    
    Args:
        process_fn: Function to process each batch
        data: Data to process in batches
        max_batch_size: Maximum items per batch
        *args, **kwargs: Additional args for process_fn
    
    Returns:
        List of results from processing each batch
    """
    batches = batch_limiter(data, max_batch_size)
    results = []
    
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} (size: {len(batch)})")
        try:
            # Process the batch
            result = process_fn(batch, *args, **kwargs)
            
            # If the result is a list, extend results
            # Otherwise append the result
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {e}")
            
            # Try again with a smaller batch if possible
            if len(batch) > 1:
                logger.info("Retrying with smaller batches...")
                half = len(batch) // 2
                
                try:
                    # Process first half
                    result1 = process_fn(batch[:half], *args, **kwargs)
                    if isinstance(result1, list):
                        results.extend(result1)
                    else:
                        results.append(result1)
                    
                    # Process second half
                    result2 = process_fn(batch[half:], *args, **kwargs)
                    if isinstance(result2, list):
                        results.extend(result2)
                    else:
                        results.append(result2)
                        
                except Exception as sub_e:
                    logger.error(f"Failed to process sub-batches: {sub_e}")
                    # Add None for each failed item
                    results.extend([None] * len(batch))
            else:
                # Add None for the failed item
                results.extend([None] * len(batch))
                
        # Clean up memory after each batch
        force_cleanup(verbose=True)
    
    return results

def safe_array_process(process_fn, arrays, mask_dimension=False, **kwargs):
    """
    Process arrays safely with proper dimension handling.
    
    This is particularly useful for functions that have specific dimensional
    requirements, such as cellpose_io.save_masks.
    
    Args:
        process_fn: Function to call
        arrays: List of arrays to process
        mask_dimension: If True, ensure masks are 2D
        **kwargs: Additional keyword arguments for process_fn
    
    Returns:
        Result from process_fn
    """
    # Handle various common dimension issues
    if mask_dimension:
        # Ensure mask dimension is 2D for each array
        prepared_arrays = []
        for arr in arrays:
            if arr is None:
                prepared_arrays.append(None)
                continue
                
            # Remove singleton dimensions
            squeezed = np.asarray(arr).squeeze()
            
            # Handle different dimensions
            if squeezed.ndim != 2:
                if squeezed.ndim > 2:
                    # Take first 2D slice if higher dimensional
                    if squeezed.shape[-1] == 1:
                        squeezed = squeezed[..., 0]
                    else:
                        squeezed = squeezed[0]
                elif squeezed.ndim < 2:
                    # Add dimensions if needed (shouldn't normally happen)
                    squeezed = np.expand_dims(squeezed, axis=0)
                    
            prepared_arrays.append(squeezed)
        arrays = prepared_arrays
    
    try:
        return process_fn(arrays, **kwargs)
    except Exception as e:
        logger.error(f"Error in safe_array_process: {e}")
        force_cleanup()
        # Return None or appropriate error value based on the context
        return None
