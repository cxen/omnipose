#!/usr/bin/env python
"""
Memory-safe batch processor for Omnipose.
This script provides a wrapper to process large batches of images safely.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnipose.memory import safe_process_batch, force_cleanup
from omnipose import core, utils
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch_processor")

def process_image_batch(image_batch, model, channels, batch_size=None, use_gpu=True, **kwargs):
    """Process a batch of images with memory safety."""
    
    # Convert to smaller batches if needed
    if batch_size is not None and len(image_batch) > batch_size:
        results = []
        for i in range(0, len(image_batch), batch_size):
            sub_batch = image_batch[i:i+batch_size]
            logger.info(f"Processing sub-batch {i//batch_size + 1}/{(len(image_batch) + batch_size - 1)//batch_size}")
            result = process_image_batch(sub_batch, model, channels, None, use_gpu, **kwargs)
            results.extend(result)
            force_cleanup()
        return results
    
    # Process the batch
    try:
        masks = model.eval(image_batch, channels=channels, **kwargs)
        return masks
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        force_cleanup(verbose=True)
        # Try again with a smaller batch if possible
        if len(image_batch) > 1:
            logger.info("Retrying with smaller batches...")
            half = len(image_batch) // 2
            return (process_image_batch(image_batch[:half], model, channels, None, use_gpu, **kwargs) + 
                    process_image_batch(image_batch[half:], model, channels, None, use_gpu, **kwargs))
        else:
            logger.error("Failed to process a single image. Skipping.")
            return [None] * len(image_batch)

def save_masks_safely(masks, imgs, flows, files, output_dir):
    """Save masks with proper dimension handling."""
    from cellpose_omni import io as cellpose_io
    
    # Prepare images and masks with proper dimensions
    batch_masks = []
    batch_imgs = []
    valid_files = []
    
    try:
        for i, (mask, img, filename) in enumerate(zip(masks, imgs, files)):
            if mask is None:
                continue
                
            # Ensure mask is 2D
            mask_2d = np.asarray(mask).squeeze()
            if mask_2d.ndim != 2:
                logger.warning(f"Adjusting mask dimensions from {mask_2d.shape}")
                if mask_2d.ndim > 2:
                    mask_2d = mask_2d[..., 0] if mask_2d.shape[-1] == 1 else mask_2d[0]
            
            batch_masks.append(mask_2d)
            
            # Prepare image with proper dimensions
            if img.ndim == 2:
                # For grayscale, add channel dimension
                batch_imgs.append(img[np.newaxis, ...])
            else:
                # For multi-channel, keep as is
                batch_imgs.append(img)
                
            valid_files.append(filename)
        
        try:
            if batch_masks:
                cellpose_io.save_masks(
                    images=batch_imgs,
                    masks=batch_masks,
                    flows=flows,
                    file_names=valid_files,
                    channels=[0,0],
                    png=False,
                    tif=True,
                    suffix='',
                    save_txt=True,
                    dir_above=output_dir
                )
                return len(batch_masks)
            return 0
        except Exception as e:
            logger.error(f"Error calling save_masks: {e}")
            # Fall back to individual saving
            count = 0
            for i, (mask, fname) in enumerate(zip(batch_masks, valid_files)):
                try:
                    save_path = os.path.join(output_dir, f"{Path(fname).stem}_mask.tif")
                    utils.imwrite(save_path, mask)
                    count += 1
                except Exception as e:
                    logger.error(f"Error saving individual mask {fname}: {e}")
            return count
    except Exception as e:
        logger.error(f"Error in save_masks_safely: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Process images with Omnipose safely')
    parser.add_argument('--input', required=True, help='Input directory with images')
    parser.add_argument('--output', required=True, help='Output directory for masks')
    parser.add_argument('--model', default='bact_phase_omni', help='Model type')
    parser.add_argument('--channels', type=int, nargs='+', default=[0,0], help='Channel indices')
    parser.add_argument('--batch_size', type=int, default=8, help='Maximum batch size')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Import here to avoid early initialization
    from cellpose_omni import models
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model = models.CellposeModel(gpu=args.gpu, model_type=args.model)
    
    # Get image files
    image_files = list(Path(args.input).glob('*.tif')) + list(Path(args.input).glob('*.png'))
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process in batches
    for i in range(0, len(image_files), args.batch_size):
        batch_files = image_files[i:i+args.batch_size]
        logger.info(f"Loading batch {i//args.batch_size + 1}/{(len(image_files) + args.batch_size - 1)//args.batch_size}")
        
        # Load images
        images = [utils.imread(str(f)) for f in batch_files]
        
        # Process images
        logger.info("Processing batch")
        masks = process_image_batch(images, model, args.channels, args.batch_size//2, args.gpu)
        
        # Save results safely
        logger.info("Saving results")
        save_masks_safely(masks, images, None, [str(f) for f in batch_files], args.output)
        
        # Clean up
        force_cleanup(verbose=True)
        del images, masks
        force_cleanup(verbose=False)

if __name__ == '__main__':
    main()
