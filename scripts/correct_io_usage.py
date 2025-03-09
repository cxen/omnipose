#!/usr/bin/env python3
"""
Helper script to fix IO function usage in Segmentation.py by:
1. Using correct scikit-image functions (io.imsave) for scikit-image
2. Using cellpose_io.imwrite for cellpose functions
3. Removing any compatibility layer confusion
"""

import os
import sys
import re

def fix_io_usage(file_path):
    """
    Fix IO function usage in the given file to use proper direct calls.
    """
    print(f"Checking {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if cellpose_omni is already imported correctly
    if 'from cellpose_omni import io' in content:
        has_cellpose_import = True
    else:
        has_cellpose_import = False
        
    # Look for problematic calls
    has_imwrite_calls = 'skimage.io.imwrite' in content
    
    if not has_imwrite_calls:
        print("No problematic skimage.io.imwrite calls found.")
        return False
    
    # Add proper import for cellpose_omni if needed
    if not has_cellpose_import and has_imwrite_calls:
        # Add cellpose_omni import at the top
        content = re.sub(
            r'from skimage import \(io,',
            'from skimage import (io as skio,',
            content
        )
        # Add cellpose_omni import
        content = re.sub(
            r'import numpy as np',
            'import numpy as np\nfrom cellpose_omni import io',
            content
        )
    
    # Replace skimage.io.imwrite with io.imwrite
    content = re.sub(
        r'skimage\.io\.imwrite\s*\(',
        'io.imwrite(',
        content
    )
    
    # Save if changes were made
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed IO usage in {file_path}")
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Microfluidics-analyses', 'Python scripts', 'Segmentation.py')
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    
    fixed = fix_io_usage(file_path)
    
    if fixed:
        print("\nFixed problematic IO calls by:")
        print("1. Using direct io.imwrite calls from cellpose_omni")
        print("2. No compatibility layers or patches applied")
        print("\nYou should now be able to run your script without errors.")
    else:
        print("\nNo changes needed.")
    
    sys.exit(0)
