#!/usr/bin/env python3
"""
Script to generate requirements.txt from dependencies.py.
This ensures consistency between the two dependency management files.
"""

import os
from dependencies import install_deps

def generate_requirements():
    """Generate requirements.txt from dependencies.py"""
    print("Generating requirements.txt from dependencies.py...")
    
    # Filter core dependencies we want in requirements.txt
    # You can adjust this list based on what you want in requirements.txt
    core_deps = [
        'numpy<=2.1',  # Preserving the Numba compatibility constraint
        'scipy',
        'torch',
        'tqdm',
        'numba',
        'scikit-image',
        'matplotlib', 
        'opencv-python-headless',
        'natsort',
        'edt'
    ]
    
    # Add these specific packages that are in the current requirements.txt
    # but not in dependencies.py
    additional_deps = [
        'ncolour',
        'aicsimageio'
    ]
    
    # Combine all dependencies
    all_deps = core_deps + additional_deps
    
    # Write the requirements.txt file
    with open('requirements.txt', 'w') as f:
        f.write("# This file is generated from dependencies.py\n")
        f.write("# Do not edit manually. Use generate_requirements.py to update.\n\n")
        for dep in all_deps:
            f.write(f"{dep}\n")
    
    print("requirements.txt has been generated successfully.")

if __name__ == "__main__":
    generate_requirements()
