#!/usr/bin/env python
"""
Script to update the Omnipose project by:
1. Migrating PyQt6 to PySide6
2. Checking PyTorch version
3. Validating GPU support
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path

def check_module(module_name):
    """Check if a module is installed and return its version."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown version')
        return version
    except ImportError:
        return None

def run_migration_script():
    """Run the PyQt to PySide migration script."""
    script_path = Path(__file__).parent / 'migrate_pyqt_to_pyside.py'
    
    if not script_path.exists():
        print(f"Migration script not found at {script_path}")
        return False
    
    try:
        print("Running PyQt6 to PySide6 migration script...")
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("Migration completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Migration failed with error: {e}")
        return False

def check_gpu_support():
    """Check GPU support for the current system."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"CUDA GPU available: {device_name} ({device_count} devices)")
            print(f"CUDA version: {torch.version.cuda}")
            return True
        
        # Check for MPS (Apple Silicon GPU) support
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print("Apple Silicon GPU (MPS) available")
                return True
        
        print("No GPU support detected. Using CPU.")
        return False
    except ImportError:
        print("PyTorch not installed. Cannot check GPU support.")
        return False

def main():
    """Main function to update the project."""
    print("Starting Omnipose project update...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check PyTorch version
    torch_version = check_module('torch')
    if torch_version:
        print(f"PyTorch version: {torch_version}")
    else:
        print("PyTorch not installed. Please install PyTorch 2.6.0+.")
    
    # Check PySide6 installation
    pyside_version = check_module('PySide6')
    if pyside_version:
        print(f"PySide6 version: {pyside_version}")
    else:
        print("PySide6 not installed. Please install PySide6.")
    
    # Run migration if PyQt6 is present
    pyqt_version = check_module('PyQt6')
    if pyqt_version:
        print(f"PyQt6 version: {pyqt_version}")
        print("PyQt6 found. Starting migration to PySide6...")
        if run_migration_script():
            print("Migration from PyQt6 to PySide6 completed.")
        else:
            print("Migration failed. Please check the logs.")
    else:
        print("PyQt6 not found. No migration needed.")
    
    # Check GPU support
    print("Checking GPU support...")
    check_gpu_support()
    
    print("\nUpdate complete! Please test the application.")
    print("If you encounter issues, please report them on the GitHub repository.")

if __name__ == "__main__":
    main()
