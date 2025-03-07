"""
Main entry point for Omnipose
"""
import sys
import os
import logging
import importlib.util
import traceback
from .gpu import gpu_help, get_device
from .cli import handle_command, get_arg_parser

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    required_deps = ["fastremap", "torch", "numpy", "scipy", "ncolor", "edt"]
    
    for dep in required_deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_deps.append(dep)
    
    return missing_deps

def main():
    # First handle any diagnostic commands
    handle_command()
    
    # Check dependencies before starting
    missing_deps = check_dependencies()
    if missing_deps:
        print("ERROR: Missing required dependencies:")
        print("\n".join(f"  - {dep}" for dep in missing_deps))
        print("\nPlease install missing packages with:")
        print(f"  pip install {' '.join(missing_deps)}")
        print("\nOr reinstall Omnipose with all dependencies:")
        print("  pip install omnipose[all]")
        sys.exit(1)
    
    try:
        try:
            # Try to import from cellpose_omni first
            from cellpose_omni.__main__ import main as cellpose_main
            # Parse arguments using the cellpose parser
            parser = get_arg_parser()
            args = parser.parse_args()
        except (ImportError, ModuleNotFoundError) as e:
            # Fallback to importing from local repackaged copy
            print(f"Notice: {e}")
            print("Trying to use local copy of cellpose_omni...")
            
            # Correct the import path for the local copy
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from cellpose_omni.__main__ import main as cellpose_main
            # Parse arguments using the cellpose parser
            parser = get_arg_parser()
            args = parser.parse_args()
            
        # Pass the parsed args to cellpose_main instead of sys.argv
        cellpose_main(args)
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Failed to run Omnipose")
        print("="*60)
        traceback.print_exc()
        
        print("\n" + "="*60)
        print("TROUBLESHOOTING TIPS:")
        print("="*60)
        print("1. Make sure all dependencies are installed:")
        print("   pip install fastremap ncolor edt torch")
        print("2. For GPU issues, run: omnipose --diagnose_gpu")
        print("3. Try running with CPU only: export OMNIPOSE_FORCE_CPU=1")
        print("4. Report this issue on GitHub with the full error message")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())

