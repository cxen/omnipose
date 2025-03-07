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

def main():
    # First handle any diagnostic commands
    handle_command()
    
    try:
        try:
            # Try to import from cellpose_omni first
            from cellpose_omni.__main__ import main as cellpose_main
            # Parse arguments using the cellpose parser
            parser = get_arg_parser()
            args = parser.parse_args()
        except ImportError:
            # Fallback to importing from local repackaged copy
            from .cellpose_omni.__main__ import main as cellpose_main
            # Parse arguments using the cellpose parser
            parser = get_arg_parser()
            args = parser.parse_args()
            
        # Pass the parsed args to cellpose_main instead of sys.argv
        cellpose_main(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running omnipose: {e}")
        print("\nTry running 'omnipose --diagnose_gpu' to troubleshoot GPU issues.")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())

