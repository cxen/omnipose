import sys
import os
import logging

from .logger import setup_logger
main_logger = setup_logger('main')

def main():
    main_logger.info("running main")
    try:
        # Use absolute imports rather than relative
        from cellpose_omni.gui import GUI
        gui = GUI.GUI()
        gui.run()
    except ImportError as e:
        main_logger.error(f"GUI error: {e}")
        print("GUI not available: {0}".format(str(e)))
        print("GUI dependencies may not be installed. Prompting...")
        install_gui_deps = input("Install GUI dependencies? (PySide6, etc.) (y/n): ")
        if install_gui_deps.lower() == 'y':
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6", "pyqtgraph", "qtpy", "superqt", "qtawesome", "pyopengl", "darkdetect", "pyqtdarktheme", "cmap"])
            # Try again after installing
            try:
                from cellpose_omni.gui import GUI
                gui = GUI.GUI()
                gui.run()
            except ImportError as e:
                main_logger.error(f"GUI still unavailable after dependency installation: {e}")
                print(f"GUI still unavailable: {e}")
                print("You may need to check your Python environment or installation.")

if __name__ == "__main__":
    main()
