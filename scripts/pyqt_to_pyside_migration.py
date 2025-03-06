import os
import re
from pathlib import Path

def migrate_file(filepath):
    """Migrate a Python file from PyQt6 to PySide6."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace imports
    content = content.replace('from PyQt6 import', 'from PySide6 import')
    content = content.replace('import PyQt6.', 'import PySide6.')
    content = content.replace('PyQt6.', 'PySide6.')
    
    # Replace signals and slots
    content = content.replace('pyqtSignal', 'Signal')
    content = content.replace('pyqtSlot', 'Slot')
    
    # Special case for pyqt progress bar comments
    content = content.replace('pyqt progress bar', 'qt progress bar')

    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Migrated: {filepath}")

def find_and_migrate_python_files(root_dir):
    """Find all Python files that mention PyQt6 and migrate them."""
    for path in Path(root_dir).rglob('*.py'):
        with open(path, 'r') as f:
            content = f.read()
            if 'PyQt6' in content:
                migrate_file(path)

if __name__ == "__main__":
    # Adjust this path to point to the root of your project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    find_and_migrate_python_files(project_root)
    print("Migration completed!")
