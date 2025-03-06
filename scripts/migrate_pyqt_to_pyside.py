#!/usr/bin/env python
"""
Migrate PyQt6 to PySide6 across the codebase.
This script updates imports and other API differences between PyQt6 and PySide6.
"""

import os
import re
import glob

def modify_file(filepath):
    """Migrate a Python file from PyQt6 to PySide6."""
    print(f"Modifying {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace imports and references
    content = content.replace('from PyQt6 import', 'from PySide6 import')
    content = content.replace('import PyQt6.', 'import PySide6.')
    content = content.replace('PyQt6.', 'PySide6.')
    
    # Replace signal and slot mechanism
    content = content.replace('pyqtSignal', 'Signal')
    content = content.replace('pyqtSlot', 'Slot')
    
    # Special case for pyqt progress bar comments
    content = content.replace('pyqt progress bar', 'qt progress bar')
    
    # Add missing translations
    # Qt.ApplicationModal -> Qt.WindowModality.ApplicationModal in PySide6
    content = re.sub(r'Qt\.ApplicationModal', r'Qt.WindowModality.ApplicationModal', content)
    content = re.sub(r'Qt\.WindowModal', r'Qt.WindowModality.WindowModal', content)
    
    # Handle changes in enums (Qt.AlignCenter -> Qt.AlignmentFlag.AlignCenter)
    content = re.sub(r'Qt\.Align([A-Za-z]+)', r'Qt.AlignmentFlag.Align\1', content)
    content = re.sub(r'Qt\.Key_([A-Za-z0-9]+)', r'Qt.Key.Key_\1', content)
    content = re.sub(r'Qt\.([A-Z][a-z]+)Button', r'Qt.MouseButton.\1Button', content)
    
    # Specific fixes for QSizePolicy
    content = re.sub(r'QSizePolicy\.([A-Za-z]+)', r'QSizePolicy.Policy.\1', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def find_and_modify_pyqt_files(base_dir='.'):
    """Find all Python files that mention PyQt6 and migrate them."""
    for pattern in ['**/*.py', '**/*.pyw']:
        for filepath in glob.glob(os.path.join(base_dir, pattern), recursive=True):
            if not os.path.isfile(filepath):
                continue
                
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            if 'PyQt6' in content:
                modify_file(filepath)

if __name__ == '__main__':
    # Get the current script directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    find_and_modify_pyqt_files(base_dir)
    print("Migration from PyQt6 to PySide6 complete!")
