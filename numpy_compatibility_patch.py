"""
This script updates all instances of np.round_ to np.round in the codebase
to make it compatible with NumPy 2.0+
"""

import os
import re
from pathlib import Path

def patch_numpy_references():
    """
    Find and replace np.round_ with np.round in all Python files in the project
    """
    base_path = Path(__file__).parent
    
    # Regular expression to find np.round_ usage
    pattern = re.compile(r'np\.round_')
    
    # Track files that were modified
    modified_files = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    # Read the file content
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Check if the pattern exists in the file
                    if pattern.search(content):
                        # Replace np.round_ with np.round
                        updated_content = pattern.sub('np.round', content)
                        
                        # Write the updated content back to the file
                        with open(filepath, 'w') as f:
                            f.write(updated_content)
                        
                        modified_files.append(filepath)
                        print(f"Updated {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    return modified_files

if __name__ == "__main__":
    modified = patch_numpy_references()
    
    if modified:
        print(f"\nPatched {len(modified)} file(s) for NumPy 2.0+ compatibility.")
    else:
        print("No files needed patching.")
