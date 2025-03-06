# Dependency Management in Omnipose

This document explains how dependencies are managed in the Omnipose project.

## Dependency Files

The project uses two main files for dependency management:

1. **`dependencies.py`** - The source of truth for all dependencies
   - Contains comprehensive dependency lists with version specifications
   - Used directly by `setup.py` and documentation building
   - Organized into categories: `install_deps`, `gui_deps`, `distributed_deps`, `doc_deps`

2. **`requirements.txt`** - Simplified requirements for direct pip installation
   - Generated from `dependencies.py`
   - Contains only core dependencies needed for basic functionality
   - Used for quick installation via `pip install -r requirements.txt`

## Updating Dependencies

When updating dependencies:

1. **Always update `dependencies.py` first**
   - Add/modify dependencies with appropriate version constraints
   - Organize dependencies into the appropriate category

2. **Generate `requirements.txt` afterward**
   - Run `python generate_requirements.py`
   - This ensures consistency between both files

Never modify `requirements.txt` directly as it will be overwritten when regenerated.

## Installation Options

- **Basic installation:** `pip install -r requirements.txt`
- **Full installation:** `pip install -e .` or `pip install .`
- **With optional dependencies:** `pip install -e .[gui]` or `pip install -e .[all]`
