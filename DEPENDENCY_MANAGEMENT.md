# Dependency Management in Omnipose

This document explains how dependencies are managed in the Omnipose project.

## Single Source of Truth

All dependencies are defined in a single file: `dependencies.py`. This file contains lists for:

- Core dependencies (`install_deps`) - required for basic functionality
- GUI dependencies (`gui_deps`) - required for the GUI interface
- Distributed computing dependencies (`distributed_deps`) - for parallel processing
- Documentation dependencies (`doc_deps`) - for building documentation

## Installation

When installing Omnipose with pip:

```bash
pip install omnipose
```

Only the core dependencies are installed.

For additional features, install with extras:

```bash
# For GUI support
pip install omnipose[gui]

# For documentation building
pip install omnipose[docs]

# For all features
pip install omnipose[all]
```

## Development Installation

When installing for development:

```bash
git clone https://github.com/kevinjohncutler/omnipose.git
cd omnipose
pip install -e .[all]
```

## Updating Dependencies

To update dependencies:

1. Edit `dependencies.py` with your changes
2. Run `python generate_requirements.py` to update requirements.txt
3. Test the installation with your changes

## Critical Dependencies

The following dependencies are critical for Omnipose functionality:

- fastremap: For efficient mask manipulation
- ncolor: For proper coloring of masks
- edt: For distance transforms 
- torch: For neural network operations

These are all included in the core `install_deps` list.
