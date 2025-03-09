"""
This file is the source of truth for all dependencies in the Omnipose project.
It's used by:
1. setup.py for package installation
2. docs/conf.py for documentation building
3. generate_requirements.py to create requirements.txt

To update requirements.txt after changing this file, run:
    python generate_requirements.py
"""

# Core dependencies required for basic functionality
install_deps = ['numpy',
                'numba',
                'scipy',
                'matplotlib',
                'scikit-image',
                'scikit-learn',
                'natsort',
                'tqdm',
                'torch',
                'torchvision',
                'opencv-python-headless',
                'fastremap',
                'tifffile',
                'zarr',
                'ome-zarr',
                'distributed',
                'networkx',
                'pandas',
                'cmapy',
                'rasterio',
                'connected-components-3d',
                'edt',
                'fire',
                'ncolor',
                'aicsimageio',
                'mgen',
                'dbscan',
                'networkit',
                'torchvf',
                'rich',
                'psutil'
                ]

# Define gui_deps directly, as imported by other modules
gui_deps = ['PySide6',
            'pyqtgraph',
            'qtpy',
            'superqt',
            'qtawesome',
            'pyopengl',
            'darkdetect',
            'pyqtdarktheme',
            'cmap']

# Define distributed_deps, as imported by other modules
distributed_deps = ['dask', 'distributed']

# Define doc_deps for completeness
doc_deps = ['sphinx',
            'sphinx_rtd_theme',
            'matplotlib',
            'sphinx-gallery',
            'sphinx-autodoc-typehints']

# All available extras
optional_deps = {
    'gui': gui_deps,
    'test': ['pytest'],
    'docs': doc_deps,
    'all': gui_deps + distributed_deps + doc_deps + ['pytest']
}
