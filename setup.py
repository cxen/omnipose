from setuptools import setup, find_packages
import sys, os
sys.path.append(os.path.dirname(__file__))

# dependencies.py is the source of truth for all dependencies
# See DEPENDENCY_MANAGEMENT.md for details on how dependencies are managed
from dependencies import install_deps, gui_deps, distributed_deps, optional_deps, doc_deps

with open("README.rst", "r") as fh:
    long_description = fh.read() 

setup(
    name="omnipose",
    use_scm_version=True,
    author="Kevin Cutler",
    author_email="kevinjohncutler@outlook.com",
    description="cell segmentation algorithm improving on the Cellpose framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinjohncutler/omnipose",
    packages=find_packages(include=['omnipose', 'cellpose_omni']),
    # Use install_deps from dependencies.py - remove duplicated hardcoded list
    install_requires=install_deps,
    extras_require=optional_deps,
    tests_require=[
      'pytest'
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts': [
          'omnipose=omnipose:run_omnipose',
        ],
    },
    py_modules=['dependencies'],
)
