import codecs
import os.path

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


name = 'fluid_utils'
__version__ = get_version("fluid_utils/_version.py")  # version is MAJOR.MINOR.PATCH
__author__ = 'Matthias Probst'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=f"{name}",
    version=__version__,
    author="Matthias Probst",
    author_email="matthias.probst@kit.edu",
    description="Library providing utilities for fluid computations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatthiasProbst/fluid_utils",
    packages=setuptools.find_packages(),
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'appdirs',
        'numpy>=1.19',
        'h5py>=3.2.0',
        'h5pyd',
        'vtk>=8.1.2',
        'matplotlib',
        'pandas',
        'tqdm',
        'opencv-python',
        'pco_tools',
        'psutil',
        'scipy',
        'netCDF4',
        'pyevtk',
        'psutil',
        'IPython',
        'matplotlib',
        'pytest',
        'pyyaml',
        'xarray',
        'pint_xarray',
        'temp',
    ],
    cmdclass={},
)
