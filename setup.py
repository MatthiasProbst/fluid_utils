import setuptools

name = 'fluid_utils'
__version__ = '0.0.0'  # version is MAJOR.MINOR.PATCH
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
    url=f"https://github.com/MatthiasProbst/fluid_utils",
    packages=setuptools.find_packages(),
    package_data={'h5wrapperpy._html': ['style.css']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
    ],
    cmdclass={},
)
