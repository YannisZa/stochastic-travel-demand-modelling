import os
import sys
import argparse

def get_root_working_directory():
    """ Returns project's root working directory (entire path).

    Returns
    -------
    string
        Path to project's root directory.

    """
    # Get current working directory
    cwd = os.getcwd()
    # Remove all children directories
    rd = os.path.join(cwd.split('stochastic-travel-demand-modelling/', 1)[0])
    # Make sure directory ends with project's name
    if not rd.endswith('stochastic-travel-demand-modelling'):
        rd = os.path.join(rd,'stochastic-travel-demand-modelling/')

    return rd

def create_directory(dir:str):
    """ Check if directory exists.

    Parameters
    ----------
    dir : str
        Directory of interest

    Returns
    -------
    boolean
        Flag for whether on not directory exists

    """
    # Check if directory already exists
    if not os.path.isdir(dir):
        # Create directory
        print('Created',dir)
        os.mkdir(dir)
    else:
        print('Directory exists',dir)

# List of all available datasets and methods
datasets = ['commuter','retail','travel']
methods = ['actual','dsf','newton_raphson','poisson_regression']

# Get project's root directory
wd = get_root_working_directory()

# Create output directory
create_directory(os.path.join(wd,'output'))

# Create output directories for each dataset
for data in datasets:
    create_directory(os.path.join(wd,'output',data))
    for method in methods:
        create_directory(os.path.join(wd,'output',data,method))
        create_directory(os.path.join(wd,'output',data,method,'figures'))
