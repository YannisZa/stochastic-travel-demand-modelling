"""
2D illustration of potential function
"""

import os
import sys

# Get current working directory and project root directory
cwd = os.getcwd()
rd = os.path.join(cwd.split('stochastic-travel-demand-modelling/', 1)[0])
if not rd.endswith('stochastic-travel-demand-modelling'):
    rd = os.path.join(cwd.split('stochastic-travel-demand-modelling/', 1)[0],'stochastic-travel-demand-modelling')
# Append project root directory to path
sys.path.append(rd)

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models.urban_model import UrbanModel
from tqdm import tqdm

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Plot potential function for given choice of parameters.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,default = 'retail',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-m", "--mode",nargs='?',type=str,default = 'stochastic',
                    help="Mode of evaluation (stochastic/determinstic)")
parser.add_argument("-as", "--alphas",nargs='?',type=list,default = [.5, 1., 1.5, 2.],
                    help="List of alpha parameters")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 1000.,
                    help="Beta parameter")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.3,
                    help="Delta parameter")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 20.,
                    help="Gamma parameter")
parser.add_argument("-k", "--kappa",nargs='?',type=float,default = 1.,
                    help="Kappa parameter")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate potential function")
args = parser.parse_args()
# Print arguments
print(json.dumps(vars(args), indent = 2))

# Define dataset directory
dataset = args.dataset_name
# Define mode (stochastic/determinstic)
mode = args.mode

# Instantiate UrbanModel
um = UrbanModel(mode,dataset,rd)

# Setup 2D model
um.cost_matrix = um.cost_matrix[:,:2]/um.cost_matrix[:,:2].sum()
um.N, um.M = np.shape(um.cost_matrix)

# Define parameters
alpha_values = np.array(args.alphas)
beta = args.beta
delta = args.delta/um.M
gamma = args.gamma
kappa = args.kappa + delta*um.M
theta = np.array([alpha_values[0], beta, delta, gamma, kappa])
grid_size = args.grid_size
space0 = -4.
space1 = .5
space = np.linspace(space0, space1, grid_size)
xx, yy = np.meshgrid(space, space)
zz = np.zeros((grid_size, grid_size))

# Run plots
plt.figure(figsize=(12,3))
for k in tqdm(range(len(alpha_values))):
    # Create a new subplot
    plt.subplot(1, 4, k+1)
    # Change the value of alpha
    theta[0] = alpha_values[k]

    # Loop over grid
    for i in range(grid_size):
        for j in range(grid_size):
            temp = np.array([xx[i, j], yy[i, j]])
            # Evaluate potential function for given point in the grid and theta parameters
            zz[i, j] = np.exp(-um.potential_value(temp,theta)[0])

    # Create a contour
    plt.contourf(xx, yy, zz, 300)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([space0, space1])
    plt.ylim([space0, space1])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(rd,'data/output/figures/2d_potential_function.png'))
print('Figure saved to {}'.format(os.path.join(rd,'data/output/figures/2d_potential_function.png')))
