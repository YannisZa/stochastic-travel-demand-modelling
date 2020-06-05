""" 2D illustration of potential function """

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Get current working directory and project root directory
def get_project_root():
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

# Append project root directory to path
sys.path.append(get_project_root())
# Import module that couldn't be imported before
from models.singly_constrained.urban_model import UrbanModel

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
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter")
parser.add_argument("-s", "--show_figure",nargs='?',type=bool,default = False,
                    help="Flag for showing resulting figure.")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate potential function")
args = parser.parse_args()
# Print arguments
print(json.dumps(vars(args), indent = 2))

# Define dataset directory
dataset = args.dataset_name
# Define mode (stochastic/determinstic)
mode = args.mode

# Get project directory
wd = get_project_root()

# Instantiate UrbanModel
um = UrbanModel(mode,dataset)

# Setup 2D model
um.cost_matrix = um.cost_matrix[:,:2]/um.cost_matrix[:,:2].sum()
um.N, um.M = np.shape(um.cost_matrix)

# Define parameters
alpha_values = np.array(args.alphas)
beta = args.beta
delta = args.delta/um.M
gamma = args.gamma
kappa = args.kappa + delta*um.M # this is Equation (2.25)
epsilon = args.epsilon
theta = np.array([alpha_values[0], beta, delta, gamma, kappa, epsilon])
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
    plt.subplot(1, len(alpha_values), k+1)
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
    # Extra settings - omit when generating nice plots for reports
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('a = {}'.format(alpha_values[k]))
    plt.colorbar()

# Extra settings - omit when generating nice plots for reports
plt.title(('beta = {}, delta = {}, gamma = {}, kappa = {}'.format(beta,delta,gamma,kappa)), y=1.30,x=-1)
# Use next line when generating nice plots for reports
# plt.tight_layout()

# Save figure to output
plt.savefig(os.path.join(wd,'data/output/{}/figures/2d_potential_function.png'.format(dataset)))

# Show figure if instructed
if args.show_figure:
    plt.show()

# Save parameters to file
with open(os.path.join(wd,'data/output/{}/figures/2d_potential_function_parameters.json'.format(dataset)), 'w') as outfile:
    json.dump(vars(args), outfile)

print('Figure saved to {}'.format(os.path.join(wd,'data/output/{}/figures/2d_potential_function.png'.format(dataset))))
