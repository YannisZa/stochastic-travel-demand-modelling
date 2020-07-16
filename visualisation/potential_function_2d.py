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

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Plot potential function for given choice of parameters.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'commuter_borough',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-cm", "--cost_matrix_type",nargs='?',type=str,choices=['','sn'],default='',
                    help="Type of cost matrix used.\
                        '': Euclidean distance based. \
                        'sn': Transportation network cost based on A and B roads only. ")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-as", "--alphas",nargs='?',type=list,default = [.5, 1., 1.5, 2.],
                    help="List of alpha parameters")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 1000.,
                    help="Beta parameter")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.3,
                    help="Delta parameter")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 20.,
                    help="Gamma parameter")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter")
parser.add_argument("-s", "--show_figure",nargs='?',type=bool,default = False,
                    help="Flag for showing resulting figure.")
parser.add_argument("-gmin", "--grid_min",nargs='?',type=float,default = -20.,
                    help="Smallest log destination W_j (x_j) to evaluate potential value.")
parser.add_argument("-gmax", "--grid_max",nargs='?',type=float,default = .2,
                    help="Largest log destination W_j (x_j) to evaluate potential value.")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate potential function")
parser.add_argument('-hide', '--hide', action='store_true')
args = parser.parse_args()
# Convert arguments to dictionary
arguments = vars(args)
# Print arguments
if not args.hide:
    print(json.dumps(arguments, indent = 2))

# Define dataset directory
dataset = args.dataset_name

# Define type of spatial interaction model
constrained = args.constrained

# Define mode (stochastic/determinstic) based on delta value
if int(args.gamma) >= 10000:
    mode = 'deterministic'
else:
    mode = 'stochastic'

# Get project directory
wd = get_project_root()

# Import selected type of spatial interaction model
if constrained == 'singly':
    from models.singly_constrained.spatial_interaction_model import SpatialInteraction
elif constrained == 'doubly':
    from models.doubly_constrained.spatial_interaction_model import SpatialInteraction
else:
    raise ValueError("{} spatial interaction model not implemented.".format(args.constrained))

# Instantiate SpatialIteraction
si = SpatialInteraction(dataset,args.cost_matrix_type)

# Normalise necessary data for learning
si.normalise_data()

# Setup 2D model
si.cost_matrix = si.cost_matrix[:,:2]/si.cost_matrix[:,:2].sum()
si.N, si.M = np.shape(si.cost_matrix)

# Define parameters
alpha_values = np.array(args.alphas)
beta = args.beta
delta = args.delta
gamma = args.gamma
kappa = 1.0 + delta*si.M # this is Equation (2.25)
epsilon = args.epsilon
theta = np.array([alpha_values[0], beta, delta, gamma, kappa, epsilon])
grid_size = args.grid_size
space0 = args.grid_min
space1 = args.grid_max
space = np.linspace(space0, space1, grid_size)
xx, yy = np.meshgrid(space, space)
zz = np.zeros((grid_size, grid_size))

# Run plots
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
# Options
params = {'text.usetex' : True,
          'font.size' : 20,
          'legend.fontsize': 20,
          'legend.handlelength': 2,
          'font.family' : 'sans-serif',
          'font.sans-serif':['Helvetica'],
          'text.latex.unicode': True
          }
plt.rcParams.update(params)
plt.style.use('classic')

# plt.figure(figsize=(12,3))
fig, axes = plt.subplots(nrows=1, ncols=len(alpha_values),figsize=(12,3))
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
            zz[i, j] = np.exp(-si.potential_value(temp,theta)[0])
            # zz[i, j] = np.exp(-si.potential_value(temp,theta)[0])

    # Create a contour
    im = plt.contourf(xx, yy, zz, 300)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([space0, space1])
    plt.ylim([space0, space1])
    plt.xticks([])
    plt.yticks([])
    # Extra settings - omit when generating nice plots for reports
    plt.xlabel(r'$x_1$',fontsize=16)
    plt.ylabel(r'$x_2$',fontsize=16)
    plt.title(r'$\alpha = {}$'.format(alpha_values[k]),fontsize=18)

# Extra settings - omit when generating nice plots for reports
# plt.title(('beta = {}, delta = {}, gamma = {}, kappa = {}'.format(beta,delta,gamma,kappa)), y=1.30,x=-1)
# Use next line when generating nice plots for reports
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) #fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im,cax=cbar_ax)
# plt.tight_layout()

# Save figure to output
plt.savefig(os.path.join(wd,f'data/output/{dataset}/potential/figures/{constrained}_{mode}_2d_potential_function{si.cost_matrix_file_extension}.png'))

# Show figure if instructed
if args.show_figure:
    plt.show()

# Save parameters to file
with open(os.path.join(wd,f'data/output/{dataset}/potential/figures/{constrained}_{mode}_2d_potential_function{si.cost_matrix_file_extension}_parameters.json'), 'w') as outfile:
    json.dump(arguments, outfile)

print('Figure saved to {}'.format(os.path.join(wd,f'data/output/{dataset}/potential/figures/{constrained}_{mode}_2d_potential_function{si.cost_matrix_file_extension}.png')))
