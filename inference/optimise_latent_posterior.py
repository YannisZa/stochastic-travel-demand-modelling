"""
Find global minimum of p(x | theta)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import minimize
from tqdm import tqdm

# Fix random seed
np.random.seed(886)

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
parser = argparse.ArgumentParser(description='Optimiser for the latent posterior given set of parameters.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'synthetic',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-a", "--alpha",nargs='?',type=float,default = 1.2,
                    help="Alpha parameter in potential function.")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 6.469646964696469,
                    help="Beta parameter in potential function.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.26666666666666666,
                    help="Delta parameter in potential function.")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 10000.,
                    help="Gamma parameter.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter.")
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

# Import selected type of spatial interaction model
if constrained == 'singly':
    from models.singly_constrained.spatial_interaction_model import SpatialInteraction
elif constrained == 'doubly':
    from models.doubly_constrained.spatial_interaction_model import SpatialInteraction
else:
    raise ValueError("{} spatial interaction model not implemented.".format(args.constrained))

# Get project directory
wd = get_project_root()

# Instantiate SpatialInteraction
si = SpatialInteraction(dataset)

# Normalise data
si.normalise_data()

# Fix random seed
np.random.seed(888)

# Set theta for high-noise model's potential value parameters
theta = [0 for i in range(6)]
theta[0] = args.alpha
theta[1] = args.beta
theta[2] = args.delta
# Set gamma for Laplace optimisation
theta[3] = args.gamma
theta[4] = 1 + args.delta*si.M
theta[5] = 1 # this is the potential values epsilon parameter which is assumed to be 1.
# Convert to np array
theta = np.array(theta)

# Get log destination sizes
xd = si.normalised_initial_destination_sizes

# Run optimization
minimum = xd
minimum_potential = np.infty
for k in range(si.M):
    delta = theta[2]
    g = np.log(delta)*np.ones(si.M)
    g[k] = np.log(1.+delta)
    f = minimize(si.potential_value,g, method='L-BFGS-B', args=(theta), jac=True, options={'disp': False})
    if(f.fun < minimum_potential):
        minimum_potential = f.fun
        minimum = f.x

# Save parameters to file
with open(os.path.join(wd,f'data/output/{dataset}/laplace/figures/{constrained}_laplace_analysis_gamma_{str(int(args.gamma))}_parameters.json'), 'w') as outfile:
    json.dump(arguments, outfile)

# Save optimal log destination sizes
np.savetxt(os.path.join(wd,f'data/output/{dataset}/laplace/{constrained}_optimal_latent_posterior_{str(theta[0])}.txt'),minimum)
