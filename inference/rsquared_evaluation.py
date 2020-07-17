"""
R2 analysis for deterministic model defined in terms of potential function.
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
from tqdm import tqdm
from scipy.optimize import minimize

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
wd = get_project_root()
sys.path.append(wd)

# Parse arguments from command line
parser = argparse.ArgumentParser(description='R^2 evaluation to find latent sizes for a given parameter choice.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'commuter_borough',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-cm", "--cost_matrix_type",nargs='?',type=str,choices=['','sn'],default='',
                    help="Type of cost matrix used.\
                        '': Euclidean distance based. \
                        'sn': Transportation network cost based on A and B roads only. ")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-a", "--alpha",nargs='?',type=float,default = 0.0,
                    help="Alpha parameter.")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 0.0,
                    help="Beta parameter.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.3,
                    help="Delta parameter.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter.")
parser.add_argument("-s", "--plot_results",action='store_true',
                    help="Flag for plotting resulting figures and tables.")
parser.add_argument('-hide', '--hide', action='store_true',
                    help="Flag for hiding print functions.")
args = parser.parse_args()

# Convert arguments to dictionary
arguments = vars(args)

# Print arguments
if not args.hide:
    print(json.dumps(arguments, indent = 2))

# Define dataset directory
dataset = args.dataset_name

# Define mode (stochastic/determinstic) based on delta value
if args.delta == 0:
    mode = 'deterministic'
else:
    mode = 'stochastic'

# Define type of spatial interaction model
constrained = args.constrained

# Get project directory
wd = get_project_root()

# Import selected type of spatial interaction model
if constrained == 'singly':
    from models.singly_constrained.spatial_interaction_model import SpatialInteraction
elif constrained == 'doubly':
    from models.doubly_constrained.spatial_interaction_model import SpatialInteraction
else:
    raise ValueError("{} spatial interaction model not implemented.".format(args.constrained))

# Instantiate SpatialInteraction model
si = SpatialInteraction(dataset,args.cost_matrix_type)

# Compute kappa
kappa = 1 + args.delta*si.M
# Define gamma (the choice is irrelevant)
gamma = 100

alpha = args.alpha
beta = args.beta
r2_value = -np.infty
potential = -np.infty
w_prediction = -np.infty

# Define theta parameters
theta = np.array([alpha, beta, args.delta, gamma, kappa, args.epsilon])

# Normalise initial log destination sizes
si.normalise_data()
xd = si.normalised_initial_destination_sizes

# Search values
last_r2 = -np.infty
last_potential = -np.infty
last_w_prediction = np.exp(xd)

# Total sum squares
w_data = np.exp(xd)
w_data_centred = w_data - np.mean(w_data)
ss_tot = np.dot(w_data_centred, w_data_centred)

# Print parameters
print('delta =',theta[2],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])


# Minimise potential function
potential_func = minimize(si.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False})

w_pred = np.exp(potential_func.x)
res = w_pred - w_data
ss_res = np.dot(res, res)

# Regression sum squares
r2_value = 1. - ss_res/ss_tot
potential = potential_func.fun
w_prediction = w_pred



print("Fitted alpha, beta and scaled beta values:")
print(alpha, beta,beta*2.0/(1.4e6))
print("R^2 and potential value:")
print(r2_value,potential)


print('Latent sizes')
print(w_prediction)

np.savetxt(os.path.join(wd,f'data/output/{dataset}/r_squared/{constrained}_{mode}_rsquared{si.cost_matrix_file_extension}_latent_sizes.txt'),w_prediction)
