"""
R2 analysis for deterministic model defined in terms of potential function.
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
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter','retail','transport'],default = 'commuter',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='doubly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-amin", "--amin",nargs='?',type=float,default = 0.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-amax", "--amax",nargs='?',type=float,default = 2.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 100.,
                    help="Gamma parameter")
parser.add_argument("-k", "--kappa",nargs='?',type=float,default = 1.3,
                    help="Kappa parameter")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter")
parser.add_argument("-s", "--show_figure",nargs='?',type=bool,default = False,
                    help="Flag for showing resulting figure.")
parser.add_argument("-gmin", "--grid_min",nargs='?',type=int,default = -20.,
                    help="Smallest log destination W_j (x_j) to evaluate potential value.")
parser.add_argument("-gmax", "--grid_max",nargs='?',type=int,default = .2,
                    help="Largest log destination W_j (x_j) to evaluate potential value.")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate potential function")
args = parser.parse_args()
# Print arguments
print(json.dumps(vars(args), indent = 2))

# Define dataset directory
dataset = args.dataset_name
# Define type of spatial interaction model
constrained = args.constrained

# Import selected type of spatial interaction model
if constrained == 'singly':
    from models.singly_constrained.spatial_interaction_model import SpatialIteraction
elif constrained == 'doubly':
    from models.doubly_constrained.spatial_interaction_model import SpatialIteraction
else:
    raise ValueError("{} spatial interaction model not implemented.".format(args.constrained))

# Get project directory
wd = get_project_root()

# Instantiate UrbanModel
si = SpatialIteraction(dataset)

# Initialize search grid
grid_n = args.grid_size
alpha_values = np.linspace(args.amin, args.amax, grid_n+1)[1:]
r2_values = np.zeros((grid_n))

# Define theta parameters
theta = np.array([alpha_values[0], 0.0, 0.0, args.gamma, args.kappa, args.epsilon])

# Search values
last_r2 = -np.infty
max_potential = -np.infty

# Normalise necessary data
si.normalise_data()
xd = si.normalised_initial_destination_sizes
# print(np.log(si.normalised_destination_demand/args.kappa)/xd)
# print(np.min(np.log(si.normalised_destination_demand/args.kappa)/xd))
# print(np.max(np.log(si.normalised_destination_demand/args.kappa)/xd))

# Total sum squares
w_data = np.exp(xd)
w_data_centred = w_data - np.mean(w_data)
ss_tot = np.dot(w_data_centred, w_data_centred)

# Perform grid evaluations
for i in tqdm(range(grid_n)):
    # Update alpha parameter
    theta[0] = alpha_values[i]
    # print('alpha =',theta[0],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])
    try:
        # Compute residual sum of squares
        w_pred = np.exp(minimize(si.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False}).x)
        res = w_pred - w_data
        ss_res = np.dot(res, res)
        # Regression sum squares
        r2_values[i] = 1. - ss_res/ss_tot

    except:
        continue

    # If minimize fails set value to previous, otherwise update previous
    if r2_values[i] == 0:
        r2_values[i] = last_r2
    else:
        last_r2 = r2_values[i]

# Output results
idx = np.unravel_index(r2_values.argmax(), r2_values.shape)
print("Fitted alpha value:")
print(alpha_values[idx], r2_values[idx])
np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/rsquared_analysis_gamma_{str(int(args.gamma))}.txt"), r2_values)
plt.plot(alpha_values, r2_values)
plt.xlim([np.min(alpha_values)-0.01, np.max(alpha_values)+0.2])
plt.ylim([np.min(r2_values)-0.3, np.max(r2_values)+0.5])
plt.ylabel("R^2")
plt.xlabel("Parameter alpha")
plt.savefig(os.path.join(wd,f'data/output/{dataset}/inverse_problem/figures/rsquared_analysis_gamma_{str(int(args.gamma))}.png'))
