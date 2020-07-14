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
sys.path.append(get_project_root())

# Parse arguments from command line
parser = argparse.ArgumentParser(description='R^2 analysis to find fitted parameters based on potential function minima.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'commuter_borough',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-cm", "--cost_matrix_type",nargs='?',type=str,choices=['','sn'],default='',
                    help="Type of cost matrix used.\
                        '': Euclidean distance based. \
                        'sn': Transportation network cost based on A and B roads only. ")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-amin", "--amin",nargs='?',type=float,default = 0.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-amax", "--amax",nargs='?',type=float,default =  2.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-bmin", "--bmin",nargs='?',type=float,default = 0.0,
                    help="Minimum beta parameter for grid search.")
parser.add_argument("-bmax", "--bmax",nargs='?',type=float,default = 400000,
                    help="Minimum beta parameter for grid search.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.3,
                    help="Delta parameter.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter.")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate potential function.")
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

# Initialize search grid
grid_n = args.grid_size
alpha_values = np.linspace(args.amin, args.amax, grid_n+1)[1:]
beta_values = np.linspace(args.bmin, args.bmax, grid_n+1)[1:]
XX, YY = np.meshgrid(alpha_values, beta_values)
r2_values = np.zeros((grid_n, grid_n))
potentials = np.zeros((grid_n, grid_n))
w_predictions = np.zeros((grid_n, grid_n, si.M))

# Define theta parameters
theta = np.array([alpha_values[0], beta_values[0], args.delta, gamma, kappa, args.epsilon])

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
# Perform grid evaluations
for i in tqdm(range(grid_n)):
    for j in range(grid_n):
        try:
            # Residiual sum squares
            theta[0] = XX[i, j]
            theta[1] = YY[i, j]

            # Minimise potential function
            potential_func = minimize(si.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False})

            w_pred = np.exp(potential_func.x)
            res = w_pred - w_data
            ss_res = np.dot(res, res)

            # Regression sum squares
            r2_values[i, j] = 1. - ss_res/ss_tot
            potentials[i, j] = potential_func.fun
            w_predictions[i,j] = w_pred

        except Exception:
            None

        # If minimize fails set value to previous, otherwise update previous
        if r2_values[i, j] == 0:
            r2_values[i, j] = last_r2
            potentials[i, j] = last_potential
            w_predictions[i,j] = last_w_prediction
        else:
            last_r2 = r2_values[i, j]
            last_potential = potentials[i, j]


# Output results
idx = np.unravel_index(r2_values.argmax(), r2_values.shape)

print("Fitted alpha, beta and scaled beta values:")
print(XX[idx], YY[idx],YY[idx]*args.amax/args.bmax)
print("R^2 and potential value:")
print(r2_values[idx],potentials[idx])

if not args.plot_results:
    # Save R^2 to file
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/r_squared/{constrained}_{mode}_rsquared_analysis{si.cost_matrix_file_extension}.txt"), r2_values)

# Save fitted values to parameters
arguments['fitted_alpha'] = XX[idx]
arguments['fitted_scaled_beta'] = YY[idx]*args.amax/args.bmax
arguments['fitted_beta'] = YY[idx]
arguments['kappa'] = kappa
arguments['R^2'] = r2_values[idx]
arguments['potential'] = potentials[idx]

# Save parameters to file
if not rgs.plot_results:
    with open(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis{si.cost_matrix_file_extension}_parameters.json'), 'w') as outfile:
        json.dump(arguments, outfile)

print('Constructing flow matrix based on fitted parameters')

# Compute estimated flows
theta[0] = XX[idx]
theta[1] = YY[idx]
estimated_flows = si.reconstruct_flow_matrix(w_predictions[idx],theta)

if not args.plot_results:
    # Save estimated flows
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/r_squared/{constrained}_{mode}_rsquared_estimated_flows{si.cost_matrix_file_extension}.txt"), estimated_flows)

print('Rendering 2D plot of R^2 variation')

# Plot options
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

# Plot R^2
fig = plt.figure(figsize=(8,8))
fig.tight_layout(pad=0.5)
plt.pcolor(XX, YY*args.amax/(args.bmax), r2_values)
# plt.pcolor(XX, YY, r2_values)
plt.xlim([np.min(XX), np.max(XX)])
plt.ylim([np.min(YY)*args.amax/(args.bmax), np.max(YY)*args.amax/(args.bmax)])
# plt.ylim([np.min(YY), np.max(YY)])
r2_cbar = plt.colorbar()
r2_cbar.set_label(r'$R^2$',rotation=90)
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\alpha$')

# Set negative R^2 values to 0
positive_r2_values = copy.deepcopy(r2_values)
positive_r2_values[positive_r2_values<0] = 0

# Show figure if requested
if args.plot_results:
    plt.show()
else:
    # Save R2 figure to file
    plt.savefig(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis{si.cost_matrix_file_extension}.png'))


print('Rendering 3D plot of R^2 variation')

# 3D plot of R^2
fig = plt.figure(figsize=(8,8))
fig.tight_layout(pad=0.5)
ax = plt.axes(projection='3d')
ax.plot_surface(XX, YY*args.amax/(args.bmax), positive_r2_values, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_ylabel(r"$\beta$")
ax.set_xlabel(r"$\alpha$")
ax.set_zlabel(r"$R^2$")
ax.set_title(r'$R^2$ variation across parameter space')


# Show figure if requested
if args.plot_results:
    plt.show()
else:
    # Save figure to file
    plt.savefig(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis_3d{si.cost_matrix_file_extension}.png'))

print('Done!')
