"""
Potential V(theta) analysis for spatial interaction model defined in terms of potential function.
"""
import os
import sys
import json
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-amin", "--amin",nargs='?',type=float,default = 0.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-amax", "--amax",nargs='?',type=float,default =  2.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-bmin", "--bmin",nargs='?',type=float,default = 0.0,
                    help="Minimum beta parameter for grid search.")
parser.add_argument("-bmax", "--bmax",nargs='?',type=float,default = 100,
                    help="Minimum beta parameter for grid search.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.3,
                    help="Delta parameter.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter.")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate potential function.")
parser.add_argument("-s", "--show_figure",nargs='?',type=bool,default = False,
                    help="Flag for showing resulting figure.")
parser.add_argument('-hide', '--hide', action='store_true')
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
si = SpatialInteraction(dataset)

# Compute kappa
kappa = 1 + args.delta*si.M

# Initialize search grid
grid_n = args.grid_size
alpha_values = np.linspace(args.amin, args.amax, grid_n+1)[1:]
beta_values = np.linspace(args.bmin, args.bmax, grid_n+1)[1:]
XX, YY = np.meshgrid(alpha_values, beta_values)
potentials = np.zeros((grid_n, grid_n))

# Define theta parameters
theta = np.array([alpha_values[0], beta_values[0], args.delta, 100, kappa, args.epsilon])

# Search values
last_potential = -np.infty

# Normalise initial log destination sizes
si.normalise_data()
xd = si.normalised_initial_destination_sizes

print('delta =',theta[2],'kappa =',theta[4],'epsilon =',theta[5])
# Perform grid evaluations
for i in tqdm(range(grid_n)):
    for j in range(grid_n):
        try:
            # Residiual sum squares
            theta[0] = XX[i, j]
            theta[1] = YY[i, j]
            # Evaluate minimum potential - i.e. maximum likelihood
            min_potential = minimize(si.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False}).fun
            potentials[i, j] = min_potential # np.exp(-potentials[i, j])

        except Exception:
            None

        # If minimize fails set value to previous, otherwise update previous
        if potentials[i, j] == 0:
            potentials[i, j] = last_potential
        else:
            last_potential = potentials[i, j]

# Output results
idx = np.unravel_index(potentials.argmax(), potentials.shape)


print("Fitted alpha, beta and scaled beta values:")
print(XX[idx], YY[idx],YY[idx]*args.amax/(args.bmax))
print("Potential value:")
print(potentials[idx])

# Save potentials to file
np.savetxt(os.path.join(wd,f"data/output/{dataset}/r_squared/{constrained}_{mode}_potential_analysis.txt"), potentials)

# Save fitted values to parameters
arguments['fitted_alpha'] = XX[idx]
arguments['fitted_scaled_beta'] = YY[idx]*args.amax/(args.bmax)
arguments['fitted_beta'] = YY[idx]
arguments['kappa'] = kappa
arguments['potential'] = potentials[idx]

# Save parameters to file
with open(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis_parameters.json'), 'w') as outfile:
    json.dump(arguments, outfile)

print('Constructing flow matrix based on fitted parameters')

# Compute estimated flows
theta[0] = XX[idx]
theta[1] = YY[idx]
estimated_flows = si.reconstruct_flow_matrix(si.normalised_initial_destination_sizes,theta)
# Save estimated flows
np.savetxt(os.path.join(wd,f"data/output/{dataset}/potential/{constrained}_{mode}_potential_estimated_flows.txt"), estimated_flows)


print('Rendering 2D plot of potential function')

# Plot options
plt.style.use('classic')
fig = plt.figure(figsize=(8,8))
fig.tight_layout(pad=0.5)

# Plot potential
plt.pcolor(XX, YY*args.amax/(args.bmax), potentials)
# plt.pcolor(XX, YY, potentials)
plt.xlim([np.min(XX), np.max(XX)])
plt.ylim([np.min(YY)*args.amax/(args.bmax), np.max(YY)*args.amax/(args.bmax)])
# plt.ylim([np.min(YY), np.max(YY)])
pot_cbar = plt.colorbar()
pot_cbar.set_label('V(theta)')
plt.ylabel("Parameter beta")
plt.xlabel("Parameter alpha")

# Save potential figure to file
plt.savefig(os.path.join(wd,f'data/output/{dataset}/potential/figures/{constrained}_{mode}_potential_analysis.png'))


print('Rendering 3D plot of potential function')

# 3D plot of potential
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.plot_surface(XX, YY*args.amax/(args.bmax), potentials, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_ylabel("beta")
ax.set_xlabel("alpha")
ax.set_zlabel("V(theta)")
ax.set_title('Potential value variation across parameter space')

# Save figure to file
plt.savefig(os.path.join(wd,f'data/output/{dataset}/potential/figures/{constrained}_{mode}_potential_analysis_3d.png'))

# Show figure if requested
if args.show_figure:
    plt.show()

print('Done!')
