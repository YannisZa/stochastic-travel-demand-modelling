"""
Evaluates p(x | theta) over grid of alpha and beta values
using the saddle point approximation for z(\theta)
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
parser = argparse.ArgumentParser(description='HMC scheme to sample from prior for latent variables of doubly constrained model.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'synthetic',
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
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.26666666666666666,
                    help="Delta parameter in potential function.")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 100.,
                    help="Gamma parameter.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter.")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate latent size posterior.")
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
theta[0] = -1
theta[1] = -1
theta[2] = args.delta
# Set gamma for Laplace optimisation
theta[3] = args.gamma
theta[4] = 1 + args.delta*si.M
theta[5] = 1 # this is the potential values epsilon parameter which is assumed to be 1.
# Convert to np array
theta = np.array(theta)

xd = si.normalised_initial_destination_sizes

# To estimate z(theta) with Laplace approximation
# def laplace_z(log_sizes,params):
#     # Find global minimum of destination sizes
#     min_sizes = minimize(si.potential_value, log_sizes, method='L-BFGS-B', args=(params), jac=True, options={'disp': False}).x
#     A = si.potential_hessian(min_sizes,params)
#     L = np.linalg.cholesky(A)
#     half_log_det_A = np.sum(np.log(np.diag(L)))
#     return  -si.potential_value(x_min)[0] + log_laplace_numerator -  half_log_det_A


# Initialize search grid
grid_n = args.grid_size
alpha_values = np.linspace(args.amin, args.amax, grid_n+1)[1:]
beta_values = np.linspace(args.bmin, args.bmax, grid_n+1)[1:]
XX, YY = np.meshgrid(alpha_values, beta_values)
log_likelihood_values = np.zeros((grid_n, grid_n))
minimum_values = np.zeros((grid_n, grid_n, si.M))
# \log [(2\pi \gamma^{-1}) ^{M/2}] = (M/2)*\log [2\pi \gamma^{-1}]
# log_laplace_numerator = 0.5*si.M*(np.log(2.*np.pi) - np.log(args.gamma))
log_laplace_numerator = 0.5*si.M*np.log(2.*np.pi)
# print('-log_laplace_numerator',-log_laplace_numerator)

# Search values
last_likelihood = -np.infty
max_potential = -np.infty


# Perform grid evaluations
for i in tqdm(range(grid_n)):
    for j in range(grid_n):
        # print("Running for " + str(i) + ", " + str(j))

        theta[0] = XX[i, j]
        theta[1] = YY[i, j]
        try:

            # Initialise arg min of potential as the true destination sizes
            minimum = xd
            minimum_values[i,j] = minimum
            # Initialise minimum potential
            minimum_potential = np.infty

            # Run L-BFGS with M different starts to ensure robustness of minimum
            for k in range(si.M):
                # Get smallest destination size
                delta = theta[2]
                # Initial guess for the minimum
                g = np.log(delta)*np.ones(si.M)
                # Expand destination k by one unit size
                g[k] = np.log(1. + delta)

                # Evaluate potential value minimum
                f = minimize(si.potential_value, g, method='L-BFGS-B', args=(theta), jac=True, options={'disp': False})

                # Update minimum if minimize found a smaller value
                if(f.fun < minimum_potential):
                    minimum_potential = f.fun
                    minimum = f.x
                    minimum_values[i,j] = f.x

            ''' Estimate likelihood with Laplace approximation '''

            # Get Hessian matrix
            A = si.potential_hessian(minimum,theta)
            # Find its cholesky decomposition Hessian = L*L^T for efficient computation
            L = np.linalg.cholesky(A)
            # Compute the log determinant of the hessian
            # det(Hessian) = det(L)*det(L^T) = det(L)^2
            # det(L) = \prod_{j=1}^M L_{jj} and
            # \log(det(L)) = \sum_{j=1}^M \log(L_{jj})
            # So \log(det(Hessian)^(1/2)) = \log(det(L))
            half_log_det_A = np.sum(np.log(np.diag(L)))
            # Compute log_normalising constant, i.e. \log(z(\theta))
            # -gamma*V(x_{minimum}) + (M/2) * \log(2\pi \gamma^{-1})
            lap =  -si.potential_value(minimum,theta)[0] + log_laplace_numerator - half_log_det_A
            # Compute log-posterior
            # \log(p(x|\theta)) = -gamma*V(x) - \log(z(\theta))
            log_likelihood_values[i, j] = -lap - si.potential_value(xd,theta)[0]

            # print(f'Log-likelihood[{str(i)},{str(j)}] = {str(log_likelihood_values[i, j])}')
            # print(f'Global-minimum[{str(i)},{str(j)}] = {str(minimum_values[i, j])}')
            # print(f'Log-likelihood potential [{str(i)},{str(j)}] = {str(- si.potential_value(xd,theta)[0])}')

        except Exception:
            None

        # If minimize fails set value to previous, otherwise update previous
        if log_likelihood_values[i, j] == 0:
            log_likelihood_values[i, j] = last_likelihood
        else:
            last_likelihood = log_likelihood_values[i, j]


# Output results
idx = np.unravel_index(log_likelihood_values.argmax(), log_likelihood_values.shape)
print("Fitted alpha, beta and scaled beta values:")
print(XX[idx], YY[idx],YY[idx]*args.amax/(args.bmax))
print("Log likelihood")
print(log_likelihood_values[idx])

# Save fitted values to parameters
arguments['fitted_alpha'] = XX[idx]
arguments['fitted_scaled_beta'] = YY[idx]*args.amax/(args.bmax)
arguments['fitted_beta'] = YY[idx]
arguments['kappa'] = 1 + args.delta*si.M
arguments['log_likelihood'] = log_likelihood_values[idx]
arguments['fitted_global_minimum'] = list(minimum_values[idx])

# Save parameters to file
with open(os.path.join(wd,f'data/output/{dataset}/laplace/figures/{constrained}_laplace_analysis_gamma_{str(int(args.gamma))}_parameters.json'), 'w') as outfile:
    json.dump(arguments, outfile)

# Save log-likelihood values to file
np.savetxt(os.path.join(wd,f'data/output/{dataset}/laplace/{constrained}_laplace_analysis_gamma_{str(int(args.gamma))}.txt'), log_likelihood_values)

# Save global minima values to file
# np.savetxt(os.path.join(wd,f'data/output/{dataset}/laplace/{constrained}_laplace_analysis_minima_gamma_{str(int(args.gamma))}.txt'), minimum_values)

# Plot options
plt.style.use('classic')
fig = plt.figure(figsize=(8,8))
fig.tight_layout(pad=0.5)

# Plot result
plt.pcolor(XX, YY*args.amax/args.bmax, log_likelihood_values)
plt.xlim([np.min(XX), np.max(XX)])
plt.ylim([np.min(YY)*args.amax/args.bmax, np.max(YY)*args.amax/args.bmax])
plt.ylabel("Parameter beta")
plt.xlabel("Parameter alpha")
plt.colorbar()

# Save figure to file
plt.savefig(os.path.join(wd,f'data/output/{dataset}/laplace/figures/{constrained}_laplace_analysis_gamma_{str(int(args.gamma))}.png'))

# Show figure if requested
if args.show_figure:
    plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.plot_surface(XX, YY, np.exp(log_likelihood_values), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_ylabel("beta")
ax.set_xlabel("alpha")
ax.set_zlabel("Likelihood")
ax.set_title('Likelihood of latent posterior variation across parameter space')

# Save figure to file
plt.savefig(os.path.join(wd,f'data/output/{dataset}/laplace/figures/{constrained}_laplace_analysis_gamma_{str(int(args.gamma))}_3d.png'))
