"""
HMC scheme to sample from prior for latent variables.
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
parser.add_argument("-a", "--alpha",nargs='?',type=float,default = 2.0,
                    help="Alpha parameter in potential function.")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 0.3*0.7e6,
                    help="Beta parameter in potential function.")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 100.,
                    help="Gamma parameter in potential function.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.3,
                    help="Delta parameter in potential function.")
parser.add_argument("-k", "--kappa",nargs='?',type=float,default = 1.3,
                    help="Kappa parameter in potential function.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter in potential function.")
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

# Instantiate SpatialIteraction
si = SpatialIteraction(dataset)

# Normalise data
si.normalise_data()

# Set theta for high-noise model
theta = [0 for i in range(5)]
theta[0] = args.alpha
theta[1] = args.beta
theta[2] = args.delta/si.M
theta[3] = args.gamma
theta[4] = args.kappa
# Convert to np array
theta = np.array(theta)

# MCMC tuning parameters
# Number of leapfrog steps
L = 10
# Leapfrog step size
eps = 0.1


# Set-up MCMC
mcmc_n = 10000
temp_n = 5
inverse_temps = np.array([1., 1./2., 1./4., 1./8., 1./16.])
samples = np.empty((mcmc_n, si.M))   # X-values


# Initialize MCMC
xx = -np.log(si.M)*np.ones((temp_n, si.M))
V = np.empty(temp_n)
gradV = np.empty((temp_n, si.M))
for j in range(temp_n):
    V[j], gradV[j] = si.potential_value(xx[j],theta)


# Counts to keep track of accept rates
ac = np.zeros(temp_n)
pc = np.zeros(temp_n)
acs = 0
pcs = 1

# MCMC algorithm
for i in tqdm(range(mcmc_n)):
    for j in range(temp_n):
        #Initialize leapfrog integrator for HMC proposal
        p = np.random.normal(0., 1., si.M)

        H = 0.5*np.dot(p, p) + inverse_temps[j]*V[j]

        # X-Proposal
        x_p = xx[j]
        p_p = p
        V_p, gradV_p = V[j], gradV[j]
        for l in range(L):
            p_p = p_p -0.5*eps*inverse_temps[j]*gradV_p
            x_p = x_p + eps*p_p
            V_p, gradV_p = si.potential_value(x_p,theta)
            p_p = p_p - 0.5*eps*inverse_temps[j]*gradV_p

        # X-accept/reject
        pc[j] += 1
        H_p = 0.5*np.dot(p_p, p_p) + inverse_temps[j]*V_p
        if np.log(np.random.uniform(0, 1)) < H - H_p:
            xx[j] = x_p
            V[j], gradV[j] = V_p, gradV_p
            ac[j] += 1

    # Perform a swap
    pcs += 1
    j0 = np.random.randint(0, temp_n-1)
    j1 = j0+1
    logA = (inverse_temps[j1]-inverse_temps[j0])*(-V[j1] + V[j0])
    if np.log(np.random.uniform(0, 1)) < logA:
        xx[[j0, j1]] = xx[[j1, j0]]
        V[[j0, j1]] = V[[j1, j0]]
        gradV[[j0, j1]] = gradV[[j1, j0]]
        acs += 1

    # Update stored Markov-chain
    samples[i] = xx[0]

    # Savedown and output details every 100 iterations
    if (i+1) % 100 == 0:
        print("Saving iteration " + str(i+1))
        np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/hmc_samples_{str(theta[0])}.txt"), samples)
        print("X AR:")
        print(ac/pc)
        print("Swap AR:" + str(float(acs)/float(pcs)))

print("Done")
