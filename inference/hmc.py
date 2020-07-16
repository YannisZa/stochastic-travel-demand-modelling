"""
HMC scheme to sample from prior for latent variables of spatial interaction model
"""

import os
import sys
import json
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
parser = argparse.ArgumentParser(description='HMC scheme to sample from posterior of latent variables of the spatial interaction model.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'synthetic',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-cm", "--cost_matrix_type",nargs='?',type=str,choices=['','sn'],default='',
                    help="Type of cost matrix used.\
                        '': Euclidean distance based. \
                        'sn': Transportation network cost based on A and B roads only. ")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-a", "--alpha",nargs='?',type=float,default = 1.2,
                    help="Alpha parameter in potential function.")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 0.3,
                    help="Beta parameter in potential function.")
parser.add_argument("-bm", "--bmax",nargs='?',type=float,default = 700000,
                    help="Maximum value for the beta parameter in potential function. This is used to normalise the above argument.")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 100.,
                    help="Gamma parameter in potential function.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.26666666666666666,
                    help="Delta parameter in potential function.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 0.1,
                    help="Leapfrog step size. This is NOT the potential value's epsilon parameter, which assumed to be 1.")
parser.add_argument("-nm", "--mcmc_n",nargs='?',type=int,default = 10000,
                    help="Number of MCMC iterations.")
parser.add_argument("-L", "--L",nargs='?',type=int,default = 10,
                    help="Number of leapfrog steps to be taken in HMC latent variable update.")
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
si = SpatialInteraction(dataset,args.cost_matrix_type)

# Normalise data
si.normalise_data()

# Fix random seed
np.random.seed(888)

# Set theta for high-noise model's potential value parameters
theta = [0 for i in range(6)]
theta[0] = args.alpha
theta[1] = args.beta*args.bmax
theta[2] = args.delta
theta[3] = args.gamma
theta[4] = 1 + args.delta*si.M
theta[5] = 1 # this is the potential values epsilon parameter which is assumed to be 1.
# Convert to np array
theta = np.array(theta)

# MCMC tuning parameters
# Number of leapfrog steps
L = args.L
# Leapfrog step size
epsilon = args.epsilon


# Inverse temperatures that go into potential energy of Hamiltonian dynamics
inverse_temperatures = np.array([1., 1./2., 1./4., 1./8., 1./16.])

# Set-up MCMC
mcmc_n = args.mcmc_n
temp_n = len(inverse_temperatures)

# Array to store X values sampled at each iteration
samples = np.empty((mcmc_n, si.M))


# Initialize MCMC
xx = -np.log(si.M)*np.ones((temp_n, si.M))

# Initiliase arrays for potential value and its gradient
V = np.empty(temp_n)
gradV = np.empty((temp_n, si.M))
# Get potential value and its gradient for the initial choice of theta and x
for j in range(temp_n):
    V[j], gradV[j] = si.potential_value(xx[j],theta)

# Counts to keep track of accept rates
# Acceptance count
ac = np.zeros(temp_n)
# Proposal count
pc = np.zeros(temp_n)
acs = 0
pcs = 1

print('alpha =',theta[0],'beta =',theta[1],'delta =',theta[2],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])

# MCMC algorithm
for i in tqdm(range(mcmc_n)):

    for j in range(temp_n):
        ''' HMC parameter/function correspondence to spatial interaction model
        q = x : position = log destination sizes
        U(x) = gamma*V(x|theta)*(1/T) :  potential energy = potential value given parameter theta times inverse temperature

        Note that the potential value function returns -gamma*V(x|theta)
        '''
        # Initialise leapfrog integrator for HMC proposal

        # Initialise momentum
        p = np.random.normal(0., 1., si.M)

        # Hamiltonian total energy function = kinetic energy + potential energy
        # at the beginning of trajectory
        # Kinetic energy K(p) = p^TM^{−1}p / 2 with M being the identity matrix
        H = 0.5*np.dot(p, p) + inverse_temperatures[j]*V[j]

        # X-Proposal (position q is the log size vector X)
        x_p = xx[j]

        # Set current momentum
        p_p = p

        # Get potential value and its Jacobian
        V_p, gradV_p = V[j], gradV[j]

        # Alternate full steps for position and momentum
        for l in range(L):
            # Make a half step for momentum in the beginning
            # inverse_temps[j]*gradV_p = grad V(x|theta)*(1/T)
            p_p = p_p - 0.5*epsilon*inverse_temperatures[j]*gradV_p

            # Make a full step for the position
            x_p = x_p + epsilon*p_p
            # Update log potential value and its gradient
            V_p, gradV_p = si.potential_value(x_p,theta)

            # print('l =',l)
            # print('V_p =',V_p)

            # Make a full step for the momentum except at the end of trajectory
            # if (l != (L-1)):
            p_p = p_p - 0.5*epsilon*inverse_temperatures[j]*gradV_p

        # sys.exit()


        # Make a falf step for momentum at the end.
        # p_p -= 0.5*epsilon*inverse_temperatures[j]*gradV_p

        # Negate momentum
        # p_p *= (-1)

        # Increment proposal count
        pc[j] += 1

        # Compute Hamiltonian total energy function at the end of trajectory
        H_p = 0.5*np.dot(p_p, p_p) + inverse_temperatures[j]*V_p

        # Accept/reject X by either returning the position at the end of the trajectory or the initial position
        if np.log(np.random.uniform(0, 1)) < H - H_p:
            xx[j] = x_p
            V[j], gradV[j] = V_p, gradV_p
            ac[j] += 1

    # Perform a swap
    pcs += 1
    # Select random index
    j0 = np.random.randint(0, temp_n-1)
    # Get neighbouring index
    j1 = j0 + 1
    # If uniform(0,1) > A  or log(uniform(0,1) < log(A) = (inverse_temperatures[j1]-inverse_temperatures[j0])*(-V[j1] + V[j0])
    # See acceptance rate in page 212 of Monte Carlo Strategies in Scientific Computing
    logA = (inverse_temperatures[j1]-inverse_temperatures[j0])*(-V[j1] + V[j0])
    if np.log(np.random.uniform(0, 1)) < logA:
        xx[[j0, j1]] = xx[[j1, j0]]
        V[[j0, j1]] = V[[j1, j0]]
        gradV[[j0, j1]] = gradV[[j1, j0]]
        acs += 1

    # Update stored Markov-chain
    samples[i] = xx[0]

    # Savedown and output details every 100 iterations
    if int(0.05*args.mcmc_n) > 0:
        if (i+1) % (int(0.05*args.mcmc_n)) == 0:
            print("Saving iteration " + str(i+1))
            np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_hmc_samples{si.cost_matrix_file_extension}_{str(theta[0])}.txt"), samples)
            print("X AR:")
            print(ac/pc)
            print("Swap AR:" + str(float(acs)/float(pcs)))

if int(0.05*args.mcmc_n) == 0:
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_hmc_samples{si.cost_matrix_file_extension}_{str(theta[0])}.txt"), samples)
    print("X AR:")
    print(ac/pc)
    print("Swap AR:" + str(float(acs)/float(pcs)))

# Save parameters to file
with open(os.path.join(wd,f'data/output/{dataset}/inverse_problem/{constrained}_hmc_samples{si.cost_matrix_file_extension}_{str(theta[0])}_parameters.json'), 'w') as outfile:
    json.dump(arguments, outfile)

print("Done")
print("Data saved to",os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_hmc_samples{si.cost_matrix_file_extension}_{str(theta[0])}.txt"))
