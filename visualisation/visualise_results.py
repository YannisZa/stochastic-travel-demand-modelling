"""
Plot data in output folder.
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
parser.add_argument("-m", "--mode",nargs='?',type=str,default = 'stochastic',
                    help="Mode of evaluation (stochastic/determinstic)")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='doubly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-a", "--alpha",nargs='?',type=float,default = 2.0,
                    help="Alpha parameter in potential function.")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 0.3*0.7e6,
                    help="Blpha parameter in potential function.")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 100.,
                    help="Gammaa parameter in potential function.")
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
# Define mode (stochastic/determinstic)
mode = args.mode
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
si = SpatialIteraction(mode,dataset)

# Normalise data
si.normalise_data()

# Read observation data
# data = np.loadtxt("data/london_n/shopping_small.txt")
# popn = np.loadtxt("data/london_n/popn.txt")


# Origin supply
# eP = popn[:, 2]/popn[:, 2].sum()
origin_supply = si.origin_supply
# Destination demand
destination_demand = si.destination_demand


print(si.origins)
sys.exit()

ret_locs = data[:, [0, 1]]
res_locs = popn[:, [0, 1]]
plt.figure(j)
plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], color='r', s=1000*np.exp(xd))

# Low noise stats
samples = np.loadtxt("output/low_noise_samples.txt")
samples2 = np.loadtxt("output/low_noise_samples2.txt")
samples3 = np.loadtxt("output/low_noise_samples3.txt")
j+=1
plt.figure(j)
plt.title("Alpha low noise")
plt.plot(samples[:, 0])
plt.xlim([0, 20000])
j+=1
plt.figure(j)
plt.title("Beta low noise")
plt.plot(samples[:, 1])
plt.xlim([0, 20000])
print("\nLow noise stats:")
alpha_mean = np.dot(samples3, samples[:, 0])/np.sum(samples3)
alpha_sd = np.sqrt(np.dot(samples3, samples[:, 0]**2)/np.sum(samples3) - alpha_mean**2)
print("Alpha mean: " + str(alpha_mean))
print("Alpha sd: " + str(alpha_sd))
beta_mean = np.dot(samples3, samples[:, 1])/np.sum(samples3)
beta_sd = np.sqrt(np.dot(samples3, samples[:, 1]**2)/np.sum(samples3) - beta_mean**2)
print("Beta mean: " + str(beta_mean))
print("Beta sd: " + str(beta_sd))

x_e = (np.exp(samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
x2_e = (np.exp(2*samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
sd = np.sqrt(x2_e - x_e**2)
mean = x_e
lower = mean-3.*sd
upper = mean+3.*sd
j+=1
plt.figure(j)
plt.title("Low noise latents")
plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*lower)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*upper)

# High noise stats
samples = np.loadtxt("output/high_noise_samples.txt")
samples2 = np.loadtxt("output/high_noise_samples2.txt")
samples3 = np.loadtxt("output/high_noise_samples3.txt")
j+=1
plt.figure(j)
plt.title("Alpha high noise")
plt.plot(samples[:, 0])
plt.xlim([0, 20000])
j+=1
plt.figure(j)
plt.title("Beta high noise")
plt.plot(samples[:, 1])
plt.xlim([0, 20000])
print("\nHigh noise stats:")
alpha_mean = np.dot(samples3, samples[:, 0])/np.sum(samples3)
alpha_sd = np.sqrt(np.dot(samples3, samples[:, 0]**2)/np.sum(samples3) - alpha_mean**2)
print("Alpha mean: " + str(alpha_mean))
print("Alpha sd: " + str(alpha_sd))
beta_mean = np.dot(samples3, samples[:, 1])/np.sum(samples3)
beta_sd = np.sqrt(np.dot(samples3, samples[:, 1]**2)/np.sum(samples3) - beta_mean**2)
print("Beta mean: " + str(beta_mean))
print("Beta sd: " + str(beta_sd))

x_e = (np.exp(samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
x2_e = (np.exp(2*samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
sd = np.sqrt(x2_e - x_e**2)
mean = x_e
lower = mean-3.*sd
upper = mean+3.*sd
j+=1
plt.figure(j)
plt.title("Low noise latents")
plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*lower)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*upper)
plt.show()

# HMC plots
for alpha in [0.5, 1.0, 1.5, 2.0]:
    samples = np.loadtxt("output/hmc_samples" + str(alpha) + ".txt")
    xx = samples[-1]
    j += 1
    plt.figure(j)
    plt.title("HMC alpha=" + str(alpha))
    plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
    plt.scatter(ret_locs[:, 1], ret_locs[:, 0], color='r', s=1000*np.exp(xx))

# Opt plots
for alpha in [0.5, 1.0, 1.5, 2.0]:
    xx = np.loadtxt("output/opt" + str(alpha) + ".txt")
    j += 1
    plt.figure(j)
    plt.title("Opt alpha=" + str(alpha))
    plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
    plt.scatter(ret_locs[:, 1], ret_locs[:, 0], color='r', s=1000*np.exp(xx))

plt.show()
