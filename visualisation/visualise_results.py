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
parser = argparse.ArgumentParser(description='Visualisation of inverse problem results. See result argument below for more info.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'commuter_borough',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='doubly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-r", "--result",nargs='?',type=str,choices=['low_noise','high_noise','hmc','opt'],default='hmc',
                    help="Type of result to visualise. \
                        low_noise: Low-noise statistics generated from MCMC \
                        high_noise: High-noise statistics generated from MCMC \
                        hmc: Statistics from HMC \
                        opt: Optimal statistics \
                        ")
parser.add_argument("-lf", "--latent_factor",nargs='?',type=int,default=1000,
                    help="Factor used to scale latent destination sizes for visualisation.")
parser.add_argument("-af", "--actual_factor",nargs='?',type=int,default=100,
                    help="Factor used to scale actual destination demand or origin supply for visualisation.")
args = parser.parse_args()
# Print arguments
print(json.dumps(vars(args), indent = 2))

# Define dataset directory
dataset = args.dataset_name
# Define type of spatial interaction model
constrained = args.constrained


# Get project directory
wd = get_project_root()


# Import selected type of spatial interaction model
if constrained == 'singly':
    from models.singly_constrained.spatial_interaction_model import SpatialIteraction

    # Instantiate SpatialIteraction
    si = SpatialIteraction(dataset)

    # Normalise data
    si.normalise_data()

    # Origin supply
    origin_supply = si.normalised_origin_supply

elif constrained == 'doubly':
    from models.doubly_constrained.spatial_interaction_model import SpatialIteraction

    # Instantiate SpatialIteraction
    si = SpatialIteraction(dataset)

    # Normalise data
    si.normalise_data()

    # Origin supply
    origin_supply = si.normalised_origin_supply
    # Destination demand
    destination_demand = si.normalised_destination_demand

else:
    raise ValueError("{} spatial interaction model not implemented.".format(args.constrained))

# Read observation data
# Origin and destination locations (lon/lat)
origin_locs = si.origin_locations
destination_locs = si.destination_locations
# Actual destination sizes
xd = si.normalised_initial_destination_sizes
# Store scale factors for visualisation
actual_factor = args.actual_factor
latent_factor= args.latent_factor


# Figure number
j = 0

plt.figure(j)
plt.scatter(destination_locs[:, 1], destination_locs[:, 0], color='w',edgecolors='r',s=latent_factor*np.exp(xd),label='Actual dest sizes')
if constrained == 'doubly':
    plt.scatter(destination_locs[:, 1], destination_locs[:, 0], color='w',edgecolors='b',s=actual_factor*destination_demand, alpha=0.5,label='Dest demands')
else:
    plt.scatter(origin_locs[:, 1], origin_locs[:, 0], color='w',edgecolors='b',s=actual_factor*origin_supply, alpha=0.5,label='Origin supply')

plt.legend()

if args.result == 'low_noise':
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
    plt.scatter(origin_locs[:, 1], origin_locs[:, 0], s=100*eP, alpha=0.5)
    plt.scatter(destination_locs[:, 1], destination_locs[:, 0], facecolors='none', edgecolors='r', s=1000*lower)
    plt.scatter(destination_locs[:, 1], destination_locs[:, 0], facecolors='none', edgecolors='r', s=1000*upper)

elif args.result == 'high_noise':
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
    plt.scatter(origin_locs[:, 1], origin_locs[:, 0], s=100*eP, alpha=0.5)
    plt.scatter(destination_locs[:, 1], destination_locs[:, 0], facecolors='none', edgecolors='r', s=1000*lower)
    plt.scatter(destination_locs[:, 1], destination_locs[:, 0], facecolors='none', edgecolors='r', s=1000*upper)
    plt.show()

elif args.result == 'hmc':
    print('I am here...')
    # HMC plots
    for alpha in tqdm([2.0]):#0.5,1.0,1.5,2.0]):
        samples = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_hmc_samples_{str(alpha)}.txt"))
        xx = samples[-1]
        j += 1

        plt.figure(j)
        plt.title("HMC alpha =" + str(alpha))
        plt.scatter(destination_locs[:, 1], destination_locs[:, 0], color='w',edgecolors='r', s=latent_factor*np.exp(xx),label='Latent dest sizes')
        if constrained == 'doubly':
            plt.scatter(destination_locs[:, 1], destination_locs[:, 0], color='w',edgecolors='b',s=actual_factor*destination_demand, alpha=0.5,label='Dest demands')
        else:
            plt.scatter(origin_locs[:, 1], origin_locs[:, 0], color='w',edgecolors='b',s=actual_factor*origin_supply, alpha=0.5,label='Origin supply')
        plt.legend()
        print('alpha=',str(alpha),'ended...')

elif args.result == 'optimal':
    # Opt plots
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        xx = np.loadtxt("output/opt" + str(alpha) + ".txt")
        j += 1
        plt.figure(j)
        plt.title("Opt alpha=" + str(alpha))
        plt.scatter(origin_locs[:, 1], origin_locs[:, 0], s=100*eP, alpha=0.5)
        plt.scatter(destination_locs[:, 1], destination_locs[:, 0], color='r', s=1000*np.exp(xx))

plt.show()
