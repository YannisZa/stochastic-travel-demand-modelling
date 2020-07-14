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
from matplotlib import rc
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
parser = argparse.ArgumentParser(description='Visualisation of inverse problem results. See result argument below for more info.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'synthetic',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-r", "--result",nargs='?',type=str,choices=['low_noise','high_noise','hmc','opt'],default='hmc',
                    help="Type of result to visualise. \
                        low_noise: Low-noise statistics generated from MCMC \
                        high_noise: High-noise statistics generated from MCMC \
                        hmc: Statistics from HMC \
                        opt: Optimal statistics \
                        ")
parser.add_argument("-lf", "--latent_factor",nargs='?',type=int,default=100,
                    help="Factor used to scale latent destination sizes for visualisation.")
parser.add_argument("-af", "--actual_factor",nargs='?',type=int,default=1000,
                    help="Factor used to scale actual destination demand or origin supply for visualisation.")
parser.add_argument('-hide', '--hide', action='store_true')
args = parser.parse_args()
# Print arguments
if not args.hide:
    print(json.dumps(vars(args), indent = 2))

# Define dataset directory
dataset = args.dataset_name
# Define type of spatial interaction model
constrained = args.constrained

# Import selected type of spatial interaction model
if constrained == 'singly':
    from models.singly_constrained.spatial_interaction_model import SpatialInteraction

    # Instantiate SpatialInteraction
    si = SpatialInteraction(dataset)

    # Normalise data
    si.normalise_data()

    # Origin supply
    origin_supply = si.normalised_origin_supply

elif constrained == 'doubly':
    from models.doubly_constrained.spatial_interaction_model import SpatialInteraction

    # Instantiate SpatialInteraction
    si = SpatialInteraction(dataset)

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


# Plot settings
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

# Figure number
j = 0

plt.figure(j)
if constrained == 'doubly':
    plt.scatter(destination_locs[:, 0], destination_locs[:, 1], color='b',edgecolors='b',s=latent_factor*destination_demand, alpha=0.5,label='Dest demands')
else:
    plt.scatter(origin_locs[:, 0], origin_locs[:, 1], color='b',edgecolors='b',s=actual_factor*origin_supply, alpha=0.5,label='Origin supply')
plt.scatter(destination_locs[:, 0], destination_locs[:, 1], color='r',edgecolors='r',s=actual_factor*np.exp(xd),label='Actual dest sizes')
plt.legend()

if args.result == 'low_noise':
    # Low noise stats
    samples = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_theta_samples.txt"))
    samples2 = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_logsize_samples.txt"))
    samples3 = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_sign_samples.txt"))

    # Import arguments
    with open(os.path.join(wd,f'data/output/{dataset}/inverse_problem/{constrained}_low_noise_mcmc_samples_parameters.json')) as infile:
        arguments = json.load(infile)


    j+=1
    plt.figure(j)
    plt.title("Alpha low noise")
    plt.plot(samples[:, 0])
    plt.xlim([arguments['mcmc_start'], arguments['mcmc_n']])

    j+=1
    plt.figure(j)
    plt.title("Beta low noise")
    plt.plot(samples[:, 1])
    plt.xlim([arguments['mcmc_start'], arguments['mcmc_n']])

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
    plt.scatter(origin_locs[:, 0], origin_locs[:, 1], s=100*si.normalised_origin_supply, alpha=0.5)
    plt.scatter(destination_locs[:, 0], destination_locs[:, 1], facecolors='none', edgecolors='r', s=1000*lower)
    plt.scatter(destination_locs[:, 0], destination_locs[:, 1], facecolors='none', edgecolors='r', s=1000*upper)

    j+=1
    plt.figure(j)
    plt.title("Residual plot")
    plt.scatter(x=np.log(mean),y=(si.normalised_initial_destination_sizes-np.log(mean)))
    plt.xlim([np.min(np.log(mean)), np.max(np.log(mean))])

    j+=1
    plt.figure(j)
    plt.title("Predictions plot")
    plt.scatter(x=np.log(mean),y=si.normalised_initial_destination_sizes)
    plt.xlim([np.min(np.log(mean)), np.max(np.log(mean))])

elif args.result == 'high_noise':
    # High noise stats
    samples = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_theta_samples.txt"))
    samples2 = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_logsize_samples.txt"))
    samples3 = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_sign_samples.txt"))

    # Import arguments
    with open(os.path.join(wd,f'data/output/{dataset}/inverse_problem/{constrained}_high_noise_mcmc_samples_parameters.json')) as infile:
        arguments = json.load(infile)


    j+=1
    plt.figure(j)
    plt.title("Alpha high noise")
    plt.plot(samples[:, 0])
    plt.xlim([arguments['mcmc_start'], arguments['mcmc_n']])

    j+=1
    plt.figure(j)
    plt.title("Beta high noise")
    plt.plot(samples[:, 1])
    plt.xlim([arguments['mcmc_start'], arguments['mcmc_n']])

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
    plt.title("High noise latents")
    plt.scatter(origin_locs[:, 0], origin_locs[:, 1], s=100*si.normalised_origin_supply, alpha=0.5)
    plt.scatter(destination_locs[:, 0], destination_locs[:, 1], facecolors='none', edgecolors='r', s=1000*lower)
    plt.scatter(destination_locs[:, 0], destination_locs[:, 1], facecolors='none', edgecolors='r', s=1000*upper)
    plt.show()

    j+=1
    plt.figure(j)
    plt.title("Residual plot")
    plt.scatter(x=np.log(mean),y=(si.normalised_initial_destination_sizes-np.log(mean)))
    plt.xlim([np.min(np.log(mean)), np.max(np.log(mean))])

    j+=1
    plt.figure(j)
    plt.title("Predictions plot")
    plt.scatter(x=np.log(mean),y=si.normalised_initial_destination_sizes)
    plt.xlim([np.min(np.log(mean)), np.max(np.log(mean))])

elif args.result == 'hmc':
    # HMC plots
    for alpha in tqdm([0.5,1.0,1.5,2.0]):
        # Import latent posterior samples
        samples = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_hmc_samples_{str(alpha)}.txt"))

        # Import arguments
        with open(os.path.join(wd,f'data/output/{dataset}/inverse_problem/{constrained}_high_noise_mcmc_samples_parameters.json')) as infile:
            arguments = json.load(infile)

        xx = samples[-1]
        j += 1

        plt.figure(j)
        plt.title(rf"Latent size posterior for $\alpha = {alpha}$")
        if constrained == 'doubly':
            plt.scatter(destination_locs[:, 0], destination_locs[:, 1], color='w',edgecolors='b',s=actual_factor*destination_demand, alpha=0.5,label='Dest demands')
        else:
            plt.scatter(origin_locs[:, 0], origin_locs[:, 1], color='w',edgecolors='b',s=actual_factor*origin_supply, alpha=0.5,label='Origin supply')
        plt.scatter(destination_locs[:, 0], destination_locs[:, 1], color='w',edgecolors='r', s=latent_factor*np.exp(xx),label='Latent dest sizes')
        plt.legend()

        plt.savefig(os.path.join(wd,f"data/output/{dataset}/inverse_problem/figures/{constrained}_hmc_samples_{str(alpha)}.png"))
        print('alpha=',str(alpha),'ended...')


elif args.result == 'opt':
    # Opt plots
    for alpha in tqdm([0.5, 1.0, 1.5, 2.0]):
        xx = np.loadtxt(os.path.join(wd,f'data/output/{dataset}/laplace/{constrained}_optimal_latent_posterior_{str(alpha)}.txt'))
        j += 1
        plt.figure(j)
        plt.title(r"Optimal latent posterior for $\alpha$ =" + str(alpha))
        plt.scatter(origin_locs[:, 0], origin_locs[:, 1], s=latent_factor*origin_supply, alpha=0.5)
        plt.scatter(destination_locs[:, 0], destination_locs[:, 1], color='r', s=actual_factor*np.exp(xx))
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(wd,f"data/output/{dataset}/laplace/figures/{constrained}_optimal_latent_posterior_{str(alpha)}.png"))
        print('alpha=',str(alpha),'ended...')

plt.show()
