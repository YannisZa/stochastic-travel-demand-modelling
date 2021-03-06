''' Infer flows for doubly constrained spatial interaction model '''

import os
import sys

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

# Ensure positivity of arguments
def check_positive(value,type):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} is an invalid positive {} value".format(value,type))
    return ivalue
def check_positive_int(value):
    ivalue = int(value)
    return check_positive(ivalue,'int')
def check_positive_float(value):
    ivalue = float(value)
    return check_positive(ivalue,'float')

import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from models.doubly_constrained.spatial_interaction_model import SpatialIteraction

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Infer flows for doubly constrained spatial interaction model based on specified method (DSF or Newton Raphson).')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,default = 'commuter',
                    help="Name of dataset (this is the directory name in data/input).")
parser.add_argument("-a", "--alpha",nargs='?',type=float,default = 0.0,
                    help="Alpha parameter - destination size power. Used only in iterative proportional filtering.")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 0.0,
                    help="Beta parameter - distance coefficient. Ignore for Poisson regression method.")
parser.add_argument("-A", "--A_factor",nargs='?',type=check_positive_float,default = 1.,
                    help="Initial value for A vector in spatial interaction model. E.g. if A_vector=1, then the A vector is an N-dimensional vector of ones.\
                    Ignore for Poisson regression method.")
parser.add_argument("-B", "--B_factor",nargs='?',type=check_positive_float,default = 1.,
                    help="Initial value for B vector in spatial interaction model. E.g. if B_vector=1, then the B vector is an M-dimensional vector of ones.\
                    Ignore for Poisson regression method.")
parser.add_argument("-m", "--method",nargs='?',type=str,choices=['actual','newton_raphson', 'dsf', 'ipf', 'poisson_regression'],default = "newton_raphson",
                    help="Method used to estimate flows. ")
parser.add_argument("-id", "--dsf_max_iters",nargs='?',type=int,default = 1000,
                    help="Maximum number of iterations of DSF procedure.")
parser.add_argument("-ii", "--ipf_max_iters",nargs='?',type=int,default = 1000,
                    help="Maximum number of iterations of Iterative proportional filtering procedure.")
parser.add_argument("-in", "--newton_raphson_max_iters",nargs='?',type=int,default = 100,
                    help="Maximum number of iterations for Newton Raphson procedure.")
parser.add_argument("-t", "--tolerance",nargs='?',type=check_positive_float,default = 1.0,
                    help="Tolerance value used to assess if total (origin + destination) errors are within acceptable limits.")
parser.add_argument("-sp", "--show_params",nargs='?',type=bool,default = False,
                    help="Flag for printing updated parameters in model.")
parser.add_argument("-sf", "--show_flows",nargs='?',type=bool,default = False,
                    help="Flag for printing updated flows in model.")
parser.add_argument("-pf", "--plot_flows",nargs='?',type=bool,default = False,
                    help="Flag for plotting resulting flows in model.")
parser.add_argument("-sod", "--show_orig_dem",nargs='?',type=bool,default = False,
                    help="Flag for printing resulting origin supplies and destination demands in model.")
parser.add_argument('-hide', '--hide', action='store_true')
args = parser.parse_args()
# Convert arguments to dictionary
arguments = vars(args)
# Print arguments
if not args.hide:
    print(json.dumps(arguments, indent = 2))


# Define dataset directory
dataset = args.dataset_name
# Define maximum number of iterations of model during learning
newton_raphson_max_iters = args.newton_raphson_max_iters
dsf_max_iters = args.dsf_max_iters
ipf_max_iters = args.ipf_max_iters


# Set error tolerance for iterative proportional filtering procedure
tolerance = args.tolerance
# Set method used to estimate flows
method = args.method
# Set flag for print statements
show_params = args.show_params
show_flows = args.show_flows
plot_flows = args.plot_flows
show_orig_dem = args.show_orig_dem



''' Infer flows '''
print('Inferring flows using {} method'.format(method.replace('_',' ')))

# Instantiate DoublyConstrainedModel
dc = SpatialIteraction(dataset)

# Initialise flows
inferred_flows = None

# Infer flows based on selected method
if method == 'newton_raphson':
    inferred_flows,_,_ = dc.flow_inference_newton_raphson(newton_raphson_max_iters,dsf_max_iters,show_params,beta=args.beta)
elif method == 'dsf':
    inferred_flows = dc.flow_inference_dsf_procedure(dsf_max_iters,show_params,show_flows,beta=args.beta)
elif method == 'poisson_regression':
    inferred_flows,_ = dc.flow_inference_poisson_regression()
elif method == 'ipf':
    inferred_flows = dc.flow_inference_ipf_procedure(tolerance,ipf_max_iters,show_flows,alpha=args.alpha,beta=args.beta,A_factor=args.A_factor,B_factor=args.B_factor)
elif method == 'actual':
    inferred_flows = dc.actual_flows

# Cast flows to integers
inferred_flows = inferred_flows.astype(int)

# Get project directory
wd = get_project_root()

if method != 'actual':
    # Save array to file
    np.savetxt(os.path.join(wd,'data/output/{}/{}/{}_flows.txt'.format(dataset,method,method)),inferred_flows)

if show_orig_dem and method != 'actual':
    print("\n")
    print('Inferred origin supply:',np.sum(inferred_flows,axis=1))
    print('Inferred destination demand:',np.sum(inferred_flows,axis=0))
    print("\n")
else:
    print("\n")
print('Total flow:',np.sum(inferred_flows))
print("\n")
print('SRMSE:',dc.SRMSE(dc.actual_flows))

''' Plot heatmap of flows '''
print('\n')
print('Plotting flow heatmap')

# Change font scaling
sns.set(font_scale=0.8)
# Set plot size
plt.figure(figsize=(20,20))
# Add heatmap
flow_heatmap = sns.heatmap(inferred_flows,
                            annot=True,
                            cmap="coolwarm",
                            fmt="d",
                            xticklabels=dc.destinations,
                            yticklabels=dc.origins,
                            linewidths=.05)
# Add x,y axes labels
plt.xlabel("Destination")
plt.ylabel("Origin")
# Add title
plt.title('Origin demand flows of {} data'.format(dataset), fontsize=20)

# Save figure to output
plt.savefig(os.path.join(wd,'data/output/{}/{}/figures/{}_flows.png'.format(dataset,method,method)))

# Plot figure if requested
if plot_flows:
    plt.show()

if method != 'actual':
    # Save parameters to file
    with open(os.path.join(wd,'data/output/{}/{}/figures/{}_flows_parameters.json'.format(dataset,method,method)), 'w') as outfile:
        json.dump(vars(args), outfile)

print('Figure saved to {}'.format(os.path.join(wd,'data/output/{}/{}/figures/{}_flows.png'.format(dataset,method,method))))
print('\n')
