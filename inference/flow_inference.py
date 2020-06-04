''' Infer flows for doubly constrained spatial interaction model '''

import os
import sys

# Get current working directory and project root directory
cwd = os.getcwd()
rd = os.path.join(cwd.split('stochastic-travel-demand-modelling/', 1)[0])
if not rd.endswith('stochastic-travel-demand-modelling'):
    rd = os.path.join(cwd.split('stochastic-travel-demand-modelling/', 1)[0],'stochastic-travel-demand-modelling')
# Append project root directory to path
sys.path.append(rd)

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
from models.doubly_constrained.spatial_interaction import DoublyConstrainedModel

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Infer flows for doubly constrained spatial interaction model based on specified method (DSF or Newton Raphson).')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,default = 'commuter',
                    help="Name of dataset (this is the directory name in data/input).")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = 0.0,
                    help="Beta parameter. Ignore for Poisson regression method.")
parser.add_argument("-A", "--A_factor",nargs='?',type=check_positive_float,default = 1.,
                    help="Initial value for A vector in spatial interaction model. E.g. if A_vector=1, then the A vector is an N-dimensional vector of ones.\
                    Ignore for Poisson regression method.")
parser.add_argument("-B", "--B_factor",nargs='?',type=check_positive_float,default = 1.,
                    help="Initial value for B vector in spatial interaction model. E.g. if B_vector=1, then the B vector is an M-dimensional vector of ones.\
                    Ignore for Poisson regression method.")
parser.add_argument("-m", "--method",nargs='?',type=str,choices=['newton_raphson', 'dsf','poisson_regression'],default = "newton_raphson",
                    help="Method used to estimate flows. ")
parser.add_argument("-id", "--dsf_max_iters",nargs='?',type=check_positive_int,default = 1000,
                    help="Maximum number of iterations of DSF procedure.")
parser.add_argument("-in", "--newton_raphson_max_iters",nargs='?',type=check_positive_int,default = 100,
                    help="Maximum number of iterations for Newton Raphson procedure.")
parser.add_argument("-sp", "--show_params",nargs='?',type=bool,default = False,
                    help="Flag for printing updated parameters in model.")
parser.add_argument("-sf", "--show_flows",nargs='?',type=bool,default = False,
                    help="Flag for printing updated flows in model.")
parser.add_argument("-pf", "--plot_flows",nargs='?',type=bool,default = False,
                    help="Flag for plotting resulting flows in model.")
parser.add_argument("-sod", "--show_orig_dem",nargs='?',type=bool,default = False,
                    help="Flag for printing resulting origin supplies and destination demands in model.")

args = parser.parse_args()
# Print arguments
print(json.dumps(vars(args), indent = 2))


# Define dataset directory
dataset = args.dataset_name
# Define maximum number of iterations of model during learning
newton_raphson_max_iters = args.newton_raphson_max_iters
dsf_max_iters = args.dsf_max_iters
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
dc = DoublyConstrainedModel(dataset,beta=args.beta,A_factor=args.A_factor,B_factor=args.B_factor)

# Initialise flows
inferred_flows = None

# Infer flows based on selected method
if method == 'newton_raphson':
    inferred_flows,_,_ = dc.flow_inference_newton_raphson(newton_raphson_max_iters,dsf_max_iters,show_params)
elif method == 'dsf':
    inferred_flows = dc.flow_inference_dsf_procedure(dsf_max_iters,show_params,show_flows)
elif method == 'poisson_regression':
    inferred_flows = dc.flow_inference_poisson_regression()

# Cast flows to integers
inferred_flows = inferred_flows.astype(int)

# Save array to file
np.savetxt(os.path.join(rd,'data/output/{}/{}_flows.txt'.format(dataset,method)),inferred_flows)

if show_orig_dem:
    print("\n")
    print('Inferred origin supply:',np.sum(inferred_flows,axis=1))
    print('Inferred destination demand:',np.sum(inferred_flows,axis=0))
    print("\n")
print('Total flow:',np.sum(inferred_flows))

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
plt.savefig(os.path.join(rd,'data/output/{}/{}/figures/{}_flows.png'.format(dataset,method,method)))

# Plot figure if requested
if plot_flows:
    plt.show()

# Save parameters to file
with open(os.path.join(rd,'data/output/{}/{}/figures/{}_flows_parameters.json'.format(dataset,method,method)), 'w') as outfile:
    json.dump(vars(args), outfile)

print('Figure saved to {}'.format(os.path.join(rd,'data/output/{}/{}/figures/{}_flows.png'.format(dataset,method,method))))
print('\n')
