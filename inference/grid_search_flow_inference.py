''' Infer flows for doubly constrained spatial interaction model using a grid search and the DSF procedure
For more details on the procedure look at  models/doubly_constrained_model.c

The search is performed over betas and not over A_factor and/or B_factor
'''

import os
import sys

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# Import module that couldn't be imported before
from models.doubly_constrained.spatial_interaction import DoublyConstrainedModel



# Ensure positivity of arguments
def check_positive(value,type):
    if value <= 0:
        raise argparse.ArgumentTypeError("{} is an invalid positive {} value".format(value,type))
    return value
def check_positive_int(value):
    ivalue = int(value)
    return check_positive(ivalue,'int')
def check_positive_float(value):
    ivalue = float(value)
    return check_positive(ivalue,'float')



# Parse arguments from command line
parser = argparse.ArgumentParser(description='Infer flows for doubly constrained spatial interaction model using a grid search and the DSF procedure\
For more details on the procedure look at  models/doubly_constrained_model.c')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,default = 'commuter',
                    help="Name of dataset (this is the directory name in data/input).")
parser.add_argument("-bmin", "--beta_minimum",nargs='?',type=float,default = -15.0,
                    help="Minimum beta parameter to be tried.")
parser.add_argument("-bmax", "--beta_maximum",nargs='?',type=float,default = 15.0,
                    help="Maximum beta parameter to be tried.")
parser.add_argument("-n", "--number_of_samples",nargs='?',type=check_positive_int,default = 500,
                    help="Number of equally spaced samples to generate between beta_minimum and beta_maximum.")
parser.add_argument("-A", "--A_factor",nargs='?',type=check_positive_float,default = 1.,
                    help="Initial value for A vector in spatial interaction model. E.g. if A_vector=1, then the A vector is an N-dimensional vector of ones.")
parser.add_argument("-B", "--B_factor",nargs='?',type=check_positive_float,default = 1.,
                    help="Initial value for B vector in spatial interaction model. E.g. if B_vector=1, then the B vector is an M-dimensional vector of ones.")
parser.add_argument("-id", "--dsf_max_iters",nargs='?',type=check_positive_int,default = 1000,
                    help="Maximum number of iterations of DSF procedure.")
parser.add_argument("-sp", "--show_params",nargs='?',type=bool,default = False,
                    help="Flag for printing updated parameters in model.")
parser.add_argument('-hide', '--hide', action='store_true')
args = parser.parse_args()
# Convert arguments to dictionary
arguments = vars(args)
# Print arguments
if not args.hide:
    print(json.dumps(arguments, indent = 2))


# Store dataset directory
dataset = args.dataset_name
# Store maximum number of iterations of model during learning
dsf_max_iters = args.dsf_max_iters
# Store number of parameter samples to generate in sample along with specified range
number_of_samples = args.number_of_samples
beta_maximum = args.beta_maximum
beta_minimum = args.beta_minimum
# Set flag for print statements
show_params = args.show_params


''' Perform grid search '''

# Initialise grid
beta_grid = np.linspace(beta_minimum,beta_maximum,number_of_samples)

# Initialise grid of inferred flows
grid_inferred_flows = {}

# Initialise min SRMSE and its corresponding beta
optimal_beta = None
optimal_SRMSE = np.inf

for i,beta in tqdm(enumerate(beta_grid),total=len(beta_grid)):
    # Initilise empty dictionary for iteration
    grid_inferred_flows[str(i)] = {}

    # Store beta
    grid_inferred_flows[str(i)]['beta'] = beta

    # Instantiate DoublyConstrainedModel
    dc = DoublyConstrainedModel(dataset,beta=beta,A_factor=args.A_factor,B_factor=args.B_factor)

    # Infer flows based on selected method
    inferred_flows = dc.flow_inference_dsf_procedure(dsf_max_iters,show_params,False)

    # Cast flows to integers
    inferred_flows = inferred_flows.astype(int)

    # Compute SRMS error
    srmse = dc.SRMSE(inferred_flows)

    # Update optimal parameter and its SRMSE if it achieves smaller error
    if srmse <= optimal_SRMSE:
        optimal_beta = beta
        optimal_SRMSE = srmse

    # Add SRMSE to dictionary
    grid_inferred_flows[str(i)]['SRMSE'] = srmse

# Convert grid of flows to dataframe
grid_inferred_flows_df = pd.DataFrame.from_dict(grid_inferred_flows)
# Take transpose
grid_inferred_flows_df = grid_inferred_flows_df.transpose()

# Get project directory
wd = get_project_root()

# Save grid to file
grid_inferred_flows_df.to_csv(os.path.join(wd,'data/output/{}/dsf_grid_srmes.csv'.format(dataset)))

if show_params:
    print("\n")
    print('Optimal beta:',optimal_beta)
    print('Optimal beta SRMSE:',optimal_SRMSE)
    print("\n")


print('Data saved to {}'.format(os.path.join(wd,'data/output/{}/dsf_grid_srmes.csv'.format(dataset))))
print('\n')
