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

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.spatial_interaction import DoublyConstrainedModel
from tqdm import tqdm

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Plot potential function for given choice of parameters.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,default = 'commuter',
                    help="Name of dataset (this is the directory name in data/input).")
parser.add_argument("-b", "--beta",nargs='?',type=float,default = -13.,
                    help="Beta parameter. Ignore for Newton Raphson method.")
parser.add_argument("-A", "--A_factor",nargs='?',type=float,default = 1.,
                    help="Initial value for A vector in spatial interaction model. E.g. if A_vector=1, then the A vector is an N-dimensional vector of ones.")
parser.add_argument("-B", "--B_factor",nargs='?',type=float,default = 1.,
                    help="Initial value for B vector in spatial interaction model. E.g. if B_vector=1, then the B vector is an M-dimensional vector of ones.")
parser.add_argument("-m", "--max_iters",nargs='?',type=int,default = 1000,
                    help="Maximum number of iterations for which A and B vectors will be recursively updated.")
parser.add_argument("-sp", "--show_params",nargs='?',type=bool,default = False,
                    help="Flag for printing updated parameters in model.")
parser.add_argument("-sf", "--show_flows",nargs='?',type=bool,default = False,
                    help="Flag for printing updated flows in model.")
parser.add_argument("-pf", "--plot_flows",nargs='?',type=bool,default = False,
                    help="Flag for plotting resulting flows in model.")

args = parser.parse_args()
# Print arguments
print(json.dumps(vars(args), indent = 2))

# Define dataset directory
dataset = args.dataset_name
# Define maximum number of iterations of model during learning
max_iterations = args.max_iters
# Set flag for print statements
show_params = args.show_params
show_flows = args.show_flows
plot_flows = args.plot_flows

# Instantiate DoublyConstrainedModel
dc = DoublyConstrainedModel(dataset,beta=args.beta,A_factor=args.A_factor,B_factor=args.B_factor)


''' Infer flows from model '''
# inferred_flows = dc.flow_inference_dsf_procedure(max_iterations,show_params,show_flows)
inferred_flows,_,_ = dc.flow_inference_newton_raphson(max_iterations,show_params)

# Cast flows to integers
inferred_flows = inferred_flows.astype(int)

# Save array to file
np.savetxt(os.path.join(rd,'data/output/{}/newton_raphson_flows.txt'.format(dataset)),inferred_flows)

print("\n")
print('Inferred origin supply:',np.sum(inferred_flows,axis=1))
print('Inferred destination demand:',np.sum(inferred_flows,axis=0))
print('Total flow:',np.sum(inferred_flows))
print("\n")


''' Plot heatmap of flows '''
print('Plotting flow heatmap...')
# Import borough names
boroughs = np.loadtxt(os.path.join(rd,'data/input/{}/origins-destinations.txt'.format(dataset)),dtype=str)
# Change font scaling
sns.set(font_scale=0.8)
# Set plot size
plt.figure(figsize=(20,20))
# Add heatmap
flow_heatmap = sns.heatmap(inferred_flows,
                            annot=True,
                            cmap="coolwarm",
                            fmt="d",
                            xticklabels=boroughs,
                            yticklabels=boroughs,
                            linewidths=.05)
# Add x,y axes labels
plt.xlabel("Destination Borough")
plt.ylabel("Origin Borough")
# Add title
plt.title('Origin demand flows of {} data'.format(dataset), fontsize=20)

# Save figure to output
plt.savefig(os.path.join(rd,'data/output/{}/figures/newton_raphson_flows.png'.format(dataset)))

# Plot figure if requested
if plot_flows:
    plt.show()

# Save parameters to file
with open(os.path.join(rd,'data/output/{}/figures/newton_raphson_flows_parameters.json'.format(dataset)), 'w') as outfile:
    json.dump(vars(args), outfile)

print('Figure saved to {}'.format(os.path.join(rd,'data/output/{}/figures/newton_raphson_flows.png'.format(dataset))))
