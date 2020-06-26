"""
R2 analysis for deterministic model defined in terms of potential function.
"""
import os
import sys
import json
import copy
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
parser = argparse.ArgumentParser(description='R^2 analysis to find fitted parameters based on potential function minima.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'commuter_borough',
                    help="Name of dataset (this is the directory name in data/input)")
# parser.add_argument("-m", "--mode",nargs='?',type=str,default = 'stochastic',
#                     help="Mode of evaluation (stochastic/determinstic)")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-amin", "--amin",nargs='?',type=float,default = 0.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-amax", "--amax",nargs='?',type=float,default =  2.0,
                    help="Minimum alpha parameter for grid search.")
parser.add_argument("-bmin", "--bmin",nargs='?',type=float,default = 0.0,
                    help="Minimum beta parameter for grid search.")
parser.add_argument("-bmax", "--bmax",nargs='?',type=float,default = 100,
# parser.add_argument("-bmax", "--bmax",nargs='?',type=float,default = 1.4e6,
                    help="Minimum beta parameter for grid search.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.3,
                    help="Delta parameter.")
parser.add_argument("-g", "--gamma",nargs='?',type=float,default = 100.,
                    help="Gamma parameter.")
# parser.add_argument("-k", "--kappa",nargs='?',type=float,default = 1.3,
#                     help="Kappa parameter = 1 + delta*M.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 1.,
                    help="Epsilon parameter.")
parser.add_argument("-s", "--show_figure",nargs='?',type=bool,default = False,
                    help="Flag for showing resulting figure.")
parser.add_argument("-n", "--grid_size",nargs='?',type=int,default = 100,
                    help="Number of points (n^2) to evaluate potential function.")
args = parser.parse_args()
# Convert arguments to dictionary
arguments = vars(args)
# Print arguments
print(json.dumps(arguments, indent = 2))

# Define dataset directory
dataset = args.dataset_name

# Define mode (stochastic/determinstic) based on delta value
if args.delta == 0:
    mode = 'determinstic'
else:
    mode = 'stochastic'

# Define type of spatial interaction model
constrained = args.constrained

# Get project directory
wd = get_project_root()


# Import selected type of spatial interaction model
if constrained == 'singly':
    from models.singly_constrained.spatial_interaction_model import SpatialInteraction

    # Instantiate SpatialInteraction model
    si = SpatialInteraction(dataset)

    # Compute kappa
    kappa = 1 + args.delta*si.M


    # Initialize search grid
    grid_n = args.grid_size
    alpha_values = np.linspace(args.amin, args.amax, grid_n+1)[1:]
    beta_values = np.linspace(args.bmin, args.bmax, grid_n+1)[1:]
    XX, YY = np.meshgrid(alpha_values, beta_values)
    r2_values = np.zeros((grid_n, grid_n))
    potentials = np.zeros((grid_n, grid_n))

    # Define theta parameters
    theta = np.array([alpha_values[0], beta_values[0], args.delta, args.gamma, kappa, args.epsilon])

    # Search values
    last_r2 = -np.infty
    last_potential = -np.infty

    # Normalise initial log destination sizes
    si.normalise_data()
    xd = si.normalised_initial_destination_sizes

    # Total sum squares
    w_data = np.exp(xd)
    w_data_centred = w_data - np.mean(w_data)
    ss_tot = np.dot(w_data_centred, w_data_centred)

    # Compute destination sizes based on optimal theta
    # theta[0] = 1.36
    # theta[1] = 5
    # print('alpha =',theta[0],'beta =',theta[1],'delta =',theta[2],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])
    # opt_w_pred = np.exp(minimize(si.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False}).x)
    # opt_res = opt_w_pred - w_data
    # opt_ss_res = np.dot(opt_res, opt_res)
    #
    # print("Optimal params alpha, beta and scaled beta:")
    # print(theta[0],theta[1],theta[1]*args.amax/(args.bmax))
    # print("Optimal R^2",1. - opt_ss_res/ss_tot)
    # sys.exit()

    print('alpha =',theta[0],'beta =',theta[1],'delta =',theta[2],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])
    # Perform grid evaluations
    for i in tqdm(range(grid_n)):
        for j in range(grid_n):
            try:
                # Residiual sum squares
                theta[0] = XX[i, j]
                theta[1] = YY[i, j]
                w_pred = np.exp(minimize(si.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False}).x)
                # print('alpha =',theta[0],'beta =',theta[1],'delta =',theta[2],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])
                # print('Potential function:', si.potential_value(xd,theta)[0])
                # print('Potential function gradient:', si.potential_value(xd,theta)[1])
                res = w_pred - w_data
                ss_res = np.dot(res, res)

                # Regression sum squares
                r2_values[i, j] = 1. - ss_res/ss_tot
                # print('i =',i,'j =',j,'R^2 =',r2_values[i, j])

                potentials[i, j] = si.potential_value(np.log(w_pred),theta)[0]

            except Exception:
                None

            # If minimize fails set value to previous, otherwise update previous
            if r2_values[i, j] == 0:
                r2_values[i, j] = last_r2
                potentials[i, j] = last_potential
            else:
                last_r2 = r2_values[i, j]
                last_potential = potentials[i, j]


    # Output results
    idx = np.unravel_index(r2_values.argmax(), r2_values.shape)

    print("Fitted alpha, beta and scaled beta values:")
    print(XX[idx], YY[idx],YY[idx]*args.amax/(args.bmax))
    print("R^2 and potential value:")
    print(r2_values[idx],potentials[idx])

    # Save R^2 to file
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/r_squared/{constrained}_{mode}_rsquared_analysis_{str(int(args.gamma))}.txt"), r2_values)

    # Plot options
    plt.style.use('classic')
    fig = plt.figure(figsize=(8,8))
    fig.tight_layout(pad=0.5)

    # Plot R^2
    plt.pcolor(XX, YY*args.amax/(args.bmax), r2_values)
    # plt.pcolor(XX, YY, r2_values)
    plt.xlim([np.min(XX), np.max(XX)])
    plt.ylim([np.min(YY)*args.amax/(args.bmax), np.max(YY)*args.amax/(args.bmax)])
    # plt.ylim([np.min(YY), np.max(YY)])
    plt.colorbar()
    plt.ylabel("Parameter beta")
    plt.xlabel("Parameter alpha")

    # Save figure to file
    plt.savefig(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis_gamma_{str(int(args.gamma))}.png'))

    # Compute estimated flows
    theta[0] = XX[idx]
    theta[1] = YY[idx]
    estimated_flows = si.reconstruct_flow_matrix(si.normalised_initial_destination_sizes,theta)
    # Save estimated flows
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/r_squared/{constrained}_{mode}_rsquared_estimated_flows_{str(int(args.gamma))}.txt"), estimated_flows)

    # Save fitted values to parameters
    arguments['fitted_alpha'] = XX[idx]
    arguments['fitted_scaled_beta'] = YY[idx]*args.amax/(args.bmax)
    arguments['fitted_beta'] = YY[idx]
    arguments['kappa'] = kappa
    arguments['R^2'] = r2_values[idx]
    arguments['potential'] = potentials[idx]

    # Set negative R^2 values to 0
    positive_r2_values = copy.deepcopy(r2_values)
    positive_r2_values[positive_r2_values<0] = 0

    # Show figure if requested
    if args.show_figure:
        plt.show()

    # 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(XX, YY, positive_r2_values, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_ylabel("beta")
    ax.set_xlabel("alpha")
    ax.set_zlabel("R^2")
    ax.set_title('R^2 variation across parameter space')

    # Save figure to file
    plt.savefig(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis_gamma_{str(int(args.gamma))}_3d.png'))

    # Save parameters to file
    with open(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis_gamma_{str(int(args.gamma))}_parameters.json'), 'w') as outfile:
        json.dump(arguments, outfile)


elif constrained == 'doubly':
    from models.doubly_constrained.spatial_interaction_model import SpatialInteraction

    # Instantiate UrbanModel
    si = SpatialInteraction(dataset)

    # Compute kappa
    kappa = 1 + args.delta*si.M

    # Initialize search grid
    grid_n = args.grid_size
    alpha_values = np.linspace(args.amin, args.amax, grid_n+1)[1:]
    r2_values = np.zeros((grid_n))
    potentials = np.zeros((grid_n))

    # Search values
    last_r2 = -np.infty
    max_potential = -np.infty

    # Normalise necessary data
    si.normalise_data()
    xd = si.normalised_initial_destination_sizes

    # Define theta parameters
    theta = np.array([alpha_values[0], 0.0, args.delta, args.gamma,kappa, args.epsilon])

    # Total sum squares
    w_data = np.exp(xd)
    w_data_centred = w_data - np.mean(w_data)
    ss_tot = np.dot(w_data_centred, w_data_centred)

    # Perform grid evaluations
    for i in tqdm(range(grid_n)):
        # Update alpha parameter
        theta[0] = alpha_values[i]
        # print('alpha =',theta[0],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])
        try:
            # Compute residual sum of squares
            w_pred = np.exp(minimize(si.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False}).x)
            res = w_pred - w_data
            ss_res = np.dot(res, res)
            # Regression sum squares
            r2_values[i] = 1. - ss_res/ss_tot

        except Exception:
            None

        # If minimize fails set value to previous, otherwise update previous
        if r2_values[i] == 0:
            r2_values[i] = last_r2
            potentials[i] = si.potential_value(np.log(w_pred),theta)[0]
        else:
            last_r2 = r2_values[i]
            last_potential = potentials[i]

    # Output results
    idx = np.unravel_index(r2_values.argmax(), r2_values.shape)
    print("Fitted alpha value:")
    print(alpha_values[idx])
    print("R^2 and potential values:")
    print(r2_values[idx],max_potential)

    # Save R^2 to file
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/r_squared/{constrained}_{mode}_rsquared_analysis_gamma_{str(int(args.gamma))}.txt"), r2_values)
    # Plot options
    plt.style.use('classic')
    fig = plt.figure(figsize=(8,8))
    fig.tight_layout(pad=0.5)

    # Plot R^2
    plt.plot(alpha_values, r2_values)
    plt.xlim([np.min(alpha_values)-0.01, np.max(alpha_values)+0.2])
    plt.ylim([np.min(r2_values)-0.3, np.max(r2_values)+0.5])
    plt.ylabel("R^2")
    plt.xlabel("Parameter alpha")

    # Save figure to file
    plt.savefig(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis_gamma_{str(int(args.gamma))}.png'))

    # Compute estimated flows
    theta[0] = alpha_values[idx]
    estimated_flows = si.reconstruct_flow_matrix(si.normalised_initial_destination_sizes,theta)
    # Save estimated flows
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/r_squared/{constrained}_{mode}_rsquared_estimated_flows_{str(int(args.gamma))}.txt"), estimated_flows)

    # Save fitted values to parameters
    arguments['fitted_alpha'] = alpha_values[idx]
    arguments['kappa'] = kappa
    arguments['R^2'] = r2_values[idx]
    arguments['max_potential'] = max_potential

    # Save parameters to file
    with open(os.path.join(wd,f'data/output/{dataset}/r_squared/figures/{constrained}_{mode}_rsquared_analysis_gamma_{str(int(args.gamma))}_parameters.json'), 'w') as outfile:
        json.dump(arguments, outfile)

    # Show figure if requested
    if args.show_figure:
        plt.show()
else:
    raise ValueError("{} spatial interaction model not implemented.".format(args.constrained))
