"""
MCMC scheme for low-noise regime.
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
parser = argparse.ArgumentParser(description='Low noise MCMC scheme to sequentially update  parameters and latent posteriors of the spatial interaction model.')
parser.add_argument("-data", "--dataset_name",nargs='?',type=str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'synthetic',
                    help="Name of dataset (this is the directory name in data/input)")
parser.add_argument("-cm", "--cost_matrix_type",nargs='?',type=str,choices=['','sn'],default='',
                    help="Type of cost matrix used.\
                        '': Euclidean distance based. \
                        'sn': Transportation network cost based on A and B roads only. ")
parser.add_argument("-c", "--constrained",nargs='?',type=str,choices=['singly','doubly'],default='singly',
                    help="Type of potential function to evaluate (corresponding to the singly or doubly constrained spatial interaction model). ")
parser.add_argument("-l", "--load_experiment",action='store_true',
                    help="Flag for loading X,Theta, sign samples from previous calls of this script. If False, it loads the initial instead of the updated samples.")
parser.add_argument("-d", "--delta",nargs='?',type=float,default = 0.26666666666666666,
                    help="Delta parameter in potential function.")
parser.add_argument("-n", "--mcmc_n",nargs='?',type=int,default = 20000,
                    help="Number of MCMC iterations.")
parser.add_argument("-s", "--mcmc_start",nargs='?',type=int,default = 1,
                    help="MCMC iteration prior to which all iterations are diregarded.")
parser.add_argument("-t", "--theta_step",nargs='?',type=float,default = 1.,
                    help="Step size used in Random Walk transition in the theta proposal.")
parser.add_argument("-e", "--epsilon",nargs='?',type=float,default = 0.02,
                    help="Leapfrog step size in HMC latent posterior update. This is NOT the potential value's epsilon parameter, which assumed to be 1.")
parser.add_argument("-L", "--L",nargs='?',type=int,default = 50,
                    help="Number of leapfrog steps to be taken in HMC latent posterior update.")
parser.add_argument('-hide', '--hide', action='store_true',
                    help="If true hide print statement of parameters used to run this script.")
parser.add_argument('-p', '--print', action='store_true',
                    help="If true allow print statements in this script.")
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
si = SpatialInteraction(dataset)

# Normalise data
si.normalise_data()

# Fix random seed
np.random.seed(None)

# Set theta for low-noise model's potential value parameters
theta = [0 for i in range(6)]
# Alpha and beta will be initialised properly later (see below)
theta[0] = -1
theta[1] = -1
theta[2] = args.delta
# Set gamma for low-noise model
theta[3] = 10000
theta[4] = 1 + args.delta*si.M
theta[5] = 1 # this is the potential values epsilon parameter which is assumed to be 1.
# Convert to np array
theta = np.array(theta)


# To compute 1/z(theta) using Saddle point approximation
def z_inverse(params):
    minimum = si.normalised_initial_destination_sizes
    minimum_potential = np.infty
    for k in range(si.M):
        delta = params[2]
        g = np.log(delta)*np.ones(si.M)
        g[k] = np.log(1.+delta)
        f = minimize(si.potential_value,g, method='L-BFGS-B', args=(params), jac=True, options={'disp': False})
        if(f.fun < minimum_potential):
            minimum_potential = f.fun
            minimum = f.x

    # Get Hessian matrix
    A = si.potential_hessian(minimum,params)
    # Find its cholesky decomposition Hessian = L*L^T for efficient computation
    L = np.linalg.cholesky(A)
    # Compute the log determinant of the hessian
    # det(Hessian) = det(L)*det(L^T) = det(L)^2
    # det(L) = \prod_{j=1}^M L_{jj} and
    # \log(det(L)) = \sum_{j=1}^M \log(L_{jj})
    # So \log(det(Hessian)^(1/2)) = \log(det(L))
    half_log_det_A = np.sum(np.log(np.diag(L)))

    # Compute log_normalising constant, i.e. \log(z(\theta))
    # -gamma*V(x_{minimum}) + (M/2) * \log(2\pi \gamma^{-1})
    # lap =  -si.potential_value(minimum,theta)[0] + lap_c1 - half_log_det_A
    # Compute log-posterior
    # \log(p(x|\theta)) = -gamma*V(x) - \log(z(\theta))
    # log_likelihood_values[i, j] = -lap - si.potential_value(xd,theta)[0]

    # Return array
    ret = np.empty(2)
    # Log z(\theta)
    ret[0] = si.potential_value(minimum,params)[0] +  half_log_det_A
    # Sign of log inverse of z(\theta)
    ret[1] = 1.
    return ret


# MCMC tuning parameters
# Randomwalk covariance
Ap = np.array([[ 0.00749674,  0.00182529], [ 0.00182529,  0.00709968]])
# Ap = np.array([[ 0.00374837,  0.000912645], [ 0.000912645,  0.00374837]])
# Number of leapfrog steps
L = args.L
# Leapfrog step size
epsilon = args.epsilon
# MCMC theta update step size
theta_step = args.theta_step
# Set-up MCMC
mcmc_start = args.mcmc_start
mcmc_n = args.mcmc_n

# Theta-values
samples = np.empty((mcmc_n, 2))
# X-values
samples2 = np.empty((mcmc_n, si.M))
# Sign-values
samples3 = np.empty(mcmc_n)

# Decide whether to start new experiment or load old one
if args.load_experiment:

    theta_path = os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_theta_samples.txt")
    logsize_path = os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_logsize_samples.txt")
    sign_path = os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_sign_samples.txt")

    if not os.path.exists(theta_path):
        raise Exception('No experiment has been run before. File does not exist in {}')
    if not os.path.exists(logsize_path):
        raise Exception('No experiment has been run before. File does not exist in {}')
    if not os.path.exists(sign_path):
        raise Exception('No experiment has been run before. File does not exist in {}')

    # Load updated sample initialisations
    samples_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_theta_samples.txt"))
    samples2_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_logsize_samples.txt"))
    samples3_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_sign_samples.txt"))
else:
    # Load initial sample initialisations
    samples_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_theta_samples_initial.txt"))
    samples2_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_logsize_samples_initial.txt"))
    samples3_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_sign_samples_initial.txt"))

# Load sample initialisations
samples[:mcmc_start+1] = samples_init[:mcmc_start+1]
samples2[:mcmc_start+1] = samples2_init[:mcmc_start+1]
samples3[:mcmc_start+1] = samples3_init[:mcmc_start+1]


# Initialize MCMC
print("Starting at " + str(mcmc_start))
# Theta initial sample
tt = samples[mcmc_start]
print('Initial theta = ',tt)
# Log size initial sample
xx = samples2[mcmc_start]
# Initial theta
theta[0] = tt[0]
theta[1] = tt[1]*0.7e6
# Compute initial log inverse z(\theta)
log_z_inverse, ss = z_inverse(theta)
# Evaluate log potential function for initial choice of \theta
V, gradV = si.potential_value(xx,theta)

# Counts to keep track of acceptance rates for Theta and X
# Theta acceptance rates
ac = 0
pc = 0
# X acceptance rates
ac2 = 0
pc2 = 0


# MCMC algorithm
for i in tqdm(range(mcmc_start, mcmc_n)):

    # print("\nIteration:" + str(i))
    ''' Theta update '''

    # Theta-proposal (random walk with reflecting boundaries)
    tt_p = tt + theta_step*np.dot(Ap, np.random.normal(0, 1, 2))
    for j in range(2):
        if tt_p[j] < 0.:
            tt_p[j] = -tt_p[j]
        elif tt_p[j] > 2.:
            tt_p[j] = 2. - (tt_p[j] - 2.)

    # Theta-accept/reject
    if tt_p.min() > 0 and tt_p.max() <= 2:
        try:
            # Update theta proposal
            theta[0] = tt_p[0]
            theta[1] = tt_p[1]*0.7e6
            # Compute inverse of z(theta)
            log_z_inverse_p, ss_p = z_inverse(theta)

            # Evaluate log potential function for theta proposal
            V_p, gradV_p = si.potential_value(xx,theta)
            # Compute log parameter posterior for choice of X and updated theta proposal
            pp_p = log_z_inverse_p - V_p
            # Compute log parameter posterior for choice of X and initial theta proposal
            pp = log_z_inverse - V

            if args.print:
                print("Proposing " + str(tt_p) + " with " + str(ss_p))
                print(str(pp_p) + " vs " + str(pp))

            pc += 1
            if np.log(np.random.uniform(0, 1)) < pp_p - pp:
                if args.print:
                    print("Theta-Accept")
                # Update initial theta
                tt = tt_p
                # Update log potential function for choice of initial theta
                V, gradV = V_p, gradV_p
                # Increment acceptance rate of theta's
                ac += 1
                # Update log inverse of z(theta) for choice of initial theta
                log_z_inverse, ss = log_z_inverse_p, ss_p

            else:
                if args.print:
                    print("Theta-Reject")
        except Exception:
            raise Exception('ERROR FOUND')


    ''' Log destination size (X) update '''

    # Reset theta for HMC
    theta[0] = tt[0]
    theta[1] = tt[1]*0.7e6


    # Initialize leapfrog integrator for HMC proposal
    p = np.random.normal(0., 1., si.M)
    # Compute log(\pi(y|x))
    VL, gradVL = si.likelihood_value(xx)
    # Compute log initial potential energy and its derivarive weighted by the likelihood function \pi(y|x)
    # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x))
    W, gradW = V + VL, gradV + gradVL
    # Initial total log Hamiltonian energy (kinetic + potential)
    H = 0.5*np.dot(p, p) + W

    # X-Proposal
    x_p = xx
    # Momentum initialisation
    p_p = p
    # Initial log potential energy and its gradient weighted by the likelihood function \pi(y|x)
    W_p, gradW_p = W, gradW
    # Leapfrog integrator
    for j in range(L):
        # Make a half step for momentum in the beginning
        # inverse_temps[j]*gradV_p = grad V(x|theta)*(1/T)
        p_p = p_p -0.5*epsilon*gradW_p

        # Make a full step for the position
        x_p = x_p + epsilon*p_p

        # Update log potential energy and its gradient
        # Compute updated log(\pi(y|x))
        VL_p, gradVL_p = si.likelihood_value(x_p)
        # Compute updated log potential function
        V_p, gradV_p = si.potential_value(x_p,theta)
        # Compute log updated potential energy and its derivarive weighted by the likelihood function \pi(y|x)
        # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x))
        W_p, gradW_p = V_p + VL_p, gradV_p + gradVL_p
        # Make a full step for the momentum except at the end of trajectory
        p_p = p_p - 0.5*epsilon*gradW_p

    # X-accept/reject
    # Increment proposal count
    pc2 += 1
    # Compute proposal log Hamiltonian energy
    H_p = 0.5*np.dot(p_p, p_p) + W_p

    # if args.print:
    #     print("Proposing " + str(x_p))
    #     print(str(H) + " vs " + str(H_p))

    # Accept/reject
    if np.log(np.random.uniform(0, 1)) < H - H_p:
        # Update initial latent variable
        xx = x_p
        # Update initial potential function and its gradient
        V, gradV = V_p, gradV_p
        # Increment acceptance count for X update
        ac2 += 1
        if args.print:
            print("X-Accept")
    else:
        if args.print:
            print("X-Reject")

    # Update stored Markov-chain
    samples[i] = tt
    samples2[i] = xx
    samples3[i] = ss

    # Savedown and output details every 100 iterations
    if int(0.05*args.mcmc_n) > 0:
        if (i+1) % (int(0.05*args.mcmc_n)) == 0:
            print("Saving")
            np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_theta_samples.txt"), samples)
            np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_logsize_samples.txt"), samples2)
            np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_sign_samples.txt"), samples3)
            print("Theta AR " + str(float(ac)/float(pc)))
            print("X AR " + str(float(ac2)/float(pc2)))
            # print(f"Net positives {str(int(np.sum(samples3[:(i+1)])))} out of {str((i+1))} = {str( int( 100*np.sum(samples3[:(i+1)])/(i+1) ) )}%")
            if dataset == 'synthetic' and float(ac)/float(pc) <= 0.7 and float(ac)/float(pc) >= 0.4:
                print('Last accepted theta = ',tt)
                theta_step *= (float(0.1*max(0.01,abs(float(ac)/float(pc)-0.55))))
                print('New theta step = ',theta_step)
            elif dataset == 'synthetic' and ((float(ac)/float(pc) > 0.7) or (float(ac)/float(pc) < 0.4)):
                # print('Last accepted theta = ',tt)
                theta_step /= (float(0.1*max(0.01,abs(float(ac)/float(pc)-0.55))))
                print('New theta step = ',theta_step)

if int(0.05*args.mcmc_n) == 0:
    print("Saving")
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_theta_samples.txt"), samples)
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_logsize_samples.txt"), samples2)
    np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_sign_samples.txt"), samples3)
    print("Theta AR " + str(float(ac)/float(pc)))
    print("X AR " + str(float(ac2)/float(pc2)))
    # print(f"Net positives {str(np.sum(samples3))} out of {str(args.mcmc_n)} = {str(int(100 * np.sum(samples3) / args.mcmc_n))}%")

print('Computing posterior summary statistics')

theta_mean = np.mean(samples,axis=0)
theta_sd = np.std(samples,axis=0)
x_mean = np.mean(samples2,axis=0)

print(f'Theta = {theta_mean} +/- {theta_sd}')

# Saving summary statistics to arguments
arguments['theta_mean'] = list(theta_mean)
arguments['theta_sd'] = list(theta_sd)
arguments['x_mean'] = list(x_mean)

print('Storing posterior samples')

# Save theta samples to file
np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_theta_samples.txt"),samples)
# Save x samples to file
np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_logsize_samples.txt"),samples2)
# Save sign samples to file
np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_low_noise_sign_samples.txt"),samples3)

# Save parameters to file
with open(os.path.join(wd,f'data/output/{dataset}/inverse_problem/{constrained}_low_noise_mcmc_samples_parameters.json'), 'w') as outfile:
    json.dump(arguments, outfile)

print("Done")
print("Data saved to",os.path.join(wd,f"data/output/{dataset}/inverse_problem/"))
