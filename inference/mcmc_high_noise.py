"""
MCMC scheme for high-noise regime.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.special import logsumexp
from scipy.optimize import minimize

# from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
num_cores = multiprocessing.cpu_count()

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
parser.add_argument("-s", "--mcmc_start",nargs='?',type=int,default = 10000,
                    help="MCMC iteration prior to which all iterations are diregarded.")
parser.add_argument("-rwsd", "--random_walk_standard_deviation",nargs='?',type=float,default = 0.3,
                    help="Random walk covariance used in Theta proposals.")
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

# Set theta for high-noise model's potential value parameters
theta = [0 for i in range(6)]
# Alpha and beta will be initialised properly later (see below)
theta[0] = -1
theta[1] = -1
theta[2] = args.delta
# Set gamma for high-noise model
theta[3] = 100
theta[4] = 1 + args.delta*si.M
theta[5] = 1 # this is the potential values epsilon parameter which is assumed to be 1.
# Convert to np array
theta = np.array(theta)

# Load random stopping times
stopping = np.loadtxt(os.path.join(wd,f"data/input/{dataset}/stopping_times.txt"))


# Annealed importance sampling - returns an importance sampling estimate of z(theta)
# See Neal, R. M. (1998). Annealed Importance Sampling. Statistics and Computing, 11(2), 125–139. Retrieved from http://arxiv.org/abs/physics/9803008
def ais_ln_z(i):
    ''' AIS sample i that is a biased estimate of log (1/z(theta)) '''

    # Initialize AIS
    np.random.seed(None)

    # Parameters for annealed importance sampling (AIS)
    # Number of samples (w^{(i)}'s) in AIS
    p_n = 10
    # Number of bridging distributions used to create w^{(i)}
    t_n = 50
    # HMC leapfrog steps
    L = 10
    # HMC leapfrog stepsize
    eps = 0.1

    # print('i =',i)

    # Initialise multiple temperatures in decreasing order
    temperatures = np.linspace(0, 1, t_n)
    minus_temperatures = 1. - temperatures

    # Initialise acceptance and proposal count for the importance sampling estimator
    ac = 0
    pc = 0

    # Initialise log importance sampling weights (their exponential sums to 1)
    log_weights = -np.log(p_n)*np.ones(p_n)

    # Unpack theta parameters
    delta = theta[2]
    gamma = theta[3]
    kappa = theta[4]

    # print('delta = ',delta,'gamma =',gamma,'kappa =',kappa)

    # For each particle
    for ip in range(p_n):

        # Initialize
        # Sample log latent sizes from prior, i.e. log-gamma model with alpha->1,beta->0
        xx = np.log(np.random.gamma(gamma*(delta+1./si.M), 1./(gamma*kappa), si.M))
        # Potential function for AIS is proportional to the original potential function in the limit of alpha to 1 and beta to 0
        # This is the proposal distribution for computing log(1/z(theta))
        V0, gradV0 = si.potential_value_annealed_importance_sampling(xx,theta)
        # Compute initial potential function
        V1, gradV1 = si.potential_value(xx,theta)

        ''' Anneal '''
        for it in range(1, t_n):
            # log(w^{(i)}) = \sum_{j=1}^t_n  \log(f_{j-1}(x)/f_{j-1}(x))
            # = \sum_{j=1}^t_n \log( (f_0(x))^temp_{j-1} (f_n(x))^temp_{j-1} /  (f_0(x))^temp_{j} (f_n(x))^temp_{j} )
            # = \sum_{j=1}^t_n (temp_{j-1}  - temp_{j}) \log(f_0(x)) + (1-temp_{j-1} -1+temp_{j}) \log(f_n(x))
            # = \sum_{j=1}^t_n (temp_{j-1}  - temp_{j}) ( \log(f_0(x))-\log(f_n(x) )
            # See page 4 equation 3 of Neal, R. M. (1998). Annealed Importance Sampling (the betas are the same as the temperatures)
            log_weights[ip] += (temperatures[it] - temperatures[it-1])*(V0 - V1)

            ''' Sample x_{it-1}  from x_{it} using HMC transition kernel T_{it-1}'''

            # Initialize HMC
            # Momentum
            p = np.random.normal(0., 1., si.M)
            # Annealed log potential function
            #  \log(f_j(x)) = temp_{j} \log(f_0(x)) + (1-temp_{j}) \log(f_n(x))
            V, gradV = minus_temperatures[it]*V0 + temperatures[it]*V1, minus_temperatures[it]*gradV0 + temperatures[it]*gradV1
            # Total initial Hamiltonian  energy (kinetic + potential)
            H = 0.5*np.dot(p, p) + V

            # HMC leapfrog integrator
            # Latent size proposal
            x_p = xx
            # Momentum proposal
            p_p = p
            # Potential and its gradient proposal
            V_p, gradV_p = V, gradV
            for j in range(L):
                # Make a half step for momentum in the beginning
                # inverse_temps[j]*gradV_p = grad V(x|theta)*(1/T)
                p_p = p_p - 0.5*eps*gradV_p
                # Make a full step for the position
                x_p = x_p + eps*p_p
                # Update log annealed potential that is proportional to the potential energy and its gradient
                V0_p, gradV0_p = si.potential_value_annealed_importance_sampling(x_p,theta)
                # Update potential and its gradient
                V1_p, gradV1_p = si.potential_value(x_p,theta)
                # Update log annealed potential energy and its gradient
                V_p, gradV_p = minus_temperatures[it]*V0_p + temperatures[it]*V1_p, minus_temperatures[it]*gradV0_p + temperatures[it]*gradV1_p
                # Make a full step for the momentum except at the end of trajectory
                p_p = p_p - 0.5*eps*gradV_p

            # HMC accept/reject
            # Increment proposal count for HMC
            pc += 1
            # Compute Hamiltonian for log annealed potential energy proposal
            H_p = 0.5*np.dot(p_p, p_p) + V_p
            # Accept reject x_{it-1}
            if np.log(np.random.uniform(0, 1)) < H - H_p:
                # Update latent size proposal
                xx = x_p
                # Update proposal potential for normalising constant log(1/z(theta))
                V0, gradV0 = V0_p, gradV0_p
                # Update actual potential for accepte latent size
                V1, gradV1 = V1_p, gradV1_p
                # Increment acceptance count for HMC
                ac += 1

    # Return \log(\sum_{i=1}^{p_n} w^{(i)})
    return logsumexp(log_weights)


# Debiasing scheme - returns unbiased esimates of 1/z(theta)
def unbiased_z_inv(cc):

    # Stopping time
    K = int(stopping[cc])
    # Pr(T\geq k) \propto k^(-1.1)
    k_pow = 1.1

    if args.print:
        print("Debiasing with K = " + str(K))

    # log_weights = np.empty(K+1)
    # for i in range(1,K+1):
        # log_weights[i] = ais_ln_z(i)

    # Initialise naive paralellisation
    pool = Pool(processes=num_cores)

    # Get importance sampling estimate of z(theta) in parallel
    log_weights = pool.map(ais_ln_z, range(K+1))
    # log_weights = Parallel(n_jobs=num_cores)(delayed(ais_ln_z)(i) for i in range(K+1))

    # Close pool
    pool.close()

    # Compute S = Y[0] + \sum_i (Y[i] - Y[i-1])/P(K > i) using logarithms
    # See equation (C4) of page 18
    ln_Y = np.empty(K+1)
    ln_Y_pos = np.empty(K+1)
    ln_Y_neg = np.empty(K)

    # For the choice of stoppping time
    for i in range(0, K+1):
        # Compute increasing average estimator
        # log (\nu_i) = \log(i+1) - \log(\sum_{j=0}^{i} w^{(j)}))
        ln_Y[i] = np.log(i+1) - logsumexp(log_weights[:i+1])

    # Store \nu_0
    ln_Y_pos[0] = ln_Y[0]
    # For the choice of stoppping time
    for i in range(1, K+1):
        # Note that Pr(T\geq i) \propto \log(i^{-1.1})
        # Compute \log(\nu_i) - \log(i^{-1.1}) = \log(\nu_i) + \log(i^{1.1})
        # Note that for i = 0 ln_Y_pos[0] = \nu_0
        ln_Y_pos[i] = ln_Y[i] + k_pow*np.log(i)
        # Compute \log(\nu_{i-1}) - \log(i^{-1.1}) = \log(\nu_i) + \log(i^{1.1})
        ln_Y_neg[i-1] = ln_Y[i-1] + k_pow*np.log(i)

    # Compute \log (\sum_{i=1}^K \nu_i / Pr(T\geq i))
    positive_sum = logsumexp(ln_Y_pos)
    # Compute \log (- \sum_{i=1}^K \nu_{i-1} / Pr(T\geq i))
    negative_sum = logsumexp(ln_Y_neg)

    ret = np.empty(2)
    # Compute \log( \nu_0 \sum_{i=1}^K \nu_i / Pr(T\geq i) - \nu_{i-1} / Pr(T\geq i) )
    # and store its sign
    if(positive_sum >= negative_sum):
        ret[0] = positive_sum + np.log(1. - np.exp(negative_sum - positive_sum))
        ret[1] = 1.
    else:
        ret[0] = negative_sum + np.log(1. - np.exp(positive_sum - negative_sum))
        ret[1] = -1.

    # Return \log(E[S]) = \log(1/z(theta)) and its sign
    return ret


# MCMC tuning parameters
# Randomwalk covariance
random_walk_sd = args.random_walk_standard_deviation
# Number of leapfrog steps
L2 = args.L
# Leapfrog step size
eps2 = args.epsilon


# Set-up MCMC
mcmc_start = args.mcmc_start
mcmc_n = args.mcmc_n

# Theta samples
samples = np.empty((mcmc_n, 2))
# Log size (X) samples
samples2 = np.empty((mcmc_n, si.M))
# Sign samples
samples3 = np.empty(mcmc_n)

# Decide whether to start new experiment or load old one
if args.load_experiment:

    theta_path = os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_theta_samples.txt")
    logsize_path = os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_logsize_samples.txt")
    sign_path = os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_sign_samples.txt")

    if not os.path.exists(theta_path):
        raise Exception('No experiment has been run before. File does not exist in {}')
    if not os.path.exists(logsize_path):
        raise Exception('No experiment has been run before. File does not exist in {}')
    if not os.path.exists(sign_path):
        raise Exception('No experiment has been run before. File does not exist in {}')

    # Load updated sample initialisations
    samples_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_theta_samples.txt"))
    samples2_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_logsize_samples.txt"))
    samples3_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_sign_samples.txt"))
else:
    # Load initial sample initialisations
    samples_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_theta_samples_initial.txt"))
    samples2_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_logsize_samples_initial.txt"))
    samples3_init = np.loadtxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_sign_samples_initial.txt"))

# Start MCMC chain from specific value
samples[:mcmc_start+1] = samples_init[:mcmc_start+1]
samples2[:mcmc_start+1] = samples2_init[:mcmc_start+1]
samples3[:mcmc_start+1] = samples3_init[:mcmc_start+1]


# Initialize MCMC
print("Starting at " + str(mcmc_start))
print("Warning max random stopping is " + str(int(stopping[mcmc_start:mcmc_n].max())))
# Initialise theta samples
tt = samples[mcmc_start]
# Initialise logsize (X) samples
xx = samples2[mcmc_start]

if args.print:
    print('initial theta', tt)
    print('initial x', xx)

# Store initial arguments to json
arguments['initial_theta'] = list(tt)
arguments['initial_x'] = list(xx)

# Initialise alpha and beta parameters
theta[0] = tt[0]
theta[1] = tt[1]*0.7e6
# Compute initial unbiased estimate of 1/z(theta)
lnzinv, ss = unbiased_z_inv(mcmc_start-1)

# Compute initial potential function
V, gradV = si.potential_value(xx,theta)


# Counts to keep track of accept rates
# Theta proposal
ac = 0
pc = 0
# Logsize X proposal
ac2 = 0
pc2 = 0


# MCMC algorithm
for i in tqdm(range(mcmc_start, mcmc_n)):

    if args.print:
        print("\nIteration:" + str(i))

    # Theta-proposal (random walk with reflecting boundaries)
    tt_p = tt + np.random.normal(0, random_walk_sd, 2)
    # Reflect boundaries if necessary
    for j in range(2):
        if tt_p[j] < 0.:
            tt_p[j] = -tt_p[j]
        elif tt_p[j] > 2.:
            tt_p[j] = 2. - (tt_p[j] - 2.)

    # Theta-accept/reject
    if tt_p.min() > 0 and tt_p.max() <= 2:
        # Update alpha and beta to new theta proposal
        theta[0] = tt_p[0]
        theta[1] = tt_p[1]*0.7e6
        # Compute unbiased estimate of 1/z(theta) for theta proposal
        lnzinv_p, ss_p = unbiased_z_inv(i)
        # Compute potential value for theta proposal
        V_p, gradV_p = si.potential_value(xx,theta)
        # Compute log likelihood log(\pi(x|\theta)) for theta proposal
        pp_p = lnzinv_p - V_p
        # Compute log likelihood log(\pi(x|\theta)) for initial theta
        pp = lnzinv - V

        if args.print:
            print("Proposing " + str(tt_p) + " with " + str(ss_p))
            print(str(pp_p) + " vs " + str(pp))

        # Increment proposal count
        pc += 1
        # Accept/reject theta
        if np.log(np.random.uniform(0, 1)) < pp_p - pp:
            if args.print:
                print("Theta-Accept")
            # If accept, update theta initial
            tt = tt_p
            # Update potential value and its gradient for initial theta
            V, gradV = V_p, gradV_p
            # Increment acceptance count
            ac += 1
            # Update log(1/z(theta)) and its sign for initial theta
            lnzinv, ss = lnzinv_p, ss_p

        else:
            if args.print:
                print("Theta-Reject")


    # Reset theta for HMC
    theta[0] = tt[0]
    theta[1] = tt[1]*0.7e6


    # Initialize leapfrog integrator for HMC proposal
    # Initial momentum
    p = np.random.normal(0., 1., si.M)
    # Initial log(\pi(y|x))
    VL, gradVL = si.likelihood_value(xx)
    # Compute log initial potential energy and its derivarive weighted by the likelihood function \pi(y|x)
    # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x))
    W, gradW = V + VL, gradV + gradVL
    # Initial Hamiltonian energy (kinetic + potential)
    H = 0.5*np.dot(p, p) + W


    # X-Proposal
    # Initial log size
    x_p = xx
    # Initial momentum
    p_p = p
    # Compute log initial potential energy and its derivarive weighted by the likelihood function \pi(y|x)
    # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x))
    W_p, gradW_p = W, gradW

    #  log initial potential energy and its derivarive weighted by the likelihood function \pi(y|x)
    # Leapfrog integrator
    for j in range(L2):
        # Make a half step for momentum in the beginning
        # inverse_temps[j]*gradV_p = grad V(x|theta)*(1/T)
        p_p = p_p -0.5*eps2*gradW_p
        # Make a full step for the position
        x_p = x_p + eps2*p_p
        # Update log potential energy and its gradient
        # Compute updated log(\pi(y|x))
        VL_p, gradVL_p = si.likelihood_value(x_p)
        # Compute updated log potential function
        V_p, gradV_p = si.potential_value(x_p,theta)
        # Compute log updated potential energy and its derivarive weighted by the likelihood function \pi(y|x)
        # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x))
        W_p, gradW_p = V_p + VL_p, gradV_p + gradVL_p
        # Make a full step for the momentum except at the end of trajectory
        p_p = p_p - 0.5*eps2*gradW_p


    # X-accept/reject
    # Increment proposal count for HMC
    pc2 += 1
    # Updated Hamiltonian energy (kinetic + potential)
    H_p = 0.5*np.dot(p_p, p_p) + W_p
    if np.log(np.random.uniform(0, 1)) < H - H_p:
        # Update initial latent variable
        xx = x_p
        # Update initial potential function and its gradient
        V, gradV = V_p, gradV_p
        # Increment acceptance count
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

    # Savedown and output details every 10 iterations
    # Savedown and output details every 100 iterations
    if (i+1) % 10 == 0:
        print("Saving")
        np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_theta_samples.txt"), samples)
        np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_logsize_samples.txt"), samples2)
        np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_sign_samples.txt"), samples3)

        arguments['last_iteration'] = i

        # Save parameters to file
        with open(os.path.join(wd,f'data/output/{dataset}/inverse_problem/{constrained}_high_noise_mcmc_samples_parameters.json'), 'w') as outfile:
            json.dump(arguments, outfile)

        print('Iteration: ',str(int(i+1)))
        print("Theta AR: " + str(float(ac)/float(pc)))
        print("X AR: " + str(float(ac2)/float(pc2)))
        print(f"Net positives: {str( int( 100*np.sum(samples3[:(i+1)])/(i+1) ) )}%")
        # print(f"Net positives {str(int(np.sum(samples3[:(i+1)])))} out of {str((i+1))} = {str( int( 100*np.sum(samples3[:(i+1)])/(i+1) ) )}%")

print('Computing posterior summary statistics')

theta_mean = np.mean(samples,axis=0)
theta_sd = np.std(samples,axis=0)
x_mean = np.mean(samples2,axis=0)

print(f'Theta = {theta_mean} +/- {theta_sd}')

# Saving summary statistics to arguments
arguments['theta_mean'] = list(theta_mean)
arguments['theta_sd'] = list(theta_sd)
arguments['x_mean'] = list(x_mean)
arguments['last_iteration'] = i

print('Storing posterior samples')

# Save theta samples to file
np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_theta_samples.txt"),samples)
# Save x samples to file
np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_logsize_samples.txt"),samples2)
# Save sign samples to file
np.savetxt(os.path.join(wd,f"data/output/{dataset}/inverse_problem/{constrained}_high_noise_sign_samples.txt"),samples3)

# Store fixed parameters to arguments
arguments['p_n'] = p_n
arguments['t_n'] = t_n
arguments['L'] = L
arguments['eps'] = eps

# Save parameters to file
with open(os.path.join(wd,f'data/output/{dataset}/inverse_problem/{constrained}_high_noise_mcmc_samples_parameters.json'), 'w') as outfile:
    json.dump(arguments, outfile)

print("Done")
print("Data saved to",os.path.join(wd,f"data/output/{dataset}/inverse_problem/"))
