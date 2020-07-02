# As defined in C function
def _logsumexp(xx,m):
    _max = xx[0]
    for j in range(m):
        if xx[j] > _max:
            _max = xx[j]

    _sum = 0;
    for j in range(m):
        _sum += np.exp(xx[j] - _max);

    return _max + np.log(_sum)

def pot_value(x,params):
    pot = 1
    a = params[0]
    b = params[1]
    d = params[2]
    g = params[3]
    k = params[4]
    e = params[5]

    a_inv = 1./a

    grad = np.zeros(M)
    wksp = np.zeros(M)

    utility = 0
    for i in range(N):
        for j in range(M):
            wksp[j] = a*x[j] - b*c[i,j]
        temp = _logsumexp(wksp,M)

        utility += -e*a_inv*o[i]*temp;

        for j in range(M):
            grad[j] += -e*o[i]*np.exp(wksp[j]-temp)

    for j in range(M):
        grad[j] += k*np.exp(x[j]) - d
        grad[j] *= g


    # Compute cost and additional utilities
    cost = k * np.sum(np.exp(x))
    additional = -d * np.sum(x)

    potential = g*e*(utility+cost+additional)

    return potential,grad

# # Fix random seed
# np.random.seed(888)

# # Set theta for high-noise model's potential value parameters
# alpha = 0.5
# beta = 4.0
# delta = 0.0
# gamma = 10000
# kappa = 1 + delta*M

# # MCMC tuning parameters
# # Number of leapfrog steps
# L = 100
# # Leapfrog step size
# epsilon = 0.1
# # Number of iterations
# mcmc_n = 10000
#
#
# # # Convert to np array
# theta = np.array([alpha,beta,delta,gamma,kappa,1])

# # Inverse temperatures that go into potential energy of Hamiltonian dynamics
# inverse_temperatures = np.array([1., 1./2., 1./4., 1./8., 1./16.])
# temp_n = len(inverse_temperatures)

# # Array to store X values sampled at each iteration
# samples = np.empty((mcmc_n, M))

# # Initialize MCMC
# xx = -np.log(M)*np.ones((temp_n, M))

# # Initiliase arrays for potential value and its gradient
# V = np.empty(temp_n)
# gradV = np.empty((temp_n, M))
# positions = np.empty((mcmc_n+1, M))
# positions[0,:] = xx[0]

# # Get potential value and its gradient for the initial choice of theta and x
# for j in range(temp_n):
#     V[j], gradV[j] = pot_value(xx[j],theta)

# # Counts to keep track of accept rates
# ac = np.zeros(temp_n)
# pc = np.zeros(temp_n)
# acs = 0
# pcs = 1


# # Iterator for number of mcmc runs
# n = 0

# # MCMC algorithm
# for i in tqdm(range(mcmc_n)):
#     for j in range(temp_n):
#         # Initialise leapfrog integrator for HMC proposal

#         ''' HMC parameter/function correspondence to spatial interaction model
#         q = x : position or log destination sizes
#         U(x) = gamma*V(x|theta)*(1/T) :  potential energy or potential value given parameter theta times inverse temperature

#         Note that the potential value function returns gamma*V(x|theta)
#         '''

#         # X-Proposal (position q is the log size vector X)
#         x_p = xx[j]

#         # Initialise momentum
#         p = np.random.normal(0., 1., M)
#         # Set current momentum
#         p_p = p
#         # Get potential value and its Jacobian
#         V_p, gradV_p = V[j], gradV[j]
#         # Make a half step for momentum in the beginning
#         # inverse_temps[j]*gradV_p = grad V(x|theta)*(1/T)
#         p_p -= 0.5*epsilon*inverse_temperatures[j]*gradV_p

#         # Hamiltonian total energy function = kinetic energy + potential energy
#         # at the beginning of trajectory
#         # Kinetic energy K(p) = p^TM^{−1}p / 2 with M being the identity matrix
#         H = 0.5*np.dot(p, p) + inverse_temperatures[j]*V[j]

#         # Alternate full steps for position and momentum
#         for l in range(L):
#             # Make a full step for the position
#             x_p += epsilon*p_p
#             # Update potential value and its gradient
#             V_p, gradV_p = pot_value(x_p,theta)
#             # Make a full step for the momentum except at the end of trajectory
#             if (l != (L-1)):
#                 p_p -= 0.5*epsilon*inverse_temperatures[j]*gradV_p

#         # Make a falf step for momentum at the end.
#         p_p -= 0.5*epsilon*inverse_temperatures[j]*gradV_p

#         # Negate momentum
#         p_p *= (-1)

#         # Store positions
# #         n += 1
# #         positions[n,:] = x_p

#         # Increment proposal count
#         pc[j] += 1

#         # Compute Hamiltonian total energy function at the end of trajectory
#         H_p = 0.5*np.dot(p_p, p_p) + inverse_temperatures[j]*V_p

#         # Accept/reject X by either returning the position at the end of the trajectory or the initial position
#         if np.log(np.random.uniform(0, 1)) < H - H_p:
#             xx[j] = x_p
#             V[j], gradV[j] = V_p, gradV_p
#             ac[j] += 1

#     # Perform a swap
#     pcs += 1
#     j0 = np.random.randint(0, temp_n-1)
#     j1 = j0+1
#     logA = (inverse_temperatures[j1]-inverse_temperatures[j0])*(-V[j1] + V[j0])
#     if np.log(np.random.uniform(0, 1)) < logA:
#         xx[[j0, j1]] = xx[[j1, j0]]
#         V[[j0, j1]] = V[[j1, j0]]
#         gradV[[j0, j1]] = gradV[[j1, j0]]
#         acs += 1

#     # Update stored Markov-chain
#     samples[i] = xx[0]

#     # Savedown and output details every 100 iterations
#     if (i+1) % (int(0.05*mcmc_n)) == 0:
#         print("Iteration " + str(i+1))
#         print("X AR:")
#         print(ac/pc)
#         print("Swap AR:" + str(float(acs)/float(pcs)))
