""" Inferring flow for doubly constrained origin-destination model using various methods. """

import os
import pysal
import ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.ctypeslib import ndpointer
from scipy.optimize import minimize

class SpatialInteraction():
    """ Object including flow (O/D) matrix inference routines of doubly constrained SIM.

    Parameters
    ----------
    dataset : str
        Name of dataset used for inference.
    **params : type
        beta [float]
            Distance coefficient.
        A [float]
            Initial value for origin effects - used to initialise flows.
        B [float]
            Initial value for destination effects - used to initialise flows.

    Attributes
    ----------
    working_directory : string
        Project's root working directory.
    data_directory : string
        Project's data directory.
    load_c_functions : function
        Loads relevant C functions for inference.
    import_data : function
        Loads relevant data for inference.

    """

    def __init__(self,dataset,cost_matrix_type=''):
        '''  Constructor '''

        # Define dataset
        self.dataset = dataset
        # Define working directory
        self.working_directory = self.get_project_root()
        # Define data directory
        self.data_directory = os.path.join(self.working_directory,'data/input/{}'.format(dataset))
        # Store cost matrix file extenstion
        self.cost_matrix_file_extension = ''
        if cost_matrix_type == 'sn':
            self.cost_matrix_file_extension = '_small_network'
        # Import data
        self.import_data()
        # Load C functions
        self.load_c_functions()

    # Get current working directory and project root directory
    def get_project_root(self):
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

    def import_data(self):
        """ Stores important data for training and validation to global variables.

        Attributes
        -------
        origins [Nx1 array]
            Names of origin zones.
        destinations [Mx1 array]
            Names of destination zones.
        cost_matrix [NxM array]
            Cost of travelling from each origin to each destination.
        origin supply [Nx1 array]
            Total supply for each origin.
        destination demand [Mx1 array]
            Total demand for each destination.
        N [int]
            Number of origin zones.
        M [int]
            Number of destination zones.
        total_flow [int]
            Total flow from every origin to every destination.
        actual_flows [np.array]
            Actual flow (O/D) matrix.
        total_cost [float]
            Total cost from every origin to every destination.
        """

        # Import ordered names of origins
        origins_file = os.path.join(self.data_directory,'origins.txt')
        self.origins = np.loadtxt(origins_file,dtype=str,ndmin=1)

        # Import ordered names of destinations
        destinations_file = os.path.join(self.data_directory,'destinations.txt')
        self.destinations = np.loadtxt(destinations_file,dtype=str,ndmin=1)

        # Import origin supply
        originsupply_file = os.path.join(self.data_directory,'origin_supply.txt')
        self.origin_supply = np.loadtxt(originsupply_file,ndmin=1).astype('float64')

        # In case origin supply is not a list
        if not isinstance(self.origin_supply,(np.ndarray, np.generic)):
            self.origin_supply = np.array([self.origin_supply])

        # Import destination demand
        destinationdemand_file = os.path.join(self.data_directory,'destination_demand.txt')
        self.destination_demand = np.loadtxt(destinationdemand_file,ndmin=1).astype('float64')

        # In case destination demand is not a list
        if not isinstance(self.destination_demand,(np.ndarray, np.generic)):
            self.destination_demand = np.array([self.destination_demand])

        # Import origin locations
        originlocations_file = os.path.join(self.data_directory,'origin_locations.txt')
        self.origin_locations = np.loadtxt(originlocations_file,ndmin=1)

        # Import destination locations
        destinationlocations_file = os.path.join(self.data_directory,'destination_locations.txt')
        self.destination_locations = np.loadtxt(destinationlocations_file,ndmin=1)

        # Import initial and final destination sizes
        initialdestinationsizes_file = os.path.join(self.data_directory,'initial_destination_sizes.txt')
        self.initial_destination_sizes = np.loadtxt(initialdestinationsizes_file,ndmin=1)

        # In case destination sizes are not a list
        if not isinstance(self.initial_destination_sizes,(np.ndarray, np.generic)):
            self.initial_destination_sizes = np.array([self.initial_destination_sizes])

        # Import N,M
        self.N = self.origin_supply.shape[0]
        self.M = self.initial_destination_sizes.shape[0]

        # Import cost matrix
        costmatrix_file = os.path.join(self.data_directory,'cost_matrix.txt')
        self.cost_matrix = np.loadtxt(costmatrix_file).astype('float64')

        # Reshape cost matrix if necessary
        if self.N == 1:
            self.cost_matrix = np.reshape(self.cost_matrix[:,np.newaxis],(self.N,self.M))
        if self.M == 1:
            self.cost_matrix = np.reshape(self.cost_matrix[np.newaxis,:],(self.N,self.M))

        # Compute total initial and final destination sizes
        self.total_initial_sizes = np.sum(self.initial_destination_sizes)

        # Compute naive total cost
        self.total_cost = 0
        for i in range(self.N):
            for j in range(self.M):
                self.total_cost += self.cost_matrix[i,j]*(self.origin_supply[i]/self.N)

    def reshape_data(self):
        """ Reshapes data into dataframe for PySAL training.

        Returns
        -------
        np.array
            Flows.
        np.array
            Origin supply.
        np.array
            Destination demand.
        np.array
            Cost matrix.
        """
        # Initialise empty dataframe
        od_data = pd.DataFrame(columns=['Origin','Destination','Cost','Flow','OriginSupply','DestinationDemand'])
        # Loop over origins and destinations to populate dataframe
        for i,orig in tqdm(enumerate(self.origins),total=len(self.origins)):
            for j,dest in enumerate(self.destinations):
                # Add row properties
                new_row = pd.Series({"Origin": orig,
                                     "Destination": dest,
                                     "Cost": self.cost_matrix[i,j],
                                     "OriginSupply": self.origin_supply[i],
                                     "DestinationDemand":self.destination_demand[j]})
                # Append row to dataframe
                od_data = od_data.append(new_row, ignore_index=True)

        # Get flatten data and et column types appropriately
        orig_supply_flat = od_data.OriginSupply.values.astype('float64')
        dest_demand_flat = od_data.DestinationDemand.values.astype('float64')
        cost_flat = od_data.Cost.values.astype('float64')

        return orig_supply_flat,dest_demand_flat,cost_flat

    def normalise(self,data,take_logs:bool=False):
        """ Normalises data for use in inverse problem.

        Parameters
        ----------
        data : np.array
            Any data e.g. destination demand
        take_logs : bool
            Flag for taking logs after normalising the vector

        Returns
        -------
        np.array
            Normalised vector.

        """

        # Normalise vector to sum up to 1
        normalised_vector = data/np.sum(data)

        # If take logs is selected, take logs
        if take_logs:
            return np.log(normalised_vector)
        else:
            return normalised_vector

    def normalise_data(self):

        # Normalise destination demand
        self.normalised_destination_demand = self.normalise(self.destination_demand,False)

        # Normalise origin supply
        self.normalised_origin_supply = self.normalise(self.origin_supply,False)

        # Normalise cost matrix
        self.normalised_cost_matrix = self.normalise(self.cost_matrix,False)

        # Normalise initial destination sizes
        self.normalised_initial_destination_sizes = self.normalise(self.initial_destination_sizes,True)


    def load_c_functions(self):
        """ Stores C functions that infer flows to global variables.

        Attributes
        -------
        infer_flows_dsf_procedure [function]
            Function used to infer flows using the DSF procedure.
        infer_flows_newton_raphson [function]
            Function used to infer flows using the Newton Raphson method.
        """

        # Load shared object
        lib = ctypes.cdll.LoadLibrary(os.path.join(self.working_directory,"models/doubly_constrained/flow_forward_models.so"))
        lib2 = ctypes.cdll.LoadLibrary(os.path.join(self.working_directory,"models/doubly_constrained/potential_function.so"))

        # Load DSF procedure flow inference
        self.infer_flows_dsf_procedure = lib.infer_flows_dsf_procedure
        self.infer_flows_dsf_procedure.restype = ctypes.c_double
        self.infer_flows_dsf_procedure.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ctypes.c_double,
                                                    ctypes.c_size_t,
                                                    ctypes.c_bool,
                                                    ctypes.c_bool]


        # Load Newton Raphson procedure flow inference
        self.infer_flows_newton_raphson = lib.infer_flows_newton_raphson
        self.infer_flows_newton_raphson.restype = None #ctypes.c_double
        self.infer_flows_newton_raphson.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_double,
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t]

        # Load Iterative proportional filtering procedure flow inference
        self.infer_flows_ipf_procedure = lib.infer_flows_ipf_procedure
        self.infer_flows_ipf_procedure.restype = ctypes.c_double
        self.infer_flows_ipf_procedure.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_size_t,
                                                    ctypes.c_double,
                                                    ctypes.c_bool]

        # Load Iterative proportional filtering procedure flow inference
        self.infer_flows_ipf_procedure_singly = lib.infer_flows_ipf_procedure_singly
        self.infer_flows_ipf_procedure_singly.restype = ctypes.c_double
        self.infer_flows_ipf_procedure_singly.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_size_t,
                                                    ctypes.c_double,
                                                    ctypes.c_bool]

        # Load potential function
        self.potential_stochastic = lib2.potential_stochastic
        self.potential_stochastic.restype = ctypes.c_double
        self.potential_stochastic.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t]



    def flow_inference_dsf_procedure(self,max_iters:int = 10000,show_params:bool = False,show_flows:bool = False,**params):
        """ Computes flows using the DSF procedure for given set of parameters and data.

        Parameters
        ----------
        max_iters : int
            Maximum number of iterations to run the DSF procedure.
        show_params : bool
            Flag for printing the inferred parameter values during inference.
        show_flows : bool
            Flag for printing the inferred flows during inference.
        params : dictionary
            Parameters used in inference.

        Returns
        -------
        np.array [NxM]
            Flows inferred from the DSF procedure.

        """
        # Define empty array of flows
        flows = np.ones((self.N,self.M)).astype('float64')

        # Define beta
        beta = params['beta']

        # Infer flows
        value = self.infer_flows_dsf_procedure(flows,self.origin_supply,self.destination_demand,self.cost_matrix,self.N,self.M,beta,max_iters,show_params,show_flows)

        return flows

    def flow_inference_newton_raphson(self,newton_raphson_max_iters:int = 100,dsf_max_iters:int = 1000,show_params:bool=False,**params):
        """ Computes flows using the Newton Raphson procedure for given set of parameters and data.
        This function makes regular calls to the DSF procedure.

        Parameters
        ----------
        newton_raphson_max_iters : int
            Maximum number of iterations to run the Newton Raphson procedure.
        dsf_max_iters : int
            Maximum number of iterations to run the DSF procedure.
        show_params : bool
            Flag for printing the inferred parameter values during inference.
        params : dictionary
            Parameters used in inference.

        Returns
        -------
        np.array [NxM]
            Flows inferred from the Newton Raphson procedure.
        np.array [newton_raphson_max_iters x 1]
            Inferred beta parameters (including initial value at position 0).
        np.array [newton_raphson_max_iters x 1]
            Inferred cost of beta parameters (they have to be monotonic).
        """

        # Define empty array of flows
        flows = np.ones((self.N,self.M)).astype('float64')

        # Define initial \beta if beta is not zero (default value)
        if params['beta'] == 0.0:
            # See suggestion in page 386 of "Gravity Models of Spatial Interaction Behavior" book
            beta = np.ones((newton_raphson_max_iters)) * 1.5 * (1./self.total_cost)
        else:
            beta = np.ones((newton_raphson_max_iters)) * params['beta']

        # Define initial cost of beta
        c_beta = np.zeros(newton_raphson_max_iters)

        # Infer flows
        value = self.infer_flows_newton_raphson(flows,beta,c_beta,self.origin_supply,self.destination_demand,self.cost_matrix,self.total_cost,self.N,self.M,dsf_max_iters,newton_raphson_max_iters)

        # Trim beta and c_beta arrays if they were not used in full
        beta = beta[~np.isnan(beta)]
        c_beta = c_beta[c_beta>=0]

        # Print parameters if requested
        if show_params:
            print('\n')
            print('Beta',beta)
            print('Cost(beta)',c_beta)

        return flows,beta,c_beta


    def flow_inference_ipf_procedure(self,tolerance:float=1,ipf_max_iters:int = 1000,show_flows:bool=False,**params):
        """ Infer flows using the Iterative proportional filtering procedure.

        Parameters
        ----------
        tolerance : double
            Tolerance value used to assess if total (origin + destination) errors are within acceptable limits.
        ipf_max_iters : int
            Maximum iterations of iterative proportional filtering procedure.
        show_flows : bool
            Flag for printing the inferred flows during inference.

        Returns
        -------
        np.array
            Inferred flows

        """

        # Define empty array of flows
        flows = np.ones((self.N,self.M)).astype('float64')

        # Define parameter vector theta
        theta = np.array([params['alpha'],params['beta']])

        # Define A and B vectors
        A_vec = np.ones(self.N)*params['A_factor'].astype('float64')
        B_vec = np.ones(self.M)*params['B_factor'].astype('float64')

        # Infer flows
        inferred_flows = self.infer_flows_ipf_procedure(flows,
                                                        self.origin_supply,
                                                        self.destination_demand,
                                                        self.cost_matrix,
                                                        self.initial_destination_sizes,
                                                        A_vec,
                                                        B_vec,
                                                        self.N,
                                                        self.M,
                                                        theta,
                                                        ipf_max_iters,
                                                        tolerance,
                                                        show_flows)


        return flows


    def flow_inference_poisson_regression(self):
        """ Infers flow using PySAL's sparse poisson regression

        Returns
        -------
        np.array [NxD]
            Flows inferred from PySAL

        """

        # Reshape data
        flows_flat,orig_supply_flat,dest_demand_flat,cost_flat = self.reshape_data()

        # Train regression model using PySAL
        model = pysal.model.spint.Doubly(flows_flat, orig_supply_flat, dest_demand_flat, cost_flat, 'exp')

        # Print optimised parameters
        print('Optimised parameters---------------------')
        print('Distance coefficient',model.params[-1:][0])
        print('y-intercept (constant) = ',model.params[0])

        # Reconstruct inferred flow matrix from model vector
        pysal_flows = np.zeros((self.N,self.M))
        for i in range(self.N):
            for j in range(self.M):
                pysal_flows[i][j] = model.yhat[i*self.M + j]

        return pysal_flows,model

    # Wrapper for potential function
    def potential_value(self,xx,theta):
        """ Computes potential value of doubly constrained model in stochastic settings.

        Parameters
        ----------
        xx : np.array[Mx1]
            Log destination W_j's (usually some measure of economic activity)
        theta : np.array[]
            List of parameter values.

        Returns
        -------
        float
            Potential function value at xx.
        np.array[Mx1]
            Potential function Jacobian at xx.

        """
        # Initialise Jacobian
        jacobian = np.zeros(self.M)

        # Compute potential function
        value = self.potential_stochastic(xx, jacobian, self.normalised_destination_demand, self.normalised_cost_matrix, theta, self.N, self.M)

        return (value, jacobian)

    # Wrapper for hessian function
    def potential_hessian(self,xx,theta):

        A = np.zeros((self.M, self.M))
        value = self.hessian_stochastic(xx, A, self.destination_demand[:,0], self.cost_matrix, theta, self.N, self.M, wksp)

        return A

    # Potential function of the likelihood
    def likelihood_value(self,xx,s2_inv:float=100.):
        diff = xx - self.initial_log_sizes
        grad = s2_inv*diff
        potential = 0.5*s2_inv*np.dot(diff, diff)
        return pot, grad

    # Potential function for aNealed importance sampling (no flows model)
    def potential_value_annealed_importance_sampling(self,xx,theta):
        delta = theta[2]
        gaM = theta[3]
        kk = theta[4]
        gaM_kk_exp_xx = gaM*kk*np.exp(xx)
        gradV = -gaM*(delta+1./self.M)*np.ones(self.M) + gaM_kk_exp_xx
        V = -gaM*(delta+1./self.M)*xx.sum() + gaM_kk_exp_xx.sum()
        return V, gradV


    def reconstruct_flow_matrix(self,xd,theta):
        """ Reconstruct flow matrices

        Parameters
        ----------
        xd : np.array
            Log destination sizes.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Estimated flow matrix.

        """

        # Estimated destination sizes
        xhat = np.exp(minimize(self.potential_value, xd, method='L-BFGS-B', jac=True, args=(theta), options={'disp': False}).x)
        # Estimated flows
        That = np.zeros((self.N,self.M))
        # Construct flow matrix
        for i in range(self.N):
            for j in range(self.M):
                _sum = 0
                # Compute denominator
                for jj in range(self.M):
                    _sum += np.exp(theta[0]*xhat[j]-theta[1]*self.normalised_cost_matrix[i,jj])
                # Compute estimated flow
                That[i,j] = self.normalised_origin_supply[i]*np.exp(theta[0]*xhat[j]-theta[1]*self.normalised_cost_matrix[i,j]) / _sum

        return That

    def SRMSE(self,t_hat:np.array,actual_flows:np.array):
        """ Computes standardised root mean square error. See equation (22) of
        "A primer for working with the Spatial Interaction modeling (SpInt) module
        in the python spatial analysis library (PySAL)" for more details.

        Parameters
        ----------
        t_hat : np.array [NxM]
            Estimated flows.
        actual_flows : np.array [NxM]
            Actual flows.

        Returns
        -------
        float
            Standardised root mean square error of t_hat.

        """
        if actual_flows.shape[0] != t_hat.shape[0]:
            raise ValueError(f'Actual flows have {actual_flows.shape[0]} rows whereas \hat{T} has {t_hat.shape[0]}.')
        if actual_flows.shape[1] != t_hat.shape[1]:
            raise ValueError(f'Actual flows have {actual_flows.shape[1]} columns whereas \hat{T} has {t_hat.shape[1]}.')

        return ((np.sum((actual_flows - t_hat)**2) / (self.N*self.M))**.5) / (np.sum(actual_flows) / (self.N*self.M))
