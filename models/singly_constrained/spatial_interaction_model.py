"""
Inferring flow for singly constrained origin-destination model using various methods.
"""

import os
import ctypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.ctypeslib import ndpointer
from scipy.optimize import minimize

class SpatialInteraction():
    """ Object including flow (O/D) matrix inference routines of singly constrained SIM.

    Parameters
    ----------
    mode : string
        Mode of inference: stochastic or deterministic.
    dataset : string
        Name of dataset used for inference.

    Attributes
    ----------
    working_directory : string
        Project's root working directory.
    data_directory : string
        Project's data directory.
    import_data : function
        Loads relevant data for inference.
    load_c_functions : function
        Loads relevant C functions for inference.

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
        cost_matrix [NxM array]
            Cost of travelling from each origin to each destination.
        initial_log_sizes [Mx1 array]
            Initial condition log sizes for ODE.
        origin supply [Nx1 array]
            Total supply for each destination.
        cost_matrix_type : string
            Type of cost matrix used ('' stands for Euclidean distance based
            'small_network' stands for small transportation network including only A and B roads)
        N [int]
            Number of origin zones.
        M [int]
            Number of destination zones.
        """

        # Import origin supply
        originsupply_file = os.path.join(self.data_directory,'origin_supply.txt')
        self.origin_supply = np.loadtxt(originsupply_file,ndmin=1)

        # In case origin supply is not a list
        if not isinstance(self.origin_supply,(np.ndarray, np.generic)):
            self.origin_supply = np.array([self.origin_supply])

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
        costmatrix_file = os.path.join(self.data_directory,f'cost_matrix{self.cost_matrix_file_extension}.txt')
        self.cost_matrix = np.loadtxt(costmatrix_file)

        # Reshape cost matrix if necessary
        if self.N == 1:
            self.cost_matrix = np.reshape(self.cost_matrix[:,np.newaxis],(self.N,self.M))
        if self.M == 1:
            self.cost_matrix = np.reshape(self.cost_matrix[np.newaxis,:],(self.N,self.M))

        # Compute total initial and final destination sizes
        self.total_initial_sizes = np.sum(self.initial_destination_sizes)

        # Compute total cost
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
                                     "Flow": self.actual_flows[i,j],
                                     "OriginSupply": self.origin_supply[i]})
                # Append row to dataframe
                od_data = od_data.append(new_row, ignore_index=True)

        # Get flatten data and et column types appropriately
        flows_flat = od_data.Flow.values.astype('int64')
        orig_supply_flat = od_data.OriginSupply.values.astype('int64')
        cost_flat = od_data.Cost.values.astype('float64')

        return flows_flat,orig_supply_flat,cost_flat

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
        potential_and_jacobian [function]
            Function used to compute the potential function and its Jacobian matrix.
        potential_hessian [function]
            Function used to compute the Hessian matrix of the potential function
        """

        # Load shared object
        lib = ctypes.cdll.LoadLibrary(os.path.join(self.working_directory,"models/singly_constrained/potential_function.so"))

        # Load potential function and its jacobian from shared object
        self.potential_and_jacobian = lib.potential_and_jacobian
        self.potential_and_jacobian.restype = ctypes.c_double
        self.potential_and_jacobian.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

        # Load hessian of potential function from shared object
        self.hessian = lib.hessian
        self.hessian.restype = None
        self.hessian.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    def potential_value(self,xx,theta):
        """ Computes potential value of singly constrained model in both deterministic and stochastic settings.

        Parameters
        ----------
        xx : np.array[Mx1]
            Values at which to evaluate the potential function.
        theta : np.array[]
            List of parameter values.

        Returns
        -------
        float
            Potential function value at xx.
        np.array[Mx1]
            Potential function Jacobian at xx.

        """
        grad = np.zeros(self.M)
        wksp = np.zeros(self.M)
        # print('alpha =',theta[0],'beta =',theta[1],'delta =',theta[2],'gamma =',theta[3],'kappa =',theta[4],'epsilon =',theta[5])
        value = self.potential_and_jacobian(xx, grad, self.normalised_origin_supply, self.normalised_cost_matrix, theta, self.N, self.M, wksp)
        return (value, grad)

    # Wrapper for hessian function
    def potential_hessian(self,xx,theta):
        A = np.zeros((self.M, self.M))
        wksp = np.zeros(self.M)

        value = self.hessian(xx, A, self.normalised_origin_supply, self.normalised_cost_matrix, theta, self.N, self.M, wksp)
        return A

    # Potential function of the likelihood
    def likelihood_value(self,xx,s2_inv:float=100.):
        diff = xx - self.normalised_initial_destination_sizes
        grad = s2_inv*diff
        potential = 0.5*s2_inv*np.dot(diff, diff)
        return pot, grad

    # Potential function for aNealed importance sampling (no flows model)
    def potential_value_annealed_importance_sampling(self,xx,theta):
        delta = theta[2]
        gamma = theta[3]
        kappa = theta[4]

        gaM_kk_exp_xx = gamma*kappa*np.exp(xx)

        gradV = -gamma*(delta+1./self.M)*np.ones(self.M) + gaM_kk_exp_xx

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
