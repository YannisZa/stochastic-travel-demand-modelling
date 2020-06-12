""" Wrapper on C code of potential functions """

import os
import sys
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer


class SpatialIteraction():
    """ Wrapper on C code of potential functions.

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

    def __init__(self,mode,dataset):
        '''  Constructor '''

        # Define mode (stochastic/deterministic)
        self.mode = mode
        # Define dataset
        self.dataset = dataset
        # Define working directory
        self.working_directory = self.get_project_root()
        # Define data directory
        self.data_directory = os.path.join(self.working_directory,'data/input/{}'.format(dataset))
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
        N [int]
            Number of origin zones.
        M [int]
            Number of destination zones.
        """

        # Import cost matrix
        costmatrix_file = os.path.join(self.data_directory,'cost_matrix.txt')
        self.cost_matrix = np.loadtxt(costmatrix_file)

        # Import origin supply
        originsupply_file = os.path.join(self.data_directory,'origin_supply.txt')
        self.origin_supply = np.loadtxt(originsupply_file)

        # Import initial log sizes
        initiallogsizes_file = os.path.join(self.data_directory,'initial_log_sizes.txt')
        self.initial_log_sizes = np.loadtxt(initiallogsizes_file)

        # Import N,M
        self.N, self.M = np.shape(self.cost_matrix)

    def load_c_functions(self):
        """ Stores C functions that infer flows to global variables.

        Attributes
        -------
        potential_deterministic [function]
            Function used to compute the deterministic potential function.
        potential_stochastic [function]
            Function used to compute the stochastic potential function.
        hessian_deterministic [function]
            Function used to compute the Hessian of the deterministic potential function.
        hessian_stochastic [function]
            Function used to compute the Hessian of the stochastic potential function.
        """

        # Load shared object
        lib = ctypes.cdll.LoadLibrary(os.path.join(self.working_directory,"models/singly_constrained/potential_functions.so"))

        # Load deterministic potential function from shared object
        self.potential_deterministic = lib.potential_deterministic
        self.potential_deterministic.restype = ctypes.c_double
        self.potential_deterministic.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

        # Load stochastic potential function from shared object
        self.potential_stochastic = lib.potential_stochastic
        self.potential_stochastic.restype = ctypes.c_double
        self.potential_stochastic.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

        # Load deterministic hessian function from shared object
        self.hessian_deterministic = lib.hessian_deterministic
        self.hessian_deterministic.restype = None
        self.hessian_deterministic.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

        # Load stochastic hessian function from shared object
        self.hessian_stochastic = lib.hessian_stochastic
        self.hessian_stochastic.restype = None
        self.hessian_stochastic.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_size_t,
                        ctypes.c_size_t,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    # Wrapper for potential function
    def potential_value(self,xx,theta):
        """ Computes potential value in both deterministic and stochastic settings.

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
        if self.mode == 'stochastic':
          value = self.potential_stochastic(xx, grad, self.origin_supply, self.cost_matrix, theta, self.N, self.M, wksp)
        else:
          value = self.potential_deterministic(xx, grad, self.origin_supply, self.cost_matrix, theta, self.N, self.M, wksp)
        return (value, grad)

    # Wrapper for hessian function
    def potential_hessian(self,xx,theta):
        A = np.zeros((self.M, self.M))
        wksp = np.zeros(self.M)
        if self.mode == 'stochastic':
          value = self.hessian_stochastic(xx, A, self.origin_supply, self.cost_matrix, theta, self.N, self.M, wksp)
        else:
          value = self.hessian_deterministic(xx, A, self.origin_supply, self.cost_matrix, theta, self.N, self.M, wksp)
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
