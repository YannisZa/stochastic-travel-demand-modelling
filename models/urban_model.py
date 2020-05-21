
"""
Wrapper on C code of potential functions
"""

import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

class UrbanModel():

  def __init__(self,mode,dataset,wd):
    '''  Constructor '''

    # Define mode (stochastic/deterministic)
    self.mode = mode
    # Define dataset
    self.dataset = dataset
    # Define working directory
    self.working_directory = wd
    # Define data directory
    self.data_directory = os.path.join(wd,'data/input/{}'.format(dataset))
    # Import data
    self.import_data()
    # Load C functions
    self.load_c_functions()

  def import_data(self):
    '''  Data import function '''

    # Import cost matrix
    costmatrix_file = os.path.join(self.data_directory,'cost_matrix.txt')
    self.cost_matrix = np.loadtxt(costmatrix_file)

    # Import origin demand
    origindemand_file = os.path.join(self.data_directory,'origin_demand.txt')
    self.origin_demand = np.loadtxt(origindemand_file)

    # Import true log sizes
    truelogsizes_file = os.path.join(self.data_directory,'true_log_sizes.txt')
    self.true_log_sizes = np.loadtxt(truelogsizes_file)

    # Import N,M
    self.N, self.M = np.shape(self.cost_matrix)

  def load_c_functions(self):
    ''' Loads C functions '''

    # Load shared object
    lib = ctypes.cdll.LoadLibrary(os.path.join(self.working_directory,"models/potential_functions.so"))

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

  # Wrapper for potential function
  def potential_value(self,xx,theta):
      grad = np.zeros(self.M)
      wksp = np.zeros(self.M)
      if self.mode == 'stochastic':
          value = self.potential_stochastic(xx, grad, self.origin_demand, self.cost_matrix, theta, self.N, self.M, wksp)
      else:
          value = self.potential_deterministic(xx, grad, self.origin_demand, self.cost_matrix, theta, self.N, self.M, wksp)
      return (value, grad)

  # Wrapper for hessian function
  def potential_hessian(self,xx,theta):
      A = np.zeros((self.M, self.M))
      wksp = np.zeros(self.M)
      if self.mode == 'stochastic':
          value = self.hessian_stochastic(xx, A, self.origin_demand, self.cost_matrix, theta, self.N, self.M, wksp)
      else:
          value = self.hessian_deterministic(xx, A, self.origin_demand, self.cost_matrix, theta, self.N, self.M, wksp)
      return A

  # Potential function of the likelihood
  def likelihood_value(self,xx,s2_inv:float=100.):
      diff = xx - self.true_log_sizes
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
