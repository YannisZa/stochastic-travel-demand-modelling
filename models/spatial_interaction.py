""" Maximum entropy flow for doubly constrained origin-destination model """

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
from models.urban_model import UrbanModel
from tqdm import tqdm
import ctypes
from numpy.ctypeslib import ndpointer

class DoublyConstrainedModel():

    def __init__(self,dataset:str,**params):
        '''  Constructor '''

        # Define working directory
        self.working_directory = rd
        # Define data directory
        self.data_directory = os.path.join(rd,'data/input/{}'.format(dataset))
        # Load C function
        self.load_c_functions()
        # Import data
        self.import_data()
        # Store parameters
        self.store_parameters(**params)

    def import_data(self):
        '''  Data import function '''

        # Import cost matrix
        costmatrix_file = os.path.join(self.data_directory,'cost_matrix.txt')
        self.cost_matrix = np.loadtxt(costmatrix_file)

        # Import origin supply
        originsupply_file = os.path.join(self.data_directory,'origin_supply.txt')
        self.origin_supply = np.loadtxt(originsupply_file).astype('int32')

        # Import origin supply
        destinationdemand_file = os.path.join(self.data_directory,'destination_demand.txt')
        self.destination_demand = np.loadtxt(destinationdemand_file).astype('int32')

        # Import N,M
        self.N, self.M = np.shape(self.cost_matrix)

        # Calculate total flow and cost
        self.total_flow = np.sum(self.origin_supply)

        # Import full flow matrix to compute total cost
        self.actual_flows =  np.loadtxt(os.path.join(self.data_directory,'od_matrix.txt'))

        # Compute total cost
        self.total_cost = 0
        for i in range(self.N):
            for j in range(self.M):
                self.total_cost += self.cost_matrix[i,j]*self.actual_flows[i,j]

    def store_parameters(self,**params):
        # Define parameters
        self.beta = params['beta']
        self.A = np.ones(self.N) * params['A_factor']
        self.B = np.ones(self.M) * params['B_factor']

    def load_c_functions(self):
        ''' Loads C function '''

        # Load shared object
        lib = ctypes.cdll.LoadLibrary(os.path.join(self.working_directory,"models/doubly_constrained_model.so"))

        # Load DSF procedure flow inference
        self.infer_flows_dsf_procedure = lib.infer_flows_dsf_procedure
        self.infer_flows_dsf_procedure.restype = ctypes.c_double
        self.infer_flows_dsf_procedure.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ctypes.c_double,
                                                    ctypes.c_size_t,
                                                    ctypes.c_bool,
                                                    ctypes.c_bool]


        # Load DSF procedure flow inference
        self.infer_flows_newton_raphson = lib.infer_flows_newton_raphson
        self.infer_flows_newton_raphson.restype = None #ctypes.c_double
        self.infer_flows_newton_raphson.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                                    ctypes.c_double,
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t,
                                                    ctypes.c_size_t]


    def flow_inference_dsf_procedure(self,max_iters:int = 10000,show_params:bool = False,show_flows:bool = False):
        ''' Returns flows for given set of parameters and data '''
        # Define empty array of flows
        flows = np.ones((self.N,self.M)).astype('float64')

        # Infer flows
        value = self.infer_flows_dsf_procedure(flows,self.origin_supply,self.destination_demand,self.cost_matrix,self.N,self.M,self.beta,max_iters,show_params,show_flows)

        return flows

    def flow_inference_newton_raphson(self,newton_raphson_max_iters:int = 100,dsf_max_iters:int = 1000,show_params:bool=False):
        ''' Returns flows for given set of parameters and data '''

        # Define empty array of flows
        flows = np.ones((self.N,self.M)).astype('float64')

        # Define initial \beta if beta is not zero (default value)
        if self.beta == 0:
            # See suggestion in page 386 of "Gravity Models of Spatial Interaction Behavior" book
            beta = np.ones((newton_raphson_max_iters)) * 1.5 * self.total_flow * (1./self.total_cost)
        else:
            beta = np.ones((newton_raphson_max_iters)) * self.beta

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

    def SRMSE(self,t_hat:np.array):
        '''
        Computes standardised root mean square error. See equation (22) of
        "A primer for working with the Spatial Interaction modeling (SpInt) module
        in the python spatial analysis library (PySAL)" for more details.

        [t] : actual/true flows
        [that] : estimated flows
        '''
        return ((np.sum((self.actual_flows - t_hat)**2) / (self.N*self.M))**.5) / (self.total_flow / (self.N*self.M))
