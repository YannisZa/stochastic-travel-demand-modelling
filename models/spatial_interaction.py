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

    def store_parameters(self,**params):
        # Define parameters
        self.beta = params['beta']
        self.A = np.ones(self.N) * params['A_factor']
        self.B = np.ones(self.M) * params['B_factor']

    def load_c_functions(self):
        ''' Loads C function '''

        # Load shared object
        lib = ctypes.cdll.LoadLibrary(os.path.join(self.working_directory,"models/doubly_constrained_model.so"))

        # Load deterministic potential function from shared object
        self.infer_flows = lib.infer_flows
        self.infer_flows.restype = ctypes.c_int
        self.infer_flows.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                    ctypes.c_size_t,
                                    ctypes.c_size_t,
                                    ctypes.c_double,
                                    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                    ctypes.c_size_t,
                                    ctypes.c_bool,
                                    ctypes.c_bool]

    def flow_inference(self,max_iters:int = 10000,show_params:bool = False,show_flows:bool = False):
        ''' Returns flows for given set of parameters and data'''
        # Define empty array of flows
        flows = np.zeros((self.N,self.M)).astype('int32')

        # Infer flows
        value = self.infer_flows(self.origin_supply,self.destination_demand,self.cost_matrix,self.N,self.M,self.beta,flows,self.A,self.B,max_iters,show_params,show_flows)

        return flows
