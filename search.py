
import numpy as np 
import tensorflow as tf 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('INFO')
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
from Regressors import * 
from utils import train_step
import pandas as pd 
from itertools import chain
import itertools
import pickle 
import multiprocessing
from multiprocessing import Pool
import tensorflow_probability as tfp
import contextlib
import functools
import time
import scipy 



PI = m.pi
_precision = tf.float64
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

borne = -1*10e40



KERNELS_LENGTH = {
    "LIN" : 2,
    "SE" : 2,
    "PER" :3,
    #"CONST" : 1,
    #"WN" : 1,
    #"RQ" : 3,
}

KERNELS = {
    "LIN" : {"parameters_lin":["lin_c","lin_sigmav"]},
    #"CONST" : {"parameters":["const_sigma"]},
    "SE" : {"parameters":["squaredexp_l","squaredexp_sigma"]},
    "PER" : {"parameters_per":["periodic_l","periodic_p","periodic_sigma"]},
    #"WN" : {"paramters_Wn":["white_noise_sigma"]},
    #"RQ" : {"parameters_rq":["rq_l","rq_sigma","rq_alpha"]},
}




KERNELS_OPS = {
    "*LIN" : "mul",
    "*SE" : "mul",
    "*PER" :"mul",
    "+LIN" : "add",
    "+SE" : "add",
    "+PER" : "add",
    #"+CONST" :"add",
    #"*CONST" : "mul",
    #"+WN" :"add",
    #"*WN" : "mul",
    #"+RQ" : "add",
    #"*RQ" : "mul",
}



def _mulkernel(kernels_name,_kernel_list,new_k):
    ''' 
        Add  new kernel to the names and in the kernel list with a * 
    inputs :
        kernels_name :  string, name of the full kernel  ex +LIN*PER
        _kernels_list : list of string, list of kernels with theirs according operation ex ["+LIN","*PER"]
        new_k : string, name of kernel to add with it according operation ex *PER
    outputs :
        kernels_name :  string, updated kernel name   ex +LIN*PER*PER
        _kernels_list : list of string, list of updated kernels ex ["+LIN","*PER","*PER"]
    '''
    kernels_name = "("+kernels_name+")" + "*"+new_k
    _kernel_list = _kernel_list + ["*"+new_k]
    return kernels_name,_kernel_list

def _addkernel(kernels_name,_kernel_list,new_k):
    ''' 
        Add  new kernel to the names and in the kernel list with a +
    inputs :
        kernels_name :  string, name of the full kernel  ex +LIN*PER
        _kernels_list : list of string, list of kernels with theirs according operation ex ["+LIN","*PER"]
        new_k : string, name of kernel to add with it according operation ex +PER
    outputs :
        kernels_name :  string, updated kernel name   ex +LIN*PER+PER
        _kernels_list : list of string, list of updated kernels ex ["+LIN","*PER","+PER"]
    '''
    kernels_name = kernels_name + "+"+new_k
    _kernel_list = _kernel_list + ["+"+new_k]
    return kernels_name,_kernel_list

def _preparekernel(_kernel_list,scipy=True):
    '''
        Receive the list of kernels with theirs operations and return a dict with the kernels names and parameters 
    inputs :
        _kernels_list : list of string, list of kernels with theirs according operation
    outputs :
        kernels : dict, dict cotaining kernel parameters (see KERNELS)
    '''
    dic = tuple([KERNELS[d[1:]] for d in _kernel_list])
    kernels={}
    i = 1
    for para in dic :
        for key,value in para.items() :
            if key in kernels.keys() :
                if key != "noise" :
                    key = key+"_"+str(i)
                    kernels.update({key:[element+"_"+str(i) for element in value]})
            else :
                kernels.update({key:[element for element in value]})
            i+=1
    return kernels



def search(kernels_name,_kernel_list,init,depth=5):
    '''
        Return all the possible combinaison of kernels starting with a '+' 
    inputs :
        kernels_name :  string, not used 
        _kernel_list : list of tuples, not used 
        init : Bool, not used 
    '''
    kerns = tuple((KERNELS_OPS.keys()))
    COMB = []
    for i in range(1,depth) :
        if i == 1 : combination =  list(itertools.combinations(kerns, i))
        else : combination = list(itertools.permutations(kerns, i)) 
        for comb in combination :
            if not comb[0][0] == "*" : 
                COMB.append(comb)
    return COMB


def _prune(tempbest,rest):
    '''
        Cut the graph in order to keep only the combinaison that correspond to the best for the moment 
    inputs :
        tempbest : list ,  names of best combinaisons of kernels already trained 
        rest : list of tuples , all the combinaisons of kernels not trained yet 
    outputs :
        new_rest : list of tuples , all the combinaisons to be trained 
    '''
    print("Prunning {} elements".format(len(rest)))
    new_rest = []
    for _element in rest :
        for _best in tempbest :
            _element = list(_element)
            if _best == _element[:len((_best))] and _element not in new_rest :
                new_rest.append(_element)
    print("Prunning ended, still {} kernels to try".format(len(new_rest)))
    return new_rest



def search_and_add(_kernel_list):
    ''' 
        Return all possible combinaison for one step
    inputs :
        _kernels_list : list, not used 
    '''
    kerns = tuple((KERNELS_OPS.keys()))
    COMB = []
    combination =  list(itertools.combinations(kerns, 1))
    for comb in combination :
        COMB.append(_kernel_list+comb)
    return COMB