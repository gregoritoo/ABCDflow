import numpy as np 
import tensorflow as tf 
from pprint import pprint
import os 
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('INFO')
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
import kernels as kernels 
import os 
import pandas as pd 
import contextlib
import functools
import os
import time
import seaborn as sn
import gpflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float64')


PI = m.pi
_jitter = 1e-7
_precision = tf.float64



KERNELS = {
    "LIN" : {"parameters_lin":["lin_c","lin_sigmav"]},
    "CONST" : {"parameters":["const_sigma"]},
    "SE" : {"parameters_se":["squaredexp_l","squaredexp_sigma"]},
    "PER" : {"parameters_per":["periodic_l","periodic_p","periodic_sigma"]},
    "WN" : {"paramters_Wn":["white_noise_sigma"]},
    "RQ" : {"parameters_rq":["rq_l","rq_sigma","rq_alpha"]},
}


KERNELS_OPS = {
    "*LIN" : "mul",
    "*SE" : "mul",
    "*PER" :"mul",
    "+LIN" : "add",
    "+SE" : "add",
    "+PER" : "add",
    "+CONST" :"add",
    "*CONST" : "mul",
    "+WN" :"add",
    "*WN" : "mul",
    "+RQ" : "add",
    "*RQ" : "mul",
}

KERNELS_LENGTH = {
    "LIN" : len(KERNELS["LIN"]["parameters_lin"]),
    "SE" : len(KERNELS["SE"]["parameters_se"]),
    "PER" :len(KERNELS["PER"]["parameters_per"]),
    "RQ" : len(KERNELS["RQ"]["parameters_rq"]),
    "CONST" : len(KERNELS["CONST"]["parameters"]),
    "WN" : len(KERNELS["WN"]["paramters_Wn"]),
}



KERNELS_FUNCTIONS = {
    "LIN" : kernels.LIN,
    "PER" : kernels.PER,
    "SE" : kernels.SE,
    "RQ" : kernels.RQ,
    "CONST" : kernels.CONST,
    "WN" :kernels.WN,

}


GPY_KERNELS = {
    "LIN" : GPy.kern.Linear(1),
    "SE" : GPy.kern.sde_Exponential(1),
    "PER" :GPy.kern.StdPeriodic(1),
    "RQ" : GPy.kern.RatQuad(1),
}




