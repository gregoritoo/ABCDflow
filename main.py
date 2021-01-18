import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
from Regressors import * 
from utils import train_step
import pandas as pd 
tf.keras.backend.set_floatx('float32')
from itertools import chain

PI = m.pi


def _mulkernel(kernels_name,_kernel_list,new_k):
    kernels_name = "("+kernels_name+")" + "*"+new_k
    _kernel_list = _kernel_list + ["*"+new_k]
    return kernels_name,_kernel_list

def _addkernel(kernels_name,_kernel_list,new_k):
    kernels_name = kernels_name + "+"+new_k
    _kernel_list = _kernel_list + ["+"+new_k]
    return kernels_name,_kernel_list

def _preparekernel(_kernel_list):
    dic = tuple([KERNELS[d[1:]] for d in _kernel_list])
    kernels={}
    i = 1
    for para in dic :
        for key,value in para.items() :
            if key in kernels.keys() :
                key = key+"_"+str(i)
                kernels.update({key:[element+"_"+str(i) for element in value]})
                i+=1
            else :
                kernels.update({key:[element for element in value]})
    return kernels

        



KERNELS_LENGTH = {
    "LIN" : 3,
    "WN" : 1,
    "SE" : 2,
    "RQ" : {},
    "PER" :3,
}

KERNELS = {
    "LIN" : {"parameters_lin":["lin_c","lin_sigmab","lin_sigmav"]},
    "WN" : {"parameters":["white_sigma"]},
    "SE" : {"parameters":["squaredexp_l","squaredexp_sigma"]},
    "RQ" : {},
    "PER" : {"parameters_per":["periodic_l","periodic_p","periodic_sigma"]},
}

def train(model,nb_iter,nb_restart,X_train,Y_train,kernels_name):
    base_model = model 
    loop = 0
    best = 10e90
    #tf.compat.v1.global_variables_initializer() 
    while loop < nb_restart :
        try :
            model = base_model
            for iteration in range(1,nb_iter):
                val = train_step(model,iteration,X_train,Y_train,kernels_name)
                sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."*int((nb_iter-iteration)/nb_iter*50)+"|"+" * log likelihood  is : {:.4f} at epoch : {:.0f} at iteration : {:.0f} / {:.0f} ".format(val[0][0],nb_iter,loop+1,nb_restart))
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as e :
            print(e)
        if val  < best :
            best =  val
            best_model = model
        loop += 1
    return best_model





if __name__ =="__main__" :
    """Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)

    X_s_num = np.arange(0, 160, 1).reshape(-1, 1)
    X_train = tf.Variable(X,dtype=tf.float32)
    Y_train = tf.Variable(Y,dtype=tf.float32)
    
    X_s = tf.Variable(X_s_num,dtype=tf.float32)
    """

    """X_train = tf.Variable(np.array(np.linspace(0,30,30)).reshape(-1, 1),dtype=tf.float32)
    Y_train = tf.Variable(np.sin(X_train.numpy().reshape(-1, 1)),dtype=tf.float32)

    X_s = tf.Variable(np.arange(-2, 28,30).reshape(-1, 1),dtype=tf.float32)
 
    nb_restart = 15
    nb_iter = 10
    model = WhiteNoiseRegressor()
    model = train(model,nb_iter,nb_restart)
    model.viewVar()

    k = GPy.kern.White(input_dim=1)       
    m = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
    m.optimize_restarts(30)
    print(m)
    m.plot()
    plt.show()
    mu,cov = model.predict(X_train,Y_train,X_s)    
    mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=1000)
    plot_gs(Y_train.numpy(),mean,X_train.numpy(),X_s.numpy(),stdp,stdi)
    plt.show()
"""

    Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)

    X_s_num = np.arange(0, 160, 1).reshape(-1, 1)
    X_train = tf.Variable(X,dtype=tf.float32)
    Y_train = tf.Variable(Y,dtype=tf.float32)
    
    X_s = tf.Variable(X_s_num,dtype=tf.float32)

    
    nb_restart = 300
    nb_iter = 4

    kernels_name,_kernel_list = _mulkernel("LIN",["+LIN"],"PER")
    kernels_name,_kernel_list =_addkernel(kernels_name,_kernel_list,"LIN")
    #kernels_name,_kernel_list =_addkernel(kernels_name,_kernel_list,"WN")
    #kernels_name,_kernel_list =_mulkernel(kernels_name,_kernel_list,"SE")
    kernels = _preparekernel(_kernel_list)
    model=CustomModel(kernels)
    model = train(model,nb_iter,nb_restart,X_train,Y_train,_kernel_list)

    model.viewVar(kernels_name)

    mu,cov = model.predict(X_train,Y_train,X_s,_kernel_list)    
    mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=1000)
    plot_gs(Y_train.numpy(),mean,X_train.numpy(),X_s.numpy(),stdp,stdi)
    plt.show()

    

