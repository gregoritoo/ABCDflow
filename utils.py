import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
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
f = open(os.devnull, 'w')
sys.stderr = f
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float64')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

PI = m.pi
OPTIMIZER = tf.optimizers.Adamax(learning_rate=0.06)
_jitter = 1e-9
_precision = tf.float64
KERNELS_LENGTH = {
    "LIN" : 3,
    "SE" : 2,
    "PER" :3,
    "RQ" : 3,
    #"CONST" : 3,
    #"WN" : 1,
}



KERNELS_FUNCTIONS = {
    "LIN" : kernels.LIN,
    "PER" : kernels.PER,
    "SE" : kernels.SE,
    "RQ" : kernels.RQ,
    #"CONST" : kernels.CONST,
    #"WN" :kernels.WN,

}

def make_df(X,stdp,stdi):
    X = np.array(X).reshape(-1)
    Y = np.array(np.arange(len(X))).reshape(-1)
    stdp = np.array(stdp).reshape(-1)
    stdi = np.array(stdi).reshape(-1)
    df = pd.DataFrame({"x":X, "y":Y,"stdp":stdp,"stdi":stdi},index=Y)
    return df 
    
def plot_gs_pretty(true_data,mean,X_train,X_s,stdp,stdi,color="blue"):
    plt.style.use('seaborn')
    try :
        true_data,X_train,X_s = true_data.numpy(),X_train.numpy(),X_s.numpy()
        mean,stdp,stdi =  mean.numpy(),stdp.numpy(), stdi.numpy()
    except Exception as e:
        pass
    true_data,X_train,X_s =  true_data.reshape(-1),X_train.reshape(-1),X_s.reshape(-1)
    mean,stdp,stdi = mean.reshape(-1),stdp.reshape(-1),stdi.reshape(-1).reshape(-1) 
    plt.style.use('seaborn')
    plt.plot(X_s,mean,color="blue",label="Predicted values")
    plt.fill_between(X_s,stdp,stdi, facecolor=CB91_Blue, alpha=0.2,label="Conf I")
    plt.plot(X_train,true_data,color=CB91_Amber,label="True data",marker="x")
    plt.legend()    
    plt.show()

def plot_gs(true_data,mean,X_train,X_s,stdp,stdi,color="blue"):
    plt.figure(figsize=(32,16), dpi=100)
    print(X_s)
    plt.style.use('seaborn')
    plt.plot(X_s,mean,color="green",label="Predicted values")
    plt.fill_between(X_s.reshape(-1,),stdp,stdi, facecolor=CB91_Blue, alpha=0.2,label="Conf I")
    plt.plot(X_train,true_data,color=CB91_Amber,label="True data")
    plt.legend()
    

def get_values(mu_s,cov_s,nb_samples=100):
    samples = np.random.multivariate_normal(mu_s,cov_s,nb_samples)
    stdp = [np.mean(samples[:,i])+1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    stdi = [np.mean(samples[:,i])-1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    mean = [np.mean(samples[:,i])for i in range(samples.shape[1])]
    return mean,stdp,stdi


def compute_posterior(y,cov,cov_s,cov_ss):
    mu = tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+_jitter*tf.eye(cov.shape[0],dtype=_precision))),y)
    cov = cov_ss - tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+_jitter*tf.eye(cov.shape[0],dtype=_precision))),cov_s)
    return mu,cov

@tf.function
def log_l(X,Y,params,kernel):
    if kernel=="PER" :
        cov = Periodic(X,Y,l=params["l"],p=params["p"],sigma=params["sigma"])+_jitter**tf.eye(X.shape[0])
    elif kernel == "LIN" :
        cov = Linear(X,Y,c=params["c"],sigmab=params["sigmab"],sigmav=params["sigmav"])+_jitter*tf.eye(X.shape[0])
    elif kernel =="SE":
        cov = exp(X,Y,l=params["l"],sigma=params["sigma"])+ _jitter*tf.eye(X.shape[0])
    loss = -0.5*tf.matmul(tf.matmul(tf.transpose(Y),tf.linalg.inv(cov)),Y) - 0.5*tf.math.log(tf.linalg.det(cov))-0.5*X.shape[0]*tf.math.log([PI*2])
    
    return -loss


def log_cholesky_l(X,Y,params,kernel):
    params_name = list(params.keys())
    par =params_name[0:0+KERNELS_LENGTH[kernel]]
    params = [params[p] for p in par]
    if kernel=="PER" :
        cov = kernels.PER(X,X,params)+ tf.eye(X.shape[0])
    elif kernel == "LIN" :
        cov = kernels.LIN(X,X,params)+ tf.eye(X.shape[0])
    elif kernel =="SE":
        cov = kernels.SE(X,X,params) + tf.eye(X.shape[0])
    elif kernel == "CONST" :
       cov = kernels.CONST(X,X,params) + tf.eye(X.shape[0]) 
    _L = tf.linalg.cholesky(cov)
    _temp = tf.linalg.solve(_L, Y)
    alpha = tf.linalg.solve(tf.transpose(_L), _temp)
    loss = 0.5*tf.matmul(tf.transpose(Y),alpha) + tf.math.log(tf.linalg.trace(_L)) +0.5*X.shape[0]*tf.math.log([PI*2])
    
    return loss


def log_cholesky_l_test(X,Y,params,kernel):
    num = 0
    params_name = list(params.keys())
    cov = 1
    for op in kernel :
        if op[0] == "+":
            method = KERNELS_FUNCTIONS[op[1:]]
            par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
            if not method:
                raise NotImplementedError("Method %s not implemented" % op[1:])
            cov += method(X,X,[params[p] for p in par])
            num += KERNELS_LENGTH[op[1:]]
        elif op[0] == "*":
            method = KERNELS_FUNCTIONS[op[1:]]
            method = KERNELS_FUNCTIONS[op[1:]]
            par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
            if not method:
                raise NotImplementedError("Method %s not implemented" % op[1:])
            cov  = tf.math.multiply(cov,method(X,X,[params[p] for p in par]))
            num += KERNELS_LENGTH[op[1:]]
    decomposed, _jitter,loop = False, 1e-4 , 0
    while not decomposed and loop < 5 :
        try :
            _L = tf.cast(tf.linalg.cholesky(tf.cast(cov+_jitter*tf.eye(X.shape[0],dtype=_precision),dtype=_precision)),dtype=_precision)
            decomposed = True 
        except Exception as e :
            loop +=1
            #print("Cholesky decomposition failed trying with a more important jitter")
            _jitter = tf.random.uniform([1], minval=1e-1, maxval=1, dtype=_precision, seed=None, name=None) + 1*(loop-1)
    _temp = tf.cast(tf.linalg.solve(_L, Y),dtype=_precision)
    alpha = tf.cast(tf.linalg.solve(tf.transpose(_L), _temp),dtype=_precision)
    loss = 0.5*tf.cast(tf.matmul(tf.transpose(Y),alpha),dtype=_precision) + tf.cast(tf.math.log(tf.linalg.trace(_L)),dtype=_precision) +0.5*tf.cast(X.shape[0]*tf.math.log([PI*2]),dtype=_precision)
    return loss



def train_step(model,iteration,X_train,Y_train,kernels_name,OPTIMIZER=tf.optimizers.Adamax(learning_rate=0.06)):
    with tf.GradientTape(persistent=False) as tape :
        tape.watch(model.variables)
        val = model(X_train,Y_train,kernels_name)
    gradient = tape.gradient(val,model.variables)
    try :
        OPTIMIZER.apply_gradients(zip(gradient, model.variables))
    except Exception as e :
        OPTIMIZER.apply_gradients(gradient,model.variables)
    return val


def train_step_single(model,iteration,X_train,Y_train,kernels_name,OPTIMIZER=tf.optimizers.Adamax(learning_rate=0.06)):
    with tf.GradientTape(persistent=False) as tape :
        tape.watch(model.variables)
        val = model(X_train,Y_train)  
    gradient = tape.gradient(val,model.variables)
    try :
        OPTIMIZER.apply_gradients(zip(gradient, model.variables))
    except Exception as e :
        OPTIMIZER.apply_gradients(gradient,model.variables)
    return val





def loss_function(x_u_train, u_train, network):
    u_pred = tf.cast(network(x_u_train), dtype=tf.float32)
    loss_value = tf.reduce_mean(tf.square(u_train - u_pred))
    return tf.cast(loss_value, dtype=tf.float32)




""" function factory is a adapted version of  :
    https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
"""

def function_factory(model, loss_f, X, Y,params,kernel):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model._opti_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """
        
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model._opti_variables[i].assign(tf.cast(tf.reshape(param, shape), dtype=_precision))

    # now create a function that will be returned by this factory

    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = model(X,Y,kernel)[0][0]

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        #f.iter.assign_add(1)
        #tf.print("Iter:", f.iter, "loss:", loss_value)
       
        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])
        return np.array(loss_value.numpy(), order='F'),np.array(grads.numpy(), order='F')

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f

