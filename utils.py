import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
import kernels
import os 
import pandas as pd 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float32')
PI = m.pi
OPTIMIZER = tf.optimizers.RMSprop(learning_rate=0.01)

KERNELS_LENGTH = {
    "LIN" : 1,
    "WN" : 1,
    "SE" : 2,
    "PER" :3,
}



KERNELS_FUNCTIONS = {
    "LIN" : kernels.LIN,
    "WN" : kernels.WN,
    "PER" : kernels.PER,
    "SE" : kernels.SE,

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
    plt.plot(X_s,mean,color="green",label="Predicted values")
    plt.fill_between(X_s.reshape(-1,),stdp,stdi, facecolor=color, alpha=0.2,label="Conf I")
    plt.plot(X_train,true_data,color="red",label="True data")
    plt.legend()    
    plt.show()

def plot_gs(true_data,mean,X_train,X_s,stdp,stdi,color="blue"):
    plt.figure(figsize=(32,16), dpi=100)
    plt.plot(X_s,mean,color="green",label="Predicted values")
    plt.fill_between(X_s.reshape(-1,),stdp,stdi, facecolor=color, alpha=0.2,label="Conf I")
    plt.plot(X_train,true_data,color="red",label="True data")
    plt.legend()
    

def get_values(mu_s,cov_s,nb_samples=100):
    samples = np.random.multivariate_normal(mu_s,cov_s,nb_samples)
    stdp = [np.mean(samples[:,i])+1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    stdi = [np.mean(samples[:,i])-1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    mean = [np.mean(samples[:,i])for i in range(samples.shape[1])]
    return mean,stdp,stdi


def compute_posterior(y,cov,cov_s,cov_ss):
    mu = tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+0.001*tf.eye(cov.shape[0]))),y)
    cov = cov_ss - tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+0.001*tf.eye(cov.shape[0]))),cov_s)
    return mu,cov

@tf.function
def log_l(X,Y,params,kernel):
    if kernel=="PER" :
        cov = Periodic(X,Y,l=params["l"],p=params["p"],sigma=params["sigma"])+1*tf.eye(X.shape[0])
    elif kernel == "LIN" :
        cov = Linear(X,Y,c=params["c"],sigmab=params["sigmab"],sigmav=params["sigmav"])+1*tf.eye(X.shape[0])
    elif kernel =="SE":
        cov = exp(X,Y,l=params["l"],sigma=params["sigma"])+ 0.001*tf.eye(X.shape[0])
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
    elif kernel == "WN" :
       cov = kernels.WN(X,X,params) + tf.eye(X.shape[0]) 
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
    _L = tf.linalg.cholesky(cov+1*tf.eye(X.shape[0]))
    _temp = tf.linalg.solve(_L, Y)
    alpha = tf.linalg.solve(tf.transpose(_L), _temp)
    loss = 0.5*tf.matmul(tf.transpose(Y),alpha) + tf.math.log(tf.linalg.trace(_L)) +0.5*X.shape[0]*tf.math.log([PI*2])
    return loss


def train_step(model,iteration,X_train,Y_train,kernels_name):
    with tf.GradientTape(persistent=False) as tape :
        tape.watch(model.variables)
        val = model(X_train,Y_train,kernels_name)
    gradient = tape.gradient(val,model.variables)
    try :
        OPTIMIZER.apply_gradients(zip(gradient, model.variables))
    except Exception as e :
        OPTIMIZER.apply_gradients(gradient,model.variables)
    return val


def train_step_single(model,iteration,X_train,Y_train):
    with tf.GradientTape(persistent=False) as tape :
        tape.watch(model.variables)
        val = model(X_train,Y_train)  
    gradient = tape.gradient(val,model.variables)
    try :
        OPTIMIZER.apply_gradients(zip(gradient, model.variables))
    except Exception as e :
        OPTIMIZER.apply_gradients(gradient,model.variables)
    return val



