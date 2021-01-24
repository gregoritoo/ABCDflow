import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
import kernels as kernels 
import os 
import pandas as pd 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float32')
PI = m.pi
OPTIMIZER = tf.optimizers.Adamax(learning_rate=0.06)
_jitter = 1e-4
_precision = tf.float64
KERNELS_LENGTH = {
    "LIN" : 3,
    "SE" : 2,
    "PER" :3,
    "RQ" : 3,
    "CONST" : 3,
    "WN" : 1,
}



KERNELS_FUNCTIONS = {
    "LIN" : kernels.LIN,
    "PER" : kernels.PER,
    "SE" : kernels.SE,
    "RQ" : kernels.RQ,
    "CONST" : kernels.CONST,
    "WN" :kernels.WN,

}

def make_df(X,stdp,stdi):
    X = np.array(X).reshape(-1)
    Y = np.array(np.arange(len(X))).reshape(-1)
    stdp = np.array(stdp).reshape(-1)
    stdi = np.array(stdi).reshape(-1)
    df = pd.DataFrame({"x":X, "y":Y,"stdp":stdp,"stdi":stdi},index=Y)
    return df 
    
def plot_gs_pretty(true_data,mean,X_train,X_s,stdp,stdi,color="blue"):
    try :
        true_data,X_train,X_s = true_data.numpy(),X_train.numpy(),X_s.numpy()
        mean,stdp,stdi =  mean.numpy(),stdp.numpy(), stdi.numpy()
    except Exception as e:
        pass
    true_data,X_train,X_s =  true_data.reshape(-1),X_train.reshape(-1),X_s.reshape(-1)
    mean,stdp,stdi = mean.reshape(-1),stdp.reshape(-1),stdi.reshape(-1).reshape(-1) 
    plt.style.use('seaborn')
    plt.plot(X_s,mean,color="green",label="Predicted values")
    plt.fill_between(X_s,stdp,stdi, facecolor=color, alpha=0.2,label="Conf I")
    plt.plot(X_train,true_data,color="red",label="True data")
    plt.legend()    
    plt.show()

def plot_gs(true_data,mean,X_train,X_s,stdp,stdi,color="blue"):
    plt.figure(figsize=(32,16), dpi=100)
    print(X_s)
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
            print("Cholesky decomposition failed trying with a more important jitter")
            _jitter = tf.random.uniform([1], minval=1e-1, maxval=1, dtype=_precision, seed=None, name=None)
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



