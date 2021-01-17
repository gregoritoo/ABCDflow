import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
from kernels import *
tf.keras.backend.set_floatx('float32')
PI = m.pi
OPTIMIZER = tf.optimizers.Adamax(learning_rate=0.1)

def plot_gs(true_data,mean,X_train,X_s,stdp,stdi,color="blue"):
    plt.figure(figsize=(32,16), dpi=100)
    plt.plot(X_s,mean,color="green",label="Predicted values")
    plt.fill_between(X_s.reshape(-1,),stdp,stdi, facecolor=color, alpha=0.2,label="Conf I")
    plt.plot(X_train,true_data,color="red",label="True data")
    plt.legend()
    

def get_values(mu_s,cov_s,nb_samples=100):
    samples = np.random.multivariate_normal(mu_s,cov_s,100)
    stdp = [np.mean(samples[:,i])+1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    stdi = [np.mean(samples[:,i])-1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    mean = [np.mean(samples[:,i])for i in range(samples.shape[1])]
    return mean,stdp,stdi

@tf.function
def compute_posterior(y,cov,cov_s,cov_ss):
    mu = tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+0.01*tf.eye(cov.shape[0]))),y)
    cov = cov_ss - tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+0.01*tf.eye(cov.shape[0]))),cov_s)
    return mu,cov

@tf.function
def log_l(X,Y,params,kernel):
    if kernel=="Periodic" :
        cov = Periodic(X,Y,l=params["l"],p=params["p"],sigma=params["sigma"])+1*tf.eye(X.shape[0])
    elif kernel == "Linear" :
        cov = Linear(X,Y,c=params["c"],sigmav=params["sigmav"])+1*tf.eye(X.shape[0])
    elif kernel =="SE":
        cov = exp(X,Y,l=params["l"],sigma=params["sigma"])+ 0.001*tf.eye(X.shape[0])
    loss = -0.5*tf.matmul(tf.matmul(tf.transpose(Y),tf.linalg.inv(cov)),Y) - 0.5*tf.math.log(tf.linalg.det(cov))-0.5*X.shape[0]*tf.math.log([PI*2])
    
    return -loss

@tf.function
def log_cholesky_l(X,Y,params,kernel):
    if kernel=="Periodic" :
        cov = Periodic(X,X,l=params["l"],p=params["p"],sigma=params["sigma"])+1*tf.eye(X.shape[0])
    elif kernel == "Linear" :
        cov = Linear(X,X,c=params["c"],sigmav=params["sigmav"])+ tf.eye(X.shape[0])
    elif kernel =="SE":
        cov = exp(X,X,l=params["l"],sigma=params["sigma"]) + tf.eye(X.shape[0])
    elif kernel == "WN" :
       cov = WhiteNoise(X,X,sigma=params["sigma"]) + tf.eye(X.shape[0]) 
    _L = tf.linalg.cholesky(cov)
    _temp = tf.linalg.solve(_L, Y)
    alpha = tf.linalg.solve(tf.transpose(_L), _temp)
    loss = 0.5*tf.matmul(tf.transpose(Y),alpha) + tf.math.log(tf.linalg.trace(_L)) +0.5*X.shape[0]*tf.math.log([PI*2])
    return loss


def train_step(model,iteration,X_train,Y_train):
    with tf.GradientTape(persistent=False) as tape :
        tape.watch(model.variables)
        val = model(X_train,Y_train)
    gradient = tape.gradient(val,model.variables)
    try :
        OPTIMIZER.apply_gradients(zip(gradient, model.variables))
    except Exception as e :
        print(model.variables)
        print(e)
        print(gradient)

        OPTIMIZER.apply_gradients(gradient, model.variables)
    return val