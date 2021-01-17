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

PI = m.pi


def train(model,nb_iter,nb_restart):
    base_model = model 
    loop = 0
    best = 10e90
    #tf.compat.v1.global_variables_initializer() 
    while loop < nb_restart :
        try :
            model = base_model
            for iteration in range(1,nb_iter):
                val = train_step(model,iteration,X_train,Y_train)
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
    Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)

    X_s_num = np.arange(0, 160, 1).reshape(-1, 1)

    X_train = tf.Variable(X,dtype=tf.float32)
    Y_train = tf.Variable(Y,dtype=tf.float32)
    
    X_s = tf.Variable(X_s_num,dtype=tf.float32)
    mu = tf.Variable(tf.zeros((1,X_train.shape[0])),dtype=tf.float32)

    nb_restart = 10
    nb_iter = 100
    model = LinearRegressor()
    model = train(model,nb_iter,nb_restart)
    model.viewVar()

    k = GPy.kern.Linear(input_dim=1)       
    m = GPy.models.GPRegression(X_train.numpy(), X_train.numpy(), k, normalizer=False)
    m.optimize_restarts(10)
    print(m)
    m.plot()
    plt.show()
    mu,cov = model.predict(X_train,Y_train,X_s)    
    mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=1000)
    plot_gs(Y_train.numpy(),mean,X_train.numpy(),X_s.numpy(),stdp,stdi)
    plt.show()

