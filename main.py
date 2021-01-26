import numpy as np 
import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
from search import *
from changepoint import *
from training import *











if __name__ =="__main__" :

    #Y = np.sin(np.linspace(0,100,100)).reshape(-1,1)
    #X = np.linspace(0,100,100).reshape(-1, 1)
    #Y = 3*(np.sin(X)).reshape(-1, 1)
    Y_a = np.array(pd.read_csv("./data/periodic.csv")["x"]).reshape(-1, 1)
    Y = Y_a[:-30]
    X = np.linspace(0,len(Y),len(Y)).reshape(-1,1)
    X_s = np.linspace(0,len(Y)+60,len(Y)+60).reshape(-1, 1)
    t0 = time.time()
    model,kernel = single_model(X,Y,X_s,["+PER","+LIN"],nb_restart=1,nb_iter=400,verbose=True,initialisation_restart=20,reduce_data=False,OPTIMIZER=tf.optimizers.RMSprop(0.01))
    #model,kernel = launch_analysis(X,Y,X_s,prune=False,straigth=True,depth=3,nb_restart=1,verbose=False,nb_iter=100,initialisation_restart=30,reduce_data=False,OPTIMIZER=tf.optimizers.RMSprop(0.01))
    mu,cov = model.predict(X,Y,X_s,kernel)
    model.plot(mu,cov,X,Y,X_s,kernel)
    plt.show()
    """model,kernel = single_model(X,Y,X_s,["+LIN","+PER"],nb_restart=1,nb_iter=100,verbose=True,initialisation_restart=5,reduce_data=False,OPTIMIZER=tf.optimizers.RMSprop(0.01))
    #model,kernel = launch_analysis(X,Y,X_s,prune=False,straigth=True,depth=3,nb_restart=1,verbose=False,nb_iter=100,initialisation_restart=5,reduce_data=False,OPTIMIZER=tf.optimizers.RMSprop(0.01))
    print('time took: {} seconds'.format(time.time()-t0))
    mu,cov = model.predict(X,Y,X_s,kernel)
    model.plot(mu,cov,X,Y,X_s,kernel)
    plt.show()
    t0 = time.time()
    k = GPy.kern.Linear(input_dim=1) + GPy.kern.StdPeriodic(input_dim=1)
    m = GPy.models.GPRegression(X, Y, k, normalizer=False)
    m.optimize_restarts(20)
    print('time took: {} seconds'.format(time.time()-t0))
    print(m)
    m.plot()
    plt.show()"""
    print('time took: {} seconds'.format(time.time()-t0))
    """k =( GPy.kern.RatQuad(input_dim=1) * GPy.kern.StdPeriodic(input_dim=1))*GPy.kern.Exponential(input_dim=1)
    m = GPy.models.GPRegression(X, Y, k, normalizer=False)
    m.optimize()
    print('time took: {} seconds'.format(time.time()-t0))
    print(m)
    m.plot()"""
    """HISTORY = pd.DataFrame(columns=["learning_rate","score"])
    for lr in lr_list :
        try :
            model,kernel = single_model(X,Y,X_s,["+RQ","+PER","*SE"],nb_restart=25,nb_iter=20,verbose=False,initialisation_restart=3,reduce_data=False,OPTIMIZER=tf.optimizers.RMSprop(learning_rate=lr))
            mu,cov = model.predict(X,Y,X_s,kernel)
            mean,_,_ = get_values(mu.numpy(),cov.numpy(),nb_samples=100)
            HISTORY.loc[len(HISTORY)+1]=[lr,mse(mean.reshape(-1)[-30 :],Y_a[-30:])]
        except Exception as e:
            print(e)
            HISTORY.loc[len(HISTORY)+1]=[lr,mse(mean[-30 :],Y_a[-30:])] 

    HISTORY.to_csv("./optimization_results/RMSprop_airline.csv")
    plt.plot(HISTORY["learning_rate"],HISTORY["score"])
    plt.show()"""
 

    """model,kernel = single_model(X,Y,X_s,["+RQ","+PER"],nb_restart=15,nb_iter=5,verbose=False,initialisation_restart=3,reduce_data=False,OPTIMIZER=tf.optimizers.Adamax(learning_rate=0.6))
    #model,kernel = launch_analysis(X,Y,X_s,prune=False,straigth=True,depth=5,nb_restart=50,nb_iter=5,initialisation_restart=5,reduce_data=False)
    print('time took: {} seconds'.format(time.time()-t0))
    mu,cov = model.predict(X,Y,X_s,kernel)
    mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=100)
    print(mse(Y[-30:],mean[:30]))"""
    """model,kernel = launch_analysis(X,Y,X_s,prune=False,reduce_data=False)
    mu,cov = model.predict(X,Y,X_s,kernel)
    model.plot(mu,cov,X,Y,X_s,kernel)
    plt.show()"""
    """model,kernel = launch_analysis(X,Y,X_s,nb_restart=11,nb_iter=5,reduce_data=False)
    mu,cov = model.predict(X,Y,X_s,kernel)
    model.plot(mu,cov,X,Y,X_s,kernel)
    plt.show()"""


    """Y = np.array(pd.read_csv("./data/periodic.csv",sep=",")["x"]).reshape(-1, 1)
    dic = changepoint_detection(Y,percent=0.05,plot=True,num_c=4)
    print(dic)
    #Y = Y[:300]
    X = np.linspace(0,len(Y),len(Y)).reshape(-1,1)
    X_s = np.linspace(0,len(Y)+40,len(Y)+41).reshape(-1, 1)
    model,kernels = launch_analysis(X,Y,X_s,prune=False,reduce_data=False)
    mu,cov = model.predict(X,Y,X_s,kernels)
    model.plot(mu,cov,X,Y,X_s,kernels)
    plt.show()"""
    """t0 = time.time()
    best_mods, all_mods, all_exprs, expanded, not_expanded = Models.modelSearch.explore_model_space(X, Y)
    print('time took: {} seconds'.format(time.time()-t0))
    preds = best_mods[0].predict(Y_train)
    m = best_mods[0]
    m.model.plot()
    plt.show()"""
    """X = np.arange(len(Y)).reshape(-1, 1)
    X_s = np.arange(0,len(Y)+20, 1).reshape(-1, 1)
    model,kernels = launch_analysis(X,Y,X_s,prune=True,nb_by_step=10,nb_restart=11,nb_iter=5)
    mu,cov = model.predict(X_train,Y_train,X_s,kernel)
    model.plot(mu,cov,X_train,Y_train,X_s,kernel)
    plt.show()"""
    """plt.plot(Y)
    plt.show()

    segment = cut_signal(Y)
    print(segment)
    """
    
    
    """Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)
    X_s = np.arange(0, 179, 1).reshape(-1, 1)
    X = np.linspace(0,100,100).reshape(-1, 1)
    Y = 3*(np.sin(X)).reshape(-1, 1)
    X_s = np.arange(-30, 130, 1).reshape(-1, 1)
    X_train,Y_train,X_s = tf.Variable(X,dtype=_precision),tf.Variable(Y,dtype=_precision),tf.Variable(X_s,dtype=_precision)
    model,kernels = launch_analysis(X,Y,X_s)
    """
    """Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)
    X_s = np.arange(0, 179, 1).reshape(-1, 1)
    X_train,Y_train,X_s = tf.Variable(X,dtype=_precision),tf.Variable(Y,dtype=_precision),tf.Variable(X_s,dtype=_precision)
    #X_train,Y_train,X_s = tf.Variable(X,dtype=_precision),tf.Variable(Y,dtype=_precision),tf.Variable(X_s,dtype=_precision)
    model,kernel = single_model(X,Y,X_s,['+PER',"*LIN"],nb_restart=50,nb_iter=10,verbose=False)
    mu,cov = model.predict(X_train,Y_train,X_s,kernel)
    model.plot(mu,cov,X_train,Y_train,X_s,kernel)
    plt.show()
    t0 = time.time()
    k = GPy.kern.StdPeriodic(input_dim=1) * GPy.kern.Linear(input_dim=1)
    m = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
    m.optimize_restarts(20)
    print('time took: {} seconds'.format(time.time()-t0))
    print(m)
    m.plot()
    plt.show()"""
    """
    #### Loading model ##########
    with open('best_model','rb') as f:
        model = pickle.load(f)
    with open('kernels','rb') as f:
        kernel = pickle.load(f)
    model.viewVar(kernel)
    mu,cov = model.predict(X_train,Y_train,X_s,kernel)
    model.plot(mu,cov,X_train,Y_train,X_s,kernel_name =kernel)
    plt.show()
    model,kernels = launch_analysis(X,Y,X_s)
    print('time took: {} seconds'.format(time.time()-t0))
    mu,cov = model.predict(X_train,Y_train,X_s,kernels)
    model.plot(mu,cov,X_train,Y_train,X_s)
    plt.show()
    t0 = time.time()
    k = GPy.kern.StdPeriodic(input_dim=1) 
    m = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
    m.optimize_restarts(20)
    print('time took: {} seconds'.format(time.time()-t0))
    print(m)
    m.plot()
    plt.show()
    k = GPy.kern.StdPeriodic(input_dim=1) 
    m = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
    m.optimize_restarts(15)
    print('time took: {} seconds'.format(time.time()-t0))
    print(m)
    m.plot()
    plt.show()"""
    
    
    
    

        