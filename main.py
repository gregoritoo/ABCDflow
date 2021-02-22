import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import math as m
import seaborn as sn
import GPy
import sys 
from kernels_utils import *
from training_utils import *
from plotting_utils import *
import pandas as pd 
from itertools import chain
import itertools
import pickle 
import contextlib
import functools
import time
import scipy 
import tensorflow as tf 
from changepoint import *
from training import *
from search import *






if __name__ =="__main__" :
    #Y = np.append(np.linspace(1,200,200),200+10*np.sin(np.linspace(1,200,200))).reshape(-1, 1)
    Y = np.array(pd.read_csv("./data/periodic.csv")["x"][:300]).reshape(-1,1)
    plt.plot(Y)
    plt.show()
    X_s = np.linspace(0,len(Y)+50,len(Y)+50).reshape(-1, 1)
    X = np.linspace(0,len(Y)-1,len(Y)).reshape(-1,1)
    t0 = time.time()
    """X = np.linspace(-10, 10, 101)[:, None].reshape(-1, 1)
    Y = np.array(np.cos( (X - 5) / 2 )**2 * X * 2 + np.random.randn(101, 1)).reshape(-1, 1)
    X_s = np.linspace(0,len(Y)+10,len(Y)+10).reshape(-1, 1)"""
    """plt.plot(Y)
    plt.show()"""
    #X_s = np.linspace(-20,20,len(X)+40).reshape(-1, 1)
    t0 = time.time()
    model,kernel= launch_analysis(X,Y,X_s,straigth=True,do_plot=True,depth=3,verbose=True,initialisation_restart=10,reduce_data=False,experimental_multiprocessing=False,GPY=False,use_changepoint=False) #straight parameters == True
    print('time took: {} seconds'.format(time.time()-t0))
    model.describe(kernel)
    mu,cov = model.predict(X,Y,X_s,kernel)
    model.plot(mu,cov,X,Y,X_s,kernel)
    #model.plot()
    plt.show()
    model.decompose(kernel,X,Y,X_s)

