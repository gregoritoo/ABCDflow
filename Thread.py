import threading
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
tf.keras.backend.set_floatx('float64')
from itertools import chain
import itertools
import pickle 
import random
import queue
tf.compat.v1.disable_eager_execution()
class Thread_Analyse(threading.Thread):
    def __init__(self,name,nb_restart,nb_iter,COMB):
        threading.Thread.__init__(self)
        self._name = name
        self._nb_restart = nb_restart
        self._nb_iter = nb_iter
        self._COMB = COMB
        print("The new thread n° : " + str(self._name) + " start running")

    def run(self):
        model,BEST_MODELS["model_list"] = self.analyse()
        self.q.put((model,BEST_MODELS["model_list"]))
        self.stop()
        print("The thread " + str(self._name) + " ended ")
        return 0

    def stop(self):
        self._stopevent.set()
    
    def analyse(self):
        nb_restart = self._nb_restart
        BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score":10e40}
        nb_iter = len(self._COMB)
        iteration=0
        for combi in self._COMB :
            iteration+=1
            BEST_MODELS = search_step(combi,BEST_MODELS,False)
            sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."*int((nb_iter-iteration)/nb_iter*50)+"|"+" * model is {} for thread {}".format(combi,self._name))
            sys.stdout.flush()
        model=BEST_MODELS["model"]
        model.viewVar(kernels_name)
        print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,_kernel_list)))
        return model,BEST_MODELS["model_list"]


