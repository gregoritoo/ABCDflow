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
tf.keras.backend.set_floatx('float32')
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


def search_step(combi,BEST_MODELS,TEMP_BEST_MODELS,nb_by_step,verbose=False):
    NEW_TEMP_BEST_MODELS = {}
    j=0
    try :
        _kernel_list = list(combi)
        kernels_name = ''.join(combi)
        if kernels_name[0] != "*" :
            kernels = _preparekernel(_kernel_list)
            model=CustomModel(kernels)
            model = train(model,nb_iter,nb_restart,X_train,Y_train,_kernel_list,verbose=False)
            if verbose :
                print("kernel = ",kernels_name )
                model.viewVar(kernels_name)
            BIC = model.compute_BIC(X_train,Y_train,_kernel_list)
            if BIC < BEST_MODELS["score"]  : 
                BEST_MODELS["model_name"] = kernels_name
                BEST_MODELS["model_list"] = _kernel_list
                BEST_MODELS["model"] = model
                BEST_MODELS["score"] = BIC
            TEMP_BEST_MODELS.update({"model_list":kernels_name,"score":BIC})
            for key, value in sorted(TEMP_BEST_MODELS.items(), key=lambda item: item[1]):
                print(key,value)
                if j < nb_by_step :
                    NEW_TEMP_BEST_MODELS.update({key, value})
                j+=1
        return BEST_MODELS,NEW_TEMP_BEST_MODELS

def _prune(tempbest,rest):
    new_rest = []
    for _element in rest :
        for _best in tempbest :
            _element = list(_element)
            if _best == _element[:len(tuple(_best))] and _element not in new_rest :
                new_rest.append(_element)
    return new_rest


def analyse(nb_restart,nb_iter,c,verbose=False):
    nb_restart = nb_restart
    nb_iter = nb_iter
    kernels_name,_kernel_list = "",[]
    COMB = search(kernels_name,_kernel_list,True)
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score":10e40}
    COMB = COMB[:15]
    nb_iter = len(COMB)
    iteration=0
    TEMP_BEST_MODELS={}
    for combi in COMB :
        iteration+=1
        BEST_MODELS,TEMP_BEST_MODELS = search_step(combi,BEST_MODELS,TEMP_BEST_MODELS,nb_by_step,verbose)
        print(TEMP_BEST_MODELS)
        """sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."*int((nb_iter-iteration)/nb_iter*50)+"|"+" * model is {} ".format(combi))
        sys.stdout.flush()
    model=BEST_MODELS["model"]
    model.viewVar(kernels_name)
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,_kernel_list)))
    return model,BEST_MODELS["model_list"]"""