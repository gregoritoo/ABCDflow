import numpy as np 
import tensorflow as tf 
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
import os 
import multiprocessing
from multiprocessing import Pool
import sys 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float32')

PI = m.pi

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)



KERNELS_LENGTH = {
    "LIN" : 1,
    "WN" : 1,
    "SE" : 2,
    "PER" :3,
}

KERNELS = {
    "LIN" : {"parameters_lin":["lin_c"]},
    "WN" : {"parameters":["white_sigma"]},
    "SE" : {"parameters":["squaredexp_l","squaredexp_sigma"]},
    "PER" : {"parameters_per":["periodic_l","periodic_p","periodic_sigma"]},
}


KERNELS_OPS = {
    "*LIN" : "mul",
    "*WN" : "mul",
    "*SE" : "mul",
    "*PER" :"mul",
    "+LIN" : "add",
    "+WN" : "add",
    "+SE" : "add",
    "+PER" : "add",
}



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
            else :
                kernels.update({key:[element for element in value]})
            i+=1
    return kernels




def train(model,nb_iter,nb_restart,X_train,Y_train,kernels_name,verbose=True):
    base_model = model 
    loop = 0
    best = 10e90
    while loop < nb_restart :
        try :
            model = base_model
            if verbose :
                for iteration in range(1,nb_iter):
                    val = train_step(model,iteration,X_train,Y_train,kernels_name)
                    sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."*int((nb_iter-iteration)/nb_iter*50)+"|"+" * log likelihood  is : {:.4f} at iteration : {:.0f} at epoch : {:.0f} / {:.0f} ".format(val[0][0],nb_iter,loop+1,nb_restart))
                    sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
            else :
                for iteration in range(1,nb_iter):
                    val = train_step(model,iteration,X_train,Y_train,kernels_name)
        except Exception as e :
            print(e)
        if val  < best :
            best =  val
            best_model = model
        loop += 1
    return best_model



def search(kernels_name,_kernel_list,init):
    kerns = tuple((KERNELS_OPS.keys()))
    COMB = []
    for i in range(1,5) :
        if i == 1 : combination =  list(itertools.combinations(kerns, i))
        else : combination = list(itertools.permutations(kerns, i)) 
        for comb in combination :
            if not comb[0][0] == "*" : 
                COMB.append(comb)
    return COMB


def _prune(tempbest,rest):
    print("Prunning ...")
    new_rest = []
    for _element in rest :
        for _best in tempbest :
            _element = list(_element)
            if _best == _element[:len((_best))] and _element not in new_rest :
                new_rest.append(_element)
    return new_rest

    



def search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose):
    j=0
    try :
        _kernel_list = list(combi)
        kernels_name = ''.join(combi)
        if kernels_name[0] != "*" :
            kernels = _preparekernel(_kernel_list)
            model=CustomModel(kernels)
            model = train(model,nb_iter,nb_restart,X_train,Y_train,_kernel_list,verbose)
            if verbose :
                model.viewVar(kernels_name)
            BIC = model.compute_BIC(X_train,Y_train,_kernel_list)
            if BIC < BEST_MODELS["score"]  : 
                BEST_MODELS["model_name"] = kernels_name
                BEST_MODELS["model_list"] = _kernel_list
                BEST_MODELS["model"] = model
                BEST_MODELS["score"] = BIC 
            TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],int(BIC.numpy()[0])]                                  ################ a supprimer pour corriger , il faut consr=erver nom et score puis découper en utilisant prune
    except Exception as e:
        print("error with kernel :",kernels_name)
        print(e)
    if prune :
        return BEST_MODELS,TEMP_BEST_MODELS
    else :
        return BEST_MODELS


def analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,verbose):
    if i == -1 :
        COMB = search("",[],True)[:10]
    else :
        name = "search/model_list_"+str(i)
        with open(name, 'rb') as f :
            COMB = pickle.load(f)
    kernels_name,_kernel_list = "",[]
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score":10e40}
    TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
    iteration=0
    full_length = len(COMB)
    for combi in COMB :
        iteration+=1
        if prune :
            if iteration % 50 == 0 :
                TEMP_BEST_MODELS = TEMP_BEST_MODELS[: nb_by_step]
                COMB = _prune(TEMP_BEST_MODELS["Name"].tolist(),COMB[iteration :])
                #print("Model to try :",COMB)
            BEST_MODELS,TEMP_BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose)
            TEMP_BEST_MODELS = TEMP_BEST_MODELS.sort_values(by=['score'],ascending=True)
            sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        else :  
            BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose)
            sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    model=BEST_MODELS["model"]
    model.viewVar(kernels_name)
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,_kernel_list)))
    return model,BEST_MODELS["model_list"]




def parralelize(X_train,Y_train,X_s,nb_workers,nb_restart,nb_iter,nb_by_step):
    multiprocessing.set_start_method('spawn', force=True)
    COMB = search("",[],True)
    poll_list=[]
    for i in range(nb_workers) :
        COMB_ = COMB[int(i*len(COMB)/nb_workers):int(i+1*len(COMB)/nb_workers)]
        name = "search/model_list_"+str(i)
        with open(name, 'wb') as f :
            pickle.dump(COMB_,f)
    i = 0
    params = [(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i) for i in range(nb_workers)]
    with Pool(nb_workers) as pool: 
        pool.starmap(analyse,params)


def launch_analysis(X_train,Y_train,X_s,nb_restart=15,nb_iter=2,do_plot=True,save_model=False,prune=False,verbose=False,nb_by_step=None,nb_workers=None,experimental_multiprocessing=False):
    if prune and nb_by_step is None : raise ValueError("As prune is True you need to precise nb_by_step")        
    X_train,Y_train,X_s = tf.Variable(X,dtype=tf.float32),tf.Variable(Y,dtype=tf.float32),tf.Variable(X_s_num,dtype=tf.float32)
    if not experimental_multiprocessing :
        i=-1
        model,kernels = analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,verbose)
        name,name_kernel = 'best_model',kernels
        if save_model :
            with open(name, 'wb') as f :
                pickle.dump(model,f)
            with open(name_kernel, 'wb') as f :
                pickle.dump(kernels,f)
        if do_plot :
            mu,cov = model.predict(X_train,Y_train,X_s,kernels)
            model.plot(mu,cov,X_train,Y_train,X_s)
            plt.show()
    elif experimental_multiprocessing :
        print("This is experimental, it is slower than monoprocessed")
        if nb_workers is None : 
            raise ValueError("Number of workers should be precise")
        parralelize(X_train,Y_train,X_s,nb_workers,nb_restart,nb_iter,nb_by_step)
    return model,kernels

    



if __name__ =="__main__" :
    Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)
    X_s_num = np.arange(0, 179, 1).reshape(-1, 1)
    model,kernels = launch_analysis(X,Y,X_s_num)
    
    

    
    
    

        
    

    