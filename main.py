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

PI = m.pi





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
    print(dic)
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


def train(model,nb_iter,nb_restart,X_train,Y_train,kernels_name,verbose=False):
    base_model = model 
    loop = 0
    best = 10e90
    #tf.compat.v1.global_variables_initializer() 
    while loop < nb_restart :
        try :
            model = base_model
            if verbose :
                for iteration in range(1,nb_iter):
                    val = train_step(model,iteration,X_train,Y_train,kernels_name)
                    sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."*int((nb_iter-iteration)/nb_iter*50)+"|"+" * log likelihood  is : {:.4f} at epoch : {:.0f} at iteration : {:.0f} / {:.0f} ".format(val[0][0],nb_iter,loop+1,nb_restart))
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


"""def search(kernels_name,_kernel_list,init):
    kerns = tuple(KERNELS.keys())
    for kern in kerns :
        op = np.random.randint(0,1)
        if init : op = 1
        if op == 0 :
            kernels_name,_kernel_list = _mulkernel(kernels_name,_kernel_list,kern)
        elif op == 1 :
            kernels_name,_kernel_list = _addkernel(kernels_name,_kernel_list,kern)
        else :
            raise EnvironmentError("Invalid values")
        if not init :  G.add_edges_from(tuple(_kernel_list))
        else : G.add_node(kern)
    print(list(G.nodes))
    return  kernels_name,_kernel_list"""

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



    

def search_step_multipro(combi,q):
    BEST_MODELS = dict({})
    try :
        _kernel_list = list(combi)
        kernels_name = ''.join(combi)

        if kernels_name[0] != "*" :
            kernels = _preparekernel(_kernel_list)
            model=CustomModel(kernels)
            model = train(model,nb_iter,nb_restart,X_train,Y_train,_kernel_list)
            BIC = model.compute_BIC(X_train,Y_train,_kernel_list)
            BEST_MODELS["model_name"] = kernels_name
            BEST_MODELS["model_list"] = _kernel_list
            BEST_MODELS["model"] = model
            BEST_MODELS["score"] = BIC
    except Exception as e :
        print("error with kernel :",kernels_name)
        print(e)
    q.put(BEST_MODELS)


def search_step(combi,BEST_MODELS,verbose=False):
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

    except Exception as e :
        print("error with kernel :",kernels_name)
        print(e)
    return BEST_MODELS
    


def analyse(nb_restart,nb_iter):
    nb_restart = nb_restart
    nb_iter = nb_iter

    kernels_name,_kernel_list = "",[]
    COMB = search(kernels_name,_kernel_list,True)
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score":10e40}


    COMB = COMB[:15]
    nb_iter = len(COMB)
    iteration=0
    for combi in COMB :
        iteration+=1
        BEST_MODELS = search_step(combi,BEST_MODELS,False)
        sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."*int((nb_iter-iteration)/nb_iter*50)+"|"+" * model is {} ".format(combi))
        sys.stdout.flush()
    model=BEST_MODELS["model"]
    model.viewVar(kernels_name)
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,_kernel_list)))
    return model,BEST_MODELS["model_list"]




if __name__ =="__main__" :
    """Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)

    X_s_num = np.arange(0, 160, 1).reshape(-1, 1)
    X_train = tf.Variable(X,dtype=tf.float32)
    Y_train = tf.Variable(Y,dtype=tf.float32)
    
    X_s = tf.Variable(X_s_num,dtype=tf.float32)
    """

    """X_train = tf.Variable(np.array(np.linspace(0,30,30)).reshape(-1, 1),dtype=tf.float32)
    Y_train = tf.Variable(np.sin(X_train.numpy().reshape(-1, 1)),dtype=tf.float32)

    X_s = tf.Variable(np.arange(-2, 28,30).reshape(-1, 1),dtype=tf.float32)

    """

    Y = np.array(pd.read_csv("periodic.csv",sep=",")["Temp"]).reshape(-1, 1)
    X = np.arange(len(Y)).reshape(-1, 1)

    X_s_num = np.arange(0, 179, 1).reshape(-1, 1)

    X_train = tf.Variable(X,dtype=tf.float32)
    Y_train = tf.Variable(Y,dtype=tf.float32)
    
    X_s = tf.Variable(X_s_num,dtype=tf.float32)
    
    nb_restart = 15
    nb_iter = 10

    model,kernels = analyse(nb_restart,nb_iter)
    mu,cov = model.predict(X_train,Y_train,X_s,kernels)
    model.plot(mu,cov,X_train,Y_train,X_s)
    plt.show()

        
    with open('best_model', 'wb') as f :
        pickle.dump(model,f)
    
    with open('kernels', 'wb') as f :
        pickle.dump(kernels,f)

    

    """with open('best_model','rb') as f:
        model = pickle.load(f)
    
    with open('kernels', 'rb') as f:
        kernels = pickle.load(f)

    model.viewVar(''.join(kernels))
    mu,cov = model.predict(X_train,Y_train,X_s,kernels)
    model.plot(mu,cov,X_train,Y_train,X_s)
    plt.show()"""



    

