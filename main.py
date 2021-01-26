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
import bocd
import ruptures as rpt
from pyinform.blockentropy import block_entropy

def mse(y,ypred):
    return np.mean(np.square(y-ypred))

PI = m.pi
_precision = tf.float64
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

lr_list = np.linspace(0.001,1,101)
borne = -1*10e40

KERNELS_LENGTH = {
    "LIN" : 3,
    "SE" : 2,
    "PER" :3,
    #"CONST" : 1,
    "WN" : 1,
    #"RQ" : 3,
}

KERNELS = {
    "LIN" : {"parameters_lin":["lin_c","lin_sigmav","lin_sigmab"]},
    #"CONST" : {"parameters":["const_sigma"]},
    "SE" : {"parameters":["squaredexp_l","squaredexp_sigma"]},
    "PER" : {"parameters_per":["periodic_l","periodic_p","periodic_sigma"]},
    "WN" : {"paramters_Wn":["white_noise_sigma"]}
    #"RQ" : {"parameters_rq":["rq_l","rq_sigma","rq_alpha"]},
}


KERNELS_OPS = {
    "*LIN" : "mul",
    "*SE" : "mul",
    "*PER" :"mul",
    "+LIN" : "add",
    "+SE" : "add",
    "+PER" : "add",
    #"+CONST" :"add",
    #"*CONST" : "mul",
    "+WN" :"add",
    "*WN" : "mul",
    #"+RQ" : "add",
    #"*RQ" : "mul",
}




def _mulkernel(kernels_name,_kernel_list,new_k):
    ''' 
        Add  new kernel to the names and in the kernel list with a * 
    inputs :
        kernels_name :  string, name of the full kernel  ex +LIN*PER
        _kernels_list : list of string, list of kernels with theirs according operation ex ["+LIN","*PER"]
        new_k : string, name of kernel to add with it according operation ex *PER
    outputs :
        kernels_name :  string, updated kernel name   ex +LIN*PER*PER
        _kernels_list : list of string, list of updated kernels ex ["+LIN","*PER","*PER"]
    '''
    kernels_name = "("+kernels_name+")" + "*"+new_k
    _kernel_list = _kernel_list + ["*"+new_k]
    return kernels_name,_kernel_list

def _addkernel(kernels_name,_kernel_list,new_k):
    ''' 
        Add  new kernel to the names and in the kernel list with a +
    inputs :
        kernels_name :  string, name of the full kernel  ex +LIN*PER
        _kernels_list : list of string, list of kernels with theirs according operation ex ["+LIN","*PER"]
        new_k : string, name of kernel to add with it according operation ex +PER
    outputs :
        kernels_name :  string, updated kernel name   ex +LIN*PER+PER
        _kernels_list : list of string, list of updated kernels ex ["+LIN","*PER","+PER"]
    '''
    kernels_name = kernels_name + "+"+new_k
    _kernel_list = _kernel_list + ["+"+new_k]
    return kernels_name,_kernel_list

def _preparekernel(_kernel_list):
    '''
        Receive the list of kernels with theirs operations and return a dict with the kernels names and parameters 
    inputs :
        _kernels_list : list of string, list of kernels with theirs according operation
    outputs :
        kernels : dict, dict cotaining kernel parameters (see KERNELS)
    '''
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



def train(model,nb_iter,nb_restart,X_train,Y_train,kernels_name,OPTIMIZER,verbose=True,mode="f"):
    '''
        Train the model according to the parameters 
    inputs :
        model : CustomModel object , Gaussian process model
        nb_iter : int, number of iterations during the training
        nb_restart : int, retrain on same data (epoch)
        X_train : Tensor, Training X
        Y_train : Tensor, Training Y
        kernels_name : string, updated kernel name   ex +LIN*PER*PER
        OPTIMIZER : tf optimizer object 
        verbose : Bool, print training process
        mode : string , training mode 
    outputs:
        best_model : dict, dictionnary containing the best model and it score
    '''
    best = borne
    loop,base_model = 0,model
    lr = 0.1
    old_val,val,lim = 0,0,1.5
    if mode == "SGD" :
        while loop < nb_restart :
            try :
                model = base_model
                if verbose :
                    for iteration in range(0,nb_iter):
                        if loop > 10 :
                            OPTIMIZER.learning_rate.assign(0.001) 
                        val = train_step(model,iteration,X_train,Y_train,kernels_name,OPTIMIZER)
                        if np.isnan(val) :
                            loop += 1
                            break 
                        sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."* int((nb_iter-iteration)/nb_iter*50)+"|" \
                            +" * log likelihood  is : {:.4f} at iteration : {:.0f} at epoch : {:.0f} / {:.0f} with lr of: {}".format(val[0][0],iteration,loop+1,nb_restart,lr))
                        sys.stdout.flush()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                else :
                    for iteration in range(1,nb_iter):
                        val = train_step(model,iteration,X_train,Y_train,kernels_name)
            except Exception as e :
                print(e)
                loop += 1
            if val  < best :
                best =  val
                best_model = model
            loop += 1
    else :
        while loop < nb_restart :
            try :
                #results = train_step_lfgbs(X_train,Y_train,model._opti_variables,kernels_name)
                func = function_factory(model, log_cholesky_l_test, X_train, Y_train,model._opti_variables,kernels_name)
                init_params = tf.dynamic_stitch(func.idx, model._opti_variables)
                results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params,tolerance=1e-8)
                best_model = model
            except Exception as e:
                print(e)
            loop+=1
    return best_model





def search(kernels_name,_kernel_list,init):
    '''
        Return all the possible combinaison of kernels starting with a '+' 
    inputs :
        kernels_name :  string, not used 
        _kernel_list : list of tuples, not used 
        init : Bool, not used 
    '''
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
    '''
        Cut the graph in order to keep only the combinaison that correspond to the best for the moment 
    inputs :
        tempbest : list ,  names of best combinaisons of kernels already trained 
        rest : list of tuples , all the combinaisons of kernels not trained yet 
    outputs :
        new_rest : list of tuples , all the combinaisons to be trained 
    '''
    print("Prunning {} elements".format(len(rest)))
    new_rest = []
    for _element in rest :
        for _best in tempbest :
            _element = list(_element)
            if _best == _element[:len((_best))] and _element not in new_rest :
                new_rest.append(_element)
    print("Prunning ended, still {} kernels to try".format(len(new_rest)))
    return new_rest

    



def search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter, \
                                        nb_by_step,prune,verbose,OPTIMIZER,unique=False,single=False,initialisation_restart=5):
    '''
        Launch the training of a gaussian process
    inputs :
        X_train : Tensor, Training X
        Y_train : Tensor, Training Y
        BEST_MODELS : dict, dictionnary containing the best model and it score
        TEMP_BEST_MODELS :  dict, dictionnary containing temporaries bests models and theirs score
        nb_iter : int, number of iterations during the training
        nb_by_step : int, number of best model to keep when prune is true 
        nb_restart : int, retrain on same data (epoch)
        kernels_name : string, updated kernel name   ex +LIN*PER*PER
        OPTIMIZER : tf optimizer object 
        verbose : Bool, print training process
        mode : string , training mode 
        unique : Bool, if only one kernel in the list ex +PER
        single : Bool, 
        initialisation_restart : int, number of restart training with different initiatlisation parameters
    outputs:
        BEST_MODELS : dict, dictionnary containing the best model and it score
    '''
    """X_full = X_train
    Y_full = Y_train"""
    j=0
    lr = 0.1
    init_values = BEST_MODELS["init_values"] 
    try :
        if not unique : _kernel_list = list(combi)
        else : _kernel_list = list([combi])
        if single : _kernel_list = combi
        kernels_name = ''.join(combi)
        true_restart = 0
        kernels = _preparekernel(_kernel_list)
        if kernels_name[0] != "*" :
            while true_restart < initialisation_restart :
                try :
                    model=CustomModel(kernels,init_values)
                    model = train(model,nb_iter,nb_restart,X_train,Y_train,_kernel_list,OPTIMIZER,verbose)
                    BIC = model.compute_BIC(X_train,Y_train,_kernel_list)
                    """mu,cov = model.predict(X_train,Y_train,X_full,_kernel_list)
                    try :
                        mean,_,_= get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=100)
                    except Exception as e:
                        print(e)
                        break
                    BIC = mse(X_full.numpy()[-30:],mean[-30:])
                    print(BIC)"""
                    if  BIC > BEST_MODELS["score"]  : 
                        BEST_MODELS["model_name"] = kernels_name
                        BEST_MODELS["model_list"] = _kernel_list
                        BEST_MODELS["model"] = model
                        BEST_MODELS["score"] = BIC 
                        BEST_MODELS["init_values"] =  model.initialisation_values
                    TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],float(BIC[0][0])]  
                    #TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],BIC]  
                    true_restart += 1     
                except Exception :
                    pass                    
    except Exception as e:
        print("error with kernel :",kernels_name)
        print(e)
    if prune :
        return BEST_MODELS,TEMP_BEST_MODELS
    else :
        return BEST_MODELS


def analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,initialisation_restart):
    '''
        Compare models for each step of the training, and keep the best model
    inputs :
        X_train : Tensor, Training X
        Y_train : Tensor, Training Y
        nb_iter : int, number of iterations during the training
        nb_by_step : int, number of best model to keep when prune is true 
        nb_restart : int, retrain on same data (epoch)
        kernels_name : string, updated kernel name   ex +LIN*PER*PER
        OPTIMIZER : tf optimizer object 
        i :  int;  kernel position in the list 
        verbose : Bool, print training process
        mode : string , training mode 
        loop_size : int, number of testing to do before prunning 
    outputs:
        model : CustomModel object, best model
        BEST_MODELS["model_list"] : list, array of best model
    '''
    if i == -1 :
        COMB = search("",[],True)
    else :
        name = "search/model_list_"+str(i)
        with open(name, 'rb') as f :
            COMB = pickle.load(f)
    kernels_name,_kernel_list = "",[]
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score": borne,"init_values":None}
    TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
    iteration=0
    j = 0
    full_length = len(COMB)
    while len(COMB) > 2 :
        try : combi = COMB[j]
        except Exception as e :break
        iteration+=1
        j+=1
        if prune :
            if iteration % loop_size == 0 :
                j=0
                TEMP_BEST_MODELS = TEMP_BEST_MODELS[: nb_by_step]
                _before_len = len(COMB)
                COMB = _prune(TEMP_BEST_MODELS["Name"].tolist(),COMB[iteration :])
                _to_add = _before_len - len(COMB)-1
                iteration += _to_add
            BEST_MODELS,TEMP_BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS, \
                                                                nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER,initialisation_restart=initialisation_restart)
            TEMP_BEST_MODELS = TEMP_BEST_MODELS.sort_values(by=['score'],ascending=True)[:nb_by_step]
            sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        else :  
            COMB,j = COMB[1 :],0
            BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart)
            sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    model=BEST_MODELS["model"]
    model.viewVar(BEST_MODELS["model_list"])
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,BEST_MODELS["model_list"])))
    return model,BEST_MODELS["model_list"]


def single_model(X_train,Y_train,X_s,kernel,OPTIMIZER=tf.optimizers.Adam(learning_rate=0.001),nb_restart=7,nb_iter=4,verbose=False,initialisation_restart=2,reduce_data=False,do_plot=False):
    """
        Train an process without the search process
    inputs :
        X_train : Tensor, Training X
        Y_train : Tensor, Training Y
        X_s :  Tensor, points to predict 
        nb_iter : int, number of iterations during the training
        nb_by_step : int, number of best model to keep when prune is true 
        nb_restart : int, retrain on same data (epoch)
        kernels : string, updated kernel name   ex +LIN*PER*PER
        OPTIMIZER : tf optimizer object  
        verbose : Bool, print training process
        reduce_data : Bool, whitenning centering data before processing
        do_plot :  Bool, plot prediction after training
    outputs:
        model : CustomModel object, best model
        kernel : list, array of best model

    """
    X_train,Y_train,X_s = tf.Variable(X,dtype=_precision),tf.Variable(Y,dtype=_precision),tf.Variable(X_s,dtype=_precision)
    if reduce_data :
            mean, var = tf.nn.moments(X_train,axes=[0])
            X_train = (X_train - mean) / var
            mean, var = tf.nn.moments(Y_train,axes=[0])
            Y_train = (Y_train - mean) / var
            mean, var = tf.nn.moments(X_s,axes=[0])
            X_s = (X_s - mean) / var
    assert kernel[0][0] == "+" , "First kernel of the list must start with + "
    iteration = 1
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score": borne,"init_values":None}
    TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
    full_length= 1
    BEST_MODELS = search_step(X_train=X_train,Y_train=Y_train,X_s=X_s,combi=kernel,BEST_MODELS=BEST_MODELS, \
        TEMP_BEST_MODELS=TEMP_BEST_MODELS,nb_restart=nb_restart,nb_iter=nb_iter,verbose = verbose,OPTIMIZER=OPTIMIZER,nb_by_step=None,prune=False,unique=True,single=True,initialisation_restart=initialisation_restart)
    sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(kernel))
    sys.stdout.flush()
    model=BEST_MODELS["model"]
    model.viewVar(kernel)
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,kernel)))
    mu,cov = model.predict(X_train,Y_train,X_s,kernel)
    if do_plot : model.plot(mu,cov,X_train,Y_train,X_s,kernel)
    return model,kernel



def parralelize(X_train,Y_train,X_s,nb_workers,nb_restart,nb_iter,nb_by_step):
    ''' 
        Create a pool of nb_workers workers
    '''
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


def launch_analysis(X_train,Y_train,X_s,nb_restart=15,nb_iter=2,do_plot=True,save_model=False,prune=False,OPTIMIZER= tf.optimizers.Adam(0.001), \
                        verbose=False,nb_by_step=None,loop_size=50,nb_workers=None,experimental_multiprocessing=False,reduce_data=False,straigth=True,depth=5,initialisation_restart=5):
    '''
        Launch the analysis
    inputs :
        X_train : Tensor, Training X
        Y_train : Tensor, Training Y
        nb_iter : int, number of iterations during the training
        nb_by_step : int, number of best model to keep when prune is true 
        prune : Bool, keep only nb_by_step best models at each loop_size step 
        nb_restart : int, retrain on same data (epoch)
        do_plot : Bool, plot the mean,cov from the best model 
        save_model : Bool, save the model in the best_model/ directory 
        OPTIMIZER : tf optimizer object 
        i :  int;  kernel position in the list 
        verbose : Bool, print training process
        mode : string , training mode 
        loop_size : int, number of testing to do before prunning 
        nb_workers : int, number of wokers 
        experimental_multiprocessing : Bool, launch multiprocessing training
        straigth : Bool, keep only the best model at eatch step 
        depth : Number of kernel to use ex depth=2 => +PER*LIN 
        initialisation_restart : int, number of restart training with different initiatlisation parameters
    outputs:
        model : CustomModel object, best model
        kernels : list, array of best model
    '''
    if prune and nb_by_step is None : raise ValueError("As prune is True you need to precise nb_by_step")
    if nb_by_step is  not None and nb_by_step > loop_size : raise ValueError("Loop size must be superior to nb_by_step")   
    if not straigth : print("You chooosed straightforward training")
    X_train,Y_train,X_s = tf.Variable(X,dtype=_precision),tf.Variable(Y,dtype=_precision),tf.Variable(X_s,dtype=_precision)
    if reduce_data :
            mean, var = tf.nn.moments(X_train,axes=[0])
            X_train = (X_train - mean) / var
            mean, var = tf.nn.moments(Y_train,axes=[0])
            Y_train = (Y_train - mean) / var
            mean, var = tf.nn.moments(X_s,axes=[0])
            X_s = (X_s - mean) / var
    t0 = time.time()
    if straigth :
        i=-1
        model,kernels = straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,depth,initialisation_restart)
        if do_plot :
            mu,cov = model.predict(X_train,Y_train,X_s,kernels)
            model.plot(mu,cov,X_train,Y_train,X_s,kernels)
            plt.show()
        return model,kernels
    if not experimental_multiprocessing :
        i=-1 
        model,kernels = analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,initialisation_restart)
        name,name_kernel = './best_models/best_model', kernels
        if save_model :
            with open(name, 'wb') as f :
                pickle.dump(model,f)
            with open(name_kernel, 'wb') as f :
                pickle.dump(kernels,f)
        print("Training ended.Took {} seconds".format(time.time()-t0))
        if do_plot :
            mu,cov = model.predict(X_train,Y_train,X_s,kernels)
            model.plot(mu,cov,X_train,Y_train,X_s,kernels)
            plt.show()
        return model,name_kernel
    elif experimental_multiprocessing :
        print("This is experimental, it is slower than monoprocessed !")
        if nb_workers is None : 
            raise ValueError("Number of workers should be precise")
        parralelize(X_train,Y_train,X_s,nb_workers,nb_restart,nb_iter,nb_by_step)
        return model,kernels


def cut_signal(signal):

    bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(300), bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))
    rt_mle = np.empty(signal.shape)
    for i, d in enumerate(signal):
        bc.update(d)
        rt_mle[i] = bc.rt
    index_changes = np.where(np.diff(rt_mle)<0)[0]
    return index_changes

def changepoint_detection(ts,percent=0.05,plot=True,num_c=4):
    length = len(ts)
    bar = int(percent*length)
    ts = np.array(ts) [bar:-bar]
    min_val,model = length, "l1" 
    algo = rpt.Dynp(model="normal").fit(np.array(ts))
    dic = {"best":[0,length]}
    try :
        for i in range(num_c) :
            my_bkps = algo.predict(n_bkps=i)
            if plot :
                rpt.show.display(np.array(ts), my_bkps, figsize=(10, 6))
                plt.show()
            start_borne,full_entro = 0,0
            for end_borne in my_bkps :
                val = block_entropy(ts[start_borne:end_borne], k=1)   
                full_entro = val + full_entro
                start_borne = end_borne
            if full_entro == 0 : break
            elif full_entro < min_val :
                min_val = full_entro
                dic["best"] = [0]+my_bkps
            else : pass 
    except Exception as e :
        print(e)
        print("Not enough point")
        return {"best":[0,length]}
    return dic

def straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,depth=10,initialisation_restart=5):
    """
        FInd best combinaison of kernel that descrive the training data , keep one best model at each step 
    inputs :
        X_train : Tensor, Training X
        Y_train : Tensor, Training Y
        X_s :  Tensor, points to predict 
        nb_iter : int, number of iterations during the training
        nb_by_step : int, number of best model to keep when prune is true 
        nb_restart : int, retrain on same data (epoch)
        kernels : string, updated kernel name   ex +LIN*PER*PER
        i :  int;  kernel position in the list 
        verbose : Bool, print training process
        loop_size : int, number of testing to do before prunning 
        OPTIMIZER : tf optimizer object  
        verbose : Bool, print training process
        reduce_data : Bool, whitenning centering data before processing
        do_plot :  Bool, plot prediction after training
        depth : Number of kernel to use ex depth=2 => +PER*LIN 
        initialisation_restart : int, number of restart training with different initiatlisation parameters
    outputs:
        model : CustomModel object, best model
        kernel : list, array of best model

    """
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score": borne,"init_values":None}
    kerns = tuple((KERNELS_OPS.keys()))
    COMB,count = [],0
    combination =  list(itertools.combinations(kerns, 1))
    train_length = depth*len(KERNELS) + len(KERNELS)/2
    for comb in combination :
        if comb[0][0] != "*" : COMB.append(comb)
    for loop in range(1,depth) :
        TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
        count += 1
        loop += 1
        if loop > 1 :
            COMB = search_and_add(tuple(BEST_MODELS["model_list"]))
            print(COMB)
        iteration=0
        j = 0
        while j <  len(COMB) :
            try : combi = COMB[j]
            except Exception as e :break
            iteration+=1
            j+=1
            TEMP_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart)
            sys.stdout.write("\r"+"="*int(count/train_length*50)+">"+"."*int((train_length-count)/train_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
        if TEMP_MODELS["score"] < BEST_MODELS["score"] :
            BEST_MODELS = TEMP_MODELS
        print("The best model is {} at layer {}".format(BEST_MODELS["model_list"],loop-1))
    model=BEST_MODELS["model"]
    model.viewVar(BEST_MODELS["model_list"])
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,BEST_MODELS["model_list"])))
    return model,BEST_MODELS["model_list"]


import numpy as np
from GPy_ABCD import *
from GPy_ABCD import Models
import pandas as pd 


def search_and_add(_kernel_list):
    ''' 
        Return all possible combinaison for one step
    inputs :
        _kernels_list : list, not used 
    '''
    kerns = tuple((KERNELS_OPS.keys()))
    COMB = []
    combination =  list(itertools.combinations(kerns, 1))
    for comb in combination :
        COMB.append(_kernel_list+comb)
    return COMB



if __name__ =="__main__" :

    #Y = np.sin(np.linspace(0,100,100)).reshape(-1,1)
    X = np.linspace(0,100,100).reshape(-1, 1)
    Y = 3*(np.sin(X)).reshape(-1, 1)
    Y_a = np.array(pd.read_csv("./data/periodic.csv")["x"]).reshape(-1, 1)
    Y = Y_a[:-30]
    X = np.linspace(0,len(Y),len(Y)).reshape(-1,1)
    X_s = np.linspace(0,len(Y)+60,len(Y)+60).reshape(-1, 1)
    t0 = time.time()
    model,kernel = single_model(X,Y,X_s,["+PER","+LIN","*SE"],nb_restart=1,nb_iter=400,verbose=True,initialisation_restart=7,reduce_data=False,OPTIMIZER=tf.optimizers.RMSprop(0.01))
    #model,kernel = launch_analysis(X,Y,X_s,prune=False,straigth=True,depth=10,nb_restart=1,verbose=False,nb_iter=100,initialisation_restart=5,reduce_data=False,OPTIMIZER=tf.optimizers.RMSprop(0.01))
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
    
    
    
    

        