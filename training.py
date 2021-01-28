import numpy as np 
import tensorflow as tf 
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('INFO')
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
from search import _preparekernel,_addkernel,_mulkernel, search,_prune,search_and_add

f = open(os.devnull, 'w')
sys.stderr = f
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
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
    "LIN" : 2,
    "SE" : 2,
    "PER" :3,
    #"CONST" : 1,
    #"WN" : 1,
    #"RQ" : 3,
}

KERNELS = {
    "LIN" : {"parameters_lin":["lin_c","lin_sigmav"]},
    #"CONST" : {"parameters":["const_sigma"]},
    "SE" : {"parameters":["squaredexp_l","squaredexp_sigma"]},
    "PER" : {"parameters_per":["periodic_l","periodic_p","periodic_sigma"]},
    #"WN" : {"paramters_Wn":["white_noise_sigma"]},
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
    #"+WN" :"add",
    #"*WN" : "mul",
    #"+RQ" : "add",
    #"*RQ" : "mul",
}

GPY_KERNELS = {
    "LIN" : GPy.kern.Linear(input_dim=1),
    "SE" : GPy.kern.sde_Exponential(input_dim=1),
    "PER" :GPy.kern.StdPeriodic(input_dim=1),
    "RQ" : GPy.kern.RatQuad(input_dim=1),
}





def train(model,nb_iter,nb_restart,X_train,Y_train,kernels_name,OPTIMIZER,verbose=True,mode="lfbgs"):
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
    best = -1*borne
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
                    loop += 1
                else :
                    for iteration in range(1,nb_iter):
                        val = train_step(model,iteration,X_train,Y_train,kernels_name)
                    loop += 1
            except Exception as e :
                loop += 1
            if val  < best :
                best =  val
                best_model = model
    elif "lfbgs" :
        while loop < nb_restart :
            try :
                nb_iter = max(nb_iter,500)
                if verbose : 
                    sys.stdout.write("\r"+"Iteration n°{}/{}".format(loop,nb_restart))
                    sys.stdout.flush()
                #results = train_step_lfgbs(X_train,Y_train,model._opti_variables,kernels_name)
                func = function_factory(model, log_cholesky_l_test, X_train, Y_train,model._opti_variables,kernels_name)
                init_params = tf.dynamic_stitch(func.idx, model._opti_variables)
                #init_params = tf.cast(init_params, dtype=_precision)
                # train the model with L-BFGS solver
                bnds = list([(1e-6, None) for _ in range(len(model.variables)-1)])
                bnds.append([1e-8,None])  # specific boundaries for the noise parameter
                #options={"maxiter":nb_iter}
                results = scipy.optimize.minimize(fun=func, x0=init_params,jac=True, method='L-BFGS-B',bounds=tuple(bnds))
                #results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params,tolerance=1e-8)
                best_model = model
            except Exception as e:
                print(e)
            loop+=1
    else :
        raise  NotImplementedError("Mode %s not available please choose between lfbgs or SBD")
    return best_model




def analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode="lfbgs"):
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
        COMB = search("",[],True,depth)
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
    while len(COMB) > 1 :
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
                                                                nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
            TEMP_BEST_MODELS = TEMP_BEST_MODELS.sort_values(by=['score'],ascending=True)[:nb_by_step]
            sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        else :  
            COMB,j = COMB[1 :],0
            BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
            sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    if not GPY :
        model=BEST_MODELS["model"]
        model.viewVar(BEST_MODELS["model_list"])
        print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,BEST_MODELS["model_list"])))
    return BEST_MODELS["model"],BEST_MODELS["model_list"]


def single_model(X_train,Y_train,X_s,kernel,OPTIMIZER=tf.optimizers.Adam(learning_rate=0.001),nb_restart=7,nb_iter=4,verbose=False,initialisation_restart=2,reduce_data=False,do_plot=False,mode="lfbgs",GPY=False):
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
    X_train,Y_train,X_s = tf.Variable(X_train,dtype=_precision),tf.Variable(Y_train,dtype=_precision),tf.Variable(X_s,dtype=_precision)
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
        TEMP_BEST_MODELS=TEMP_BEST_MODELS,nb_restart=nb_restart,nb_iter=nb_iter,verbose = verbose,OPTIMIZER=OPTIMIZER,nb_by_step=None,prune=False,unique=True,single=True,mode=mode,initialisation_restart=initialisation_restart,GPY=GPY)
    sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(kernel))
    sys.stdout.flush()
    if not GPY :
        model=BEST_MODELS["model"]
        model.viewVar(kernel)
        print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,kernel)))
        mu,cov = model.predict(X_train,Y_train,X_s,kernel)
    if do_plot : model.plot(mu,cov,X_train,Y_train,X_s,kernel)
    return BEST_MODELS["model"],kernel



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


def launch_analysis(X_train,Y_train,X_s,nb_restart=15,nb_iter=2,do_plot=False,save_model=False,prune=False,OPTIMIZER= tf.optimizers.Adam(0.001), \
                        verbose=False,nb_by_step=None,loop_size=50,nb_workers=None,experimental_multiprocessing=False,reduce_data=False,straigth=True,depth=5,initialisation_restart=5,GPY=False,mode="lfbgs"):
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
    if  straigth : print("You chooosed straightforward training")
    X_train,Y_train,X_s = tf.Variable(X_train,dtype=_precision),tf.Variable(Y_train,dtype=_precision),tf.Variable(X_s,dtype=_precision)
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
        model,kernels = straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,depth,initialisation_restart,GPY=GPY,mode=mode)
        if do_plot :
            mu,cov = model.predict(X_train,Y_train,X_s,kernels)
            model.plot(mu,cov,X_train,Y_train,X_s,kernels)
            plt.show()
        return model,kernels
    if not experimental_multiprocessing :
        i=-1 
        model,kernels = analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,initialisation_restart,GPY=GPY,depth=depth,mode=mode)
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




def straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,depth=5,initialisation_restart=5,GPY=False,mode="lfbgs"):
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
        loop += 1
        if loop > 1 :
            COMB = search_and_add(tuple(BEST_MODELS["model_list"]))
            print("Next combinaison to try : {}".format(COMB))
        iteration=0
        j = 0
        while j <  len(COMB) :
            count += 1
            try : combi = COMB[j]
            except Exception as e :break
            iteration+=1
            j+=1
            TEMP_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
            sys.stdout.write("\r"+"="*int(count/train_length*50)+">"+"."*int((train_length-count)/train_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
        if TEMP_MODELS["score"] > BEST_MODELS["score"] :
            BEST_MODELS = TEMP_MODELS
        print("The best model is {} at layer {}".format(BEST_MODELS["model_list"],loop-1))
    if not GPY :
        model=BEST_MODELS["model"]
        model.viewVar(BEST_MODELS["model_list"])
        print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,BEST_MODELS["model_list"])))
    return BEST_MODELS["model"],BEST_MODELS["model_list"]


def search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter, \
                                        nb_by_step,prune,verbose,OPTIMIZER,mode="lfbgs",unique=False,single=False,initialisation_restart=5,GPY=False):
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
        if mode == "lfbgs" : kernels = _preparekernel(_kernel_list,scipy=True)
        else : kernels = _preparekernel(_kernel_list)
        if kernels_name[0] != "*" :
            if not GPY : 
                while true_restart < initialisation_restart :
                        try :
                            model=CustomModel(kernels,init_values)
                            model = train(model,nb_iter,nb_restart,X_train,Y_train,_kernel_list,OPTIMIZER,verbose,mode=mode)
                            BIC = model.compute_BIC(X_train,Y_train,_kernel_list)
                            """mu,cov = model.predict(X_train,Y_train,X_full,_kernel_list)
                            try :
                                mean,_,_= get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=100)     // uncomment to do cross validation to avoid overfitting when using cross validation
                            except Exception as e:
                                print(e)
                                break
                            BIC = mse(X_full.numpy()[-30:],mean[-30:])
                            print(BIC)"""
                            if  BIC > BEST_MODELS["score"] and BIC != float("inf") : 
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
            else :
                k = gpy_kernels_from_names(_kernel_list)
                model = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
                model.optimize_restarts(initialisation_restart)
                BIC = -model.objective_function()
                if  BIC > BEST_MODELS["score"]  : 
                    BEST_MODELS["model_name"] = kernels_name
                    BEST_MODELS["model_list"] = _kernel_list
                    BEST_MODELS["model"] = model
                    BEST_MODELS["score"] = BIC 
                    BEST_MODELS["init_values"] =  model.param_array()
                    TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],BIC]                
    except Exception as e:
        print("error with kernel :",kernels_name)
    if prune :
        return BEST_MODELS,TEMP_BEST_MODELS
    else :
        return BEST_MODELS


def gpy_kernels_from_names(_kernel_list):
    kernel = GPY_KERNELS[_kernel_list[0][1 :]]
    for j in range(1,len(_kernel_list)) :
        if _kernel_list[j][0] == "+" :
            kernel = kernel + GPY_KERNELS[_kernel_list[j][1 :]]
        elif _kernel_list[j][0] == "*" :
            kernel = kernel * GPY_KERNELS[_kernel_list[j][1 :]]
        else :
            raise ValueError("Illicite operation")
    return kernel



