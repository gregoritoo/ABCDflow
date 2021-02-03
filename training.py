import numpy as np 
import tensorflow as tf 
import os
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
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
from multiprocessing.dummy import Pool as ThreadPool 
import contextlib
import functools
import time
import scipy 
from search import preparekernel,addkernel,mulkernel,search,prune,search_and_add,replacekernel
from utils import KERNELS,KERNELS_LENGTH,KERNELS_OPS,GPY_KERNELS


PI = m.pi
_precision = tf.float64
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.inter_op_parallelism_threads = 8
config.intra_op_parallelism_threads = 8
session = tf.compat.v1.Session(config=config)

lr_list = np.linspace(0.001,1,101)
borne = -1*10e40





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
        try :
            nb_iter = max(nb_iter,100)
            #results = train_step_lfgbs(X_train,Y_train,model._opti_variables,kernels_name)
            func = function_factory(model, log_cholesky_l_test, X_train, Y_train,model._opti_variables,kernels_name)
            init_params = tf.dynamic_stitch(func.idx, model._opti_variables)
            #init_params = tf.cast(init_params, dtype=_precision)
            # train the model with L-BFGS solver
            bnds = list([(1e-6, None) for _ in range(len(model.variables)-1)])
            bnds.append([1e-8,None])  # specific boundaries for the noise parameter
            #options={"maxiter":nb_iter}
            results = scipy.optimize.minimize(fun=func, x0=init_params,jac=True, method='L-BFGS-B',bounds=tuple(bnds),options={"maxiter":nb_iter})
            #results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params,tolerance=1e-8)
            best_model = model
        except Exception as e:
            print(e)
    else :
        raise  NotImplementedError("Mode not available please choose between lfbgs or SBD")
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
    kernels_name,kernel_list = "",[]
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score": borne,"init_values":None}
    TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
    iteration=0
    j = 0
    full_length = len(COMB)
    loop_size = 24
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
                COMB = prune(TEMP_BEST_MODELS["Name"].tolist(),COMB[iteration :])
                _to_add = _before_len - len(COMB)-1
                iteration += _to_add
            BEST_MODELS,TEMP_BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS, \
                                                                nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
            TEMP_BEST_MODELS = TEMP_BEST_MODELS.sort_values(by=['score'],ascending=True)[:nb_by_step]
            print_trainning_steps(iteration,full_length,combi)
        else :  
            COMB,j = COMB[1 :],0
            BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
            print_trainning_steps(iteration,full_length,combi)
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
            X_train = whitenning_datas(X_train)
            Y_train = whitenning_datas(Y_train)
            X_s = whitenning_datas(X_s)
    assert kernel[0][0] == "+" , "First kernel of the list must start with + "
    iteration = 1
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score": borne,"init_values":None}
    TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
    full_length= 1
    BEST_MODELS = search_step(X_train=X_train,Y_train=Y_train,X_s=X_s,combi=kernel,BEST_MODELS=BEST_MODELS,TEMP_BEST_MODELS=TEMP_BEST_MODELS,\
        nb_restart=nb_restart,nb_iter=nb_iter,verbose = verbose,OPTIMIZER=OPTIMIZER,nb_by_step=None,prune=False,unique=True,single=True,mode=mode,initialisation_restart=initialisation_restart,GPY=GPY)
    print_trainning_steps(iteration,full_length,kernel)
    if not GPY :
        model=BEST_MODELS["model"]
        model.viewVar(kernel)
        print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,kernel)))
        mu,cov = model.predict(X_train,Y_train,X_s,kernel)
    if do_plot :
        model.plot(mu,cov,X_train,Y_train,X_s,kernel)
    return BEST_MODELS["model"],kernel





def launch_analysis(X_train,Y_train,X_s,nb_restart=15,nb_iter=2,do_plot=False,save_model=False,prune=False,OPTIMIZER= tf.optimizers.Adam(0.001), \
                        verbose=False,nb_by_step=None,loop_size=10,experimental_multiprocessing=False,reduce_data=False,straigth=True,depth=5,initialisation_restart=5,GPY=False,mode="lfbgs"):
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
            X_train = whitenning_datas(X_train)
            Y_train = whitenning_datas(Y_train)
            X_s = whitenning_datas(X_s)
    i=-1 
    if experimental_multiprocessing :
        model,kernels = multithreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                        verbose,OPTIMIZER,depth,initialisation_restart,GPY,mode,experimental_multiprocessing,do_plot,save_model)
        return model,kernels
    if straigth :
        model,kernels = monothreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,experimental_multiprocessing,save_model,do_plot)
        return model,kernels
    if not experimental_multiprocessing :
        model,kernels = griddy_search(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,experimental_multiprocessing,save_model,do_plot)
        return model,name_kernel
   
def save_and_plot(func):
    def wrapper_func(*args,**kwargs):
        model,kernels = func(*args,**kwargs)
        name,name_kernel = './best_models/best_model', kernels
        do_plot,save_model=args[-2],args[-1]
        X_train,Y_train,X_s = args[0],args[1],args[2]
        if do_plot :
            mu,cov = model.predict(X_train,Y_train,X_s,kernels)
            model.plot(mu,cov,X_train,Y_train,X_s,kernels)
            plt.show()
        if save_model :
            with open(name, 'wb') as f :
                pickle.dump(model,f)
            with open(name_kernel, 'wb') as f :
                pickle.dump(kernels,f)
        return model,name_kernel
    return wrapper_func


@save_and_plot
def multithreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                        verbose,OPTIMIZER,depth,initialisation_restart,GPY,mode,parralelize_code,do_plot,save_model):
    print("This is experimental, the accuracy may varie a lot  !")
    try :
        model,kernels = straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,depth,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode,parralelize_code=parralelize_code)
    except Exception as e :
        print(e)
    return model,kernels

@save_and_plot
def monothreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,parralelize_code,do_plot,save_model):
    model,kernels = straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                                verbose,OPTIMIZER,depth,initialisation_restart,GPY=GPY,mode=mode)
    return model,kernels

@save_and_plot
def griddy_search(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,parralelize_code,do_plot,save_model):
    
    model,kernels = analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY=GPY,depth=depth,mode=mode,parralelize_code=parralelize_code)
    return model,kernels


def parralelize(X_train,Y_train,X_s,combi,BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER,initialisation_restart,GPY,mode):
    ''' 
        Use class multiprocessing.dummies to multithread one step train
    '''
    params = [[X_train,Y_train,X_s,comb,BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER,mode,False,False,initialisation_restart,GPY] for comb in combi]
    pool = ThreadPool()
    outputs = pool.starmap(search_step_parrallele,params)
    pool.close()
    pool.join()
    return outputs


def straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,depth=5,initialisation_restart=5,GPY=False,mode="lfbgs",parralelize_code=False):
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
    train_length = (depth+1)*len(KERNELS) + len(KERNELS)
    for comb in combination :
        if comb[0][0] != "*" : 
            COMB.append(comb)
    for loop in range(depth) :
        TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
        loop += 1
        if loop > 1 :
            COMB = search_and_add(tuple(BEST_MODELS["model_list"]))
            print("Next combinaison to try : {}".format(COMB))
        iteration,j=0,0
        if parralelize_code :
            outputs = parralelize(X_train,Y_train,X_s,COMB,BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
            BEST_MODELS = update_best_model_after_parallelized_step(outputs,BEST_MODELS)
            print("The best model is {} at layer {}".format(BEST_MODELS["model_list"],loop))
            if loop > 1 :
                new_COMB = replacekernel(BEST_MODELS["model_list"])   #swap step 
                print("Trying to switch kernels : trying {} ".format(new_COMB))
                outputs = parralelize(X_train,Y_train,X_s,new_COMB,BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
                BEST_MODELS = update_best_model_after_parallelized_step(outputs,BEST_MODELS)
        else :
            while j <  len(COMB) :
                try : combi = COMB[j]
                except Exception as e :break
                iteration+=1
                j+=1
                count += 1
                TEMP_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,\
                                                    verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
                print_trainning_steps(count,train_length,combi)
            if TEMP_MODELS["score"] > BEST_MODELS["score"] :
                BEST_MODELS = TEMP_MODELS
            print("The best model is {} at layer {}".format(BEST_MODELS["model_list"],loop))
            if loop > 2 and len(BEST_MODELS["model_list"]) > 2:
                new_COMB = replacekernel(BEST_MODELS["model_list"])   #swap step 
                print("Trying to switch kernels : trying {} ".format(new_COMB))
                j=0
                while j <  len(new_COMB) :
                    try : combi = new_COMB[j]
                    except Exception as e :break
                    j+=1
                    TEMP_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,\
                                                prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
                    print_trainning_steps(count,train_length,combi)
                if TEMP_MODELS["score"] > BEST_MODELS["score"] :
                    BEST_MODELS = TEMP_MODELS
        print("The best model is {} after swap ".format(BEST_MODELS["model_list"]))
    if not GPY  and verbose :
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
        if not unique : kernel_list = list(combi)
        else : kernel_list = list([combi])
        if single : kernel_list = combi
        kernels_name = ''.join(combi)
        true_restart = 0
        if mode == "lfbgs" : kernels = preparekernel(kernel_list,scipy=True)
        else : kernels = preparekernel(kernel_list)
        if kernels_name[0] != "*" :
            if not GPY : 
                while true_restart < initialisation_restart :
                    try :
                        model=CustomModel(kernels,init_values)
                        model = train(model,nb_iter,nb_restart,X_train,Y_train,kernel_list,OPTIMIZER,verbose,mode=mode)
                        BIC = model.compute_BIC(X_train,Y_train,kernel_list)
                        BEST_MODELS = update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name)
                        TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],float(BIC[0][0])]  
                        true_restart += 1     
                    except Exception :
                        pass  
            else :
                k = gpy_kernels_from_names(kernel_list)
                model = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
                model.optimize_restarts(initialisation_restart)
                BIC = -model.objective_function()
                BEST_MODELS = update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name,GPy=True)
                TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],BIC]                
    except Exception as e:
        print("error with kernel :",kernels_name)
    if prune :
        return BEST_MODELS,TEMP_BEST_MODELS
    else :
        return BEST_MODELS


def search_step_parrallele(X_train,Y_train,X_s,combi,TEMP_BEST_MODELS,nb_restart,nb_iter, \
                                        nb_by_step,prune,verbose,OPTIMIZER,mode="lfbgs",unique=False,single=False,initialisation_restart=10,GPY=False):
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
    true_restart=0
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score": borne,"init_values":None}
    init_values = TEMP_BEST_MODELS["init_values"] 
    try :
        if not unique : kernel_list = list(combi)
        else : kernel_list = list([combi])
        if single : kernel_list = combi
        kernels_name = ''.join(combi)
        true_restart = 0
        if mode == "lfbgs" : kernels = preparekernel(kernel_list)
        else : kernels = preparekernel(kernel_list)
        if kernels_name[0] != "*" :
            while true_restart < initialisation_restart :
                try :
                    model=CustomModel(kernels,init_values)
                    model = train(model,nb_iter,nb_restart,X_train,Y_train,kernel_list,OPTIMIZER,verbose,mode=mode)
                    BIC = model.compute_BIC(X_train,Y_train,kernel_list)
                    BEST_MODELS = update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name)
                    true_restart += 1     
                except Exception :
                    pass              
    except Exception as e:
        print("error with kernel :",kernels_name)
    else :
        return BEST_MODELS



