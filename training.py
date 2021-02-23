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
from Regressor import * 
from Regressor_GPy import * 
from kernels_utils import *
from training_utils import *
from plotting_utils import *
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
from search import preparekernel,addkernel,mulkernel,search,prune,search_and_add,replacekernel,gpy_kernels_from_names,first_kernel
from kernels_utils import KERNELS,KERNELS_LENGTH,KERNELS_OPS,GPY_KERNELS
from termcolor import colored

PI = m.pi
_precision = tf.float64
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0,'CPU':4})
config.gpu_options.allow_growth = True
config.inter_op_parallelism_threads = 3
config.intra_op_parallelism_threads = 3
session = tf.compat.v1.Session(config=config)
borne = -1*10e40



def define_boundaries(model,X_train):
    bnds=[]
    for param in model._opti_variables_name :
        if param == "cp_x0" :
            bnds.append([5,len(X_train)])
        elif param == "cp_s" :
            bnds.append([0.99,1])
        elif param == "periodic_p" or param == "periodic_l" or param == "squaredexp_l"  :
            bnds.append([1e-8,len(X_train)])
        else :
            bnds.append([1e-8,None])
    return bnds 



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
            func = function_factory(model, log_cholesky_l_test, X_train, Y_train,model._opti_variables,kernels_name)
            init_params = tf.dynamic_stitch(func.idx, model._opti_variables)
            # train the model with L-BFGS-B solver
            bnds = define_boundaries(model,X_train)
            results = scipy.optimize.minimize(fun=func, x0=init_params,jac=True, method='L-BFGS-B',bounds=tuple(bnds),options={"maxiter":nb_iter})
            best_model = model
        except Exception as e:
            print(e)
    else :
        raise  NotImplementedError("Mode not available please choose between lfbgs or SBD")
    return best_model




def analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode="lfbgs",use_changepoint=False):
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
        COMB = search("",[],True,depth,use_changepoint)
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
        print(colored('[STATE]', 'red'),"model BIC is {}".format(model.compute_BIC(X_train,Y_train,BEST_MODELS["model_list"])))
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
    X_train,Y_train,X_s = tf.convert_to_tensor(X_train,dtype=_precision),tf.convert_to_tensor(Y_train,dtype=_precision),tf.convert_to_tensor(X_s,dtype=_precision)
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
        print(colored('[STATE]', 'red'),"model BIC is {}".format(model.compute_BIC(X_train,Y_train,kernel)))
        mu,cov = model.predict(X_train,Y_train,X_s,kernel)
    if do_plot :
        model.plot(mu,cov,X_train,Y_train,X_s,kernel)
    return BEST_MODELS["model"],kernel





def launch_analysis(X_train,Y_train,X_s,nb_restart=15,nb_iter=2,do_plot=False,save_model=False,prune=False,OPTIMIZER= tf.optimizers.Adam(0.001), \
                        verbose=False,nb_by_step=None,loop_size=10,experimental_multiprocessing=False,reduce_data=False,straigth=True,depth=5,initialisation_restart=5,GPY=False,mode="lfbgs",use_changepoint=False):
    '''
        Main function wich launch the search in the model space 
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
    X_train,Y_train,X_s = tf.convert_to_tensor(X_train,dtype=_precision),tf.convert_to_tensor(Y_train,dtype=_precision),tf.convert_to_tensor(X_s,dtype=_precision)
    if reduce_data :
            X_train = whitenning_datas(X_train)
            Y_train = whitenning_datas(Y_train)
            X_s = whitenning_datas(X_s)
    i=-1 
    if experimental_multiprocessing and not GPY:
        model,kernels = multithreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                        verbose,OPTIMIZER,depth,initialisation_restart,GPY,mode,experimental_multiprocessing,do_plot,save_model,use_changepoint)
        return model,kernels
    if straigth :
        model,kernels = monothreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,experimental_multiprocessing,save_model,do_plot,use_changepoint)
        return model,kernels
    if not experimental_multiprocessing :
        model,kernels = griddy_search(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,experimental_multiprocessing,save_model,do_plot,use_changepoint)
        return model,name_kernel
   
def save_and_plot(func):
    '''
        Decorator to plot the figure and pickle the model if user validate it.
    '''
    def wrapper_func(*args,**kwargs):
        model,kernels = func(*args,**kwargs)
        name = './best_models/best_model'
        do_plot,save_model=args[-2],args[-1]
        X_train,Y_train,X_s = args[0],args[1],args[2]
        if do_plot :
            try :
                mu,cov = model.predict(X_train,Y_train,X_s,kernels)
                model.plot(mu,cov,X_train,Y_train,X_s,kernels)
            except :
                model.plot()
            plt.show()
        if save_model :
            with open(name, 'wb') as f :
                pickle.dump(model,f)
            with open( './best_models/kernels', 'wb') as f :
                pickle.dump(kernels,f)
        return model,kernels
    return wrapper_func


@save_and_plot
def multithreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                        verbose,OPTIMIZER,depth,initialisation_restart,GPY,mode,parralelize_code,do_plot,save_model,use_changepoint):
    print("This is experimental, the speed may varie a lot !")
    try :
        model,kernels = straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,depth,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode,parralelize_code=parralelize_code,use_changepoint=use_changepoint)
    except Exception as e :
        print(e)
    return model,kernels

@save_and_plot
def monothreaded_straight_analysis(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,parralelize_code,do_plot,save_model,use_changepoint):
    model,kernels = straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                                verbose,OPTIMIZER,depth,initialisation_restart,GPY=GPY,mode=mode,use_changepoint=use_changepoint)
    return model,kernels

@save_and_plot
def griddy_search(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY,depth,mode,parralelize_code,do_plot,save_model,use_changepoint):
    
    model,kernels = analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,\
                                    verbose,OPTIMIZER,initialisation_restart,GPY=GPY,depth=depth,mode=mode,parralelize_code=parralelize_code,use_changepoint=use_changepoint)
    return model,kernels

def parralelize(X_train,Y_train,X_s,combi,BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER,initialisation_restart,GPY,mode):
    ''' 
        Use class multiprocessing.dummies to multithread one step train , keep all the score of every models in order to compared them lately 
    '''
    params = [[X_train,Y_train,X_s,comb,BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER,mode,False,False,initialisation_restart,GPY] for comb in combi]
    pool = ThreadPool(3)
    outputs = pool.starmap(search_step_parrallele,params)
    pool.close()
    pool.join()
    return outputs



def straigth_analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER,depth=5,initialisation_restart=10,GPY=False,mode="lfbgs",parralelize_code=False,use_changepoint=False):
    """
        FInd best combinaison of kernel that descrive the training data , keep one best model at each step contrary to the greedy seach of the analyse function
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
    count,constant,old_model = 0,0,""
    train_length = (depth+1)*len(KERNELS) + len(KERNELS)
    COMB = first_kernel(use_changepoint)
    print(colored('[STATE]', 'red')," starting with ",COMB)
    for loop in range(depth) :
        TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
        loop += 1
        if loop > 1 :
            #COMB = search_and_add(tuple(BEST_MODELS["model_list"]),use_changepoint)
            COMB = search_and_add(tuple(BEST_MODELS["model_list"]),False)
            print(colored('[STATE]', 'red')," Next combinaison to try : {}".format(COMB))
        iteration,j=0,0
        if parralelize_code :
            outputs = parralelize(X_train,Y_train,X_s,COMB,BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER=OPTIMIZER,initialisation_restart=initialisation_restart,GPY=GPY,mode=mode)
            BEST_MODELS = update_best_model_after_parallelized_step(outputs,BEST_MODELS)
            print(colored('[INTERMEDIATE RESULTS] ', 'green'),'The best model is {} at layer {}'.format(BEST_MODELS["model_list"],loop))
            if old_model == BEST_MODELS['model_name'] :
                constant += 1
            old_model = BEST_MODELS['model_name']
            if constant > 1 :
                return BEST_MODELS["model"],BEST_MODELS["model_list"]
            if loop > 2 :
                new_COMB = replacekernel(BEST_MODELS["model_list"])   #swap step 
                print(colored('[STATE]', 'red')," Trying to switch kernels : trying {} ".format(new_COMB)) 
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
            print(colored('[INTERMEDIATE RESULTS] ', 'green'),'The best model is {} at layer {}'.format(BEST_MODELS["model_list"],loop))
            if loop > 2 and len(BEST_MODELS["model_list"]) > 2:
                new_COMB = replacekernel(BEST_MODELS["model_list"])   #swap step 
                print(colored('[STATE]', 'red')," Trying to switch kernels : trying {} ".format(new_COMB))
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
        print(colored('[STATE]', 'red')," The best model is {} after swap ".format(BEST_MODELS["model_list"]))
    if not GPY  and verbose :
        model=BEST_MODELS["model"]
        model.viewVar(BEST_MODELS["model_list"])
        print(colored('[STATE]', 'red')," model BIC is {}".format(model.compute_BIC(X_train,Y_train,BEST_MODELS["model_list"])))
    return BEST_MODELS["model"],BEST_MODELS["model_list"]



def search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter, \
                                        nb_by_step,prune,verbose,OPTIMIZER,mode="lfbgs",unique=False,single=False,initialisation_restart=5,GPY=False):
    '''
        Launch the training of a single gaussian process : Create model, initialize params with the params of the previous best models, launch the optimization
        compute BIC and update the best models array if BIC > best_model's score  
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
    j=0
    lr = 0.1
    init_values = BEST_MODELS["init_values"] 
    try :
        if not unique : kernel_list = list(combi)
        else : kernel_list = list([combi])
        if single : kernel_list = combi
        kernels_name = ''.join(combi)
        true_restart = 0
        kernels = preparekernel(kernel_list)
        failed = 0
        if kernels_name[0] != "*" :
            if not GPY : 
                while true_restart < initialisation_restart and failed < initialisation_restart * 2 :
                    try :
                        model = CustomModel(kernels,init_values,X_train)
                        model = train(model,nb_iter,nb_restart,X_train,Y_train,kernel_list,OPTIMIZER,verbose,mode=mode)
                        BIC = model.compute_BIC(X_train,Y_train,kernel_list)
                        print(BIC)
                        if verbose :  print("[STATE] ",BIC)
                        BEST_MODELS = update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name)
                        if  BIC > BEST_MODELS["score"] and prune : TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],float(BIC[0][0])] 
                        if math.isnan(BIC) == False or math.isinf(BIC) == False : 
                            true_restart += 1   
                        del model  
                    except Exception as e:
                        print(e)
                        failed +=1  

            else :
                k = gpy_kernels_from_names(kernel_list)
                model = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
                model.optimize_restarts(initialisation_restart)
                BIC = -model.objective_function()
                BEST_MODELS = update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name,GPy=True)
                if  BIC > BEST_MODELS["score"] and prune : TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],BIC]               
    except Exception as e:
        print(colored('[ERROR]', 'red')," error with kernel :",kernels_name)
    if prune :
        return BEST_MODELS,TEMP_BEST_MODELS
    else :
        return BEST_MODELS


def search_step_parrallele(X_train,Y_train,X_s,combi,TEMP_BEST_MODELS,nb_restart,nb_iter, \
                                        nb_by_step,prune,verbose,OPTIMIZER,mode="lfbgs",unique=False,single=False,initialisation_restart=10,GPY=False):
    '''
        Launch the training of a single gaussian process : Create model, initialize params with the params of the previous best models, launch the optimization
        compute BIC and keep the score in a dictionnary (not to use shared BEST_MODELS dictionnary variable)
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
    del TEMP_BEST_MODELS
    try :
        if not unique : kernel_list = list(combi)
        else : kernel_list = list([combi])
        if single : kernel_list = combi
        kernels_name = ''.join(combi)
        true_restart = 0
        kernels = preparekernel(kernel_list)
        failed = 0
        if kernels_name[0] != "*" :
            if not GPY :
                while true_restart < initialisation_restart and failed < 20:
                    try :
                        model = CustomModel(kernels,init_values,X_train)
                        model = train(model,nb_iter,nb_restart,X_train,Y_train,kernel_list,OPTIMIZER,verbose,mode=mode)
                        BIC = model.compute_BIC(X_train,Y_train,kernel_list)
                        BEST_MODELS = update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name)
                        true_restart += 1  
                        del model    
                    except Exception :
                        failed += 1    
            else :
                try :
                    k = gpy_kernels_from_names(kernel_list)
                    model = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
                    model.optimize_restarts(initialisation_restart)
                    BIC = -model.objective_function()
                    BEST_MODELS = update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name,GPy=True)
                except Exception as e :
                    print(e)
                    pass         
    except Exception as e:
        print(colored('[ERROR]', 'red')," error with kernel :",kernels_name)
    else :
        return BEST_MODELS




def update_current_best_model(BEST_MODELS,model,BIC,kernel_list,kernels_name,GPy=False):
    '''
        Update the BEST_MODELS dictionnary if the specific input model has a higher BIC score
    '''
    if  BIC > BEST_MODELS["score"] and BIC != float("inf") : 
        BEST_MODELS["model_name"] = kernels_name
        BEST_MODELS["model_list"] = kernel_list
        BEST_MODELS["score"] = BIC 
        if not GPy :
            BEST_MODELS["model"] = model
            BEST_MODELS["init_values"] =  model.initialisation_values
        else :
            BEST_MODELS["model"] = GPyWrapper(model,kernel_list)
            BEST_MODELS["init_values"] =  model.param_array()
    return BEST_MODELS