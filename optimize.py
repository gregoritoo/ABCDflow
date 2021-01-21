import numpy as np 
import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float32')
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


PI = m.pi
_precision = tf.float64
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)



KERNELS_LENGTH = {
    "LIN" : 1,
    "CONST" : 1,
    "SE" : 2,
    "PER" :3,
}

KERNELS = {
    "LIN" : {"parameters_lin":["lin_c"]},
    "CONST" : {"parameters":["const_sigma"]},
    "SE" : {"parameters":["squaredexp_l","squaredexp_sigma"]},
    "PER" : {"parameters_per":["periodic_l","periodic_p","periodic_sigma"]},
}


KERNELS_OPS = {
    "*LIN" : "mul",
    "*CONST" : "mul",
    "*SE" : "mul",
    "*PER" :"mul",
    "+LIN" : "add",
    "+CONST" : "add",
    "+SE" : "add",
    "+PER" : "add",
}



def make_val_and_grad_fn(value_fn):
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return tfp.math.value_and_gradient(value_fn,x)
  return val_and_grad


@contextlib.contextmanager
def timed_execution():
  t0 = time.time()
  yield
  dt = time.time() - t0
  print('Evaluation took: %f seconds' % dt)


def np_value(tensor):
  """Get numpy value out of possibly nested tuple of tensors."""
  if isinstance(tensor, tuple):
    return type(tensor)(*(np_value(t) for t in tensor))
  else:
    return tensor.numpy()

def run(optimizer):
  optimizer()  # Warmup.
  with timed_execution():
    result = optimizer()
  return np_value(result)



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



def train(model,nb_iter,nb_restart,X_train,Y_train,kernels_name,OPTIMIZER,verbose=True,mode="SGD"):
    base_model = model 
    loop = 0
    best = 10e90
    @make_val_and_grad_fn
    def log_cholesky_lbfgs(params):
        params = list([tf.reshape(ele,(1,-1)) for ele in params])
        print("=========",params)
        num = 0
        X = X_train 
        Y = Y_train
        kernel = kernels_name
        cov = 1
        params_name = list(model._variables)
        for op in kernel :
            if op[0] == "+":
                method = KERNELS_FUNCTIONS[op[1:]]
                par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError("Method %s not implemented" % op[1:])
                cov += method(X,X,tf.reshape(params[num:num+KERNELS_LENGTH[op[1:]]],(1,-1)))
                num += KERNELS_LENGTH[op[1:]]
            elif op[0] == "*":
                method = KERNELS_FUNCTIONS[op[1:]]
                method = KERNELS_FUNCTIONS[op[1:]]
                par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError("Method %s not implemented" % op[1:])
                cov  = tf.math.multiply(cov,method(X,X,[params[p] for p in par]))
                num += KERNELS_LENGTH[op[1:]]
        _L = tf.linalg.cholesky(cov+_jitter*tf.eye(X.shape[0]))
        _temp = tf.linalg.solve(_L, Y)
        alpha = tf.linalg.solve(tf.transpose(_L), _temp)
        loss = 0.5*tf.matmul(tf.transpose(Y),alpha) + tf.math.log(tf.linalg.trace(_L)) +0.5*X.shape[0]*tf.math.log([PI*2])
        print("=====================losssssssssssssssssssss",loss)
        return tf.reshape(loss,(1,-1))

    def l2_regression_with_lbfgs():
        return tfp.optimizer.lbfgs_minimize(log_cholesky_lbfgs,initial_position=model._opti_variables,tolerance=1e-8)

    if mode == "SGD" :
        while loop < nb_restart :
            try :
                model = base_model
                if verbose :
                    for iteration in range(1,nb_iter):
                        val = train_step(model,iteration,X_train,Y_train,kernels_name,OPTIMIZER)
                        sys.stdout.write("\r"+"="*int(iteration/nb_iter*50)+">"+"."* int((nb_iter-iteration)/nb_iter*50)+"|" \
                            +" * log likelihood  is : {:.4f} at iteration : {:.0f} at epoch : {:.0f} / {:.0f} ".format(val[0][0],nb_iter,loop+1,nb_restart))
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
    else :
        params = run(l2_regression_with_lbfgs)
        #params = run(l2_regression_with_lbfgs).converged.numpy()
        print("==================== params ===========================",params)
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
                                        nb_by_step,prune,verbose,OPTIMIZER,unique=False,single=False):
    j=0
    try :
        if not unique : _kernel_list = list(combi)
        else : _kernel_list = list([combi])
        if single : _kernel_list = combi
        kernels_name = ''.join(combi)
        if kernels_name[0] != "*" :
            kernels = _preparekernel(_kernel_list)
            model=CustomModel(kernels)
            model = train(model,nb_iter,nb_restart,X_train,Y_train,_kernel_list,OPTIMIZER,verbose)
            BIC = model.compute_BIC(X_train,Y_train,_kernel_list)
            if BIC < BEST_MODELS["score"]  : 
                BEST_MODELS["model_name"] = kernels_name
                BEST_MODELS["model_list"] = _kernel_list
                BEST_MODELS["model"] = model
                BEST_MODELS["score"] = BIC 
            TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS)+1]=[[kernels_name],int(BIC.numpy()[0])]                           
    except Exception as e:
        print("error with kernel :",kernels_name)
        print(e)
    if prune :
        return BEST_MODELS,TEMP_BEST_MODELS
    else :
        return BEST_MODELS


def analyse(X_train,Y_train,X_s,nb_restart,nb_iter,nb_by_step,i,prune,loop_size,verbose,OPTIMIZER):
    if i == -1 :
        COMB = search("",[],True)
    else :
        name = "search/model_list_"+str(i)
        with open(name, 'rb') as f :
            COMB = pickle.load(f)
    kernels_name,_kernel_list = "",[]
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score":10e40}
    TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
    iteration=0
    j = 0
    while len(COMB) > 2 :
        full_length = len(COMB)
        try : 
            combi = COMB[j]
        except Exception as e :
            break
        iteration+=1
        j+=1
        if prune :
            if iteration % loop_size == 0 :
                j=0
                TEMP_BEST_MODELS = TEMP_BEST_MODELS[: nb_by_step]
                COMB = _prune(TEMP_BEST_MODELS["Name"].tolist(),COMB[iteration :])
                #print("Model to try :",COMB)
            BEST_MODELS,TEMP_BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS, \
                                                                nb_restart,nb_iter,nb_by_step,prune,verbose,OPTIMIZER)
            TEMP_BEST_MODELS = TEMP_BEST_MODELS.sort_values(by=['score'],ascending=True)[:nb_by_step]
            sys.stdout.write("\r"+"="*int(j/full_length*50)+">"+"."*int((full_length-j)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        else :  
            COMB,j = COMB[1 :],0
            BEST_MODELS = search_step(X_train,Y_train,X_s,combi,BEST_MODELS,TEMP_BEST_MODELS,nb_restart,nb_iter,nb_by_step,prune,verbose)
            sys.stdout.write("\r"+"="*int(j/full_length*50)+">"+"."*int((full_length-j)/full_length*50)+"|"+" * model is {} ".format(combi))
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    model=BEST_MODELS["model"]
    model.viewVar(BEST_MODELS["model_list"])
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,BEST_MODELS["model_list"])))
    return model,BEST_MODELS["model_list"]


def single_model(X_train,Y_train,X_s,kernel,OPTIMIZER,nb_restart=150,nb_iter=50,verbose=False):
    X_train,Y_train,X_s = tf.Variable(X,dtype=_precision),tf.Variable(Y,dtype=_precision),tf.Variable(X_s,dtype=_precision)
    iteration = 0
    BEST_MODELS = {"model_name":[],"model_list":[],'model':[],"score":10e400}
    TEMP_BEST_MODELS = pd.DataFrame(columns=["Name","score"])
    full_length= 1
    iteration+=1
    BEST_MODELS = search_step(X_train=X_train,Y_train=Y_train,X_s=X_s,combi=kernel,BEST_MODELS=BEST_MODELS, \
        TEMP_BEST_MODELS=TEMP_BEST_MODELS,nb_restart=nb_restart,nb_iter=nb_iter,verbose = verbose,OPTIMIZER=OPTIMIZER,nb_by_step=None,prune=False,unique=True,single=True)
    sys.stdout.write("\r"+"="*int(iteration/full_length*50)+">"+"."*int((full_length-iteration)/full_length*50)+"|"+" * model is {} ".format(kernel))
    sys.stdout.flush()
    model=BEST_MODELS["model"]
    model.viewVar(kernel)
    print("model BIC is {}".format(model.compute_BIC(X_train,Y_train,kernel)))
    return model,kernel




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


def launch_analysis(X_train,Y_train,X_s,nb_restart=15,nb_iter=2,do_plot=True,save_model=False,prune=False,OPTIMIZER = tf.optimizers.Adamax(learning_rate=0.06), \
                        verbose=False,nb_by_step=None,loop_size=50,nb_workers=None,experimental_multiprocessing=False,reduce_data=True,initialisation_restart=1):
    if prune and nb_by_step is None : raise ValueError("As prune is True you need to precise nb_by_step")
    if nb_by_step is  not None and nb_by_step > loop_size : raise ValueError("Loop size must be superior to nb_by_step")   
    X_train,Y_train,X_s = tf.Variable(X,dtype=_precision),tf.Variable(Y,dtype=_precision),tf.Variable(X_s,dtype=_precision)
    if reduce_data :
            mean, var = tf.nn.moments(X_train,axes=[0])
            X_train = (X_train - mean) / var
            mean, var = tf.nn.moments(Y_train,axes=[0])
            Y_train = (Y_train - mean) / var
            mean, var = tf.nn.moments(X_s,axes=[0])
            X_s = (X_s - mean) / var
    t0 = time.time()
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
            start_borne = 0
            full_entro = 0
            for borne in my_bkps :
                val = block_entropy(ts[start_borne:borne], k=1)   
                full_entro = val + full_entro
                start_borne = borne
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

GRID_SEARCH = {
        "algo" : [tf.optimizers.Adamax,tf.optimizers.Adadelta,tf.optimizers.Adam,tf.optimizers.RMSprop] ,
        "lr"  : np.linspace(1e-05,1,100) ,
}
OPTIMIZER = tf.optimizers.Adamax(learning_rate=0.06)


if __name__ =="__main__" :

    X = np.linspace(0,100,100).reshape(-1, 1)
    Y = 3*(np.sin(X)).reshape(-1, 1)
    X_s = np.arange(-30, 130, 1).reshape(-1, 1)
    """HISTORY = pd.DataFrame(columns=["OPTIMIZER","learning_rate","score"])
    for element in GRID_SEARCH["algo"] :
        for lr in GRID_SEARCH["lr"] :
            OPTIMIZER = element(lr)
            print("kefzleznfje")
            print(OPTIMIZER)
            try :
                model,kernel = single_model(X,Y,X_s,['+PER'],OPTIMIZER,nb_restart=15,nb_iter=10,verbose=False)
                BIC = model.compute_BIC(X,Y,kernel).numpy()
                print(int(BIC[0]))
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,int(BIC[0])]
            except Exception :
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,0]
    HISTORY.to_csv("./optimization_results/PER.csv")
    HISTORY = pd.DataFrame(columns=["OPTIMIZER","learning_rate","score"])
    for element in GRID_SEARCH["algo"] :
        for lr in GRID_SEARCH["lr"] :
            OPTIMIZER = element(lr)
            print("kefzleznfje")
            print(OPTIMIZER)
            try :
                model,kernel = single_model(X,Y,X_s,['+CONST'],OPTIMIZER,nb_restart=15,nb_iter=10,verbose=False)
                BIC = model.compute_BIC(X,Y,kernel).numpy()
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,float(BIC[0])]
            except Exception :
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,0]
    HISTORY.to_csv("./optimization_results/CONST.csv")
    HISTORY = pd.DataFrame(columns=["OPTIMIZER","learning_rate","score"])
    for element in GRID_SEARCH["algo"] :
        for lr in GRID_SEARCH["lr"] :
            OPTIMIZER = element(lr)
            print("kefzleznfje")
            print(OPTIMIZER)
            try :
                model,kernel = single_model(X,Y,X_s,['+SE'],OPTIMIZER,nb_restart=15,nb_iter=10,verbose=False)
                BIC = model.compute_BIC(X,Y,kernel).numpy()
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,float(BIC[0])]
            except Exception :
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,0]
    HISTORY.to_csv("./optimization_results/SE.csv")
    HISTORY = pd.DataFrame(columns=["OPTIMIZER","learning_rate","score"])
    for element in GRID_SEARCH["algo"] :
        for lr in GRID_SEARCH["lr"] :
            OPTIMIZER = element(lr)
            print("kefzleznfje")
            print(OPTIMIZER)
            try :
                model,kernel = single_model(X,Y,X_s,['+LIN'],OPTIMIZER,nb_restart=15,nb_iter=10,verbose=False)
                BIC = model.compute_BIC(X,Y,kernel).numpy()
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,float(BIC[0])]
            except Exception :
                HISTORY.loc[len(HISTORY)+1]=[str(element)[-15:-2],lr,0]
    HISTORY.to_csv("./optimization_results/LIN.csv")"""

    df_lin = pd.read_csv("./optimization_results/LIN.csv")
    df_se = pd.read_csv("./optimization_results/SE.csv")
    df_per = pd.read_csv("./optimization_results/PER.csv")
    df_const = pd.read_csv("./optimization_results/CONST.csv")

    opti_lin = df_lin["OPTIMIZER"].unique()
    opti_se = df_se["OPTIMIZER"].unique()
    opti_per = df_per["OPTIMIZER"].unique()
    opti_const = df_const["OPTIMIZER"].unique()

    """for element in opti_lin :
        df = df_lin[df_lin["OPTIMIZER"]==element]
        df[df['score'] == 0]=100000
        plt.plot(df["learning_rate"],df["score"],label=element+"_lin")
    
    for element in opti_se :
        df = df_se[df_se["OPTIMIZER"]==element]
        plt.plot(df["learning_rate"],df["score"],label=element+"_se")"""

    
    for element in opti_per :
        df = df_per[df_per["OPTIMIZER"]==element]
        df_lin = df_lin[df_lin["OPTIMIZER"]==element]
        df_const = df_const[df_const["OPTIMIZER"]==element]
        plt.plot(df_lin["learning_rate"],df_lin["score"].max()-df["score"],label=element+"_per")
        plt.plot(df["learning_rate"],df["score"].max()-df["score"],label=element+"_per")
        plt.legend()
        plt.show()
    
    """for element in opti_const :
        df = df_const[df_const["OPTIMIZER"]==element]
        plt.plot(df["learning_rate"],df["score"],label=element+"_const")"""
    

    
    
    

        