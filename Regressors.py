import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
from utils import *
import os 
import kernels as kernels 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float32')
PI = m.pi
_precision = tf.float64

KERNELS_FUNCTIONS = {
    "LIN" : kernels.LIN,
    "PER" : kernels.PER,
    "SE" : kernels.SE,
    "RQ" : kernels.RQ,
    "CONST" : kernels.CONST,
    "WN": kernels.WN,

}

class PeriodicRegressor(object):
    def __init__(self):
        self._l = tf.compat.v1.get_variable('l',
                   dtype=_precision,
                   shape=(1,),
                   initializer=tf.random_uniform_initializer(minval=1e-5, maxval=1.))
        self._p = tf.compat.v1.get_variable('p',
                    dtype=_precision,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1e-5, maxval=1.))
        self._sigma = tf.compat.v1.get_variable('sigma',
                    dtype=_precision,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1e-5, maxval=1.))


    @tf.function
    def __call__(self,X_train,Y_train):
        params={"l":self._l,"p":self._p,"sigma":self._sigma}
        return log_cholesky_l(X_train,Y_train,params,kernel="PER")


    @tf.function
    def predict(self,X_train,Y_train,X_s):
        cov = PER(X_train,X_train,l=self._l,p=self._p,sigma=self._sigma)
        cov_ss =  PER(X_s,X_s,l=self._l,p=self._p,sigma=self._sigma)
        cov_s  = PER(X_train,X_s,l=self._l,p=self._p,sigma=self._sigma)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    @property
    def variables(self):
        return self._l,self._p,self._sigma
    
    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()[0]))))+"{}".format(var.numpy()[0]))


class LinearRegressor(object) :

    def __init__(self) :
        self._c = tf.compat.v1.get_variable('c',
                    dtype=_precision,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1e-5, maxval=1.))
                   
    @tf.function
    def __call__(self,X_train,Y_train):
        params={"c":self._c}
        return log_cholesky_l(X_train,Y_train,params,kernel="LIN")

    @property
    def variables(self):
        return self._c


    def predict(self,X_train,Y_train,X_s):
        X_train = tf.cast(X_train,dtype=_precision)
        Y_train = tf.cast(Y_train,dtype=_precision)
        X_s = tf.cast(X_s,dtype=_precision)
        cov = kernels.LIN(X_train,X_train,c=self._c)
        cov_ss =  kernels.LIN(X_s,X_s,c=self._c)
        cov_s  = kernels.LIN(X_train,X_s,c=self._c)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()[0]))))+"{}".format(var.numpy()[0]))



class SquaredExpRegressor(object) :

    def __init__(self) :
        self._l = tf.compat.v1.get_variable('l',
                    dtype=_precision,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1e-5, maxval=1.))
        self._sigma = tf.compat.v1.get_variable('sigmab',
                    dtype=_precision,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1e-5, maxval=1.))
                    

    @tf.function
    def __call__(self,X_train,Y_train):
        params={"l":self._l,"sigma":self._sigma}
        return log_cholesky_l(X_train,Y_train,params,kernel="SE")

    @tf.function
    def predict(self,X_train,Y_train,X_s):
        cov = exp(X_train,X_train,l=self._l,sigma=self._sigma)
        cov_ss =  exp(X_s,X_s,l=self._l,sigma=self._sigma)
        cov_s  = exp(X_train,X_s,l=self._l,sigma=self._sigma)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    @property
    def variables(self):
        return self._l,self._sigma

    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()[0]))))+"{}".format(var.numpy()[0]))
        

class WhiteNoiseRegressor(object) :

    def __init__(self) :
        self._sigma = tf.compat.v1.get_variable('sigma',
                    dtype=_precision,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1e-5, maxval=1.))
        self._N = tf.compat.v1.get_variable('unused_variable',
                    dtype=_precision,
                    initializer=0.0)
                    
    @tf.function
    def __call__(self,X_train,Y_train):
        params={"sigma":self._sigma}
        return log_cholesky_l(X_train,Y_train,params,kernel="CONST")


    def predict(self,X_train,Y_train,X_s):
        cov = WhiteNoise(X_train,X_train,sigma=self._sigma)
        cov_ss =  WhiteNoise(X_s,X_s,sigma=self._sigma)
        cov_s  = WhiteNoise(X_train,X_s,sigma=self._sigma)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    @property
    def variables(self):
        return self._sigma,self._N


    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()))))+"{}".format(var.numpy()))


class CustomModel(object):
    def __init__(self,params,existing=None):
        for attr in params.keys() :
            pars = params[attr]
            for var in pars :
                if existing is  None :
                    self.__dict__[var] = tf.compat.v1.get_variable(var,
                            dtype=_precision,
                            shape=(1,),
                            initializer=tf.random_uniform_initializer(minval=1, maxval=10.))
                else :
                    if var in existing.keys() :
                        self.__dict__[var] = tf.Variable(existing[var],dtype=_precision)
                    else :
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                            dtype=_precision,
                            shape=(1,),
                            initializer=tf.random_uniform_initializer(minval=1, maxval=10.))

    @property
    def initialisation_values(self):
        return dict({k:v for k,v in zip(list(vars(self).keys()),list(vars(self).values()))})

    @property
    def variables(self):
        return vars(self).values()

    @property
    def _variables(self):
        return vars(self)
    @property
    def _opti_variables(self):
        return list(vars(self).values())
    
    @property
    def _opti_variables_name(self):
        return vars(self).keys()
    

    #@tf.function
    def __call__(self,X_train,Y_train,kernels_name):
        params=vars(self)
        return log_cholesky_l_test(X_train,Y_train,params,kernel=kernels_name)

    def viewVar(self,kernels):
        list_vars = self.variables
        print("\n Parameters of  : {}".format(kernels))
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()))))+"{}".format(var.numpy()))



    def predict(self,X_train,Y_train,X_s,kernels_name):
        params= self._variables
        X_train = tf.Variable(X_train,dtype=_precision)
        Y_train = tf.Variable(Y_train,dtype=_precision)
        X_s = tf.Variable(X_s,dtype=_precision)
        cov = self._get_cov(X_train,X_train,kernels_name,params)
        cov_ss =  self._get_cov(X_s,X_s,kernels_name,params)
        cov_s  =  self._get_cov(X_train,X_s,kernels_name,params)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    def _get_cov(self,X,Y,kernel,params):
        params_name = list(params.keys())
        cov = 0
        num = 0
        for op in kernel :
            if op[0] == "+":
                method = KERNELS_FUNCTIONS[op[1:]]
                par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError("Method %s not implemented" % op[1:])
                cov += method(X,Y,[params[p] for p in par])
                num += KERNELS_LENGTH[op[1:]]
            elif op[0] == "*":
                method = KERNELS_FUNCTIONS[op[1:]]
                par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError("Method %s not implemented" % op[1:])
                cov  = tf.math.multiply(cov,method(X,Y,[params[p] for p in par]))
                num += KERNELS_LENGTH[op[1:]]
        return cov


    def compute_BIC(self,X_train,Y_train,kernels_name):
        params= self._variables
        n =tf.Variable(X_train.shape[0],dtype=_precision)
        k = tf.Variable(len(kernels_name),dtype=_precision)
        ll = log_cholesky_l_test(X_train,Y_train,params,kernel=kernels_name)
        return  -ll - 0.5*k*tf.math.log(k)

    def plot(self,mu,cov,X_train,Y_train,X_s,kernel_name=None):
        try :
            Y_train,X_train,X_s = Y_train,X_train,X_s
        except Exception as e :
            print(e)
        mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=100)
        if kernel_name is not None : plt.title("kernel :"+''.join(kernel_name)[1:])
        plot_gs_pretty(Y_train,np.array(mean),X_train,X_s,np.array(stdp),np.array(stdi))
        plt.show()

        